"""Slot-based Differentiable Weightless Network (DWN) with multi-head LUT layers."""

import logging
from collections import OrderedDict
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from torch_dwn.binarization import DistributiveThermometer
from torch_dwn.lut_layer import LUTLayer
from torch_dwn.utils import GroupSum

logger = logging.getLogger(__name__)

Mapping = Union[str, torch.Tensor, Sequence]


class MultiHeadDWNLayer(nn.Module):
    """
    Several independent LUT layers over the same input, concatenated on the last dim.
    `mapping` may be a string, a single tensor shared by all heads, or a
    list/tuple with one mapping per head.
    """

    def __init__(self,
                 num_heads: int,
                 in_dim: int,
                 out_dim: int,
                 n: int,
                 mapping: Mapping = 'random',
                 ste: bool = True,
                 lm_tau: float = 0.001,
                 subset_lm_mode: str = 'random',
                 pool_size: Optional[int] = None):
        super().__init__()
        per_head = isinstance(mapping, (list, tuple))
        self.heads = nn.ModuleList(
            LUTLayer(input_size=in_dim, output_size=out_dim // num_heads, n=n,
                     mapping=mapping[h] if per_head else mapping, ste=ste, lm_tau=lm_tau,
                     subset_lm_mode=subset_lm_mode, pool_size=pool_size)
            for h in range(num_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class SlotDWN(nn.Module):
    """
    DWN classifier operating on object-centric slot representations.

    Slots are context-normalised, thermometer-encoded into `dwn_levels` bits per
    feature and fed to a stack of LUT layers. Two first-layer mappings are supported:

    * ``'paired'``: one shared set of LUTs is applied to every slot pair
      (all unique cross-object pairs if ``use_object_pairs`` else each slot with itself).
    * ``'one-to-one'``: every LUT sees the same bit of all slots.

    Input shape: ``(batch, n_slots, slot_dim)``
    Call `fit_binarization` before use.
    """

    def __init__(self,
                 slot_dim: int,
                 n_slots: int,
                 dwn_levels: int,
                 dwn_heads: int = 4,
                 dwn_num_layers: int = 2,
                 n: int = 6,
                 out_size: int = 1,
                 tau: float = 0.3,
                 lm_tau: float = 0.001,
                 mapping_type: str = 'paired',
                 dwn_hid: int = 1024,
                 first_layer_heads: int = 4,
                 use_subset_lm: bool = False,
                 subset_lm_mode: str = 'random',
                 pool_size: Optional[int] = None,
                 apply_context_norm: bool = True,
                 use_object_pairs: bool = True):
        super().__init__()
       
        if mapping_type not in ('paired', 'one-to-one'):
            raise ValueError(f"Mapping '{mapping_type}' not implemented!")

        self.n_levels = dwn_levels
        self.B_size = slot_dim * dwn_levels          # binary features per slot
        self.num_slots = n_slots
        self.mapping = mapping_type
        self.apply_ctx_norm = apply_context_norm
        self.use_object_pairs = use_object_pairs
        self.binarization: Optional[DistributiveThermometer] = None

        # First layer: weights shared across pairs ('paired') or across slots ('one-to-one').
        if mapping_type == 'paired':
            in_dim, n_first, first_map = 2 * self.B_size, 2, self._paired_mapping()
        else:
            in_dim, n_first, first_map = n_slots * self.B_size, n_slots, self._one_to_one_mapping()
        hid_dim = self.B_size * first_layer_heads
        layers = [MultiHeadDWNLayer(first_layer_heads, in_dim, hid_dim, n_first, first_map)]
        logger.info('Output neurons in dwn layer 0: %d', hid_dim)

        if mapping_type == 'paired':
            n_pairs = n_slots * (n_slots - 1) // 2 if use_object_pairs else n_slots
            hid_dim *= n_pairs
            pool_size = hid_dim // 8 if pool_size is None else pool_size
            logger.info('Pool size: %d', pool_size)

        # Remaining layers: learnable mapping first, random afterwards; width halves each layer.
        for i in range(dwn_num_layers - 1):
            in_dim, hid_dim = hid_dim, (dwn_hid if i == 0 else hid_dim // 2)
            if i > 0:
                mapping = 'random'
            else:
                mapping = 'subset-learnable' if use_subset_lm else 'learnable'
            layers.append(MultiHeadDWNLayer(dwn_heads, in_dim, hid_dim, n, mapping,
                                            lm_tau=lm_tau, subset_lm_mode=subset_lm_mode,
                                            pool_size=pool_size))
            logger.info('Output neurons in dwn layer %d: %d', i + 1, hid_dim)

        self.dwn = nn.Sequential(OrderedDict([
            ('layers', nn.Sequential(*layers)),
            ('class_head', GroupSum(k=out_size, tau=1 / tau)),
        ]))

    # ------------------------------------------------------------------ mappings
    def _one_to_one_mapping(self) -> torch.Tensor:
        """(B_size, n_slots): row i selects bit i of every slot."""
        offsets = torch.arange(self.num_slots) * self.B_size
        return (torch.arange(self.B_size).unsqueeze(1) + offsets).int()

    def _paired_mapping(self) -> torch.Tensor:
        """(B_size, 2): row k selects bit k of the left and of the right slot."""
        k = torch.arange(self.B_size, dtype=torch.int32)
        return torch.stack([k, k + self.B_size], dim=1)

    # ------------------------------------------------------------ preprocessing
    @staticmethod
    def context_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalise each feature across the slots"""
        if x.shape[-2] == 1:
            return x
        mu = x.mean(dim=-2, keepdim=True)
        sigma = (x.var(dim=-2, keepdim=True) + eps).sqrt()
        return (x - mu) / sigma

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_norm(x) if self.apply_ctx_norm else x

    def fit_binarization(self, x: torch.Tensor) -> None:
        """Fit the thermometer encoder on slots of shape (..., n_slots, slot_dim)."""
        x = self._preprocess(x)
        self.binarization = DistributiveThermometer(num_bits=self.n_levels).fit(x.flatten(end_dim=-2))

    def create_slot_pairs(self, x: torch.Tensor) -> torch.Tensor:
        """(..., n_slots, D) -> (..., n_pairs, 2D): concatenate left/right slots of each pair."""
        n = x.shape[-2]
        if self.use_object_pairs:
            i, j = torch.triu_indices(n, n, offset=1, device=x.device)   # unique pairs, i < j
        else:
            i = j = torch.arange(n, device=x.device)                      # slot with itself
        return torch.cat([x[..., i, :], x[..., j, :]], dim=-1)

    # ------------------------------------------------------------------- public
    def clip_luts(self) -> None:
        """Clip LUT weights of all layers (call after each optimiser step)."""
        self.dwn.layers.apply(lambda m: getattr(m, 'clip_weights', lambda: None)())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slot_bins = self.binarization.binarize(self._preprocess(x)).flatten(start_dim=-2)  # (b, slots, B_size)
        if self.mapping != 'paired':
            return self.dwn(slot_bins.flatten(start_dim=-2))

        # paired processing
        pairs = self.create_slot_pairs(slot_bins)                  # (b, n_pairs, 2*B_size)
        *batch_shape, n_pairs, pair_dim = pairs.shape
        first, rest = self.dwn.layers[0], self.dwn.layers[1:]
        # Same LUTs applied to every pair: fold pairs into the batch dim, then unfold into features.
        out = first(pairs.reshape(-1, pair_dim))
        out = out.reshape(*batch_shape, n_pairs, -1).flatten(start_dim=-2)
        return self.dwn.class_head(rest(out))
