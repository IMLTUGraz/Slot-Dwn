from torch import nn
import torch
from torch.nn.functional import softmax, relu
from conv import SlotDecoder, SlotEncoder

class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int, 
        input_dim: int,
        hidden_dim: int, 
        iters: int = 3, 
        eps: float = 1e-8,
        **_
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = hidden_dim ** -0.5
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, hidden_dim)))

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(input_dim, hidden_dim)
        self.to_v = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.norm_input  = nn.LayerNorm(input_dim)
        self.norm_slots  = nn.LayerNorm(hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(hidden_dim)
        self.register_buffer('fixed_slots', None)

    def freeze(self, num_slots=None, seed=None):
        """Fix slot initialization to a single sample drawn from the trained mu/sigma."""
        n_s = num_slots if num_slots is not None else self.num_slots
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.slots_mu.device).manual_seed(seed)
        mu = self.slots_mu.squeeze(0).expand(n_s, -1)
        sigma = self.slots_sigma.squeeze(0).expand(n_s, -1)
        self.fixed_slots = torch.normal(mu, sigma, generator=gen).detach()

    def unfreeze(self):
        self.fixed_slots = None

    def init(self, batch_size: int) -> torch.Tensor:
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = torch.normal(mu, sigma)
        return slots 
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x (torch.Tensor ~ [batch_size, num_feats, dim]): Encoder features and positions combined.

        Returns:
            torch.Tensor ~ [batch_size, num_slots, dim]: Slot embeddings.
            torch.Tensor ~ [batch_size, num_slots, num_feats]: Last unnormalized attention scores.
        """
        b = x.shape[0]
        if self.fixed_slots is not None:
            slots = self.fixed_slots.unsqueeze(0).expand(b, -1, -1)
        else:
            slots = self.init(b)
        x = self.norm_input(x) 
        k, v = self.to_k(x), self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            with torch.autocast(device_type='cuda', enabled=False):
                raw_attn = softmax(dots.float(), dim=1) + self.eps
                attn = raw_attn / raw_attn.sum(dim=-1, keepdim=True)
            attn = attn.to(v.dtype)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(updates.reshape(-1, self.hidden_dim), slots_prev.reshape(-1, self.hidden_dim))
            slots = slots.reshape(b, self.num_slots, self.hidden_dim)
            slots = slots + self.fc2(relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn, raw_attn
    

class SlotAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        enc_channels: list[int],
        dec_channels: list[int],
        target_res: int,
        slot_dim: int,
        num_slots: int,
        iters: int,
        dec_hid_res: int = None,
        **_
    ):
        super().__init__()
        self.img_size = img_size
        self.num_slots = num_slots
        self.iters = iters

        self.encoder = SlotEncoder(img_size, target_res=target_res, channels=enc_channels)
        hidden_dim = self.encoder.hidden_dim
        self.latent_res = self.encoder.resolution
        dec_res = (dec_hid_res, dec_hid_res) if dec_hid_res is not None else self.latent_res

        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.enc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # slot-attention layer
        self.attn = SlotAttention(num_slots=num_slots, input_dim=hidden_dim, hidden_dim=slot_dim,
                                  iters=iters)

        # create the decoder
        self.decoder = SlotDecoder(dim=slot_dim, hid_res=dec_res, target_res=img_size[1],
                                   channels=dec_channels, out_ch=img_size[0])

    def forward(self, img: torch.Tensor):
        """Forward pass.

        Args:
            imgs (torch.Tensor ~ [batch_size, input_dim, height, width]): Input images.

        Returns:
            torch.Tensor ~ [batch_size, input_dim, height, width]: Reconstructed images.
            torch.Tensor ~ [batch_size, num_slots, height, width, input_dim]: Reconstruction per slot.
            torch.Tensor ~ [batch_size, num_slots, height, width, 1]: Alpha masks per slot.
            torch.Tensor ~ [batch_size, num_slots, hidden_dim]: Encoder features pooled per slot.
            torch.Tensor ~ [batch_size, num_slots, latent_h, latent_w]: Unnormalized attention map, to use in post-processing of slots.
        """
        x, feat, _ = self.encoder(img)
        feat = feat.flatten(1, 2)

        y = self.enc_mlp(self.enc_norm(x))
        slots, attn, raw_attn = self.attn(y)
        feat_slots = torch.einsum('bjd,bij->bid', feat, attn)

        out = self.decoder(slots)
        h, w = self.latent_res
        attn_map = raw_attn.reshape(raw_attn.shape[0], raw_attn.shape[1], h, w)
        return (*out, feat_slots, attn_map)
    
    def get(self, img:torch.Tensor):    # slot encoding part only
        x, feat, _ = self.encoder(img)
        feat = feat.flatten(1, 2)

        y = self.enc_mlp(self.enc_norm(x))
        _, attn, raw_attn = self.attn(y)
        feat_slots = torch.einsum('bjd,bij->bid', feat, attn)
        h, w = self.latent_res
        attn_map = raw_attn.reshape(raw_attn.shape[0], raw_attn.shape[1], h, w)
        return feat_slots, attn_map


    def decode(self, slot: torch.Tensor) -> torch.Tensor:
        return self.decoder(slot)
    
    def freeze(self, seed=None):
        self.attn.freeze(seed=seed)

    def unfreeze(self):
        self.attn.unfreeze()
    
    