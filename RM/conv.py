from torch import nn
import torch
from torch.nn.functional import softmax

class SoftGridEmbedding(nn.Module):
    def __init__(self, dim: int, resolution: tuple[int, int]):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.embed = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, device: torch.device):
        """Compute soft grid embeddings.

        Args:
            device (torch.device): CUDA device.

        Returns:
            torch.Tensor ~ [height, width, dim]: Grid embeddings.
        """
        rows = torch.linspace(-1, 1, self.resolution[0], device=device)
        cols = torch.linspace(-1, 1, self.resolution[1], device=device)
        base = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).flatten(0, 1)
        grid = self.embed(torch.cat([base, -base], dim=-1)).reshape(*self.resolution, -1)
        return grid

class ConvLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        **_
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, **_)
        self.act = nn.ReLU()
        self.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}, {self.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding}, act={self.act})'


    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

class ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        act = 'relu',
        **_
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, **_)
        self.act = nn.ReLU() if act == 'relu' else  nn.Identity()
        self.reset_parameters(act)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}, {self.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding}, act={self.act})'


    def reset_parameters(self, act):
        if act == 'relu':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SlotEncoder(nn.Module):
    def __init__(
        self, 
        img_size: tuple[int, int, int], 
        target_res:int,
        channels: list[int] = (64, 64, 64, 64),
        kernel_size: int = 5,
    ):
        super().__init__()
        self.img_size = img_size

        size = min(img_size[1:])
        dim = img_size[0]
        layers = []
        for hdim in channels:
            s = 2 if size > target_res else 1
            size = size // s
            layers += [
                ConvLayer(dim, hdim, kernel_size, padding=kernel_size//2, stride=s),
            ]
            dim = hdim 

        assert size == target_res, f"reached {size}, expected target_res={target_res}: add/remove layers"
        self.hidden_dim = channels[-1]
        self.output_size = (dim, target_res, target_res)
        self.layers = nn.Sequential(*layers)
        
        # positional information
        self.pemb = SoftGridEmbedding(dim=self.hidden_dim, resolution=self.resolution)
    
    @property
    def num_feats(self) -> int:
        return self.output_size[1]*self.output_size[2]
    
    @property
    def resolution(self) -> int:
        return (self.output_size[1], self.output_size[2])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x).permute(0, 2, 3, 1) # b, h, w, dim
        pos = self.pemb(x.device).unsqueeze(0)  # 1, h, w, dim
        return (x + pos).flatten(1, 2), x, pos

class SlotDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hid_res: tuple[int, int],
        target_res: int,
        channels: list[int] = (64, 64, 64, 64),
        out_ch: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()
        padding = kernel_size // 2

        layers = []
        in_ch = dim
        res = hid_res[0]
        for ch in channels:
            if res < target_res:
                stride, output_padding = 2, 1
                res *= 2
            else:
                stride, output_padding = 1, 0
            layers.append(ConvTransposeLayer(
                in_ch, ch, kernel_size,
                stride=stride, padding=padding, output_padding=output_padding
            ))
            in_ch = ch

        assert res == target_res, f"reached {res}, expected target_res={target_res}: add/remove layers"
        layers.append(ConvTransposeLayer(in_ch, out_ch + 1, 3, stride=1, padding=1, act='linear'))
        self.layers = nn.Sequential(*layers)
        self.pemb = SoftGridEmbedding(dim=dim, resolution=hid_res)
        self.hid_res = hid_res
        self.resolution = (target_res, target_res)
        self.out_ch = out_ch

    def forward(self, slot: torch.Tensor)  -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            slot (torch.Tensor ~ [batch_size, num_slots, hidden_dim]): Slot embeddings.

        Returns:
            torch.Tensor ~ [batch_size, output_dim, height, width]: Reconstructed image.
            torch.Tensor ~ [batch_size, num_slots, height, width, output_dim]: Reconstructed image per slot.
            torch.Tensor ~ [batch_size, num_slots, height, width, 1]: Normalized alpha masks per slot.
        """
        b, k = slot.shape[:2]
        x = slot.reshape(-1, slot.shape[-1], 1, 1).expand(-1, -1, *self.hid_res) + self.pemb(slot.device).permute(2, 0, 1).unsqueeze(0)
        recons, masks = self.layers(x).split([self.out_ch, 1], dim=1)
        # reshape recons & mask to -> b, s, h, w, c
        recons = recons.view(b, k, self.out_ch, *self.resolution).permute(0, 1, 3, 4, 2)
        masks = masks.view(b, k, 1, *self.resolution).permute(0, 1, 3, 4, 2)
        with torch.autocast(device_type='cuda', enabled=False):
            masks = softmax(masks.float(), dim=1)
        masks = masks.to(slot.dtype)
        recon_combined = (recons * masks).sum(dim=1).permute(0, 3, 1, 2) # b, c, h, w
        return recon_combined, recons, masks
