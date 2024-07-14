
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from model.mamba.mamba import BiMamba

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)


class MambaModule(nn.Module):
    def __init__(self, d_model=64, layer_idx=0, dtype=torch.float32, device=None):
        super(MambaModule, self).__init__()
        self.mamba = BiMamba(d_model=d_model, layer_idx=layer_idx, dtype=dtype, device=device)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layernorm(x)
        x= self.mamba(x)
        return x


class ConvMambaBlock(nn.Module):
    def __init__(self, d_model=64, layer_idx=0, dtype=torch.float32, device=None,
                 ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., ccm_dropout=0.):
        super(ConvMambaBlock, self).__init__()
        self.ffm1 = FeedForwardModule(d_model, ffm_mult, dropout=ffm_dropout)
        self.mamba = MambaModule(d_model=d_model, layer_idx=layer_idx, dtype=dtype, device=device)
        self.ccm = ConvModule(d_model, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(d_model, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.mamba(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x


def main():
    x = torch.ones(10, 100, 64)
    conformer = ConvMambaBlock(dim=64)
    print(conformer(x))


if __name__ == '__main__':
    main()
