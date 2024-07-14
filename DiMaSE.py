import torch
from torch import nn

from convmamba import ConvMambaBlock
import math
from torch.utils.checkpoint import checkpoint

def checkpoint_fn(module, input):
    #output = checkpoint(module, input)
    output = module(input)
    return output
    
def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), int((kernel_size[1] * dilation[1] - dilation[1]) / 2))


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True
        self.act = nn.Sigmoid() 

    def forward(self, x): 
        return self.beta * self.act(self.slope * x)


class DenseBlock(nn.Module):
    def __init__(self, dense_channel=64, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(dense_channel * (i + 1), dense_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, dense_channel=64, in_channel=2):
        super(DenseEncoder, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        self.dense_block = DenseBlock(dense_channel, depth=4)  # [b, dense_channel, ndim_time, n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self,discriminative=False, dense_channel=64, dim_freq=256, beta=2.0, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 4), (1, 2), (0, 1)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        
        self.lsigmoid = LearnableSigmoid_2d(dim_freq, beta=beta) if discriminative else nn.Conv1d(dim_freq, dim_freq, 1)

    def forward(self, x):  # [B, 64, T, F//2]
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)  # [B,1,T,F]
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, dense_channel=64, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 4), (1, 2), (0, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class TFMambaBlock(nn.Module):
    def __init__(self, discriminative=True, layer_idx=0, dense_channel=64, dim_freq = 256, bimamba_type='v1', skip=True, device=None):
        super(TFMambaBlock, self).__init__()
        self.discriminative = discriminative
        self.skip = skip
        
        self.time_mamba = ConvMambaBlock(d_model=dense_channel, layer_idx=layer_idx * 2, dtype=torch.float32, device=device)
        self.freq_mamba = ConvMambaBlock(d_model=dense_channel, layer_idx=layer_idx * 2 + 1, dtype=torch.float32, device=device)
        if not self.discriminative:
            self.time_proj = nn.Sequential(
                nn.Linear(dense_channel , dense_channel * 2),
                nn.SiLU(),
                nn.Linear(dense_channel * 2, dense_channel),
            )
        if self.skip:
            self.skip_linear = nn.Linear(dense_channel*2 , dense_channel)
            
    def forward(self, x, temb=None, skip=None):
        if self.skip and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=1).permute(0, 2, 3, 1).contiguous())
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
        B, T, F, C = x.shape   
        if not self.discriminative:
            temb = self.time_proj(temb)
            temb = temb.view(B, 1, 1, C).expand(B, T, F, C)
            x = x + temb
                   
        x = x.permute(0, 2, 1, 3).contiguous().view(B * F, T, C)
        x = x + checkpoint_fn(self.time_mamba,x) 
        x = x.view(B, F, T, C).permute(0, 2, 1, 3).contiguous().view(B * T, F, C)
        x = x + checkpoint_fn(self.freq_mamba,x)
        x = x.view(B, T, F, C).permute(0, 3, 1, 2)
        return x


class DiMaSE(nn.Module):

    def __init__(self, discriminative=True, dense_channel=64, dim_freq=256, num_block=4, device=None):
        super().__init__()
        self.num_block = num_block
        self.dense_encoder = DenseEncoder(dense_channel=dense_channel, in_channel=2)
        self.discriminative = discriminative
        
        if not self.discriminative:
            self.time_embedding = TimestepEmbedder(dense_channel)
            nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
            nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)
                    
        self.in_blocks = nn.ModuleList([])
        for i in range(num_block//2):
            self.in_blocks.append(TFMambaBlock(discriminative=discriminative, layer_idx=i, dense_channel=dense_channel, skip=False, device=device))

        self.mid_block = TFMambaBlock(discriminative=discriminative, layer_idx=num_block//2+1, dense_channel=dense_channel, skip=False, device=device)

        self.out_blocks = nn.ModuleList([])
        for i in range(num_block//2):
            self.out_blocks.append(TFMambaBlock(discriminative=discriminative, layer_idx=num_block//2+1+i, dense_channel=dense_channel, device=device))

        self.mask_decoder = MaskDecoder(discriminative=discriminative, dense_channel=dense_channel, dim_freq=dim_freq, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel=dense_channel, out_channel=1)

    def forward(self, x, t=None, cond=None):  # [B,1,F,T]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        mag, phase = torch.abs(x).permute(0, 1, 3, 2), torch.angle(x).permute(0, 1, 3, 2)  # [B,1,T,F]

        x = torch.cat([mag, phase], dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)  # [B, C, T, F//2]

        if not self.discriminative:
            temb = self.time_embedding(t)   
        else:
            temb = None

        skips = []
        for block in self.in_blocks:
            x = block(x, temb)
            skips.append(x)

        x =  self.mid_block(x, temb)

        for block in self.out_blocks:
            x = block(x, temb, skip=skips.pop())

        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 1, 3, 2)
        denoised_pha = self.phase_decoder(x).permute(0, 1, 3, 2)
        x = torch.complex(denoised_mag * torch.cos(denoised_pha),
                          denoised_mag * torch.sin(denoised_pha))
        return x


def trainable_parameters(model):
    print(f"Trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6 : .2f} M")


if __name__ == '__main__':
    model = DiMaSE(discriminative=True, dense_channel=196, num_block=4, device=self.device).to('cuda:0')
    trainable_parameters(model)

    x = torch.randn([4, 1, 256, 500], dtype=torch.complex64).to('cuda:0')
    t = torch.rand([4, 1]).to('cuda:0')
    x = model(x, t)
