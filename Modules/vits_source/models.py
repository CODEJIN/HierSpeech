import copy
import math
import torch
from torch import nn

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import torchaudio

from typing import List, Tuple

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]
        n_ffts = [1024, 2048, 512, 300, 1200]

        discs = [STFT_Discriminator(n_fft= n_fft) for n_fft in n_ffts]
        discs = discs + [Period_Discriminator(period= i) for i in periods]
        
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Scale_Discriminator(torch.nn.Module):
    def __init__(
        self,
        channels_list: List[int]= [16, 64, 256, 1024, 1024, 1024],
        kernel_size_list: List[int]= [15, 41, 41, 41, 41, 5],
        stride_list: List[int]= [1, 4, 4, 4, 4, 1],
        groups_list: List[int]= [1, 4, 16, 64, 256, 1],
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        previous_channels = 1
        self.blocks = torch.nn.ModuleList()
        for channels, kernel_size, stride, groups in zip(
            channels_list,
            kernel_size_list,
            stride_list,
            groups_list,
            ):
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv1d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    stride= stride,
                    groups= groups,
                    padding= (kernel_size - 1) // 2
                    )),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels
        
        # Postnet
        self.blocks.append(torch.nn.utils.weight_norm(Conv1d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 3,
            padding= 1
            )))

    def forward(self, audios: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x = audios.unsqueeze(1) # [Batch, 1, Audio_t]
        x = audios
        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = x.flatten(start_dim= 1)
        
        return x, feature_maps

class Period_Discriminator(torch.nn.Module):
    def __init__(
        self,
        period,
        channels_list: List[int]= [32, 128, 512, 1024, 1024],
        kernel_size: int= 5,
        stride: int= 3,
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        self.period = period

        previous_channels = 1
        self.blocks = torch.nn.ModuleList()
        for channels in channels_list:
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv2d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= (kernel_size, 1),
                    stride= (stride, 1),
                    padding= ((kernel_size - 1) // 2, 0)
                    )),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels
        
        # Postnet
        self.blocks.append(torch.nn.utils.weight_norm(Conv2d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= (3, 1),
            padding= (1, 0)
            )))

    def forward(self, audios: torch.Tensor):
        x = audios

        # dividable by period
        if audios.size(2) % self.period != 0: 
            n_pad = self.period - (audios.size(2) % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(x.size(0), x.size(1), x.size(2) // self.period, self.period) # [Batch, 1, Audio_d // Period, Period]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = x.flatten(start_dim= 1)
        
        return x, feature_maps
    
class STFT_Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        channels_list: List[int]= [32, 128, 512, 1024, 1024],
        kernel_size: int= 5,
        stride: int= 3,
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        self.prenet = torchaudio.transforms.Spectrogram(
            n_fft= n_fft,
            hop_length= n_fft // 4,
            win_length= n_fft,
            window_fn=torch.hann_window,
            normalized= True,
            center= False,
            pad_mode= None,
            power= None,
            return_complex= True
            )

        previous_channels = 2
        self.blocks = torch.nn.ModuleList()
        for channels in channels_list:
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv2d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= kernel_size, # (kernel_size, 1),
                    stride= stride, # (stride, 1),
                    padding= (kernel_size - 1) // 2 # ((kernel_size - 1) // 2, 0)
                    )),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels
        
        # Postnet
        self.blocks.append(torch.nn.utils.weight_norm(Conv2d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= (3, 1),
            padding= (1, 0)
            )))

    def forward(self, audios: torch.Tensor):
        x = self.prenet(audios.squeeze(1)).permute(0, 2, 1)    # [Batch, Feature_t, Feature_d]
        x = torch.stack([x.real, x.imag], dim= 1)   # [Batch, 2, Feature_t, Feature_d]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = x.flatten(start_dim= 1)
        
        return x, feature_maps