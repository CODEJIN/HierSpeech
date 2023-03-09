import torch, torchaudio
from torch.nn import Conv1d, Conv2d

from typing import List, Tuple

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

LRELU_SLOPE = 0.1

class Discriminator(torch.nn.Module):
    def __init__(
        self,
        stft_n_fft_list: List[int],
        stft_win_size_list: List[int],
        stft_channels_list: List[int]= [32, 32, 32, 32, 32],
        stft_kernel_size_list: List[Tuple[int, int]] = [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3)],
        stft_stride_list: List[Tuple[int, int]] = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1)],
        stft_dilation_list: List[Tuple[int, int]] = [(1, 1), (1, 1), (2, 1), (4, 1), (1, 1)],
        scale_channels_list: List[int]= [128, 128, 256, 512, 1024, 1024, 1024],
        scale_kernel_size_list: List[int] = [15, 41, 41, 41, 41, 41, 5],
        scale_stride_list: List[int] = [1, 2, 2, 4, 4, 1, 1],
        scale_gropus_list: List[int] = [1, 4, 16, 16, 16, 16, 1],
        leaky_relu_negative_slope: float= 0.2
        ):
        super().__init__()

        self.multi_stft_discriminator =  Multi_STFT_Discriminator(
            n_fft_list= stft_n_fft_list,
            win_size_list= stft_win_size_list,
            channels_list= stft_channels_list,
            kernel_size_list= stft_kernel_size_list,
            stride_list= stft_stride_list,
            dilation_list= stft_dilation_list,
            leaky_relu_negative_slope= leaky_relu_negative_slope
            )
        self.scale_discriminator = Scale_Discriminator(
            channels_list= scale_channels_list,
            kernel_size_list= scale_kernel_size_list,
            stride_list= scale_stride_list,
            gropus_list= scale_gropus_list,
            leaky_relu_negative_slope= leaky_relu_negative_slope,
            )
        
    def forward(self, audios: torch.Tensor):
        discriminations_list, feature_maps_list = [], []
        
        discriminations, feature_maps = self.multi_stft_discriminator.forward(audios)
        discriminations_list.extend(discriminations)
        feature_maps_list.extend(feature_maps)

        discriminations, feature_maps = self.scale_discriminator.forward(audios)
        discriminations_list.append(discriminations)
        feature_maps_list.extend(feature_maps)

        return discriminations_list, feature_maps_list

class STFT_Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_size: int,
        channels_list: List[int]= [32, 32, 32, 32, 32],
        kernel_size_list: List[Tuple[int, int]] = [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3)],
        stride_list: List[Tuple[int, int]] = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1)],
        dilation_list: List[Tuple[int, int]] = [(1, 1), (1, 1), (2, 1), (4, 1), (1, 1)],
        leaky_relu_negative_slope: float= 0.2
        ):
        super().__init__()
        self.n_fft = n_fft
        self.win_size = win_size
        hop_size = win_size // 4

        self.prenet = torchaudio.transforms.Spectrogram(
            n_fft= n_fft,
            hop_length= hop_size,
            win_length= win_size,
            window_fn=torch.hann_window,
            normalized= True,
            center= False,
            pad_mode= None,
            power= None
            )

        self.blocks = torch.nn.ModuleList()
        previous_channels= 2    # real + imag
        for channels, kernel_size, stride, dilation in zip(
            channels_list,
            kernel_size_list,
            stride_list,
            dilation_list
            ):
            block = torch.nn.Sequential(
                Conv2d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    stride= stride,
                    dilation= dilation,
                    padding= (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)
                    ),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels

        self.postnet = Conv2d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 3,
            padding= 1
            )
        
    def forward(self, audios: torch.Tensor):
        x = self.prenet(audios).unsqueeze(1)   # [Batch, 1, Feature_d, Feature_t]
        x = torch.cat([x.real, x.imag], dim= 1).permute(0, 1, 3, 2)   # [Batch, 2, Feature_t, Feature_d]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = self.postnet(x).flatten(start_dim= 1)

        return x, feature_maps

class Multi_STFT_Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_fft_list: List[int],
        win_size_list: List[int],
        channels_list: List[int]= [32, 32, 32, 32, 32],
        kernel_size_list: List[Tuple[int, int]] = [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3)],
        stride_list: List[Tuple[int, int]] = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1)],
        dilation_list: List[Tuple[int, int]] = [(1, 1), (1, 1), (2, 1), (4, 1), (1, 1)],
        leaky_relu_negative_slope: float= 0.2
        ):
        super().__init__()

        self.discriminators = torch.nn.ModuleList()
        for n_fft, win_size in zip(n_fft_list, win_size_list):
            self.discriminators.append(STFT_Discriminator(
                n_fft= n_fft,
                win_size= win_size,
                channels_list= channels_list, 
                kernel_size_list= kernel_size_list,
                stride_list= stride_list,
                dilation_list= dilation_list,
                leaky_relu_negative_slope= leaky_relu_negative_slope
                ))
            
    def forward(self, audios: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        discriminations_list = []
        feature_maps_list = []
        for discriminator in self.discriminators:
            discriminations, feature_maps = discriminator(audios)
            discriminations_list.append(discriminations)
            feature_maps_list.extend(feature_maps)

        return discriminations_list, feature_maps_list

class Scale_Discriminator(torch.nn.Module):
    def __init__(
        self,      
        channels_list: List[int]= [128, 128, 256, 512, 1024, 1024, 1024],
        kernel_size_list: List[int] = [15, 41, 41, 41, 41, 41, 5],
        stride_list: List[int] = [1, 2, 2, 4, 4, 1, 1],
        gropus_list: List[int] = [1, 4, 16, 16, 16, 16, 1],
        leaky_relu_negative_slope: float= 0.2
        ):
        super().__init__()

        previous_channels = 1
        self.blocks = torch.nn.ModuleList()
        for channels, kernel_size, stride, groups in zip(
            channels_list,
            kernel_size_list,
            stride_list,
            gropus_list,
            ):
            block = torch.nn.Sequential(
                Conv1d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    stride= stride,
                    groups= groups,
                    padding= (kernel_size - 1) // 2
                    ),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels

        self.postnet = Conv1d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 3,
            padding= 1
            )

    def forward(self, audios: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = audios.unsqueeze(1) # [Batch, 1, Audio_t]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = self.postnet(x).flatten(start_dim= 1)
        
        return x, feature_maps

def Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake):
    return torch.stack([
        torch.mean(torch.abs(feature_maps_for_real - feature_maps_for_fake))
        for feature_maps_for_real, feature_maps_for_fake in zip(
            feature_maps_list_for_real,
            feature_maps_list_for_fake
            )
        ]).sum() * 2.0


def Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake):
    return torch.stack([
        (1 - discriminations_for_real).pow(2.0).mean() + discriminations_for_fake.pow(2.0).mean()
        for discriminations_for_real, discriminations_for_fake in zip(
            discriminations_list_for_real,
            discriminations_list_for_fake
            )
        ]).sum()


def Generator_Loss(discriminations_list_for_fake):
    return torch.stack([
        (1 - discriminations_for_fake).pow(2.0).mean()
        for discriminations_for_fake in discriminations_list_for_fake
        ]).sum()

class R1_Regulator(torch.nn.Module):
    def forward(
        self,
        discriminations_list: List[torch.Tensor],
        audios: torch.Tensor
        ):
        x = torch.autograd.grad(
            outputs= [
                discriminations.sum()
                for discriminations in discriminations_list
                ],
            inputs= audios,
            create_graph= True,
            retain_graph= True,
            only_inputs= True
            )[0].pow(2)
        x = (x.view(audios.size(0), -1).norm(2, dim=1) ** 2).mean()

        return x