# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from scipy.signal import get_window
import librosa
from librosa.util import pad_center

class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with hifigan """

    def __init__(self, hifigan, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros', **infer_kw):
        super().__init__()
        for name, p in hifigan.named_parameters():
            if name.endswith('.weight'):
                dtype = p.dtype
                device = p.device
                break

        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length/n_overlap),
                         win_length=win_length).to(device= device)

        mel_init = {'zeros': torch.zeros, 'normal': torch.randn}[mode]
        mel_input = mel_init((1, 80, 88), dtype=dtype, device=device)

        with torch.no_grad():
            bias_audio = hifigan(mel_input, **infer_kw).float() / 32768.0
            if len(bias_audio.size()) > 2:
                bias_audio = bias_audio.squeeze(0)
            elif len(bias_audio.size()) < 2:
                bias_audio = bias_audio.unsqueeze(0)
            assert len(bias_audio.size()) == 2

            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :].copy())

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

        # Compute the squared window at the desired length
        win_sq = get_window(self.window, self.win_length, fftbins=True)
        win_sq = librosa.util.normalize(win_sq, norm= None)**2
        win_sq = librosa.util.pad_center(win_sq, size= self.filter_length)
        self.register_buffer('win_sq', torch.from_numpy(win_sq))

    def transform(self, input_data):
        # similar to librosa, reflect-pad the input
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1).unsqueeze(2),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect').squeeze(1)

        forward_transform = torch.nn.functional.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        with torch.no_grad():
            inverse_transform = torch.nn.functional.conv_transpose1d(
                recombine_magnitude_phase, self.inverse_basis,
                stride=self.hop_length, padding=0)

        if self.window is not None:
            window_sum = self.window_sumsquare(torch.ones_like(magnitude[0, 0]).sum().long())

            # remove modulation effects
            approx_nonzero_indices = (window_sum > torch.finfo().tiny).nonzero().squeeze(1)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

    def window_sumsquare(self, n_frames):
        n_fft = torch.tensor(self.filter_length)
        hop_length = torch.tensor(self.hop_length)

        n = n_fft + hop_length * (n_frames - 1)
        x = torch.zeros(n, dtype= torch.float, device= n_frames.device)

        for i in torch.arange(0, n_frames, 1):
            sample = i * hop_length
            x[sample:torch.minimum(n, sample + n_fft)] += \
                self.win_sq[:torch.maximum(torch.tensor(0), torch.minimum(n_fft, n - sample))]

        return x
