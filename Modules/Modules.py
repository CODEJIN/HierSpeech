from argparse import Namespace
import torch, torchaudio, torchvision
import math
from typing import Optional, List, Dict, Tuple, Union

from transformers import Wav2Vec2ForCTC

from .LinearAttention import LinearAttention
from .Monotonic_Alignment_Search import Calc_Duration
from .Stochastic_Duration_Predictor import Stochastic_Duration_Predictor
from .Flow import FlowBlock, WaveNet
from .Layer import Conv1d, ConvTranspose1d, Linear, Lambda, RMSNorm
from meldataset import spectrogram_to_mel, mel_spectrogram

class HierSpeech(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.text_encoder = Text_Encoder(self.hp)
        self.speaker = torch.nn.Embedding(
            num_embeddings= self.hp.Speakers,
            embedding_dim= self.hp.Encoder.Size,
            )
        embedding_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.hp.Speakers + self.hp.Encoder.Size))
        self.speaker.weight.data.uniform_(-embedding_variance, embedding_variance)

        self.decoder = Decoder(self.hp)
        
        self.linguistic_encoder = Linguistic_Encoder(self.hp)
        self.linguistic_flow = FlowBlock(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            flow_stack= self.hp.Linguistic_Flow.Stack,
            flow_wavenet_conv_stack= self.hp.Linguistic_Flow.Conv_Stack,
            flow_wavenet_kernel_size= self.hp.Linguistic_Flow.Kernel_Szie,
            flow_wavnet_dilation_rate= self.hp.Linguistic_Flow.Dilation_Rate,
            flow_wavenet_dropout_rate= self.hp.Linguistic_Flow.Dropout_Rate,
            condition_channels= self.hp.Encoder.Size
            )

        self.acoustic_encoder = Acoustic_Encoder(self.hp)
        self.acoustic_flow = FlowBlock(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            flow_stack= self.hp.Acoustic_Flow.Stack,
            flow_wavenet_conv_stack= self.hp.Acoustic_Flow.Conv_Stack,
            flow_wavenet_kernel_size= self.hp.Acoustic_Flow.Kernel_Szie,
            flow_wavnet_dilation_rate= self.hp.Acoustic_Flow.Dilation_Rate,
            flow_wavenet_dropout_rate= self.hp.Acoustic_Flow.Dropout_Rate,
            condition_channels= self.hp.Encoder.Size
            )

        self.variance_block = Variance_Block(self.hp)

        self.token_predictor = Token_Predictor(self.hp)
        
        self.segment = Segment()

    def forward(
        self,
        tokens: torch.LongTensor,
        token_lengths: torch.LongTensor,
        speakers: torch.LongTensor,
        linear_spectrograms: torch.FloatTensor,
        linear_spectrogram_lengths: torch.LongTensor,
        audios: torch.FloatTensor
        ):
        encoding_means, encoding_log_stds, encodings = self.text_encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        speakers = self.speaker(speakers)
        
        linguistic_means, linguistic_log_stds = self.linguistic_encoder(audios, linear_spectrogram_lengths)
        linguistic_samples = linguistic_means + linguistic_log_stds.exp() * torch.randn_like(linguistic_log_stds)
        linguistic_flows = self.linguistic_flow(
            x= linguistic_samples,
            lengths= linear_spectrogram_lengths,
            conditions= speakers,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]
        
        acoustic_means, acoustic_log_stds = self.acoustic_encoder(linear_spectrograms, linear_spectrogram_lengths)
        acoustic_samples = acoustic_means + acoustic_log_stds.exp() * torch.randn_like(acoustic_log_stds)
        acoustic_flows = self.acoustic_flow(
            x= acoustic_samples,
            lengths= linear_spectrogram_lengths,
            conditions= speakers,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]

        durations, alignments = Calc_Duration(
            encoding_means= encoding_means,
            encoding_log_stds= encoding_log_stds,
            encoding_lengths= token_lengths,
            decodings= linguistic_flows,
            decoding_lengths= linear_spectrogram_lengths,
            )
        _, duration_losses = self.variance_block(
            encodings= encodings,
            encoding_lengths= token_lengths,
            conditions= speakers,
            durations= durations
            )

        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)

        acoustic_samples_slice, offsets = self.segment(
            patterns= acoustic_samples.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= linear_spectrogram_lengths
            )
        acoustic_samples_slice = acoustic_samples_slice.permute(0, 2, 1)    # [Batch, Enc_d, Feature_st]

        mels = spectrogram_to_mel(
            linear_spectrograms,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            win_size= self.hp.Sound.Frame_Length,
            fmin= 0,
            fmax= None,
            use_denorm= False
            )
        mels_slice, _ = self.segment(
            patterns= mels.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        mels_slice = mels_slice.permute(0, 2, 1)    # [Batch, Mel_d, Feature_st]
        
        audios_slice, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Frame_Shift,
            offsets= offsets * self.hp.Sound.Frame_Shift
            )   # [Batch, Audio_st(Feature_st * Hop_Size)]

        audio_predictions_slice = self.decoder(
            encodings= acoustic_samples_slice,
            lengths= torch.full_like(linear_spectrogram_lengths, self.hp.Train.Segment_Size)
            )

        mel_predictions_slice = mel_spectrogram(
            audio_predictions_slice,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Length,
            fmin= 0,
            fmax= None
            )

        token_predictions = self.token_predictor(
            encodings= linguistic_samples
            )

        return \
            audio_predictions_slice, audios_slice, mel_predictions_slice, mels_slice, \
            encoding_means, encoding_log_stds, linguistic_flows, linguistic_log_stds, \
            linguistic_means, linguistic_log_stds, acoustic_flows, acoustic_log_stds, \
            token_predictions, duration_losses, alignments

    def Inference(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        speakers: torch.LongTensor,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0
        ):
        length_scales = self.Scale_to_Tensor(tokens= tokens, scale= length_scales)

        encoding_means, encoding_log_stds, encodings = self.text_encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        speakers = self.speaker(speakers)

        alignments, _ = self.variance_block(
            encodings= encodings,
            encoding_lengths= token_lengths,
            conditions= speakers,
            length_scales= length_scales
            )
        feature_lengths = alignments.sum(dim= [1, 2])
        
        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)

        linguistic_samples = self.linguistic_flow(
            x= encoding_samples,
            lengths= feature_lengths,
            conditions= speakers,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        acoustic_samples = self.acoustic_flow(
            x= linguistic_samples,
            lengths= feature_lengths,
            conditions= speakers,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        audio_predictions = self.decoder(
            encodings= acoustic_samples,
            lengths= feature_lengths
            )

        return audio_predictions, alignments

    def Scale_to_Tensor(
        self,
        tokens: torch.Tensor,
        scale: Union[float, List[float], torch.Tensor]
        ):
        if isinstance(scale, float):
            scale = torch.FloatTensor([scale,]).unsqueeze(0).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, list):
            if len(scale) != tokens.size(0):
                raise ValueError(f'When scale is a list, the length must be same to the batch size: {len(scale)} != {tokens.size(0)}')
            scale = torch.FloatTensor(scale).unsqueeze(1).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, torch.Tensor):
            if scale.ndim != 2:
                raise ValueError('When scale is a tensor, ndim must be 2.')
            elif scale.size(0) != tokens.size(0):
                raise ValueError(f'When scale is a tensor, the dimension 0 of tensor must be same to the batch size: {scale.size(0)} != {tokens.size(0)}')
            elif scale.size(1) != tokens.size(1):
                raise ValueError(f'When scale is a tensor, the dimension 1 of tensor must be same to the token length: {scale.size(1)} != {tokens.size(1)}')

        return scale.to(tokens.device)

class Text_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size,
            )
        embedding_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.hp.Tokens + self.hp.Encoder.Size))
        self.token_embedding.weight.data.uniform_(-embedding_variance, embedding_variance)

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                feedforward_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                feedforward_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                )
            for index in range(self.hp.Encoder.Transformer.Stack)
            ])

        self.projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        tokens: [Batch, Time]
        '''
        encodings = self.token_embedding(tokens).permute(0, 2, 1)
        
        for block in self.blocks:
            encodings = block(encodings, lengths)

        means, stds = self.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        log_stds = torch.nn.functional.softplus(stds).log()

        return means, log_stds, encodings

class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        feedforward_kernel_size: int,
        feedforward_dropout_rate: float= 0.1
        ) -> None:
        super().__init__()

        self.attention = LinearAttention(
            query_channels= channels,
            key_channels= channels,
            value_channels= channels,
            calc_channels= channels,
            num_heads= num_head
            )
        
        self.ffn = FFN(
            channels= channels,
            kernel_size= feedforward_kernel_size,
            dropout_rate= feedforward_dropout_rate
            )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())    # [Batch, Time]
        float_masks = (~masks).unsqueeze(1).float()   # float mask

        # Attention + Dropout + Norm
        x = self.attention(
            queries= x,
            keys= x,
            values= x,
            key_padding_masks= masks
            )
        
        # FFN + Dropout + Norm
        x = self.ffn(x, float_masks)

        return x * float_masks

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'relu'
            )
        self.silu = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm = RMSNorm(
            num_features= channels,
            )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        residuals = x

        x = self.conv_0(x * masks)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)
        x = self.dropout(x)
        x = self.norm(x + residuals)

        return x * masks


class Decoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Decoder.Upsample.Base_Size,
            kernel_size= self.hp.Decoder.Prenet.Kernel_Size,
            padding= (self.hp.Decoder.Prenet.Kernel_Size - 1) // 2,
            # w_init_gain= 'leaky_relu'   # Don't use this line.
            )

        self.upsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Decoder.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Decoder.Upsample.Rate,
            self.hp.Decoder.Upsample.Kernel_Size
            )):
            current_channels = self.hp.Decoder.Upsample.Base_Size // (2 ** (index + 1))
            upsample_block = torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ),
                torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
                    in_channels= previous_channels,
                    out_channels= current_channels,
                    kernel_size= kernel_size,
                    stride= upsample_rate,
                    padding= (kernel_size - upsample_rate) // 2
                    ))
                )
            self.upsample_blocks.append(upsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Decoder.Residual_Block.Kernel_Size,
                self.hp.Decoder.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Decoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.postnet = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Decoder.Postnet.Kernel_Size,
                padding= (self.hp.Decoder.Postnet.Kernel_Size - 1) // 2,
                bias= False
                ),
            torch.nn.Tanh(),
            Lambda(lambda x: x.squeeze(1))
            )

        # This is critical when using weight normalization.
        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)
        self.upsample_blocks.apply(weight_norm_initialize_weight)
        self.residual_blocks.apply(weight_norm_initialize_weight)
            
    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(encodings[0, 0]).sum())).unsqueeze(1).float()

        decodings = self.prenet(encodings) * masks
        for upsample_block, residual_blocks, upsample_rate in zip(
            self.upsample_blocks,
            self.residual_blocks,
            self.hp.Decoder.Upsample.Rate
            ):
            decodings = upsample_block(decodings)
            lengths = lengths * upsample_rate
            masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(decodings[0, 0]).sum())).unsqueeze(1).float()
            decodings = torch.stack(
                [block(decodings, masks) for block in residual_blocks],
                # [block(decodings) for block in residual_block],
                dim= 1
                ).mean(dim= 1)
            
        predictions = self.postnet(decodings)

        return predictions

class Decoder_Residual_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Union[List, Tuple],
        negative_slope: float= 0.1
        ):
        super().__init__()

        self.in_convs = torch.nn.ModuleList()
        self.out_convs = torch.nn.ModuleList()
        for dilation in dilations:
            self.in_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= (kernel_size * dilation - dilation) // 2
                )))
            self.out_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels,
                kernel_size= kernel_size,
                dilation= 1,
                padding= (kernel_size - 1) // 2
                )))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope= negative_slope)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ):
        for in_conv, out_conv in zip(self.in_convs, self.out_convs):
            residuals = x
            x = self.leaky_relu(x) * masks
            x = in_conv(x) * masks
            x = self.leaky_relu(x) * masks
            x = out_conv(x) * masks
            x = x + residuals
        
        return x * masks


class Acoustic_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv1d(
            in_channels= self.hp.Sound.N_FFT // 2 + 1,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        self.wavenet = WaveNet(
            calc_channels= self.hp.Encoder.Size,
            conv_stack= self.hp.Acoustic_Encoder.Conv_Stack,
            kernel_size= self.hp.Acoustic_Encoder.Kernel_Size,
            dilation_rate= self.hp.Acoustic_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Acoustic_Encoder.Dropout_Rate,
            )        
        self.projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
            
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        features: [Batch, Feature_d, Feature_t], Spectrogram
        lengths: [Batch]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(features[0, 0]).sum())).unsqueeze(1).float()  # [Batch, 1, Feature_t]

        encodings = self.prenet(features) * masks   # [Batch, Acoustic_d, Feature_t]
        encodings = self.wavenet(encodings, masks)
        means, stds = self.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        log_stds = torch.nn.functional.softplus(stds).log()
        
        return means, log_stds

class Variance_Block(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.duration_predictor = Stochastic_Duration_Predictor(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Duration_Predictor.Kernel_Size,
            conv_stack= self.hp.Duration_Predictor.Conv_Stack,
            flow_stack= self.hp.Duration_Predictor.Flow_Stack,
            dropout_rate= self.hp.Duration_Predictor.Dropout_Rate,
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        conditions: torch.Tensor,
        durations: Optional[torch.Tensor]= None,
        length_scales: Optional[torch.Tensor]= None,  # None when training
        ):
        if not conditions is None and conditions.ndim == 2:
            conditions = conditions.unsqueeze(2)    # [Batch, Cond_d, 1]

        encodings = (encodings + conditions).detach()
        if not durations is None:
            alignments = None
            _, duration_loss = self.duration_predictor(
                encodings= encodings,
                encoding_lengths= encoding_lengths,
                durations= durations,
                reverse= False
                )
        else:
            duration_loss = None
            durations, _ = self.duration_predictor(
                encodings= encodings,
                encoding_lengths= encoding_lengths,
                reverse= True
                )
            if not length_scales is None:
                durations = durations * length_scales
            alignments = self.Length_Regulate(durations)

        return alignments, duration_loss
    
    def Length_Regulate(self, durations):
        """If target=None, then predicted durations are applied"""
        repeats = (durations.float() + 0.5).long()
        decoding_lengths = repeats.sum(dim=1)

        max_decoding_length = decoding_lengths.max()
        reps_cumsum = torch.cumsum(torch.nn.functional.pad(repeats, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]

        range_ = torch.arange(max_decoding_length)[None, :, None].to(durations.device)
        alignments = ((reps_cumsum[:, :, :-1] <= range_) &
                (reps_cumsum[:, :, 1:] > range_))
        
        return alignments.float()

class Variance_Predictor(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        lstm_features: int,
        lstm_stack: int,
        dropout_rate: float,
        ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size= in_features,
            hidden_size= lstm_features,
            num_layers= lstm_stack,
            bidirectional= True
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= dropout_rate
            )

        self.projection = torch.nn.Sequential(
            Linear(
                in_features= lstm_features * 2,
                out_features= 1,
                w_init_gain= 'linear'
                ),
            Lambda(lambda x: x.squeeze(2))
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        '''
        unpacked_length = encodings.size(2)

        encodings = encodings.permute(2, 0, 1)    # [Enc_t, Batch, Enc_d]        
        if self.training:
            encodings = torch.nn.utils.rnn.pack_padded_sequence(
                encodings,
                encoding_lengths.cpu().numpy(),
                enforce_sorted= False
                )
        
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0]

        if self.training:
            encodings = torch.nn.utils.rnn.pad_packed_sequence(
                sequence= encodings,
                total_length= unpacked_length
                )[0]
        
        encodings = encodings.permute(1, 0, 2)    # [Batch, Enc_t, Enc_d]
        encodings = self.lstm_dropout(encodings)

        variances = self.projection(encodings)  # [Batch, Enc_t]

        return variances

class Segment(torch.nn.Module):
    def forward(
        self,
        patterns: torch.Tensor,
        segment_size: int,
        lengths: torch.Tensor= None,
        offsets: torch.Tensor= None
        ):
        '''
        patterns: [Batch, Time, ...]
        lengths: [Batch]
        segment_size: an integer scalar
        '''
        if offsets is None:
            offsets = (torch.rand_like(patterns[:, 0, 0]) * (lengths - segment_size)).long()
        segments = torch.stack([
            pattern[offset:offset + segment_size]
            for pattern, offset in zip(patterns, offsets)
            ], dim= 0)
        
        return segments, offsets



class Linguistic_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        
        self.hp = hyper_parameters
        wav2vec2_feature_size = 1024
        
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-xls-r-300m')
        self.feature_extractor = lambda x: self.wav2vec2(x, output_hidden_states= True).hidden_states[11]
        self.norm = RMSNorm(
            num_features= wav2vec2_feature_size
            )
        
        self.prenet = Conv1d(
            in_channels= wav2vec2_feature_size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        self.wavenet = WaveNet(
            calc_channels= self.hp.Encoder.Size,
            conv_stack= self.hp.Linguistic_Encoder.Conv_Stack,
            kernel_size= self.hp.Linguistic_Encoder.Kernel_Size,
            dilation_rate= self.hp.Linguistic_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Linguistic_Encoder.Dropout_Rate,
            )        
        self.projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

    def forward(
        self,
        audios: torch.Tensor,
        lengths: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        audios: [Batch, Audio_t], Raw waveform
        lengths: [Batch], * This is feature length, not waveform length.
        '''
        with torch.no_grad():
            audios = torchaudio.functional.resample(audios, self.hp.Sound.Sample_Rate, 16000)
            features = self.feature_extractor(audios)
            features = torchvision.transforms.functional.resize(
                features.unsqueeze(1),
                [lengths.max(), 1024]
                ).squeeze(1).permute(0, 2, 1)
        features = self.norm(features)
        
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(features[0, 0]).sum())).unsqueeze(1).float()  # [Batch, 1, Feature_t]

        encodings = self.prenet(features) * masks   # [Batch, Acoustic_d, Feature_t]
        encodings = self.wavenet(encodings, masks)
        means, stds = self.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        log_stds = torch.nn.functional.softplus(stds).log()
        
        return means, log_stds

    def train(self, mode: bool= True):
        super().train(mode)
        self.wav2vec2.eval()

class Token_Predictor(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.lstm = torch.nn.LSTM(
            input_size= self.hp.Encoder.Size,
            hidden_size= self.hp.Token_Predictor.Size,
            num_layers= self.hp.Token_Predictor.LSTM.Stack,
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= self.hp.Token_Predictor.LSTM.Dropout_Rate,
            )

        self.projection = Conv1d(
            in_channels= self.hp.Token_Predictor.Size,
            out_channels= self.hp.Tokens + 1,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
            
    def forward(
        self,
        encodings: torch.Tensor,
        ) -> torch.Tensor:
        '''
        features: [Batch, Feature_d, Feature_t], Spectrogram
        lengths: [Batch]
        '''
        encodings = encodings.permute(2, 0, 1)    # [Feature_t, Batch, Enc_d]
        
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0] # [Feature_t, Batch, LSTM_d]
        
        predictions = self.projection(encodings.permute(1, 2, 0))
        predictions = torch.nn.functional.log_softmax(predictions, dim= 1)

        return predictions

def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]
