from argparse import Namespace
import torch, torchaudio, torchvision
import math
from typing import Optional, List, Dict, Tuple, Union

from transformers import Wav2Vec2ForCTC

from .Nvidia_Alignment_Leraning_Framework import Alignment_Learning_Framework
from .Gaussian_Upsampler import Gaussian_Upsampler
from .Flow import FlowBlock
from .Layer import Conv1d, ConvTranspose1d, Linear, Lambda, LayerNorm

class HierSpeech(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        if self.hp.Feature_Type == 'Spectrogram':
            feature_size = self.hp.Sound.N_FFT // 2 + 1
        elif self.hp.Feature_Type == 'Mel':
            feature_size = self.hp.Sound.Mel_Dim
        else:
            raise ValueError('Unknown feature type: {}'.format(self.hp.Feature_Type))

        self.text_encoder = Text_Encoder(self.hp)
        self.alignment_learning_framework = Alignment_Learning_Framework(
            feature_size= feature_size,
            encoding_size= self.hp.Encoder.Size
            )
        self.variance_block = Variance_Block(self.hp)

        self.linguistic_encoder = Linguistic_Encoder(self.hp)
        self.linguistic_flow = FlowBlock(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            flow_stack= self.hp.Linguistic_Flow.Stack,
            flow_wavenet_conv_stack= self.hp.Linguistic_Flow.Conv_Stack,
            flow_wavenet_kernel_size= self.hp.Linguistic_Flow.Kernel_Szie,
            flow_wavnet_dilation_rate= self.hp.Linguistic_Flow.Dilation_Rate,
            flow_wavenet_dropout_rate= self.hp.Linguistic_Flow.Dropout_Rate
            )
        self.token_predictor = Token_Predictor(self.hp)
        
        self.acoustic_encoder = Acoustic_Encoder(self.hp)
        self.acoustic_flow = FlowBlock(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            flow_stack= self.hp.Acoustic_Flow.Stack,
            flow_wavenet_conv_stack= self.hp.Acoustic_Flow.Conv_Stack,
            flow_wavenet_kernel_size= self.hp.Acoustic_Flow.Kernel_Szie,
            flow_wavnet_dilation_rate= self.hp.Acoustic_Flow.Dilation_Rate,
            flow_wavenet_dropout_rate= self.hp.Acoustic_Flow.Dropout_Rate
            )
        self.segment = Segment()
        self.decoder = Decoder(self.hp)

    def forward(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        features: torch.FloatTensor= None,
        feature_lengths: torch.Tensor= None,
        audios: torch.Tensor= None,
        linear_spectrograms: torch.Tensor= None,
        attention_priors: torch.Tensor= None,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0,
        ):
        if not features is None and not feature_lengths is None:    # train
            return self.Train(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths,
                audios= audios,
                linear_spectrograms= linear_spectrograms,
                attention_priors= attention_priors
                )
        else:   #  inference
            return self.Inference(
                tokens= tokens,
                token_lengths= token_lengths,
                length_scales= length_scales
                )

    def Train(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        features: torch.FloatTensor,
        feature_lengths: torch.Tensor,
        audios: torch.Tensor,
        linear_spectrograms: torch.Tensor,
        attention_priors: torch.Tensor,
        ):
        encoding_means, encoding_stds, encodings = self.text_encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        durations, attention_softs, attention_hards, attention_logprobs = self.alignment_learning_framework(
            token_embeddings= self.text_encoder.token_embedding(tokens).permute(0, 2, 1),
            encodings= encodings,
            encoding_lengths= token_lengths,
            features= features,
            feature_lengths= feature_lengths,
            attention_priors= attention_priors
            )
        alignments, log_duration_predictions, durations = self.variance_block(
            encodings= encodings,
            encoding_lengths= token_lengths,
            durations= durations
            )

        encoding_distributions_prior = torch.distributions.Normal(
            loc= encoding_means @ alignments,
            scale= torch.clamp(encoding_stds @ alignments, min= 1e-5)
            )   # [Batch, Enc_d, Enc_t] @ [Batch, Enc_t, Feature_t] -> [Batch, Enc_d, Feature_t]

        linguistic_distributions_prior = self.linguistic_encoder(
            audios= audios,
            lengths= feature_lengths
            )
        lingustic_samples = linguistic_distributions_prior.rsample()
        linguistic_flows = self.linguistic_flow(
            x= lingustic_samples,
            lengths= feature_lengths,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]
        encoding_distributions_posterior = torch.distributions.Normal(
            loc= linguistic_flows,
            scale= linguistic_distributions_prior.scale
            )
        
        token_predictions = self.token_predictor(
            encodings= lingustic_samples,
            lengths= feature_lengths
            )
        
        acoustic_distributions = self.acoustic_encoder(
            features= linear_spectrograms,
            lengths= feature_lengths
            )
        acoustic_samples = acoustic_distributions.rsample() # [Batch, Enc_d, Feature_t]
        acoustic_flows = self.acoustic_flow(
            x= acoustic_samples,
            lengths= feature_lengths,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]
        linguistic_distributions_posterior = torch.distributions.Normal(
            loc= acoustic_flows,
            scale= acoustic_distributions.scale
            )

        acoustic_samples_slice, offsets = self.segment(
            patterns= acoustic_samples.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= feature_lengths
            )        
        acoustic_samples_slice = acoustic_samples_slice.permute(0, 2, 1)    # [Batch, Enc_d, Feature_st]

        audios_slice, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Frame_Shift,
            offsets= offsets * self.hp.Sound.Frame_Shift
            )   # [Batch, Audio_st(Feature_st * Hop_Size)]

        audio_predictions_slice = self.decoder(
            encodings= acoustic_samples_slice,
            lengths= torch.full_like(feature_lengths, self.hp.Train.Segment_Size)
            )
        
        return \
            audio_predictions_slice, audios_slice, token_predictions, acoustic_distributions, \
            encoding_distributions_prior, encoding_distributions_posterior, linguistic_distributions_prior, linguistic_distributions_posterior, \
            log_duration_predictions, durations, attention_softs, attention_hards, attention_logprobs

    def Inference(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        length_scales: Union[float, List[float], torch.Tensor]= 1.0
        ):
        length_scales = self.Scale_to_Tensor(tokens= tokens, scale= length_scales)

        encoding_means, encoding_stds, encodings = self.text_encoder(
            tokens= tokens,
            lengths= token_lengths
            )
        alignments, _, durations = self.variance_block(
            encodings= encodings,
            encoding_lengths= token_lengths,
            length_scales= length_scales
            )
        feature_lengths = torch.stack([
            duration[:token_length].sum()
            for duration, token_length in zip(durations, token_lengths)
            ])

        encoding_distributions = torch.distributions.Normal(
            loc= encoding_means @ alignments,
            scale= torch.clamp(encoding_stds @ alignments, min= 1e-3)
            )   # [Batch, Enc_d, Enc_t] @ [Batch, Enc_t, Feature_t] -> [Batch, Enc_d, Feature_t]
        
        linguistic_samples = self.linguistic_flow(
            x= encoding_distributions.rsample(),
            lengths= feature_lengths,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]
        acoustic_samples = self.acoustic_flow(
            x= linguistic_samples,
            lengths= feature_lengths,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]
        
        audio_predictions = self.decoder(
            encodings= acoustic_samples,
            lengths= feature_lengths
            )

        return \
            audio_predictions, None, None, \
            None, None, None, None, \
            durations, None, None, None

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


class Feature_Encoder(torch.nn.Module): 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavenet_stack: int,
        conv_stack: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float
        ):
        super().__init__()

        self.prenet = Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            )

        self.wavenet_block = torch.nn.ModuleList([
            WaveNet(
                calc_channels= out_channels,
                conv_stack= conv_stack,
                kernel_size= kernel_size,
                dilation_rate= dilation_rate,
                dropout_rate= dropout_rate,
                )
            for _ in range(wavenet_stack)
            ])
        
        self.projection = Conv1d(
            in_channels= out_channels,
            out_channels= out_channels * 2,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
            
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.distributions.Normal:
        '''
        features: [Batch, Feature_d, Feature_t], Spectrogram
        lengths: [Batch]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(features[0, 0]).sum())).unsqueeze(1).float()  # [Batch, 1, Feature_t]

        encodings = self.prenet(features) * masks   # [Batch, Acoustic_d, Feature_t]
        for block in self.wavenet_block:
            encodings = block(encodings, masks)

        means, stds = self.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        stds = torch.nn.functional.softplus(stds)
        distributions = torch.distributions.Normal(loc= means, scale= stds) # distribution of [Batch, Acoustic_d, Feature_t]

        return distributions

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        calc_channels: int,
        conv_stack: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float= 0.0
        ):
        super().__init__()
        self.conv_stack = conv_stack
        
        self.input_convs = torch.nn.ModuleList()
        self.residual_and_skip_convs = torch.nn.ModuleList()
        for index in range(conv_stack):
            dilation = dilation_rate ** index
            padding = (kernel_size - 1) * dilation // 2
            self.input_convs.append(Conv1d(
                in_channels= calc_channels,
                out_channels= calc_channels * 2,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= padding,
                w_init_gain= 'gate'
                ))
            self.residual_and_skip_convs.append(Conv1d(
                in_channels= calc_channels,
                out_channels= calc_channels * 2,
                kernel_size= 1,
                w_init_gain= 'linear'
                ))
        
        self.dropout = torch.nn.Dropout(p= dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ):
        skips_list = []
        for in_conv, residual_and_skip_conv in zip(self.input_convs, self.residual_and_skip_convs):
            ins = in_conv(x)
            ins_tanh, ins_sigmoid = ins.chunk(chunks= 2, dim= 1)
            acts = ins_tanh.tanh() * ins_sigmoid.sigmoid()
            acts = self.dropout(acts)
            residuals, skips = residual_and_skip_conv(acts).chunk(chunks= 2, dim= 1)
            x = (x + residuals) * masks
            skips_list.append(skips)

        skips = torch.stack(skips_list, dim= 1).sum(dim= 1) * masks

        return skips

class Acoustic_Encoder(Feature_Encoder):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        self.hp = hyper_parameters

        super().__init__(
            in_channels= self.hp.Sound.N_FFT // 2 + 1,  # Spectrogram
            out_channels= self.hp.Encoder.Size,
            wavenet_stack= self.hp.Acoustic_Encoder.WaveNet_Stack,
            conv_stack= self.hp.Acoustic_Encoder.Conv_Stack,
            kernel_size= self.hp.Acoustic_Encoder.Kernel_Size,
            dilation_rate= self.hp.Acoustic_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Acoustic_Encoder.Dropout_Rate,
            )

class Linguistic_Encoder(Feature_Encoder):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        self.hp = hyper_parameters
        
        super().__init__(
            in_channels= 512,
            out_channels= self.hp.Encoder.Size,
            wavenet_stack= self.hp.Linguistic_Encoder.WaveNet_Stack,
            conv_stack= self.hp.Linguistic_Encoder.Conv_Stack,
            kernel_size= self.hp.Linguistic_Encoder.Kernel_Size,
            dilation_rate= self.hp.Linguistic_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Linguistic_Encoder.Dropout_Rate,
            )
        
        wav2vec2 = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-xls-r-2b')
        wav2vec2.freeze_feature_encoder()
        self.feature_extractor = wav2vec2.wav2vec2.feature_extractor
        self.norm = LayerNorm(
            num_features= 512
            )

    def forward(
        self,
        audios: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.distributions.Normal:
        '''
        audios: [Batch, Audio_t], Raw waveform
        lengths: [Batch], * This is feature length, not waveform length.
        '''
        with torch.no_grad():
            audios = torchaudio.functional.resample(audios, self.hp.Sound.Sample_Rate, 16000)
            features = self.feature_extractor(audios)
            features = torchvision.transforms.functional.resize(
                features.unsqueeze(1),
                [512, lengths.max()]
                ).squeeze(1)
        features = self.norm(features)

        return super().forward(
            features= features,
            lengths= lengths
            )


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
            w_init_gain= 'linear'
            )

        self.upsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Decoder.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Decoder.Upsample.Rate,
            self.hp.Decoder.Upsample.Kernel_Size
            )):
            upsample_block = torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ),
                ConvTranspose1d(
                    in_channels= previous_channels,
                    out_channels= self.hp.Decoder.Upsample.Base_Size // (2 ** (index + 1)),
                    kernel_size= kernel_size,
                    stride= upsample_rate,
                    padding= (kernel_size - upsample_rate) // 2,
                    w_init_gain= 'leaky_relu'
                    )
                )
            self.upsample_blocks.append(upsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Decoder.Residual_Block.Kernel_Size,
                self.hp.Decoder.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Decoder_Residual_Block(
                    channels= self.hp.Decoder.Upsample.Base_Size // (2 ** (index + 1)),
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = self.hp.Decoder.Upsample.Base_Size // (2 ** (index + 1))

        self.postnet = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Decoder.Postnet.Kernel_Size,
                padding= (self.hp.Decoder.Postnet.Kernel_Size - 1) // 2,
                w_init_gain= 'tanh'
                ),
            torch.nn.Tanh(),
            Lambda(lambda x: x.squeeze(1))
            )
            
    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(encodings[0, 0]).sum())).unsqueeze(1).float()

        decodings = self.prenet(encodings) * masks
        for upsample_block, residual_blocks in zip(
            self.upsample_blocks,
            self.residual_blocks
            ):
            decodings = upsample_block(decodings)
            decodings = torch.stack(
                [residual_block(decodings) for residual_block in residual_blocks],
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

        self.conv = torch.nn.ModuleList()
        for dilation in dilations:
            conv = torch.nn.Sequential(                
                torch.nn.LeakyReLU(negative_slope= negative_slope),
                Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    dilation= dilation,
                    padding= (kernel_size * dilation - dilation) // 2,
                    w_init_gain= 'leaky_relu'
                    ),
                torch.nn.LeakyReLU(negative_slope= negative_slope),
                Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    dilation= 1,
                    padding= (kernel_size - 1) // 2
                    )
                )
            self.conv.append(conv)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x) + x

        return x


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
                dropout_rate= self.hp.Encoder.Transformer.Dropout_Rate,
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
        ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        '''
        tokens: [Batch, Time]
        '''
        encodings = self.token_embedding(tokens).permute(0, 2, 1)
        for block in self.blocks:
            encodings = block(encodings, lengths)

        means, stds = self.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        stds = torch.nn.functional.softplus(stds)

        return means, stds, encodings

class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        feedforward_kernel_size: int,
        dropout_rate: float= 0.1,
        feedforward_dropout_rate: float= 0.1
        ) -> None:
        super().__init__()

        self.attention = LinearAttention(
            channels= channels,
            calc_channels= channels,
            num_heads= num_head,
            dropout_rate= dropout_rate
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
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())).unsqueeze(1).float()   # float mask

        # Attention + Dropout + LayerNorm
        x = self.attention(x)
        
        # FFN + Dropout + LayerNorm
        x = self.ffn(x, masks)

        return x * masks

class LinearAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        num_heads: int,
        dropout_rate: float= 0.1,
        use_scale: bool= True,
        use_residual: bool= True,
        use_norm: bool= True
        ):
        super().__init__()
        assert calc_channels % num_heads == 0
        self.calc_channels = calc_channels
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.prenet = Conv1d(
            in_channels= channels,
            out_channels= calc_channels * 3,
            kernel_size= 1,
            bias=False,
            w_init_gain= 'linear'
            )
        self.projection = Conv1d(
            in_channels= calc_channels,
            out_channels= channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        
        if use_scale:
            self.scale = torch.nn.Parameter(torch.zeros(1))

        if use_norm:
            self.norm = LayerNorm(num_features= channels)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        '''
        x: [Batch, Enc_d, Enc_t]
        '''
        residuals = x

        x = self.prenet(x)  # [Batch, Calc_d * 3, Enc_t]        
        x = x.view(x.size(0), self.num_heads, x.size(1) // self.num_heads, x.size(2))    # [Batch, Head, Calc_d // Head * 3, Enc_t]
        queries, keys, values = x.chunk(chunks= 3, dim= 2)  # [Batch, Head, Calc_d // Head, Enc_t] * 3
        keys = (keys + 1e-3).softmax(dim= 3)

        contexts = keys @ values.permute(0, 1, 3, 2)   # [Batch, Head, Calc_d // Head, Calc_d // Head]
        contexts = contexts.permute(0, 1, 3, 2) @ queries   # [Batch, Head, Calc_d // Head, Enc_t]
        contexts = contexts.view(contexts.size(0), contexts.size(1) * contexts.size(2), contexts.size(3))   # [Batch, Calc_d, Enc_t]
        contexts = self.projection(contexts)    # [Batch, Enc_d, Enc_t]

        if self.use_scale:
            contexts = self.scale * contexts

        contexts = self.dropout(contexts)

        if self.use_residual:
            contexts = contexts + residuals

        if self.use_norm:
            contexts = self.norm(contexts)

        return contexts

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
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm = LayerNorm(
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
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)
        x = self.dropout(x)
        x = self.norm(x + residuals)

        return x * masks


class Variance_Block(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.duration_predictor = Duration_Predictor(self.hp)

        self.gaussian_upsampler = Gaussian_Upsampler(
            encoding_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Variance_Block.Gaussian_Upsampler.Kernel_Size,
            range_lstm_stack= self.hp.Variance_Block.Gaussian_Upsampler.Range_Predictor.Stack,
            range_dropout_rate= self.hp.Variance_Block.Gaussian_Upsampler.Range_Predictor.Dropout_Rate
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        durations: torch.Tensor= None,  # None when inference
        length_scales: torch.Tensor= None,  # None when training
        ):
        log_duration_predictions = self.duration_predictor(
            encodings= encodings,
            encoding_lengths= encoding_lengths
            )   # [Batch, Enc_t]
        
        if not length_scales is None:
            log_duration_predictions = (((log_duration_predictions.exp() - 1) * length_scales) + 1).log()
        
        if durations is None:
            durations = (log_duration_predictions.exp() - 1).ceil().long() # [Batch, Enc_t]
            max_duration_sum = durations.sum(dim= 1).max()

            for duration, length in zip(durations, encoding_lengths):
                duration[length - 1] = duration[length - 1] + max_duration_sum - duration.sum()

        alignments = self.gaussian_upsampler(
            encodings= encodings,
            encoding_lengths= encoding_lengths,
            durations= durations
            )   # [Batch, Enc_t, Feature_t]

        return alignments, log_duration_predictions, durations

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

class Duration_Predictor(Variance_Predictor):
    def __init__(self, hyper_parameters: Namespace):        
        self.hp = hyper_parameters
        super().__init__(
            in_features= self.hp.Encoder.Size,
            lstm_features= self.hp.Encoder.Size // 2,
            lstm_stack= self.hp.Variance_Block.Duration_Predictor.Stack,
            dropout_rate= self.hp.Variance_Block.Duration_Predictor.Dropout_Rate
            )

    def forward(self, encodings: torch.Tensor, encoding_lengths: torch.Tensor):
        log_duration_predictions = super().forward(encodings, encoding_lengths)

        return log_duration_predictions


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
        lengths: torch.Tensor,
        ) -> torch.distributions.Normal:
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


def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]