import torch
import math
from typing import Optional, Union

from .Layer import Conv1d, LayerNorm
from .Stochastic_Duration_Predictor_Transforms import piecewise_rational_quadratic_transform

class Stochastic_Duration_Predictor(torch.nn.Module):
    def __init__(
        self,
        channels: int,  # enc_d
        calc_channels: int, # 192
        kernel_size: int,   # 3
        conv_stack: int,    # 3
        flow_stack: int,    # 4        
        dropout_rate: float= 0.5,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()
        
        self.prenet_pre = Conv1d(
            in_channels= channels,
            out_channels= calc_channels,
            kernel_size= 1,            
            )
        self.prenet_ddsc = Deep_Depthwise_Separable_Conv(
            channels= calc_channels,
            kernel_size= kernel_size,
            stack= conv_stack,
            dropout_rate= dropout_rate
            )
        self.prenet_post = Conv1d(
            in_channels= calc_channels,
            out_channels= calc_channels,
            kernel_size= 1
            )

        self.postnet_pre = Conv1d(
            in_channels= 1,
            out_channels= calc_channels,
            kernel_size= 1,
            )
        self.postnet_ddsc = Deep_Depthwise_Separable_Conv(
            channels= calc_channels,
            kernel_size= kernel_size,
            stack= conv_stack,
            dropout_rate= dropout_rate
            )
        self.postnet_post = Conv1d(
            in_channels= calc_channels,
            out_channels= calc_channels,
            kernel_size= 1
            )
        
        self.postnet_flows = torch.nn.ModuleList()
        self.postnet_flows.append(Elementwise_Affine(channels= 2))
        for index in range(flow_stack):
            self.postnet_flows.append(ConvFlow(
                channels= 2,
                calc_channels= calc_channels,
                kernel_size= kernel_size,
                conv_stack= conv_stack
                ))
            self.postnet_flows.append(Flip())

        self.flow_log = Log()
        self.flows = torch.nn.ModuleList()
        self.flows.append(Elementwise_Affine(channels= 2))
        for index in range(flow_stack):
            self.flows.append(ConvFlow(
                channels= 2,
                calc_channels= calc_channels,
                kernel_size= kernel_size,
                conv_stack= conv_stack
                ))
            self.flows.append(Flip())

        if not condition_channels is None:
            self.condition = Conv1d(
                in_channels= condition_channels,
                out_channels= calc_channels,
                kernel_size= 1
                )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        durations: torch.Tensor= None,
        conditions: Optional[torch.Tensor]= None,
        reverse: bool= False,
        noise_scale: float=1.0
        ):
        masks = (~Mask_Generate(lengths= encoding_lengths, max_length= torch.ones_like(encodings[0, 0]).sum())).unsqueeze(1).float()   # float mask

        x = encodings.detach()
        x = self.prenet_pre(x)
        if not conditions is None:
            conditions = conditions.detach()
            if conditions.ndim == 2:
                conditions = conditions.unsqueeze(2)    # [Batch, Cond_d, 1]
            conditions = self.condition(conditions)
            x = x + conditions        
        x = self.prenet_ddsc(
            x= x,
            masks= masks,
            conditions= conditions
            )
        x = self.prenet_post(x) * masks # [Batch, Calc_d, Enc_t]

        if not reverse:
            if durations.ndim == 2:
                durations = durations.unsqueeze(1)
            duration_hiddens = self.postnet_pre(durations)
            duration_hiddens = self.postnet_ddsc(
                x= duration_hiddens,
                masks= masks,
                conditions= conditions
                )
            duration_hiddens = self.postnet_post(duration_hiddens) * masks  # [Batch, Calc_d, Enc_t]
            
            e_q = torch.randn(
                size= (x.size(0), 2, x.size(2)),
                dtype= x.dtype,
                device= x.device
                ) * masks
            z_q = e_q
            logdet_q_sum = 0
            for flow in self.postnet_flows:
                z_q, logdet_q = flow(
                    x= z_q,
                    masks= masks,
                    conditions= x + duration_hiddens
                    )
                logdet_q_sum += logdet_q
            z_u, z_1 = z_q.chunk(chunks= 2, dim= 1)
            u = z_u.sigmoid() * masks
            z_0 = (durations - u) * masks
            logdet_q_sum += ((z_u.sigmoid().log() + (-z_u).sigmoid().log()) * masks).sum(dim= [1, 2])
            logs_q = (-0.5 * (math.log(2.0 * math.pi) + e_q.pow(2.0)) * masks).sum(dim= [1, 2]) - logdet_q_sum

            logdet_sum = 0
            z_0, logdet = self.flow_log(z_0, masks)
            logdet_sum += logdet
            z = torch.cat([z_0, z_1], dim= 1)
            for flow in self.flows:
                z, logdet = flow(
                    x= z,
                    masks= masks,
                    conditions= x,
                    reverse= False
                    )
                logdet_sum += logdet
            nll = (0.5 * (math.log(2.0 * math.pi) + z.pow(2.0)) * masks).sum(dim= [1, 2]) - logdet_sum
            
            duration_losses = (nll + logs_q) / masks.sum()

            return None, duration_losses  # [Batch]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]    # Initial convflow is removed
            z = torch.randn(
                size= (x.size(0), 2, x.size(2)),
                dtype= x.dtype,
                device= x.device
                ) * noise_scale
            for flow in flows:
                z, _ = flow(
                    x= z,
                    masks= masks,
                    conditions= x,
                    reverse= True
                    )
            durations = (z[:, 0].exp() * masks).clamp(0.0).ceil().long()

            return durations, None

class Deep_Depthwise_Separable_Conv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stack: int,
        dropout_rate: float= 0.0
        ):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for index in range(stack):
            dilation = kernel_size ** index
            padding = (kernel_size - 1) * dilation // 2
            self.blocks.append(torch.nn.Sequential(
                Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    groups= channels,
                    dilation= dilation,
                    padding= padding
                    ),
                LayerNorm(num_features= channels),
                torch.nn.GELU(),
                Conv1d(
                    in_channels= channels,
                    out_channels= channels,
                    kernel_size= 1,
                    ),
                LayerNorm(num_features= channels),
                torch.nn.GELU(),
                torch.nn.Dropout(p= dropout_rate)
                ))


    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        conditions: Optional[torch.Tensor]= None
        ):
        if conditions is not None:
            x = x + conditions

        for block in self.blocks:
            x = block(x * masks) + x

        return x * masks

class Elementwise_Affine(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.means = torch.nn.Parameter(torch.zeros(channels, 1))
        self.log_stds = torch.nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, masks, reverse=False, **kwargs):
        if not reverse:
            x = self.means + torch.exp(self.log_stds) * x
            x = x * masks
            logdet = (self.log_stds * masks).sum(dim= [1, 2])
            return x, logdet
        else:
            x = (x - self.means) * torch.exp(-self.log_stds) * masks            
            return x, None

class ConvFlow(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        kernel_size: int,
        conv_stack: int,
        num_bins: int= 10,
        tail_bound: float=5.0
        ):
        super().__init__()
        self.channels = channels
        self.calc_channels = calc_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.prenet = Conv1d(
            in_channels= channels // 2,
            out_channels= calc_channels,
            kernel_size= 1
            )
        self.ddsc = Deep_Depthwise_Separable_Conv(
            channels= calc_channels,
            kernel_size= kernel_size,
            stack= conv_stack
            )
        self.projection = Conv1d(
            in_channels= calc_channels,
            out_channels= (channels // 2) * (num_bins * 3 - 1),
            kernel_size= 1,
            w_init_gain= 'zero'
            )

    def forward(self, x, masks, conditions= None, reverse=False):
        x_0, x_1 = x.chunk(chunks= 2, dim= 1)
        x_hiddens = self.prenet(x_0)
        x_hiddens = self.ddsc(
            x= x_hiddens,
            masks= masks,
            conditions= conditions
            )
        x_hiddens = self.projection(x_hiddens) * masks  # [Batch, Dim // 2 * (Bins * 3 - 1), Enc_t]
        x_hiddens = x_hiddens.view(x_hiddens.size(0), self.channels // 2, self.num_bins * 3 - 1, x_hiddens.size(2))  # [Batch, Dim // 2, Bins * 3 - 1, Enc_t]
        x_hiddens = x_hiddens.permute(0, 1, 3, 2)   # [Batch, Dim // 2, Enc_t, Bins * 3 - 1]

        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = x_hiddens.split([self.num_bins, self.num_bins, self.num_bins - 1], dim= 3)
        unnormalized_widths = unnormalized_widths / math.sqrt(self.calc_channels)
        unnormalized_heights = unnormalized_heights / math.sqrt(self.calc_channels)

        x_1, logabsdet = piecewise_rational_quadratic_transform(
            inputs= x_1,
            unnormalized_widths= unnormalized_widths,
            unnormalized_heights= unnormalized_heights,
            unnormalized_derivatives= unnormalized_derivatives,
            inverse= reverse,
            tails= 'linear',
            tail_bound= self.tail_bound
            )

        x = torch.cat([x_0, x_1], dim= 1) * masks        
        logdet = (logabsdet * masks).sum(dim= [1, 2])

        return x, logdet

class Flip(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        reverse: bool= False, 
        *args,
        **kwargs
        ):
        x = x.flip(dims= [1,])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype= x.dtype, device= x.device)
            return x, logdet
        else:
            return x, None

class Log(torch.nn.Module):
    def forward(self, x: torch.Tensor, masks: torch.Tensor, reverse: bool= False, **kwargs):
        if not reverse:
            x = x.clamp(1e-5).log() * masks
            logdet = -x.sum(dim= [1, 2])
            return x, logdet
        else:
            x = torch.exp(x) * masks
            return x

def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]







