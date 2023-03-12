import torch
import numpy as np
from numba import jit
import math


def Calc_Duration(
    encoding_means: torch.Tensor,
    encoding_stds: torch.Tensor,
    encoding_lengths: torch.Tensor,
    decodings: torch.Tensor,
    decoding_lengths: torch.Tensor
    ):
    with torch.no_grad():
        encoding_masks = (~Mask_Generate(
            lengths= encoding_lengths,
            max_length= torch.ones_like(encoding_means[0, 0]).sum()
            )).unsqueeze(1).float()
        decoding_masks = (~Mask_Generate(
            lengths= decoding_lengths,
            max_length= torch.ones_like(decodings[0, 0]).sum()
            )).unsqueeze(1).float()
        
        # negative cross-entropy
        stds_p_sq_r = encoding_stds.pow(-2.0)  # [Batch, Enc_d, Enc_t]
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - encoding_stds.clamp(min= 1e-3).log(), [1], keepdim=True) # [Batch, 1, Enc_t]
        neg_cent2 = torch.matmul(-0.5 * (decodings ** 2).permute(0, 2, 1), stds_p_sq_r) # [Batch, Dec_t, Enc_d] x [Batch, Enc_d, Enc_t] -> [Batch, Dec_t, Enc_t]
        neg_cent3 = torch.matmul(decodings.permute(0, 2, 1), (encoding_means * stds_p_sq_r)) # [Batch, Dec_t, Enc_d] x [b, Enc_d, Enc_t] -> [Batch, Dec_t, Enc_t]
        neg_cent4 = torch.sum(-0.5 * (encoding_means ** 2) * stds_p_sq_r, [1], keepdim=True) # [Batch, 1, Enc_t]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4    # [Batch, Dec_t, Enc_t]

        attention_masks = encoding_masks * decoding_masks.permute(0, 2, 1)  # [Batch, 1, Enc_t] x [Batch, Dec_t, 1] -> [Batch, Dec_t, Enc_t]
        attentions = Maximum_Path_Generator(neg_cent, attention_masks).detach()
        durations = attentions.sum(dim= 1).long()    # [Batch, Enc_t]

    return durations

def Maximum_Path_Generator(neg_cent, mask):
    '''
    x: [Batch, Dec_t, Enc_t]
    mask: [Batch, Dec_t, Enc_t]
    '''
    neg_cent *= mask
    device, dtype = neg_cent.device, neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    mask = mask.data.cpu().numpy()

    token_lengths = mask.sum(axis= 2)[:, 0].astype('int32')   # [Batch]
    feature_lengths = mask.sum(axis= 1)[:, 0].astype('int32')   # [Batch]

    paths = Calc_Paths(neg_cent, token_lengths, feature_lengths)

    return torch.from_numpy(paths).to(device= device, dtype= dtype)

def Calc_Paths(neg_cent, token_lengths, feature_lengths):
    return np.stack([
        Calc_Path(x, token_length, feature_length)
        for x, token_length, feature_length in zip(neg_cent, token_lengths, feature_lengths)
        ], axis= 0)

@jit(nopython=True)
def Calc_Path(x, token_length, feature_length):
    path = np.zeros_like(x, dtype= np.int32)
    for feature_index in range(feature_length):
        for token_index in range(max(0, token_length + feature_index - feature_length), min(token_length, feature_index + 1)):
            if feature_index == token_index:
                current_q = -1e+9
            else:
                current_q = x[feature_index - 1, token_index]   # Stayed current token
            if token_index == 0:
                if feature_index == 0:
                    prev_q = 0.0
                else:
                    prev_q = -1e+9
            else:
                prev_q = x[feature_index - 1, token_index - 1]  # Moved to next token
            x[feature_index, token_index] = x[feature_index, token_index] + max(prev_q, current_q)

    token_index = token_length - 1
    for feature_index in range(feature_length - 1, -1, -1):
        path[feature_index, token_index] = 1
        if token_index != 0 and token_index == feature_index or x[feature_index - 1, token_index] < x[feature_index - 1, token_index - 1]:
            token_index = token_index - 1

    return path

def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]
