import torch
import numpy as np
from numba import jit
import math

from typing import Tuple


def Calc_Duration(
    encoding_means: torch.Tensor,
    encoding_log_stds: torch.Tensor,
    encoding_lengths: torch.Tensor,
    decodings: torch.Tensor,
    decoding_lengths: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
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
        stds_p_sq_r = (-2.0 * encoding_log_stds).exp()  # [Batch, Enc_d, Enc_t]
        neg_cent1 = torch.sum(-0.5 * math.log(2.0 * math.pi) - encoding_log_stds, [1], keepdim=True) # [Batch, 1, Enc_t]
        neg_cent2 = torch.matmul(-0.5 * (decodings ** 2.0).permute(0, 2, 1), stds_p_sq_r) # [Batch, Dec_t, Enc_d] x [Batch, Enc_d, Enc_t] -> [Batch, Dec_t, Enc_t]
        neg_cent3 = torch.matmul(decodings.permute(0, 2, 1), (encoding_means * stds_p_sq_r)) # [Batch, Dec_t, Enc_d] x [b, Enc_d, Enc_t] -> [Batch, Dec_t, Enc_t]
        neg_cent4 = torch.sum(-0.5 * (encoding_means ** 2.0) * stds_p_sq_r, [1], keepdim=True) # [Batch, 1, Enc_t]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4    # [Batch, Dec_t, Enc_t]

        attention_masks = encoding_masks * decoding_masks.permute(0, 2, 1)  # [Batch, 1, Enc_t] x [Batch, Dec_t, 1] -> [Batch, Dec_t, Enc_t]
        alignments = Maximum_Path_Generator(neg_cent, attention_masks).detach() # [Batch, Feature_t, Enc_t]
        durations = alignments.sum(dim= 1)  # [Batch, Enc_t]

    return durations, alignments

def Maximum_Path_Generator(neg_cent, masks) -> torch.Tensor:
    '''
    x: [Batch, Dec_t, Enc_t]
    mask: [Batch, Dec_t, Enc_t]
    '''
    neg_cent *= masks
    device, dtype = neg_cent.device, neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    masks = masks.data.cpu().numpy()

    token_lengths = masks.sum(axis= 2)[:, 0].astype('int32')   # [Batch]
    feature_lengths = masks.sum(axis= 1)[:, 0].astype('int32')   # [Batch]

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

def Mask_Generate(lengths: torch.Tensor, max_length: int= None) -> torch.Tensor:
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]
