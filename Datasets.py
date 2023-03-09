from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging
from typing import Dict, List

from Pattern_Generator import Text_Filtering, Phonemize
from Modules.Nvidia_Alignment_Leraning_Framework import Attention_Prior_Generator

def Text_to_Token(text: str, token_dict: Dict[str, int]):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens: List[np.ndarray], token_dict, max_length: int= None):
    max_token_length = max_length or max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Feature_Stack(features: List[np.ndarray], max_length: int= None):
    max_feature_length = max_length or max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -1.0) for feature in features],
        axis= 0
        )
    return features

def Audio_Stack(audios: List[np.ndarray], max_length: int= None):
    max_audio_length = max_length or max([energy.shape[0] for energy in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )
    return audios

def Attention_Prior_Stack(attention_priors: List[np.ndarray], max_token_length: int, max_feature_length: int):
    attention_priors_padded = np.zeros(
        shape= (len(attention_priors), max_feature_length, max_token_length),
        dtype= np.float32
        )    
    for index, attention_prior in enumerate(attention_priors):
        attention_priors_padded[index, :attention_prior.shape[0], :attention_prior.shape[1]] = attention_prior

    return attention_priors_padded


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        feature_range_info_dict: Dict[str, Dict[str, float]],
        linear_spectrogram_range_info_dict: Dict[str, Dict[str, float]],
        pattern_path: str,
        metadata_file: str,
        feature_type: str,
        feature_length_min: int,
        feature_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.token_dict = token_dict
        self.feature_range_info_dict = feature_range_info_dict
        self.linear_spectrogram_range_info_dict = linear_spectrogram_range_info_dict
        self.feature_type = feature_type
        self.pattern_path = pattern_path
        
        if feature_type == 'Mel':
            feature_length_dict = 'Mel_Length_Dict'
        elif feature_type == 'Spectrogram':
            feature_length_dict = 'Spectrogram_Length_Dict'

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict[feature_length_dict][x] >= feature_length_min,
                metadata_dict[feature_length_dict][x] <= feature_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch

        self.attention_prior_generator = Attention_Prior_Generator()

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))
        speaker = pattern_dict['Speaker']

        token = Text_to_Token(pattern_dict['Pronunciation'], self.token_dict)

        feature_min = self.feature_range_info_dict[speaker]['Min']
        feature_max = self.feature_range_info_dict[speaker]['Max']
        feature = (pattern_dict[self.feature_type] - feature_min) / (feature_max - feature_min) * 2.0 - 1.0

        feature_min = self.linear_spectrogram_range_info_dict[speaker]['Min']
        feature_max = self.linear_spectrogram_range_info_dict[speaker]['Max']
        linear_spectrogram = (pattern_dict['Spectrogram'] - feature_min) / (feature_max - feature_min) * 2.0 - 1.0

        attention_prior = self.attention_prior_generator.get_prior(feature.shape[0], token.shape[0])

        return token, feature, pattern_dict['Audio'], linear_spectrogram, attention_prior

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        texts: List[str],
        ):
        super().__init__()
        self.token_dict = token_dict

        pronunciations = Phonemize(texts, language= 'English')

        self.patterns = []
        for index, (text, pronunciation) in enumerate(zip(texts, pronunciations)):
            text = Text_Filtering(text)
            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue            

            self.patterns.append((text, pronunciation))

    def __getitem__(self, idx):
        text, pronunciation = self.patterns[idx]

        return Text_to_Token(pronunciation, self.token_dict), text, pronunciation

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        hop_size: int,
        compatible_length: int
        ):
        self.token_dict = token_dict
        self.hop_size = hop_size
        self.compatible_length = compatible_length

    def __call__(self, batch):
        tokens, features, audios, linear_spectrograms, attention_priors = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        feature_lengths = np.array([feature.shape[0] for feature in features])

        max_token_length = max(token_lengths)
        max_feature_length = max(feature_lengths)
        if max_feature_length % self.compatible_length != 0:
            max_feature_length += self.compatible_length - max_feature_length % self.compatible_length

        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict
            )
        features = Feature_Stack(
            features= features,
            max_length= max_feature_length
            )
        audios = Audio_Stack(
            audios= audios,
            max_length= max_feature_length * self.hop_size
            )
        linear_spectrograms = Feature_Stack(
            features= linear_spectrograms,
            max_length= max_feature_length
            )
        attention_priors = Attention_Prior_Stack(
            attention_priors= attention_priors,
            max_token_length= max_token_length,
            max_feature_length= max_feature_length
            )
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Featpure_t]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]        
        audios = torch.FloatTensor(audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size
        linear_spectrograms = torch.FloatTensor(linear_spectrograms).permute(0, 2, 1)   # [Batch, Spectrogram_d, Feature_t]
        attention_priors = torch.FloatTensor(attention_priors) # [Batch, Token_t, Feature_t]

        return tokens, token_lengths, features, feature_lengths, audios, linear_spectrograms, attention_priors

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, texts, pronunciations = zip(*batch)

        token_lengths = np.array([token.shape[0] for token in tokens])
        
        tokens = Token_Stack(tokens, self.token_dict)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        
        return tokens, token_lengths, texts, pronunciations