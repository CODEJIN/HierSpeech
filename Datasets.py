from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging
from typing import Dict, List, Optional
import functools

from Pattern_Generator import Text_Filtering, Phonemize

def Text_to_Token(text: str, token_dict: Dict[str, int]):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens: List[np.ndarray], token_dict, max_length: Optional[int]= None):
    max_token_length = max_length or max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Feature_Stack(features: List[np.ndarray], max_length: Optional[int]= None):
    max_feature_length = max_length or max([feature.shape[1] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, 0], [0, max_feature_length - feature.shape[1]]], constant_values= feature.min()) for feature in features],
        axis= 0
        )
    return features

def Audio_Stack(audios: List[np.ndarray], max_length: Optional[int]= None):
    max_audio_length = max_length or max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )
    return audios

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        speaker_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        feature_length_min: int,
        feature_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.token_dict = token_dict
        self.speaker_dict = speaker_dict
        self.pattern_path = pattern_path

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
                metadata_dict['Spectrogram_Length_Dict'][x] >= feature_length_min,
                metadata_dict['Spectrogram_Length_Dict'][x] <= feature_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch

        if use_pattern_cache:
            self.Pattern_LRU_Cache = functools.lru_cache(maxsize= None)(self.Pattern_LRU_Cache)
    
    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        return self.Pattern_LRU_Cache(path)
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))
        
        # padding between tokens
        token = ['<P>'] * (len(pattern_dict['Pronunciation']) * 2 - 1)
        token[0::2] = pattern_dict['Pronunciation']
        token = Text_to_Token(token, self.token_dict)

        speaker = self.speaker_dict[pattern_dict['Speaker']]

        return token, speaker, pattern_dict['Spectrogram'], pattern_dict['Audio']

    def __len__(self):
        return len(self.patterns)    

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        speaker_dict: Dict[str, int],
        texts: List[str],
        speakers: List[str],
        languages: List[str]
        ):
        super().__init__()
        self.token_dict = token_dict
        self.speaker_dict = speaker_dict

        language_text_dict = {
            language: []
            for language in set(languages)
            }
        for text, language in zip(texts, languages):
            language_text_dict[language].append(text)

        pronunciation_dict = {}
        for language, language_texts in language_text_dict.items():
            language_pronunciations = Phonemize(language_texts, language)
            pronunciation_dict.update({
                text: pronunciation
                for text, pronunciation in zip(language_texts, language_pronunciations)
                })

        self.patterns = []
        for index, (text, speaker) in enumerate(zip(texts, speakers)):
            text = Text_Filtering(text)
            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            if speaker is None or not speaker in self.speaker_dict.keys():
                logging.warning('The speaker of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            self.patterns.append((text, pronunciation_dict[text], speaker))

    def __getitem__(self, idx):
        text, pronunciation, speaker = self.patterns[idx]

        token = ['<P>'] * (len(pronunciation) * 2 - 1)
        token[0::2] = pronunciation
        pronunciation = [(x if x != '<P>' else '') for x in token]
        token = Text_to_Token(token, self.token_dict)

        return token, self.speaker_dict[speaker], text, pronunciation, speaker

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        hop_size: int,
        ):
        self.token_dict = token_dict
        self.hop_size = hop_size

    def __call__(self, batch):
        tokens, speakers, linear_spectrograms, audios = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        linear_spectrogram_lengths = np.array([feature.shape[1] for feature in linear_spectrograms])

        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict
            )
        linear_spectrograms = Feature_Stack(
            features= linear_spectrograms
            )
        audios = Audio_Stack(
            audios= audios
            )
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        speakers = torch.LongTensor(speakers)   # [Batch]
        linear_spectrograms = torch.FloatTensor(linear_spectrograms)  # [Batch, Feature_d, Featpure_t]
        linear_spectrogram_lengths = torch.LongTensor(linear_spectrogram_lengths)   # [Batch]
        audios = torch.FloatTensor(audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size

        return tokens, token_lengths, speakers, linear_spectrograms, linear_spectrogram_lengths, audios

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, speakers, texts, pronunciations, speaker_labels = zip(*batch)

        token_lengths = np.array([token.shape[0] for token in tokens])
        
        tokens = Token_Stack(tokens, self.token_dict)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        speakers = torch.LongTensor(speakers)   # [Batch]
                
        return tokens, token_lengths, speakers, texts, pronunciations, speaker_labels