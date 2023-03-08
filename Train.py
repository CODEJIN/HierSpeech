from email.policy import strict
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Modules.Modules import HierSpeech, Mask_Generate
from Modules.Discriminator import Discriminator, Feature_Map_Loss, Generator_Loss, Discriminator_Loss
from Modules.Nvidia_Alignment_Leraning_Framework import AttentionBinarizationLoss, AttentionCTCLoss

from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Noam_Scheduler
from Logger import Logger

from meldataset import mel_spectrogram
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }

            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model_dict['HierSpeech'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        if self.hp.Feature_Type == 'Spectrogram':
            self.feature_range_info_dict = yaml.load(open(self.hp.Spectrogram_Range_Info_Path), Loader=yaml.Loader)
        if self.hp.Feature_Type == 'Mel':
            self.feature_range_info_dict = yaml.load(open(self.hp.Mel_Range_Info_Path), Loader=yaml.Loader)
        linear_spectrogram_range_info_dict = yaml.load(open(self.hp.Spectrogram_Range_Info_Path), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            feature_range_info_dict= self.feature_range_info_dict,
            linear_spectrogram_range_info_dict= linear_spectrogram_range_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            feature_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            feature_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Train_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Train_Pattern.Text_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            feature_range_info_dict= self.feature_range_info_dict,
            linear_spectrogram_range_info_dict= linear_spectrogram_range_info_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            feature_length_min= max(self.hp.Train.Train_Pattern.Feature_Length.Min, self.hp.Train.Segment_Size),
            feature_length_max= self.hp.Train.Eval_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Eval_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Eval_Pattern.Text_Length.Max
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            texts= self.hp.Train.Inference_in_Train.Text,
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict,
            hop_size= self.hp.Sound.Frame_Shift,
            compatible_length= 1
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_dict = {
            'HierSpeech': HierSpeech(self.hp).to(self.device),
            'Discriminator': Discriminator(
                stft_n_fft_list= self.hp.Discriminator.STFT.N_FFT,
                stft_win_size_list= self.hp.Discriminator.STFT.Win_Size,
                ).to(self.device),
            }
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduce= None).to(self.device),
            'MAE': torch.nn.L1Loss(reduce= None).to(self.device),
            'Feature_Map': Feature_Map_Loss,
            'Discriminator': Discriminator_Loss,
            'Generator': Generator_Loss,
            'TokenCTC': torch.nn.CTCLoss(
                blank= self.hp.Tokens,  # == Token length
                zero_infinity=True
                ),
            'Attention_Binarization': AttentionBinarizationLoss(),
            'Attention_CTC': AttentionCTCLoss()
            }
        self.optimizer_dict = {
            'HierSpeech': torch.optim.NAdam(
                params= self.model_dict['HierSpeech'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                ),
            'Discriminator': torch.optim.NAdam(
                params= self.model_dict['Discriminator'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                ),
            }
        self.scheduler_dict = {
            'HierSpeech': Noam_Scheduler(
                optimizer= self.optimizer_dict['HierSpeech'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step,
                ),
            'Discriminator': Noam_Scheduler(
                optimizer= self.optimizer_dict['Discriminator'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step,
                ),
            }

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        # if self.gpu_id == 0:
        #     logging.info(self.model_dict['HierSpeech'])

    def Train_Step(self, tokens, token_lengths, features, feature_lengths, audios, linear_spectrograms, attention_priors):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)
        linear_spectrograms = linear_spectrograms.to(self.device, non_blocking=True)
        attention_priors = attention_priors.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            audio_predictions_slice, audios_slice, token_predictions, acoustic_distributions, \
            encoding_distributions_prior, encoding_distributions_posterior, linguistic_distributions_prior, linguistic_distributions_posterior, \
            log_duration_predictions, durations, attention_softs, attention_hards, attention_logprobs = self.model_dict['HierSpeech'](
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths,
                audios= audios,
                linear_spectrograms= linear_spectrograms,
                attention_priors= attention_priors
                )

            if self.steps >= self.hp.Train.Discrimination_Step:
                discriminations_list_for_real, _ = self.model_dict['Discriminator'](audios_slice)
                discriminations_list_for_fake, _ = self.model_dict['Discriminator'](audio_predictions_slice.detach())
        
        if self.steps >= self.hp.Train.Discrimination_Step:
            loss_dict['Discrimination'] = self.criterion_dict['Discriminator'](discriminations_list_for_real, discriminations_list_for_fake)

            self.optimizer_dict['Discriminator'].zero_grad()
            self.scaler.scale(loss_dict['Discrimination']).backward()

            if self.hp.Train.Gradient_Norm > 0.0:
                self.scaler.unscale_(self.optimizer_dict['Discriminator'])
                torch.nn.utils.clip_grad_norm_(
                    parameters= self.model_dict['Discriminator'].parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )

            self.scaler.step(self.optimizer_dict['Discriminator'])
            self.scaler.update()
            self.scheduler_dict['Discriminator'].step()

        token_masks = Mask_Generate(
            lengths= token_lengths,
            max_length= tokens.size(1)
            ).to(tokens.device)

        loss_dict['STFT'] = self.criterion_dict['MSE'](
            mel_spectrogram(
                audio_predictions_slice,
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= self.hp.Sound.Mel_F_Min,
                fmax= self.hp.Sound.Mel_F_Max
                ),
            mel_spectrogram(
                audios_slice,
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= self.hp.Sound.Mel_F_Min,
                fmax= self.hp.Sound.Mel_F_Max
                )
            ).mean()
        loss_dict['Log_Duration'] = (self.criterion_dict['MSE'](
            log_duration_predictions,
            (durations.to(log_duration_predictions.dtype) + 1).log()
            ) * ~token_masks).mean()
        loss_dict['TokenCTC'] = self.criterion_dict['TokenCTC'](
                log_probs= token_predictions.permute(2, 0, 1),  # [Feature_t, Batch, Token_n]
                targets= tokens,   # Remove <S>, <E>
                input_lengths= feature_lengths,
                target_lengths= token_lengths - 2
                )
        loss_dict['Encoding_KLD'] = torch.distributions.kl_divergence(
            encoding_distributions_prior,
            encoding_distributions_posterior
            ).mean()
        loss_dict['Linguistic_KLD'] = torch.distributions.kl_divergence(
            linguistic_distributions_prior,
            linguistic_distributions_posterior
            ).mean()
        loss_dict['Acoustic_KLD'] = torch.distributions.kl_divergence(
            acoustic_distributions,
            torch.distributions.Normal(
                loc= 0.0,
                scale= 1.0
                )
            ).mean()
        loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
        loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, feature_lengths)
        
        loss = \
            loss_dict['STFT'] + loss_dict['Log_Duration'] + loss_dict['TokenCTC'] + \
            loss_dict['Encoding_KLD'] + loss_dict['Linguistic_KLD'] + loss_dict['Acoustic_KLD'] + \
            loss_dict['Attention_Binarization'] + loss_dict['Attention_CTC']

        if self.steps >= self.hp.Train.Discrimination_Step:
            with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
                discriminations_list_for_real, feature_maps_list_for_real = self.model_dict['Discriminator'](audios_slice)
                discriminations_list_for_fake, feature_maps_list_for_fake = self.model_dict['Discriminator'](audio_predictions_slice)

            loss_dict['Adversarial'] = \
                self.criterion_dict['Feature_Map'](feature_maps_list_for_real, feature_maps_list_for_fake) + \
                self.criterion_dict['Generator'](discriminations_list_for_fake)
            loss = loss + loss_dict['Adversarial']
            
        self.optimizer_dict['HierSpeech'].zero_grad()
        self.scaler.scale(loss).backward()

        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer_dict['HierSpeech'])
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['HierSpeech'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['HierSpeech'])
        self.scaler.update()
        self.scheduler_dict['HierSpeech'].step()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, token_lengths, features, feature_lengths, audios, linear_spectrograms, attention_priors in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths,
                audios= audios,
                linear_spectrograms= linear_spectrograms,
                attention_priors= attention_priors
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['HierSpeech'].get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(self, tokens, token_lengths, features, feature_lengths, audios, linear_spectrograms, attention_priors):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        feature_lengths = feature_lengths.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)
        linear_spectrograms = linear_spectrograms.to(self.device, non_blocking=True)
        attention_priors = attention_priors.to(self.device, non_blocking=True)

        audio_predictions_slice, audios_slice, token_predictions, acoustic_distributions, \
        encoding_distributions_prior, encoding_distributions_posterior, linguistic_distributions_prior, linguistic_distributions_posterior, \
        log_duration_predictions, durations, attention_softs, attention_hards, attention_logprobs = self.model_dict['HierSpeech'](
            tokens= tokens,
            token_lengths= token_lengths,
            features= features,
            feature_lengths= feature_lengths,
            audios= audios,
            linear_spectrograms= linear_spectrograms,
            attention_priors= attention_priors
            )

        token_masks = Mask_Generate(
            lengths= token_lengths,
            max_length= tokens.size(1)
            ).to(tokens.device)

        loss_dict['STFT'] = self.criterion_dict['MSE'](
            mel_spectrogram(
                audio_predictions_slice,
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= self.hp.Sound.Mel_F_Min,
                fmax= self.hp.Sound.Mel_F_Max
                ),
            mel_spectrogram(
                audios_slice,
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= self.hp.Sound.Mel_F_Min,
                fmax= self.hp.Sound.Mel_F_Max
                )
            ).mean()
        loss_dict['Log_Duration'] = (self.criterion_dict['MSE'](
            log_duration_predictions,
            (durations.to(log_duration_predictions.dtype) + 1).log()
            ) * ~token_masks).mean()
        loss_dict['TokenCTC'] = self.criterion_dict['TokenCTC'](
                log_probs= token_predictions.permute(2, 0, 1),  # [Feature_t, Batch, Token_n]
                targets= tokens[:, 1:-1],   # Remove <S>, <E>
                input_lengths= feature_lengths,
                target_lengths= token_lengths - 2
                )
        loss_dict['Encoding_KLD'] = torch.distributions.kl_divergence(
            encoding_distributions_prior,
            encoding_distributions_posterior
            ).mean()
        loss_dict['Linguistic_KLD'] = torch.distributions.kl_divergence(
            linguistic_distributions_prior,
            linguistic_distributions_posterior
            ).mean()
        loss_dict['Acoustic_KLD'] = torch.distributions.kl_divergence(
            acoustic_distributions,
            torch.distributions.Normal(
                loc= 0.0,
                scale= 1.0
                )
            ).mean()
        loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
        loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, feature_lengths)
        
        if self.steps >= self.hp.Train.Discrimination_Step:
            discriminations_list_for_real, feature_maps_list_for_real = self.model_dict['Discriminator'](audios_slice)
            discriminations_list_for_fake, feature_maps_list_for_fake = self.model_dict['Discriminator'](audio_predictions_slice)

            loss_dict['Discrimination'] = self.criterion_dict['Discriminator'](discriminations_list_for_real, discriminations_list_for_fake)
            loss_dict['Adversarial'] = \
                self.criterion_dict['Feature_Map'](feature_maps_list_for_real, feature_maps_list_for_fake) + \
                self.criterion_dict['Generator'](discriminations_list_for_fake)
            
        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        for model in self.model_dict.values():
            model.eval()

        for step, (tokens, token_lengths, features, feature_lengths, audios, linear_spectrograms, attention_priors) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            self.Evaluation_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                features= features,
                feature_lengths= feature_lengths,
                audios= audios,
                linear_spectrograms= linear_spectrograms,
                attention_priors= attention_priors
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['HierSpeech'], 'HierSpeech', self.steps, delete_keywords=[])
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['Discriminator'], 'Discriminator', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))

            with torch.inference_mode():
                prediction_audios, *_, duration_predictions, _, _, _ = self.model_dict['HierSpeech'](
                    tokens= tokens[index].unsqueeze(0).to(self.device),
                    token_lengths= token_lengths[index].unsqueeze(0).to(self.device)
                    )
                prediction_audios = prediction_audios.clamp(-1.0, 1.0)
            
            target_feature_length = feature_lengths[index]
            prediction_feature_length = int(duration_predictions[0, :token_lengths[index] - 1].sum()) * self.hp.Sound.Frame_Shift
            target_audio_length = target_feature_length * self.hp.Sound.Frame_Shift
            prediction_audio_length = prediction_feature_length * self.hp.Sound.Frame_Shift

            target_audio = audios[index, :target_audio_length]
            prediction_audio = prediction_audios[0, :prediction_audio_length]

            target_feature = mel_spectrogram(
                target_audio.unsqueeze(0),
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= self.hp.Sound.Mel_F_Min,
                fmax= self.hp.Sound.Mel_F_Max
                ).squeeze(0).cpu().numpy()
            prediction_feature = mel_spectrogram(
                prediction_audio.unsqueeze(0),
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.Mel_Dim,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Frame_Shift,
                win_size= self.hp.Sound.Frame_Length,
                fmin= self.hp.Sound.Mel_F_Min,
                fmax= self.hp.Sound.Mel_F_Max
                ).squeeze(0).cpu().numpy()
            
            target_audio = target_audio.cpu().numpy()
            prediction_audio = prediction_audio.cpu().numpy()

            duration_prediction = duration_predictions[0, :token_lengths[index]]
            duration_prediction = torch.arange(duration_prediction.size(0)).repeat_interleave(duration_prediction.cpu()).numpy()
            
            image_dict = {
                'Feature/Target': (target_feature, None, 'auto', None, None, None),
                'Feature/Prediction': (prediction_feature, None, 'auto', None, None, None),
                'Duration/Prediction': (duration_prediction[:feature_lengths[index]], None, 'auto', (0, features.size(2)), (0, tokens.size(1)), None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (prediction_audio, self.hp.Sound.Sample_Rate),
                }

            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Feature.Target': wandb.Image(target_feature),
                        'Evaluation.Feature.Prediction': wandb.Image(prediction_feature),
                        'Evaluation.Duration': wandb.plot.line_series(
                            xs= np.arange(feature_lengths[index].cpu().numpy()),
                            ys= [
                                    duration_prediction[:feature_lengths[index]]
                                    ],
                            keys= ['Prediction'],
                            title= 'Duration',
                            xname= 'Feature_t'
                            ),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Linear': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Linear_Audio'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()

    @torch.inference_mode()
    def Inference_Step(self, tokens, token_lengths, texts, decomposed_texts, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)

        audio_predictions, *_, durations, _, _, _ = self.model_dict['HierSpeech'](
            tokens= tokens,
            token_lengths= token_lengths,
            )
        audio_predictions = audio_predictions.clamp(-1.0, 1.0)

        feature_lengths = [
            int(duration[:token_length - 1].sum())
            for duration, token_length in zip(durations, token_lengths)
            ]
        audio_lengths = [
            length * self.hp.Sound.Frame_Shift
            for length in feature_lengths
            ]

        files = []
        for index in range(tokens.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        durations = [
            torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
            for duration in durations
            ]

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            audio,
            duration,
            feature_length,
            audio_length,
            text,
            decomposed_text,
            file
            ) in enumerate(zip(
            audio_predictions.cpu().numpy(),
            durations,
            feature_lengths,
            audio_lengths,
            texts,
            decomposed_texts,
            files
            )):
            title = 'Text: {}'.format(text if len(text) < 90 else text[:90] + '…')
            new_figure = plt.figure(figsize=(20, 5 * 5), dpi=100)
            ax = plt.subplot2grid((3, 1), (0, 0))
            plt.plot(audio[:audio_length])
            plt.title('Prediction    {}'.format(title))
            plt.margins(x= 0)
            ax = plt.subplot2grid((3, 1), (1, 0), rowspan= 2)
            plt.plot(duration[:feature_length])
            plt.title('Duration    {}'.format(title))
            plt.margins(x= 0)
            plt.yticks(
                range(len(decomposed_text) + 2),
                ['<S>'] + list(decomposed_text) + ['<E>'],
                fontsize = 10
                )
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio
                )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, token_lengths, texts, decomposed_texts) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, token_lengths, texts, decomposed_texts, start_index= step * batch_size)

        for model in self.model_dict.values():
            model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model_dict['HierSpeech'].load_state_dict(state_dict['Model']['HierSpeech'])
        self.model_dict['Discriminator'].load_state_dict(state_dict['Model']['Discriminator'])
        self.optimizer_dict['HierSpeech'].load_state_dict(state_dict['Optimizer']['HierSpeech'])
        self.optimizer_dict['Discriminator'].load_state_dict(state_dict['Optimizer']['Discriminator'])
        self.scheduler_dict['HierSpeech'].load_state_dict(state_dict['Scheduler']['HierSpeech'])
        self.scheduler_dict['Discriminator'].load_state_dict(state_dict['Scheduler']['Discriminator'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': {
                'HierSpeech': self.model_dict['HierSpeech'].state_dict(),
                'Discriminator': self.model_dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'HierSpeech': self.optimizer_dict['HierSpeech'].state_dict(),
                'Discriminator': self.optimizer_dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'HierSpeech': self.scheduler_dict['HierSpeech'].state_dict(),
                'Discriminator': self.scheduler_dict['Discriminator'].state_dict(),
                },
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)

    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)

        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl'
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()