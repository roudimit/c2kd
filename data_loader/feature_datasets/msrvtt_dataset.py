from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pickle
from torch.utils.data.dataloader import default_collate
import random
import torch
from transformers import AutoTokenizer
from data_loader.feature_datasets.howto_dataset import create_audio_features, create_text_features, \
    create_video_features, _tokenize_text
from data_loader.feature_datasets.youcook_dataset import cut_into_clips


class MSRVTT_Dataset(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            data_path,
            we,
            data_path_3D=None,
            we_dim=300,
            max_words=30,
            training=True,
            n_video_tokens=15,
            num_audio_frames=1024,
            num_frames_multiplier=1,
            video_sampling_strategy='max_pool',
            sample_audio_clips=False,
            sample_audio_video_clips=False,
            use_2D=True,
            use_3D=True,
            key_2d='2d',
            key_3d='3d',
    ):
        """
        Args:
        """
        assert use_2D or use_3D
        self.data = pickle.load(open(data_path, 'rb'))
        if use_3D:
            self.data_3D = self.data if data_path_3D is None else pickle.load(open(data_path_3D, 'rb'))
        else:
            self.data_3D = None
        self.use_2D = use_2D
        self.use_3D = use_3D
        self.key_2d = key_2d
        self.key_3d = key_3d

        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.training = training

        self.n_video_tokens = n_video_tokens
        self.num_audio_frames = num_audio_frames
        self.num_frames_multiplier = num_frames_multiplier
        self.video_sampling_strategy = video_sampling_strategy
        self.sample_audio_clips = sample_audio_clips
        self.sample_audio_video_clips = sample_audio_video_clips
        assert self.sample_audio_clips is False or self.sample_audio_video_clips is False
        if self.sample_audio_video_clips:
            assert video_sampling_strategy == 'clip'

        if not training and len(self.data) == 968: # update for vatex
            assert len(self.data) == 968
            self.complete_dataset_size = 1000
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE', local_files_only=False)
        self.tokenizer_xlm = AutoTokenizer.from_pretrained('xlm-roberta-base', local_files_only=False) 
        self.tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=False)
        self.tokenizer_infoxlm = AutoTokenizer.from_pretrained("microsoft/infoxlm-base", local_files_only=False)
        self.tokenizer_xlm_align = AutoTokenizer.from_pretrained("microsoft/xlm-align-base", local_files_only=False)
        self.tokenizer_distill = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2', local_files_only=False)
        self.tokenizer_sim_cse = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base", local_files_only=False)
        self.langs = ['de', 'fr', 'cs', 'zh', 'ru', 'vi', 'sw', 'es', 'en']
        self.multilingual = True

    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def __getitem__(self, idx):
        if self.sample_audio_clips:
            target_nvideo_tokens = self.n_video_tokens
        elif self.sample_audio_video_clips:
            target_nvideo_tokens = self.n_video_tokens * self.num_frames_multiplier
        else:
            target_nvideo_tokens = self.n_video_tokens

        # load 2d and 3d features
        id_ = self.data[idx]['id']
        feat_2d = None
        feat_3d = None
        if self.use_2D:
            feat_2d = torch.from_numpy(self.data[idx][self.key_2d]).float()
        if self.use_3D:
            feat_3d = torch.from_numpy(self.data_3D[idx][self.key_3d]).float()
        video, video_mask = create_video_features(feat_2d, feat_3d, target_nvideo_tokens, strategy=self.video_sampling_strategy)
        # load audio
        if 'audio' in self.data[idx]:
            audio = self.data[idx]['audio']
            target_nframes = self.num_audio_frames * self.num_frames_multiplier
            audio, audio_mask, nframes = create_audio_features(audio, target_nframes)
        else: # no audio in vatex
            audio, audio_mask, nframes = torch.zeros((40, 10)), torch.zeros(10), 10

        # choose a caption
        if self.training:
            cap_idx = random.choice(range(len(self.data[idx]['caption_en'])))
            caption = self.data[idx]['caption_en'][cap_idx]
        else:
            caption = self.data[idx]['eval_caption_en']
        words = _tokenize_text(caption)
        text, text_mask, raw_text = create_text_features(words, self.max_words, self.we, self.we_dim)

        # meta data
        unroll_fake_1st_dim = 0
        dataset = 'MSRVTT'

        if self.sample_audio_clips or self.sample_audio_video_clips:
            video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, id_, dataset = \
                cut_into_clips(video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, id_, dataset,
                               n_clips=self.num_frames_multiplier, flag_cut_video=self.sample_audio_video_clips)
            unroll_fake_1st_dim = 1

        tokens_en = self.tokenizer(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en['input_ids'] = tokens_en['input_ids'][0] # fix dimensions
        if 'token_type_ids' in tokens_en:
            del tokens_en['token_type_ids'] # fix for Multilingual-CLIP-BERT
        
        tokens_en_xlm = self.tokenizer_xlm(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_xlm['input_ids'] = tokens_en_xlm['input_ids'][0] # fix dimensions

        tokens_en_mbert = self.tokenizer_mbert(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_mbert['input_ids'] = tokens_en_mbert['input_ids'][0] # fix dimensions
        if 'token_type_ids' in tokens_en_mbert:
            del tokens_en_mbert['token_type_ids'] # fix for mbert

        tokens_en_infoxlm = self.tokenizer_infoxlm(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_infoxlm['input_ids'] = tokens_en_infoxlm['input_ids'][0] # fix dimensions

        tokens_en_xlm_align = self.tokenizer_xlm_align(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_xlm_align['input_ids'] = tokens_en_xlm_align['input_ids'][0] # fix dimensions

        tokens_en_distill = self.tokenizer_distill(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_distill['input_ids'] = tokens_en_distill['input_ids'][0] # fix dimensions

        tokens_en_sim_cse = self.tokenizer_sim_cse(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_sim_cse['input_ids'] = tokens_en_sim_cse['input_ids'][0] # fix dimensions

        if self.multilingual:
            multilingual_tokens = {i: '' for i in self.langs if 'caption_{}'.format(i) in self.data[idx] or 'eval_caption_{}'.format(i) in self.data[idx]}
            multilingual_tokens_xlm = {}
            multilingual_tokens_mbert = {}
            multilingual_tokens_infoxlm = {}
            multilingual_tokens_xlm_align = {}
            multilingual_tokens_distill = {}
            multilingual_tokens_sim_cse = {}
            
            for lang in multilingual_tokens:
                if self.training:
                    caption = self.data[idx]['caption_{}'.format(lang)][cap_idx] # use same index for all langs
                else:
                    caption = self.data[idx]['eval_caption_{}'.format(lang)]
                tokens = self.tokenizer(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
                tokens['input_ids'] = tokens['input_ids'][0] # fix dimensions
                if 'token_type_ids' in tokens:
                    del tokens['token_type_ids'] # fix for Multilingual-CLIP-BERT
                multilingual_tokens[lang] = tokens

                tokens_xlm = self.tokenizer_xlm(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
                tokens_xlm['input_ids'] = tokens_xlm['input_ids'][0] # fix dimensions
                multilingual_tokens_xlm['{}_xlm'.format(lang)] = tokens_xlm

                tokens_mbert = self.tokenizer_mbert(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
                tokens_mbert['input_ids'] = tokens_mbert['input_ids'][0] # fix dimensions
                if 'token_type_ids' in tokens_mbert:
                    del tokens_mbert['token_type_ids'] # firx for mbert
                multilingual_tokens_mbert['{}_mbert'.format(lang)] = tokens_mbert

                tokens_infoxlm = self.tokenizer_infoxlm(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
                tokens_infoxlm['input_ids'] = tokens_infoxlm['input_ids'][0] # fix dimensions
                multilingual_tokens_infoxlm['{}_infoxlm'.format(lang)] = tokens_infoxlm

                tokens_xlm_align = self.tokenizer_xlm_align(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
                tokens_xlm_align['input_ids'] = tokens_xlm_align['input_ids'][0] # fix dimensions
                multilingual_tokens_xlm_align['{}_xlm_align'.format(lang)] = tokens_xlm_align

                tokens_distill = self.tokenizer_distill(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
                tokens_distill['input_ids'] = tokens_distill['input_ids'][0] # fix dimensions
                multilingual_tokens_distill['{}_distill'.format(lang)] = tokens_distill

                multilingual_tokens_sim_cse['{}_sim_cse'.format(lang)] = tokens_en_sim_cse # Hack to get the rest of the code to work


        return {**{'video': video, 'audio': audio, 'text': text, 'nframes': nframes,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_fake_1st_dim': unroll_fake_1st_dim,
                'meta': {'paths': id_, 'ids': id_, 'dataset': dataset}, 
                'tokens': tokens_en, 'tokens_xlm': tokens_en_xlm, 'tokens_mbert': tokens_en_mbert, 
                'tokens_infoxlm': tokens_en_infoxlm, 'tokens_xlm_align': tokens_en_xlm_align,
                'tokens_distill': tokens_en_distill, 'tokens_sim_cse': tokens_en_sim_cse}, 
                **multilingual_tokens, **multilingual_tokens_xlm, **multilingual_tokens_mbert, 
                **multilingual_tokens_infoxlm, **multilingual_tokens_xlm_align,
                **multilingual_tokens_distill, **multilingual_tokens_sim_cse}