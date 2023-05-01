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


class RUDDER_Dataset(Dataset):
    """RUDDER dataset loader."""

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
        self.complete_dataset_size = len(self.data)

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE', local_files_only=False)
        self.tokenizer_xlm = AutoTokenizer.from_pretrained('xlm-roberta-base', local_files_only=False) 
        self.tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=False)
        self.tokenizer_distill = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2', local_files_only=False)
        self.tokenizer_sim_cse = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base", local_files_only=False)
        # self.langs = ['en', 'hi', 'mr', 'ta', 'te', 'kn', 'ml']
        self.langs = ['en', 'hi', 'mr', 'kn']

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

        # meta data
        unroll_fake_1st_dim = 0
        dataset = 'RUDDER'

        langs_available = [i for i in self.data[idx]['captions'] if i in self.langs]
        multilingual_tokens, multilingual_audio = {}, {}
        multilingual_tokens_xlm = {}
        multilingual_tokens_mbert = {}
        multilingual_tokens_distill = {}
        multilingual_tokens_sim_cse = {}
        for lang in langs_available:
            caption = self.data[idx]['captions']['{}'.format(lang)]
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

            tokens_distill = self.tokenizer_distill(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
            tokens_distill['input_ids'] = tokens_distill['input_ids'][0] # fix dimensions
            multilingual_tokens_distill['{}_distill'.format(lang)] = tokens_distill

            tokens_sim_cse = self.tokenizer_sim_cse(caption, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
            tokens_sim_cse['input_ids'] = tokens_sim_cse['input_ids'][0] # fix dimensions
            multilingual_tokens_sim_cse['{}_sim_cse'.format(lang)] = tokens_sim_cse

            # add audio for each language
            audio_tokens = {}
            audio = self.data[idx]['audios']['{}'.format(lang)]
            target_nframes = self.num_audio_frames * self.num_frames_multiplier
            audio, audio_mask, nframes = create_audio_features(audio, target_nframes)
            audio_tokens['audio'] = audio
            audio_tokens['nframes'] = nframes
            audio_tokens['audio_mask'] = audio_mask
            audio_tokens['use_lang'] = True # use this to decide to use the lang or not
            multilingual_audio[lang + '_audio'] = audio_tokens
        for lang in set(self.langs) - set(langs_available): # handle missing langs
            multilingual_tokens[lang] = tokens # works because at least one lang had tokens
            audio_tokens_clone = audio_tokens.copy()
            audio_tokens_clone['use_lang'] = False
            multilingual_audio[lang + '_audio'] = audio_tokens_clone

            multilingual_tokens_xlm['{}_xlm'.format(lang)] = tokens_xlm
            multilingual_tokens_mbert['{}_mbert'.format(lang)] = tokens_mbert
            multilingual_tokens_distill['{}_distill'.format(lang)] = tokens_distill
            multilingual_tokens_sim_cse['{}_sim_cse'.format(lang)] = tokens_sim_cse

        # handle english - use another lang if no captions available
        lang = 'en' if 'en' in langs_available else langs_available[0]
        caption = self.data[idx]['captions']['{}'.format(lang)]
        words = _tokenize_text(caption)
        text, text_mask, raw_text = create_text_features(words, self.max_words, self.we, self.we_dim)
        tokens_en = multilingual_tokens[lang]
        audio = multilingual_audio[lang + '_audio']['audio']
        audio_mask = multilingual_audio[lang + '_audio']['audio_mask']
        nframes = multilingual_audio[lang + '_audio']['nframes']

        tokens_en_xlm = multilingual_tokens_xlm['{}_xlm'.format(lang)]
        tokens_en_mbert = multilingual_tokens_mbert['{}_mbert'.format(lang)]
        tokens_en_distill = multilingual_tokens_distill['{}_distill'.format(lang)]
        tokens_en_sim_cse = multilingual_tokens_sim_cse['{}_sim_cse'.format(lang)]

        return {**{'video': video, 'audio': audio, 'text': text, 'nframes': nframes,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_fake_1st_dim': unroll_fake_1st_dim,
                'meta': {'paths': id_, 'ids': id_, 'dataset': dataset}, 
                'tokens': tokens_en, 'tokens_xlm': tokens_en_xlm, 'tokens_mbert': tokens_en_mbert, 
                'tokens_distill': tokens_en_distill, 'tokens_sim_cse': tokens_en_sim_cse}, 
                **multilingual_tokens, **multilingual_audio, 
                **multilingual_tokens_xlm, **multilingual_tokens_mbert, 
                **multilingual_tokens_distill, **multilingual_tokens_sim_cse}