from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pickle
from torch.utils.data.dataloader import default_collate
import torch
from transformers import AutoTokenizer
from data_loader.feature_datasets.howto_dataset import create_audio_features, create_text_features, \
    create_video_features, _tokenize_text


class Youcook_Dataset(Dataset):
    """Youcook dataset loader."""

    def __init__(
            self,
            we,
            data_path=None,
            data_path_2D=None,
            data_path_3D=None,
            we_dim=300,
            max_words=30,
            n_video_tokens=15,
            num_audio_frames=1024,
            num_frames_multiplier=1,
            video_sampling_strategy='max_pool',
            sample_audio_clips=False,
            sample_audio_video_clips=False,
            use_2D=True,
            use_3D=True,
            key_2d='2d_full',
            key_3d='3d_full',
    ):
        """
        Args:
        """
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
        if data_path_2D is None:
            data_2D = data
        else:
            data_2D = pickle.load(open(data_path_2D, 'rb'))
        if data_path_3D is None:
            data_3D = data
        else:
            data_3D = pickle.load(open(data_path_3D, 'rb'))

        self.data = data
        self.data_2D = []
        self.data_3D = []
        for i in range(len(data)):
            if '2d_full' in data[i]:
                self.data_2D.append(data_2D[i])
                self.data_3D.append(data_3D[i])
        if len(self.data_2D) == 0:
            self.data_2D = data_2D # fix for multilingual
        if len(self.data_3D) == 0:
            self.data_3D = data_3D # fix for multilingual

        self.use_2D = use_2D
        self.use_3D = use_3D
        self.key_2d = key_2d
        self.key_3d = key_3d

        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words

        self.num_audio_frames = num_audio_frames
        self.n_video_tokens = n_video_tokens
        self.num_frames_multiplier = num_frames_multiplier
        self.video_sampling_strategy = video_sampling_strategy
        self.sample_audio_clips = sample_audio_clips
        self.sample_audio_video_clips = sample_audio_video_clips
        assert self.sample_audio_clips is False or self.sample_audio_video_clips is False
        if self.sample_audio_video_clips:
            assert video_sampling_strategy == 'clip'

        # assert len(self.data) == 3339
        self.complete_dataset_size = 3350
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE', local_files_only=False)
        self.tokenizer_xlm = AutoTokenizer.from_pretrained('xlm-roberta-base', local_files_only=False) 
        self.tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=False)
        self.tokenizer_distill = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2', local_files_only=False)
        self.tokenizer_sim_cse = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base", local_files_only=False)
        self.langs = ['de', 'fr', 'cz', 'zh', 'ru', 'vi', 'es', 'en', 'ja'] # no sw and cs is labeled cz
        self.multilingual = True

    def __len__(self):
        return len(self.data_2D)

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
        feat_2d = None
        feat_3d = None
        if self.use_2D:
            feat_2d = torch.from_numpy(self.data_2D[idx][self.key_2d]).float()
        if self.use_3D:
            feat_3d = torch.from_numpy(self.data_3D[idx][self.key_3d]).float()
        video, video_mask = create_video_features(feat_2d, feat_3d, target_nvideo_tokens,
                                                  strategy=self.video_sampling_strategy,
                                                  )
        # load audio
        audio = self.data_2D[idx]['audio']
        target_nframes = self.num_audio_frames * self.num_frames_multiplier
        audio, audio_mask, nframes = create_audio_features(audio, target_nframes)

        # load text
        caption = self.data_2D[idx]['caption']
        words = _tokenize_text(caption)
        text, text_mask, raw_text= create_text_features(words, self.max_words, self.we, self.we_dim)

        id_ = str(self.data_2D[idx]['id'])
        unroll_fake_1st_dim = 0
        dataset = 'YouCook2'

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

        tokens_en_distill = self.tokenizer_distill(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_distill['input_ids'] = tokens_en_distill['input_ids'][0] # fix dimensions

        tokens_en_sim_cse = self.tokenizer_sim_cse(raw_text, padding='max_length', max_length=self.max_words, truncation=True, return_tensors='pt')
        tokens_en_sim_cse['input_ids'] = tokens_en_sim_cse['input_ids'][0] # fix dimensions

        multilingual_tokens = {}
        multilingual_tokens_xlm = {}
        multilingual_tokens_mbert = {}
        multilingual_tokens_distill = {}
        multilingual_tokens_sim_cse = {}
        # multi_raw_text = {'raw_text_en': self.data[idx]['caption_en']}
        if self.multilingual:
            for lang in self.langs:
                caption = self.data[idx]['caption_{}'.format(lang)]
                # multi_raw_text['raw_text_{}'.format(lang)] = caption
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

            multilingual_tokens['cs'] = multilingual_tokens['cz']
            del multilingual_tokens['cz'] # fix Czech language code

            multilingual_tokens_xlm['cs_xlm'] = multilingual_tokens_xlm['cz_xlm']
            del multilingual_tokens_xlm['cz_xlm'] # fix Czech language code

            multilingual_tokens_mbert['cs_mbert'] = multilingual_tokens_mbert['cz_mbert']
            del multilingual_tokens_mbert['cz_mbert'] # fix Czech language code

            multilingual_tokens_distill['cs_distill'] = multilingual_tokens_distill['cz_distill']
            del multilingual_tokens_distill['cz_distill'] # fix Czech language code

            multilingual_tokens_sim_cse['cs_sim_cse'] = multilingual_tokens_sim_cse['cz_sim_cse']
            del multilingual_tokens_sim_cse['cz_sim_cse'] # fix Czech language code

        return {**{'video': video, 'audio': audio, 'text': text, 'nframes': nframes,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_fake_1st_dim': unroll_fake_1st_dim,
                'meta': {'paths': id_, 'ids': id_, 'dataset': dataset},
                'tokens': tokens_en, 'tokens_xlm': tokens_en_xlm, 'tokens_mbert': tokens_en_mbert, 
                'tokens_distill': tokens_en_distill, 'tokens_sim_cse': tokens_en_sim_cse}, 
                **multilingual_tokens, **multilingual_tokens_xlm, **multilingual_tokens_mbert, 
                **multilingual_tokens_distill, **multilingual_tokens_sim_cse}

def cut_into_clips(video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, id_, dataset,
                   n_clips, flag_cut_video):
    # create audio clips
    num_audio_frames = int(audio_mask.shape[0] // n_clips)
    audio = audio.permute(1, 0) \
        .view(n_clips, num_audio_frames, audio.size(0)) \
        .permute(0, 2, 1)
    audio_mask = audio_mask.view(n_clips, num_audio_frames)

    # create video clips
    n_video_tokens = int(video_mask.shape[0] // n_clips)
    if flag_cut_video:
        video = video.view(n_clips, n_video_tokens, video.size(-1))
        video_mask = video_mask.view(n_clips, n_video_tokens)
    else:
        video = video.unsqueeze(0).expand(n_clips, -1, -1)
        video_mask = video_mask.unsqueeze(0).expand(n_clips, -1)

    # copy text
    text = text.unsqueeze(0).expand(n_clips, -1, -1)
    text_mask = text_mask.unsqueeze(0).expand(n_clips, -1)

    # determine nframes
    new_n_frames = []
    new_id = []
    for i in range(n_clips):
        left_frames = nframes - i * num_audio_frames
        if (i == 0) or (left_frames > 0.7 * num_audio_frames):
            new_n_frames.append(min(num_audio_frames, left_frames))
            new_id.append(id_)
        else:
            new_n_frames.append(num_audio_frames)
            new_id.append('-1')
    nframes = torch.tensor(new_n_frames)
    id_ = new_id
    dataset = [dataset] * n_clips
    raw_text = [raw_text] * n_clips
    return video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, id_, dataset
