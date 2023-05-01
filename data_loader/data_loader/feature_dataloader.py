from functools import partial

import braceexpand
import numpy as np
import torch

from base import BaseDataLoaderExplicitSplit
from data_loader.feature_datasets.msrvtt_dataset import MSRVTT_Dataset
from data_loader.feature_datasets.rudder_dataset import RUDDER_Dataset
from data_loader.feature_datasets.youcook_dataset import Youcook_Dataset
import pickle
from io import BytesIO

caption = None
we = None


class FeatureDataloader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 dataset_kwargs,
                 drop_last=False,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True,
                 split=None, # TODO: delete
                 sliding_window_stride=None):

        from gensim.models.keyedvectors import KeyedVectors
        word2vec_path = dataset_kwargs.pop('word2vec_path')
        global we
        if we is None:
            we = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            print('Using loaded we')

        if 'MSRVTT' in dataset_name:
            dataset = MSRVTT_Dataset(**dataset_kwargs, we=we)
        elif 'RUDDER' in dataset_name:
            dataset = RUDDER_Dataset(**dataset_kwargs, we=we)
        elif 'YouCook2' in dataset_name:
            dataset = Youcook_Dataset(**dataset_kwargs, we=we)
        else:
            raise NotImplementedError(f"Dataset: {dataset_name} not found.")

        super().__init__(dataset, batch_size, shuffle, num_workers,
                         drop_last=drop_last)
        self.dataset_name = dataset_name

    def resample_dataset(self):
        self.dataset.sample_dataset()


def _sample_from_dataset(sample, we, dataset_kwargs):
    video_id = sample['__key__']

    features_2d = sample['2d.pth'].float()
    features_3d = sample['3d.pth'].float()
    caption = sample['captions.pyd']
    mel_spec = sample['audio.pth'].numpy()

    video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, paths, ids, starts, ends = \
        _sample_video_audio_text(caption, mel_spec, features_2d, features_3d, video_id, we=we, **dataset_kwargs)
    # print(video.shape, video_mask.shape, audio.shape, audio_mask.shape, text.shape, text_mask.shape, len(nframes), len(ids), len(paths))
    return {'video': video, 'audio': audio, 'text': text,
            'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
            'raw_text': raw_text,
            'unroll_fake_1st_dim': 1, 'nframes': nframes,
            'paths': paths, 'ids': ids, 'dataset': ['HowTo100M'] * len(video)}