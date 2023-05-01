from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch as th
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import re
import librosa


def _tokenize_text(sentence):
    w = re.findall(r"[\w']+", str(sentence))
    return w


def create_audio_features(mel_spec, num_audio_frames):
    audio = np.zeros((mel_spec.shape[0], num_audio_frames), dtype=np.float32)
    audio_mask = np.zeros(num_audio_frames, dtype=np.float32)

    nframes = min(mel_spec.shape[1], num_audio_frames)
    audio[:, :nframes] = mel_spec[:, :nframes]
    audio_mask[:nframes] = 1

    audio = torch.from_numpy(audio).float()
    audio_mask = torch.from_numpy(audio_mask).float()
    return audio, audio_mask, nframes


def create_text_features(words, max_words, we, we_dim):
    raw_text = ' '.join(words)
    words = [word for word in words if word in we.vocab]
    text = np.zeros((max_words, we_dim), dtype=np.float32)
    text_mask = np.zeros(max_words, dtype=np.float32)
    nwords = min(len(words), max_words)
    if nwords > 0:
        text[:nwords] = we[words][:nwords]
        text_mask[:nwords] = 1
    text = torch.from_numpy(text).float()
    text_mask = torch.from_numpy(text_mask).float()

    return text, text_mask, raw_text


def create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip'):
    if n_tokens == 0:
        feat_2d = F.normalize(torch.max(feat_2d, dim=0)[0], dim=0) if len(feat_2d) else torch.zeros(feat_2d.shape[1])
        feat_3d = F.normalize(torch.max(feat_3d, dim=0)[0], dim=0) if len(feat_3d) else torch.zeros(feat_3d.shape[1])
        video = torch.cat((feat_2d, feat_3d))
        video_mask = torch.ones(1) # TODO: not exactly 0
        return video, video_mask
    else:
        if strategy == 'clip':
            if feat_2d is None:
                video = torch.zeros(n_tokens, feat_3d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                cur_n_tokens_3d, dim_3d = feat_3d.shape
                video[:cur_n_tokens_3d] = F.normalize(feat_3d[:n_tokens], dim=1)
                video_mask[:cur_n_tokens_3d] = 1
                return video, video_mask
            elif feat_3d is None:
                video = torch.zeros(n_tokens, feat_2d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                video[:cur_n_tokens_2d] = F.normalize(feat_2d[:n_tokens], dim=1)
                video_mask[:cur_n_tokens_2d] = 1
                return video, video_mask
            else:
                video = torch.zeros(n_tokens, feat_2d.shape[-1] + feat_3d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                cur_n_tokens_3d, dim_3d = feat_3d.shape

                if cur_n_tokens_2d != 0 and cur_n_tokens_3d != 0:
                    feat_2d = torch.nn.functional.interpolate(
                        feat_2d.permute(1, 0).unsqueeze(0),
                        size=cur_n_tokens_3d,
                        mode='nearest').squeeze(0).permute(1, 0)

                    video[:cur_n_tokens_3d, :dim_2d] = F.normalize(feat_2d[:n_tokens], dim=1)
                    video[:cur_n_tokens_3d, dim_2d:] = F.normalize(feat_3d[:n_tokens], dim=1)
                    video_mask[:cur_n_tokens_3d] = 1
                return video, video_mask
        elif strategy == 'nearest':
            if feat_2d is None:
                cur_n_tokens_3d, dim_3d = feat_3d.shape
                if cur_n_tokens_3d <= n_tokens:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')
                feat_3d = torch.nn.functional.interpolate(
                    feat_3d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                video = F.normalize(feat_3d, dim=1)
                video_mask = torch.ones(n_tokens)
                return video, video_mask
            elif feat_3d is None:
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                if cur_n_tokens_2d <= n_tokens:
                    return create_video_features(feat_2d, feat_2d, n_tokens, strategy='clip')
                feat_2d = torch.nn.functional.interpolate(
                    feat_2d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                video = F.normalize(feat_2d, dim=1)
                video_mask = torch.ones(n_tokens)
                return video, video_mask
            else:
                cur_n_tokens_2d, dim_2d = feat_2d.shape
                cur_n_tokens_3d, dim_3d = feat_3d.shape
                if cur_n_tokens_3d <= n_tokens or cur_n_tokens_2d == 0:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')

                video = torch.zeros(n_tokens, feat_2d.shape[-1] + feat_3d.shape[-1])
                video_mask = torch.zeros(n_tokens)
                feat_2d = torch.nn.functional.interpolate(
                    feat_2d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                feat_3d = torch.nn.functional.interpolate(
                    feat_3d.permute(1, 0).unsqueeze(0),
                    size=n_tokens,
                    mode='nearest').squeeze(0).permute(1, 0)
                video[:, :dim_2d] = F.normalize(feat_2d, dim=1)
                video[:, dim_2d:] = F.normalize(feat_3d, dim=1)
                video_mask[:] = 1
                return video, video_mask

        elif strategy == 'max_pool':
            if feat_2d is None:
                cur_n_tokens_3d = feat_3d.shape[0]
                if cur_n_tokens_3d <= n_tokens:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')
                kernel_size_3d = int(np.floor(cur_n_tokens_3d / n_tokens))
                if kernel_size_3d <= 1:  # we don't have what to max pool
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')
                feat_3d = torch.nn.functional.max_pool1d(feat_3d.permute(1, 0), kernel_size=kernel_size_3d).permute(1,
                                                                                                                    0)
                return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')
            elif feat_3d is None:
                cur_n_tokens_2d = feat_2d.shape[0]
                if cur_n_tokens_2d <= n_tokens:
                    return create_video_features(feat_2d, feat_2d, n_tokens, strategy='clip')
                kernel_size_2d = int(np.floor(cur_n_tokens_2d / n_tokens))
                if kernel_size_2d <= 1:  # we don't have what to max pool
                    return create_video_features(feat_2d, feat_2d, n_tokens, strategy='nearest')
                feat_2d = torch.nn.functional.max_pool1d(feat_2d.permute(1, 0), kernel_size=kernel_size_2d).permute(1,
                                                                                                                    0)
                return create_video_features(feat_2d, feat_2d, n_tokens, strategy='nearest')
            else:
                cur_n_tokens_2d = feat_2d.shape[0]
                cur_n_tokens_3d = feat_3d.shape[0]
                if cur_n_tokens_3d <= n_tokens or cur_n_tokens_2d == 0:
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='clip')

                kernel_size_3d = int(np.floor(cur_n_tokens_3d / n_tokens))
                kernel_size_2d = int(np.floor(cur_n_tokens_2d / n_tokens))

                if kernel_size_2d <= 1 or kernel_size_3d <= 1: # we don't have what to max pool
                    return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')

                feat_2d = torch.nn.functional.max_pool1d(feat_2d.permute(1, 0), kernel_size=kernel_size_2d).permute(1, 0)
                feat_3d = torch.nn.functional.max_pool1d(feat_3d.permute(1, 0), kernel_size=kernel_size_3d).permute(1, 0)
                return create_video_features(feat_2d, feat_3d, n_tokens, strategy='nearest')
        else:
            raise NotImplementedError


def _crop_audio_from_mel_spec(start, end, mel_spec):
    frames = librosa.core.time_to_frames([start, end], sr=16000, hop_length=160,
                                         n_fft=400)
    mel_spec = mel_spec[:, max(0, frames[0]): frames[1]]
    return mel_spec


def _get_single_random_text_and_audio(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_frames, ind=None,
                                      flag_text_audio_misaligned=False, flag_text_audio_misaligned_hard=False,
                                      flag_text_audio_misaligned_random=False):
    if ind is None:
        n_caption = len(caption['start'])
        if n_caption == 0:  # TODO: this a fix for tarred data
            caption['start'] = np.array([0])
            caption['end'] = np.array([0])
            caption['text'] = np.array([''])
            n_caption = 1
            print('Warning: no captions for the video')
        ind = np.random.choice(range(n_caption))

    start, end = ind, ind
    words = _tokenize_text(caption['text'][ind])
    diff = caption['end'][end] - caption['start'][start]
    # Extend the video clip if shorter than the minimum desired clip duration
    while diff < min_time:
        if start > 0 and end < len(caption['end']) - 1:
            next_words = _tokenize_text(caption['text'][end + 1])
            prev_words = _tokenize_text(caption['text'][start - 1])
            d1 = caption['end'][end + 1] - caption['start'][start]
            d2 = caption['end'][end] - caption['start'][start - 1]
            # Use the closest neighboring video clip
            if d2 <= d1:
                start -= 1
                words.extend(prev_words)
            else:
                end += 1
                words.extend(next_words)
        # If no video clips after it, use the clip before it
        elif start > 0:
            words.extend(_tokenize_text(caption['text'][start - 1]))
            start -= 1
        # If no video clips before it, use the clip after it.
        elif end < len(caption['end']) - 1:
            words.extend(_tokenize_text(caption['text'][end + 1]))
            end += 1
        # If there's no clips before or after
        else:
            break
        diff = caption['end'][end] - caption['start'][start]

    start, end = caption['start'][start], caption['end'][end]
    if flag_text_audio_misaligned:
        if np.random.rand() > 0:
            start_audio = start + 0.5 * min_time
            end_audio = start_audio + min_time
        else:
            start_audio = max(0, start - 0.5 * min_time)
            end_audio = start_audio + min_time
    elif flag_text_audio_misaligned_hard:
        start_audio = start +  min_time
        end_audio = start_audio + min_time
    elif flag_text_audio_misaligned_random:
        start_audio = start + (np.random.rand() - 0.5) * min_time
        end_audio = start_audio + min_time
    else:
        start_audio = start
        end_audio = end

    # create padded audio features
    mel_spec = _crop_audio_from_mel_spec(start_audio, end_audio, mel_spec)
    audio, audio_mask, nframes = create_audio_features(mel_spec, num_audio_frames)

    # create padded text features
    text, text_mask, raw_text = create_text_features(words, max_words, we, we_dim)

    return audio, audio_mask, nframes, start, end, text, text_mask, raw_text


def _get_single_random_audio_and_text(caption, mel_spec, min_time,  max_words, we, we_dim, num_audio_frames,
                                      flag_text_audio_misaligned=False,
                                      flag_text_audio_misaligned_hard=False,
                                      flag_text_audio_misaligned_random=False):
    video_duration_seconds = int(librosa.core.frames_to_time(mel_spec.shape[1], sr=16000, hop_length=160, n_fft=400))
    # Sample clips that end before the end of the video
    # If the video is shorter than the desired window, use the entire video
    start = np.random.rand() * max(0, video_duration_seconds - min_time)
    end = start + min_time

    ind_start = max(0, np.searchsorted(caption['start'], start, side='left') - 1)
    ind_end = max(0, np.searchsorted(caption['start'], end, side='left') - 1)
    if caption['start'][ind_start] <= start <= caption['end'][ind_start]:
        return _get_single_random_text_and_audio(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_frames,
                                                 ind=ind_start,
                                                 flag_text_audio_misaligned=flag_text_audio_misaligned,
                                                 flag_text_audio_misaligned_hard=flag_text_audio_misaligned_hard,
                                                 flag_text_audio_misaligned_random=flag_text_audio_misaligned_random)
    elif caption['start'][ind_end] <= end <= caption['end'][ind_end]:
        return _get_single_random_text_and_audio(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_frames,
                                                 ind=ind_end,
                                                 flag_text_audio_misaligned=flag_text_audio_misaligned,
                                                 flag_text_audio_misaligned_hard=flag_text_audio_misaligned_hard,
                                                 flag_text_audio_misaligned_random=flag_text_audio_misaligned_random)
    else:
        words = [] # no words in this clip
        start_frame = max(0, librosa.core.time_to_frames(start, sr=16000, hop_length=160, n_fft=400))
        mel_spec = mel_spec[:, start_frame:start_frame + num_audio_frames]

        # padded audio features
        audio, audio_mask, nframes = create_audio_features(mel_spec, num_audio_frames)

        # create padded text features
        text, text_mask, raw_text = create_text_features(words, max_words, we, we_dim)

        return audio, audio_mask, nframes, start, end, text, text_mask, raw_text


def _get_audio_and_text(caption, n_pair, mel_spec, min_time, max_words, we, we_dim, num_audio_frames,
                        flag_random_audio_start=False,
                        flag_text_audio_misaligned=False,
                        flag_text_audio_misaligned_hard=False,
                        flag_text_audio_misaligned_random=False):
    starts = np.zeros(n_pair)
    ends = np.zeros(n_pair)
    text = [0 for _ in range(n_pair)]
    text_mask = [0 for _ in range(n_pair)]
    raw_text = [0 for _ in range(n_pair)]
    audio = [0 for _ in range(n_pair)]
    audio_mask = [0 for _ in range(n_pair)]
    nframes = np.zeros(n_pair)

    for i in range(n_pair):
        if flag_random_audio_start:
            audio[i], audio_mask[i], nframes[i], starts[i], ends[i], text[i], text_mask[i], raw_text[i] = \
                _get_single_random_audio_and_text(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_frames,
                                                  flag_text_audio_misaligned=flag_text_audio_misaligned,
                                                  flag_text_audio_misaligned_hard=flag_text_audio_misaligned_hard,
                                                  flag_text_audio_misaligned_random=flag_text_audio_misaligned_random)
        else:
            audio[i], audio_mask[i], nframes[i], starts[i], ends[i], text[i], text_mask[i], raw_text[i]  = \
                _get_single_random_text_and_audio(caption, mel_spec, min_time, max_words, we, we_dim, num_audio_frames,
                                                  flag_text_audio_misaligned=flag_text_audio_misaligned,
                                                  flag_text_audio_misaligned_hard=flag_text_audio_misaligned_hard,
                                                  flag_text_audio_misaligned_random=flag_text_audio_misaligned_random)

    audio = th.stack(audio, dim=0)
    audio_mask = th.stack(audio_mask, dim=0)
    text = th.stack(text, dim=0)
    text_mask = th.stack(text_mask, dim=0)
    return audio, audio_mask, nframes, starts, ends, text, text_mask, raw_text


def _get_video(features_2d, features_3d, fps_2d, fps_3d, starts, ends, n_video_tokens, video_sampling_strategy='clip',
               accurate_borders=False):
    def get_slice(features, fps, start, end):
        if accurate_borders:
            start = int(np.floor(start * fps))
            end = int(np.ceil(end * fps))
        else:
            # this was in baseline code
            start = int(start * fps)
            end = int(end * fps) + 1
        if features is not None:
            return features[start:end]
        else:
            return None

    all_videos = []
    all_video_masks = []
    for i in range(len(starts)):
        slice_2d = get_slice(features_2d, fps_2d, starts[i], ends[i])
        slice_3d = get_slice(features_3d, fps_3d, starts[i], ends[i])
        video, video_mask = create_video_features(slice_2d, slice_3d, n_video_tokens, strategy=video_sampling_strategy)
        all_videos.append(video)
        all_video_masks.append(video_mask)
    all_videos = torch.stack(all_videos, dim=0)
    all_video_masks = torch.stack(all_video_masks, dim=0)

    return all_videos, all_video_masks


def _sample_video_audio_text(caption, mel_spec, features_2d, features_3d, vid_path, n_pair, fps_2d, fps_3d,
                             min_time, max_words, we, we_dim, num_audio_frames, n_video_tokens,
                             video_sampling_strategy='clip',
                             flag_random_audio_start=False,
                             flag_text_audio_misaligned=False,
                             flag_text_audio_misaligned_hard=False,
                             flag_text_audio_misaligned_random=False
                             ):
    audio, audio_mask, nframes, starts, ends, text, text_mask, raw_text = \
        _get_audio_and_text(caption, n_pair, mel_spec,
                            min_time, max_words, we, we_dim, num_audio_frames,
                            flag_random_audio_start=flag_random_audio_start,
                            flag_text_audio_misaligned=flag_text_audio_misaligned,
                            flag_text_audio_misaligned_hard=flag_text_audio_misaligned_hard,
                            flag_text_audio_misaligned_random=flag_text_audio_misaligned_random)

    video, video_mask = _get_video(features_2d, features_3d, fps_2d, fps_3d, starts, ends, n_video_tokens, video_sampling_strategy)

    nframes = -np.ones(len(nframes))

    paths = [vid_path] * len(video)
    ids = [vid_path + str(start) for start in starts]

    return video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, paths, ids, starts, ends


class HowTo_Dataset(Dataset):
    """Youtube dataset loader."""

    def __init__(
            self,
            csv,
            features_path,
            features_path_audio,
            caption,
            we,
            # word2vec_path,
            features_path_3D=None,
            min_time=10.0,
            n_video_tokens=None,
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            we_dim=300,
            max_words=30,
            min_words=0,
            n_pair=1,
            num_audio_frames=1024,
            random_audio_windows=False,
            dataset_length_quantile=1,
            non_zero_attention_mask=False,
            video_sampling_strategy='clip',
            flag_random_audio_start=False,
            flag_text_audio_misaligned=False,
            flag_text_audio_misaligned_hard=False,
            flag_text_audio_misaligned_random=False,
            use_2D=True,
            use_3D=True,
    ):
        """
        Args:
        """
        self.original_csv = ''
        self.video_sampling_strategy = video_sampling_strategy
        self.flag_random_audio_start = flag_random_audio_start
        assert not flag_text_audio_misaligned or not flag_text_audio_misaligned_hard
        self.flag_text_audio_misaligned = flag_text_audio_misaligned
        self.flag_text_audio_misaligned_hard = flag_text_audio_misaligned_hard
        self.flag_text_audio_misaligned_random = flag_text_audio_misaligned_random
        self.features_path = features_path
        self.features_path_audio = features_path_audio if features_path_audio != "" \
            else features_path
        self.caption = caption
        self.min_time = min_time
        self.n_video_tokens = n_video_tokens if n_video_tokens is not None else int(1.5 * min_time)
        self.feature_framerate = feature_framerate
        self.feature_framerate_3D = feature_framerate_3D
        self.we_dim = we_dim
        self.max_words = max_words
        self.min_words = min_words
        self.num_audio_frames = num_audio_frames
        # we = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.we = we
        self.n_pair = n_pair
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = {'2d': features_path}
        if features_path_3D is not None:
            self.feature_path['3d'] = features_path_3D
        else:
            self.feature_path['3d'] = features_path
        self.random_audio_windows = random_audio_windows
        self.use_2D = use_2D
        self.use_3D = use_3D

        self.dataset_length_quantile = dataset_length_quantile
        self.length = int(len(self.original_csv) * self.dataset_length_quantile)
        self.sample_dataset()

    def sample_dataset(self):
        if self.dataset_length_quantile != 1:
            self.csv = self.original_csv.sample(self.length, replace=True)
        else:
            self.csv = self.original_csv

    def __len__(self):
        return self.length

    # def _get_audio_random(self, n_pair_max, mel_spec):
    #     k = n_pair_max
    #     starts = np.zeros(k)
    #     ends = np.zeros(k)
    #     audio = [0 for i in range(k)]
    #     nframes = np.zeros(k)
    #     video_duration_seconds = int(
    #         librosa.core.frames_to_time(mel_spec.shape[1], sr=16000, hop_length=160, n_fft=400))
    #     num_audio_seconds = int(librosa.core.frames_to_time(self.num_audio_frames, sr=16000, hop_length=160, n_fft=400))
    #     # Sample clips that end before the end of the video
    #     # If the video is shorter than the desired window, use the entire video
    #     start_seconds = np.random.choice(range(max(1, video_duration_seconds - (num_audio_seconds + 1))), k,
    #                                      replace=True)
    #
    #     for i in range(k):
    #         start_frame = max(0, librosa.core.time_to_frames(start_seconds[i], sr=16000, hop_length=160, n_fft=400))
    #         audio_window = mel_spec[:, start_frame: start_frame + self.num_audio_frames]
    #         # Pad in the case that the audio wasn't long enough
    #         padded_mel_spec, nframes_spec = _zero_pad_audio(audio_window, self.num_audio_frames)
    #         end_second = start_seconds[i] + num_audio_seconds
    #         audio[i], nframes[i], starts[i], ends[i] = th.from_numpy(padded_mel_spec), nframes_spec, start_seconds[
    #             i], end_second
    #
    #     audio = th.cat([i.unsqueeze(0) for i in audio], dim=0)
    #     return audio, nframes, starts, ends

    def __getitem__(self, idx):
        # load video
        vid_path = self.csv['path'].values[idx].replace("None/", "")
        video_id = vid_path.split("/")[-1]

        features_2d = None
        features_3d = None
        if self.use_2D:
            path = os.path.join(self.feature_path['2d'], vid_path, video_id + "_2d.npz")
            if not os.path.exists(path):
                path = os.path.join(self.feature_path['2d'], vid_path + "_2d.npz")

            features_2d = np.load(path)['features']
            features_2d = th.from_numpy(features_2d).float()
        if self.use_3D:
            path = os.path.join(self.feature_path['3d'], vid_path, video_id + "_3d.npz")
            if not os.path.exists(path):
                path = os.path.join(self.feature_path['3d'], vid_path, video_id + ".npz")
            features_3d = np.load(path)['features']
            features_3d = th.from_numpy(features_3d).float()

        # load audio
        audio_path = os.path.join(self.features_path_audio, vid_path, video_id + "_spec.npz")
        mel_spec = np.load(audio_path)['arr_0']

        # load text
        caption = self.caption[video_id]

        video, video_mask, audio, audio_mask, text, text_mask, raw_text, nframes, paths, ids, starts, ends = \
            _sample_video_audio_text(caption, mel_spec, features_2d, features_3d, vid_path, self.n_pair,
                                     self.fps['2d'], self.fps['3d'],
                                     self.min_time, self.max_words, self.we, self.we_dim,
                                     self.num_audio_frames, self.n_video_tokens,
                                     video_sampling_strategy=self.video_sampling_strategy,
                                     flag_random_audio_start=self.flag_random_audio_start,
                                     flag_text_audio_misaligned=self.flag_text_audio_misaligned,
                                     flag_text_audio_misaligned_hard=self.flag_text_audio_misaligned_hard,
                                     flag_text_audio_misaligned_random=self.flag_text_audio_misaligned_random)

        return {'video': video, 'audio': audio, 'text': text,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_fake_1st_dim': 1, 'nframes': nframes,
                'meta': {'paths': paths, 'ids': ids, 'dataset': ['HowTo100M'] * len(video)}}


