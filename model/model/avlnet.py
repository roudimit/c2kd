from email.headerregistry import DateHeader
import torch
from timm.models.layers import trunc_normal_
from torch import nn as nn
import numpy as np
from transformers import AutoModel

from model.fusion_transformer import FusionTransformer
from model.layers import Gated_Embedding_Unit, Sentence_Maxpool
from model.loss import normalize_embeddings


def get_projection(input_dim, output_dim, projection_type):
    if projection_type == 'minimal':
        return nn.Linear(input_dim, output_dim)
    if projection_type == 'gated':
        return Gated_Embedding_Unit(input_dim, output_dim)
    elif projection_type == '':
        return nn.Identity()
    else:
        raise NotImplementedError


class BaselineModel(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            amd=False,
            davenet_v2=False
    ):
        super().__init__()
        from model.model.davenet import load_DAVEnet

        self.DAVEnet = load_DAVEnet(amd=amd, v2=davenet_v2)
        self.DAVEnet_projection = nn.Linear(1024, embd_dim)
        self.GU_audio = Gated_Embedding_Unit(1024, 1024)
        self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)
        self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
        self.GU_text_captions = Gated_Embedding_Unit(embd_dim, embd_dim)

    def forward(self, data, task=None):
        output = {}

        output['video_nonempty_input_mask'] = data['video_mask'].sum(-1) != 0
        output['text_nonempty_input_mask'] = data['text_mask'].sum(-1) != 0
        output['audio_nonempty_input_mask'] = data['video_mask'].sum(-1) != 0

        output["text_embed"] = self.GU_text_captions(self.text_pooling_caption(data['text']))

        video = data['video']
        if len(video.shape) == 3:
            video = torch.nn.functional.normalize(torch.max(video, dim=1)[0], dim=1)
        output["video_embed"] = self.GU_video(video)

        if 'audio' in data:
            audio_input = data['audio']
            nframes = data['nframes']

            audio = self.DAVEnet(audio_input)
            if nframes[0] != -1:
                # if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
                # Mean-pool audio embeddings and disregard embeddings from input 0 padding
                pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
                nframes = nframes / pooling_ratio
                audioPoolfunc = torch.nn.AdaptiveAvgPool2d((1, 1))
                audio_outputs = audio.unsqueeze(2)
                pooled_audio_outputs_list = []
                for idx in range(audio.shape[0]):
                    nF = max(1, nframes[idx].cpu().item())
                    pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:int(nF)]).unsqueeze(0))
                audio = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
            else:
                audio = audio.mean(dim=2)  # this averages features from 0 padding too
            # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training
            audio = self.GU_audio(audio)
            output["audio_embed"] = self.DAVEnet_projection(audio)

        return output


class BaselineModelMoreLogic(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            amd=False,
            davenet_v2=False
    ):
        super().__init__()
        from model.model.davenet import load_DAVEnet

        self.DAVEnet = load_DAVEnet(amd=amd, v2=davenet_v2)
        self.GU_audio = Gated_Embedding_Unit(1024, embd_dim)
        self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)
        self.GU_text_captions = Gated_Embedding_Unit(embd_dim, embd_dim)

    def forward(self, data, task=None):
        output = {}

        output["text_embed"] = self.GU_text_captions(torch.max(data['text'], dim=1)[0])

        video = data['video']
        # if len(video.shape) == 3:
        #     video = torch.nn.functional.normalize(torch.max(video, dim=1)[0], dim=1)
        output["video_embed"] = self.GU_video(video)

        if 'audio' in data:
            audio_input = data['audio']
            nframes = data['nframes']

            audio = self.DAVEnet(audio_input)
            if nframes[0] != -1:
                # if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
                # Mean-pool audio embeddings and disregard embeddings from input 0 padding
                pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
                nframes = nframes / pooling_ratio
                audioPoolfunc = torch.nn.AdaptiveAvgPool2d((1, 1))
                audio_outputs = audio.unsqueeze(2)
                pooled_audio_outputs_list = []
                for idx in range(audio.shape[0]):
                    nF = max(1, nframes[idx].cpu().item())
                    pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:int(nF)]).unsqueeze(0))
                audio = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
            else:
                audio = audio.mean(dim=2)  # this averages features from 0 padding too
            output["audio_embed"] = self.GU_audio(audio)
        return output


class BaselineLinearModel(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            amd=False,
            davenet_v2=False
    ):
        super().__init__()
        from model.model.davenet import load_DAVEnet

        self.DAVEnet = load_DAVEnet(amd=amd, v2=davenet_v2)
        self.audio_proj = nn.Linear(1024, embd_dim)
        self.video_proj = nn.Linear(video_dim, embd_dim)
        self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
        self.text_proj = nn.Linear(embd_dim, embd_dim)

    def forward(self, data, task=None):
        output = {}

        output["text_embed"] = self.text_proj(self.text_pooling_caption(data['text']))

        video = data['video']
        if len(video.shape) == 3:
            video = torch.nn.functional.normalize(torch.max(video, dim=1)[0], dim=1)
        output["video_embed"] = self.video_proj(video)

        if 'audio' in data:
            audio_input = data['audio']
            nframes = data['nframes']

            audio = self.DAVEnet(audio_input)
            if nframes[0] != -1:
                # if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
                # Mean-pool audio embeddings and disregard embeddings from input 0 padding
                pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
                nframes = nframes / pooling_ratio
                audioPoolfunc = torch.nn.AdaptiveAvgPool2d((1, 1))
                audio_outputs = audio.unsqueeze(2)
                pooled_audio_outputs_list = []
                for idx in range(audio.shape[0]):
                    nF = max(1, nframes[idx].cpu().item())
                    pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:int(nF)]).unsqueeze(0))
                audio = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
            else:
                audio = audio.mean(dim=2)  # this averages features from 0 padding too
            # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training
            output["audio_embed"] = self.audio_proj(audio)

        return output


class BaselineNonLinearModel(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            amd=False,
            davenet_v2=False
    ):
        super().__init__()
        from model.model.davenet import load_DAVEnet

        self.DAVEnet = load_DAVEnet(amd=amd, v2=davenet_v2)
        self.audio_proj = nn.Sequential(
            nn.Linear(1024, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, embd_dim),
        )
        self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
        self.text_proj = nn.Linear(embd_dim, embd_dim)

    def forward(self, data, task=None):
        output = {}

        output["text_embed"] = self.text_proj(self.text_pooling_caption(data['text']))

        video = data['video']
        if len(video.shape) == 3:
            video = torch.nn.functional.normalize(torch.max(video, dim=1)[0], dim=1)
        output["video_embed"] = self.video_proj(video)

        if 'audio' in data:
            audio_input = data['audio']
            nframes = data['nframes']

            audio = self.DAVEnet(audio_input)
            if nframes[0] != -1:
                # if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
                # Mean-pool audio embeddings and disregard embeddings from input 0 padding
                pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
                nframes = nframes / pooling_ratio
                audioPoolfunc = torch.nn.AdaptiveAvgPool2d((1, 1))
                audio_outputs = audio.unsqueeze(2)
                pooled_audio_outputs_list = []
                for idx in range(audio.shape[0]):
                    nF = max(1, nframes[idx].cpu().item())
                    pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:int(nF)]).unsqueeze(0))
                audio = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
            else:
                audio = audio.mean(dim=2)  # this averages features from 0 padding too
            # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training
            output["audio_embed"] = self.audio_proj(audio)

        return output


class BaselineWithFusion(BaselineModel):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            fusion_params={},
            cross_modal=True,
            apply_fusion_to_a_single_mod=True,
            amd=False,
            davenet_v2=False

    ):
        super().__init__(embd_dim=embd_dim, video_dim=video_dim, we_dim=we_dim, amd=amd, davenet_v2=davenet_v2)

        self.fusion = FusionTransformer(**fusion_params)
        self.cross_modal = cross_modal
        self.apply_fusion_to_a_single_mod = apply_fusion_to_a_single_mod

    def forward(self, data, task=None):
        output = super().forward(data, task)
        text_raw_embed = {'all_tokens': output['text_embed'].unsqueeze(1)}
        video_raw_embed = {'all_tokens': output['video_embed'].unsqueeze(1)}

        if self.apply_fusion_to_a_single_mod:
            text = self.fusion(text=text_raw_embed)
            video = self.fusion(video=video_raw_embed)
            output["text_embed"] = text['text']['cls']
            output["video_embed"] = video['video']['cls']

        if 'audio_embed' in output:
            audio_raw_embed = {'all_tokens': output['audio_embed'].unsqueeze(1)}
            if self.apply_fusion_to_a_single_mod:
                audio = self.fusion(audio=audio_raw_embed)
                output["audio_embed"] = audio['audio']['cls']

        if self.cross_modal and ('audio_embed' in output):
                ta = self.fusion(text=text_raw_embed, audio=audio_raw_embed)
                va = self.fusion(video=video_raw_embed, audio=audio_raw_embed)
                tv = self.fusion(text=text_raw_embed, video=video_raw_embed)
                output["ta_embed"] = ta['text_audio']['cls']
                output["va_embed"] = va['video_audio']['cls']
                output["tv_embed"] = tv['text_video']['cls']
        return output


def create_audio_tokens(audio, audio_mask, nframes, n_tokens, strategy='avg_pool'):
    if torch.is_tensor(nframes):
        nframes = int(nframes.cpu().item())
    if strategy == 'clip':
        return audio[:n_tokens], audio_mask[:n_tokens]
    elif strategy == 'nearest':
        if nframes <= n_tokens:
            return create_audio_tokens(audio, audio_mask, nframes, n_tokens, strategy='clip')
        audio = audio[:nframes]
        audio = torch.nn.functional.interpolate(
            audio.permute(1, 0).unsqueeze(0),
            size=n_tokens,
            mode='nearest').squeeze(0).permute(1, 0)
        return audio, audio_mask[:n_tokens]
    elif strategy == 'max_pool':
        if nframes <= n_tokens:
            return create_audio_tokens(audio, audio_mask, nframes, n_tokens, strategy='clip')
        audio = audio[:nframes]
        audio = torch.nn.functional.adaptive_max_pool1d(
            audio.permute(1, 0).unsqueeze(0),
            output_size=n_tokens).squeeze(0).permute(1, 0)
        return audio, audio_mask[:n_tokens]
    elif strategy == 'avg_pool':
        if nframes <= n_tokens:
            return create_audio_tokens(audio, audio_mask, nframes, n_tokens, strategy='clip')
        audio = audio[:nframes]
        audio = torch.nn.functional.adaptive_avg_pool1d(
            audio.permute(1, 0).unsqueeze(0),
            output_size=n_tokens).squeeze(0).permute(1, 0)
        return audio, audio_mask[:n_tokens]
    elif strategy == 'none':
        return audio, audio_mask
    else:
        raise NotImplementedError

class BaselineTokenFusionModel(nn.Module):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        super().__init__()

        self.fusion = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.add_masking_token_when_no_tokens = add_masking_token_when_no_tokens
        self.n_masking_token_when_no_tokens = n_masking_token_when_no_tokens
        self.add_masking_token_when_modality_is_missing = add_masking_token_when_modality_is_missing
        self.normalize_before_avg_with_ind_proj = normalize_before_avg_with_ind_proj
        self.mlfm_ratio = mlfm_ratio
        self.mvm_ratio = mvm_ratio
        self.mam_ratio = mam_ratio
        self.individual_projections = individual_projections
        self.use_positional_emb = use_positional_emb
        self.add_pos_embeds = add_pos_embeds
        self.apply_projection = apply_projection

        embed_dim = fusion_params['embed_dim']

        if use_cls_tokens:
            self.video_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.text_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.video_cls_token = None
            self.text_cls_token = None
            self.audio_cls_token = None

        if use_norm_layers:
            self.video_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
            self.text_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
            self.audio_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        else:
            self.video_norm_layer = None
            self.text_norm_layer = None
            self.audio_norm_layer = None

        self.stategy_audio_pooling = stategy_audio_pooling

        # audio token preprocess
        from model.model.davenet import load_DAVEnet
        self.davenet = load_DAVEnet(amd=amd, v2=davenet_v2)

        if self.use_positional_emb:
            video_max_tokens = video_params['max_tokens']
            text_max_tokens = text_params['max_tokens']
            max_num_audio_frames = audio_params.get('max_num_audio_frames')
            audio_max_tokens = audio_params.get('max_tokens')
            assert max_num_audio_frames is None or audio_max_tokens is None
            if audio_max_tokens is None:
                if davenet_v2:
                    audio_max_tokens = int(max_num_audio_frames / 64)
                else:
                    audio_max_tokens = int(max_num_audio_frames / 16)
            self.audio_max_tokens = audio_max_tokens
            self.video_pos_embed = nn.Parameter(torch.zeros(1, video_max_tokens + use_cls_tokens, embed_dim))
            self.text_pos_embed = nn.Parameter(torch.zeros(1, text_max_tokens + use_cls_tokens, embed_dim))
            self.audio_pos_embed = nn.Parameter(torch.zeros(1, self.audio_max_tokens + use_cls_tokens, embed_dim))
        else:
            self.video_pos_embed = None
            self.text_pos_embed = None
            self.audio_pos_embed = None

            max_num_audio_frames = audio_params.get('max_num_audio_frames')
            audio_max_tokens = audio_params.get('max_tokens')

            # assert max_num_audio_frames is None or audio_max_tokens is None
            if (audio_max_tokens is None) and (max_num_audio_frames is not None):
                if davenet_v2:
                    audio_max_tokens = int(max_num_audio_frames / 64)
                else:
                    audio_max_tokens = int(max_num_audio_frames / 16)
            self.audio_max_tokens = audio_max_tokens

        self.cross_modal = cross_modal

        video_embed_dim = video_params['embed_dim']
        text_embed_dim = text_params['embed_dim']
        audio_embed_dim = 4096 if davenet_v2 else 1024

        self.video_token_proj = get_projection(video_embed_dim, embed_dim, pre_projection)
        self.text_token_proj = get_projection(text_embed_dim, embed_dim, pre_projection)
        self.audio_token_proj = get_projection(audio_embed_dim, embed_dim, pre_projection)
        if not self.individual_projections:
            self.proj = get_projection(embed_dim, projection_dim, projection)

            if use_norm_layers_before_proj:
                self.proj = nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    self.proj
                )
        else:
            self.video_proj = get_projection(embed_dim, projection_dim, projection)
            self.text_proj = get_projection(embed_dim, projection_dim, projection)
            self.audio_proj = get_projection(embed_dim, projection_dim, projection)

            if use_norm_layers_before_proj:
                self.video_proj = nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    self.video_proj
                )
                self.text_proj = nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    self.text_proj
                )
                self.audio_proj = nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    self.audio_proj
                )

        self.init_weights()
        if load_checkpoint is not None:
            raise NotImplementedError()

        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            self.freeze_backbone()
        self.normalize_input = normalize_input

    def freeze_backbone(self):
        for param in self.davenet.parameters():
            param.requires_grad = False
        for param in self.text_token_proj.parameters():
            param.requires_grad = False
        for param in self.video_token_proj.parameters():
            param.requires_grad = False
        for param in self.audio_token_proj.parameters():
            param.requires_grad = False

        if self.use_positional_emb:
            self.video_pos_embed.requires_grad = False
            self.audio_pos_embed.requires_grad = False
            self.text_pos_embed.requires_grad = False
        # for param in self.fusion.parameters():
        #     param.requires_grad = False
        self.frozen_backbone = True

    def init_weights(self):
        for weights in [self.video_cls_token, self.video_pos_embed,
                        self.audio_cls_token, self.audio_pos_embed,
                        self.text_cls_token, self.text_pos_embed]:
            if weights is not None:
                trunc_normal_(weights, std=.02)

    def _add_cls_token(self, x, attention_mask, cls_token, pos_embed, add_pos_embed=True):
        if cls_token is not None:
            cls_token = cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

            cls_token_mask = torch.ones((1, 1)).to(attention_mask.device).expand(x.shape[0], -1)
            attention_mask = torch.cat((cls_token_mask, attention_mask), dim=1)
            special_token_mask = torch.cat((cls_token_mask > 0, attention_mask == 0), dim=1)
        else:
            special_token_mask = attention_mask == 0
        if add_pos_embed and pos_embed is not None:
            x = x + pos_embed
        return x, attention_mask, special_token_mask

    def _get_masking_token(self, mod):
        if self.fusion.masking_token_per_modality:
            ids = {'text': 0, 'video': 1, 'audio': 2}
            return self.fusion.masking_token[ids[mod]]
        else:
            return self.fusion.masking_token

    def _check_if_zero_input(self, x, attention_mask, mod):
        nonempty_input_mask = attention_mask.sum(-1) != 0

        if self.add_masking_token_when_no_tokens:
            zero_input_mask = nonempty_input_mask == 0
            masking_token = self._get_masking_token(mod)
            x[zero_input_mask, :self.n_masking_token_when_no_tokens] = masking_token.type(x.dtype)
            attention_mask[zero_input_mask, :self.n_masking_token_when_no_tokens] = 1
        return x, attention_mask, nonempty_input_mask

    def extract_video_tokens(self, video, attention_mask, add_pos_embed=True):
        if self.normalize_input:
            video = torch.nn.functional.normalize(video, dim=-1)

        x = self.video_token_proj(video)

        if self.video_norm_layer is not None:
            x = self.video_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_if_zero_input(x, attention_mask, 'video')
        x, attention_mask, special_token_mask = self._add_cls_token(x, attention_mask, self.video_cls_token,
                                                                    self.video_pos_embed, add_pos_embed=add_pos_embed)

        return {'all_tokens': x, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def extract_audio_tokens(self, audio, attention_mask, nframes, add_pos_embed=True):
        audio = self.davenet(audio)
        audio = audio.permute(0, 2, 1)

        coef = int(np.ceil(attention_mask.shape[1] / audio.shape[1]))
        attention_mask = torch.nn.functional.max_pool1d(attention_mask.unsqueeze(0), kernel_size=coef).squeeze(0)
        nframes = (nframes / coef).int()

        if (self.audio_max_tokens is not None) and (audio.shape[1] > self.audio_max_tokens):
            new_audio, new_audio_mask = [], []
            for i in range(len(audio)):
                cur_audio, cur_audio_mask = create_audio_tokens(
                    audio[i], attention_mask[i], nframes[i], self.audio_max_tokens, strategy=self.stategy_audio_pooling)
                new_audio.append(cur_audio)
                new_audio_mask.append(cur_audio_mask)
            audio = torch.stack(new_audio, dim=0)
            attention_mask = torch.stack(new_audio_mask, dim=0)

        if self.normalize_input:
            audio = torch.nn.functional.normalize(audio, dim=-1)

        audio = self.audio_token_proj(audio)

        if self.audio_norm_layer is not None:
            audio = self.audio_norm_layer(audio)

        audio, attention_mask, nonempty_input_mask = self._check_if_zero_input(audio, attention_mask, 'audio')
        audio, attention_mask, special_token_mask = self._add_cls_token(audio, attention_mask, self.audio_cls_token,
                                                                        self.audio_pos_embed, add_pos_embed=add_pos_embed)
        return {'all_tokens': audio, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def extract_text_tokens(self, text, attention_mask, add_pos_embed=True):
        if self.normalize_input:
            text = torch.nn.functional.normalize(text, dim=-1)

        x = self.text_token_proj(text)

        if self.text_norm_layer is not None:
            x = self.text_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_if_zero_input(x, attention_mask, 'text')
        x, attention_mask, special_token_mask = self._add_cls_token(x, attention_mask, self.text_cls_token,
                                                                    self.text_pos_embed, add_pos_embed=add_pos_embed)
        return {'all_tokens': x, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def get_missing_mod(self, modality, batch_size):
        if not self.add_masking_token_when_modality_is_missing:
            return None
        else:
            masking_token = self._get_masking_token(modality)
            all_tokens = masking_token.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_masking_token_when_no_tokens, -1)
            attention_mask = torch.ones(batch_size, self.n_masking_token_when_no_tokens).to(all_tokens.device) > 0
            return {'all_tokens': all_tokens, 'attention_mask': attention_mask}

    def forward(self, data, task=None, v_ratio=None):
        if task is None:
            task = 'Retrieval'

        output = {}

        if self.frozen_backbone:
            torch.set_grad_enabled(False)

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'], add_pos_embed=False)
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'], add_pos_embed=False)
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'], add_pos_embed=False)
        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        if task == 'Retrieval_with_masking':
            for raw_embed, ratio, mod in [
                (text_raw_embed, self.mlfm_ratio, 'text'),
                (video_raw_embed, self.mvm_ratio, 'video'),
                (audio_raw_embed, self.mam_ratio, 'audio'),
            ]:
                if ratio > 0:
                    masking_token = self._get_masking_token(mod)
                    raw_embed['all_tokens'], _, _ = \
                        random_video_masking(raw_embed['all_tokens'], masking_token, ratio, raw_embed['special_token_mask'])

        if self.frozen_backbone:
            torch.set_grad_enabled(True)

        # add positional embedding after masking
        if self.use_positional_emb and self.add_pos_embeds:
            text_raw_embed['all_tokens'] = text_raw_embed['all_tokens'] + self.text_pos_embed
            video_raw_embed['all_tokens'] = video_raw_embed['all_tokens'] + self.video_pos_embed
            audio_raw_embed['all_tokens'] = audio_raw_embed['all_tokens'] + self.audio_pos_embed

        bs = text_raw_embed['all_tokens'].shape[0]
        text = self.fusion(text=text_raw_embed,
                           video=self.get_missing_mod('video', bs),
                           audio=self.get_missing_mod('audio', bs))['text']
        video = self.fusion(text=self.get_missing_mod('text', bs),
                            video=video_raw_embed,
                            audio=self.get_missing_mod('audio', bs)
                            )['video']
        audio = self.fusion(text=self.get_missing_mod('text', bs),
                            video=self.get_missing_mod('video', bs),
                            audio=audio_raw_embed)['audio']

        if self.apply_projection:
            if not self.individual_projections:
                output["text_embed"] = self.proj(text['cls'])
                output["video_embed"] = self.proj(video['cls'])
                output["audio_embed"] = self.proj(audio['cls'])
            else:
                output["text_embed"] = self.text_proj(text['cls'])
                output["video_embed"] = self.video_proj(video['cls'])
                output["audio_embed"] = self.audio_proj(audio['cls'])
        else:
            output["text_embed"] = text['cls']
            output["video_embed"] = video['cls']
            output["audio_embed"] = audio['cls']

        if self.cross_modal or 'cross' in task:
            if task != 'cross_video_audio':
                tv = self.fusion(text=text_raw_embed,
                                 video=video_raw_embed,
                                 audio=self.get_missing_mod('audio', bs))
                ta = self.fusion(text=text_raw_embed,
                                 video=self.get_missing_mod('video', bs),
                                 audio=audio_raw_embed)
            va = self.fusion(text=self.get_missing_mod('text', bs),
                             video=video_raw_embed,
                             audio=audio_raw_embed)

            if self.apply_projection:
                if self.fusion.cls_token is not None:
                    assert not self.individual_projections
                    assert not self.normalize_before_avg_with_ind_proj
                    if task != 'cross_video_audio':
                        output["tv_embed"] = self.proj(tv['text_video']['cls'])
                        output["ta_embed"] = self.proj(ta['text_audio']['cls'])
                    output["va_embed"] = self.proj(va['video_audio']['cls'])
                else:
                    if self.individual_projections:
                        text_proj, video_proj, audio_proj = self.text_proj, self.video_proj, self.audio_proj
                    else:
                        text_proj, video_proj, audio_proj = self.proj, self.proj, self.proj

                    if task != 'cross_video_audio':
                        output["text_tv_embed"] = text_proj(tv['text']['cls'])
                        output["video_tv_embed"] = video_proj(tv['video']['cls'])

                        if self.normalize_before_avg_with_ind_proj:
                            output["tv_embed"] = (normalize_embeddings(output["text_tv_embed"]) +
                                                  normalize_embeddings(output["video_tv_embed"])) / 2
                        else:
                            output["tv_embed"] = (output["text_tv_embed"] + output["video_tv_embed"]) / 2

                        output["text_ta_embed"] = text_proj(ta['text']['cls'])
                        output["audio_ta_embed"] = audio_proj(ta['audio']['cls'])
                        if self.normalize_before_avg_with_ind_proj:
                            output["ta_embed"] = (normalize_embeddings(output["text_ta_embed"]) +
                                                  normalize_embeddings(output["audio_ta_embed"])) / 2
                        else:
                            output["ta_embed"] = (output["text_ta_embed"] + output["audio_ta_embed"]) / 2

                    output["video_va_embed"] = video_proj(va['video']['cls'])
                    output["audio_va_embed"] = audio_proj(va['audio']['cls'])
                    if self.normalize_before_avg_with_ind_proj:
                        video_w, audio_w = (0.5, 0.5) if v_ratio is None else (v_ratio, 1 - v_ratio)
                        output["va_embed"] = video_w * normalize_embeddings(output["video_va_embed"]) + \
                                              audio_w * normalize_embeddings(output["audio_va_embed"])
                    else:
                        output["va_embed"] = (output["video_va_embed"] + output["audio_va_embed"]) / 2
            else:
                if task != 'cross_video_audio':
                    output["tv_embed"] = tv['text_video']['cls']
                    output["ta_embed"] = ta['text_audio']['cls']
                output["va_embed"] = va['video_audio']['cls']
        return output


class BaselineTokenFusionPerModModel(BaselineTokenFusionModel):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )

        self.fusion_text = self.fusion
        self.fusion_video = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.fusion_audio = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)

    def forward(self, data, task=None):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        text = self.fusion_text(text=text_raw_embed)['text']
        output["text_embed"] = self.text_proj(text['cls'])

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])

        audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        output["audio_embed"] = self.audio_proj(audio['cls'])

        return output

class BaselineTokenFusionPerModModelHuggingface(BaselineTokenFusionModel):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )

        del self.fusion # clear the fusion text transformer
        # print("Using roberta large")
        # self.fusion_text_xlm = AutoModel.from_pretrained('roberta-large', local_files_only=False)
        self.fusion_text_xlm = AutoModel.from_pretrained('xlm-roberta-large', local_files_only=False)
        # keep only first 12 layers out of 24 https://github.com/huggingface/transformers/issues/2483
        print("Using first 12 layers of XLM-RoBERTa-Large")
        newModuleList = nn.ModuleList()
        for i in range(0, 12):
            newModuleList.append(self.fusion_text_xlm.encoder.layer[i])
        self.fusion_text_xlm.encoder.layer = newModuleList
        # freeze first 9 layers https://discuss.huggingface.co/t/how-to-freeze-some-layers-of-bertmodel/917
        print("Freezing 9 layers of XLM-RoBERTa-Large")
        for i in range(0, 9):
            for param in self.fusion_text_xlm.encoder.layer[i].parameters():
                param.requires_grad = False
        # Add two more layers
        print("Adding two transformer layers from xlm-roberta-large config") 
        from transformers import RobertaModel
        from copy import deepcopy
        new_config = deepcopy(self.fusion_text_xlm.config)
        self.tp = RobertaModel(new_config)
        self.fusion_text_xlm.encoder.layer += nn.ModuleList([self.tp.encoder.layer[0], self.tp.encoder.layer[1]])

        self.text_proj = nn.Sequential(
            nn.LayerNorm(1024, eps=1e-6),
            get_projection(1024, projection_dim, 'gated')
        )
        self.fusion_video = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.fusion_audio = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.fusion = self.fusion_video # hack to get the code to work while saving memory

    def forward(self, data, task=None):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        embs = self.fusion_text_xlm(**data['tokens'])['last_hidden_state']
        att = data['tokens']['attention_mask'].squeeze(1)
        text_cls = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None] # mean pool
        # text_cls = embs[:, 0] # use first token
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])

        audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        output["audio_embed"] = self.audio_proj(audio['cls'])

        return output

    def forward_lang(self, data, lang='de'):
        output = {}
        embs = self.fusion_text_xlm(**data[lang])['last_hidden_state']
        att = data[lang]['attention_mask'].squeeze(1)
        text_cls = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None] # mean pool
        output[lang] = self.text_proj(text_cls)
        return output

class BaselineTokenFusionPerModModelLaBSE(BaselineTokenFusionModel):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )

        del self.fusion # clear the fusion text transformer
        self.fusion_text = AutoModel.from_pretrained('sentence-transformers/LaBSE', local_files_only=False)
        print("Adding projection layer")
        self.text_proj = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            get_projection(768, projection_dim, 'gated')
        )
        self.text_proj_t2t = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            get_projection(768, projection_dim, 'gated')
        )
        self.fusion_video = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.fusion = self.fusion_video # hack to get the code to work while saving memory

        # self.learnable_weights = torch.nn.parameter.Parameter(torch.tensor([0.33, 0.33, 0.33]))        
        
    def mean_pool(self, embs, att):
        # mean pool using the attention mask
        # embs[:, 0] # this would use first token
        att = att.squeeze(1) # fix dims
        return (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]

    def random_mask(self, shape, prob=0.05):
        # randomly mask each token with prob
        rand_mask = (1 - (torch.rand(shape) < 0.05).int()).cuda()
        return rand_mask

    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']

        # import ipdb; ipdb.set_trace()
        embs = self.fusion_text(**data[lang1])
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)
        output["text_t2t"] = embs['pooler_output']

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']

        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output
    
    def forward_lang(self, data, lang='de', audio_fwd=False):
        output = {}
        embs = self.fusion_text(**data[lang])
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        output['t2t'] = embs['pooler_output']
        if audio_fwd:
            audio_raw_embed = self.extract_audio_tokens(data[lang + '_audio']['audio'], 
                                                        data[lang + '_audio']['audio_mask'], 
                                                        data[lang + '_audio']['nframes'])
            audio = self.fusion_audio(audio=audio_raw_embed)['audio']
            output[lang + '_audio_embed'] = self.audio_proj(audio['cls']) # NOTE: might want to make lang-specifc proj
            output[lang + '_audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
            output[lang + '_audio_padding_mask'] = audio_raw_embed['attention_mask']
        # output['t2t'] = self.text_proj_t2t(text_cls)
        return output

class BaselineTokenFusionPerModModelMBert(BaselineTokenFusionPerModModelLaBSE):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        del self.fusion_text
        self.fusion_text = AutoModel.from_pretrained('bert-base-multilingual-uncased', local_files_only=False)

    # NOTE: overwriting the forward pass for the frozen teacher mode
    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']

        # import ipdb; ipdb.set_trace()
        embs = self.fusion_text(**data['tokens_mbert']) # NOTE: using the mbert tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']

        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output

    def forward_lang(self, data, lang='de', audio_fwd=False):
        output = {}
        embs = self.fusion_text(**data['{}_mbert'.format(lang)])
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        return output

class BaselineTokenFusionPerModModelXLMbase(BaselineTokenFusionPerModModelLaBSE):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        del self.fusion_text
        self.fusion_text = AutoModel.from_pretrained('xlm-roberta-base', local_files_only=False)

    # NOTE: overwriting the forward pass for the frozen teacher mode
    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']

        # import ipdb; ipdb.set_trace()
        embs = self.fusion_text(**data['tokens_xlm']) # NOTE: using the xlm tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']
        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output

    def forward_lang(self, data, lang='de', audio_fwd=False):
        output = {}
        embs = self.fusion_text(**data['{}_xlm'.format(lang)])
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        return output

class BaselineTokenFusionPerModModelInfoXLM(BaselineTokenFusionPerModModelLaBSE):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        del self.fusion_text
        self.fusion_text = AutoModel.from_pretrained('microsoft/infoxlm-base', local_files_only=False)

    # NOTE: overwriting the forward pass for the frozen teacher mode
    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']

        # import ipdb; ipdb.set_trace()
        embs = self.fusion_text(**data['tokens_infoxlm']) # NOTE: using the xlm tokens
        # embs = self.fusion_text(**data[lang1]) # NOTE - need to use this to learn the student
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']

        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output

    def forward_lang(self, data, lang='de', audio_fwd=False):
        output = {}
        embs = self.fusion_text(**data['{}_infoxlm'.format(lang)]) # NOTE: using the xlm tokens
        # embs = self.fusion_text(**data[lang]) # NOTE: need to use this to learn the student
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        return output

class BaselineTokenFusionPerModModelXLMAlign(BaselineTokenFusionPerModModelLaBSE):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        del self.fusion_text
        self.fusion_text = AutoModel.from_pretrained("microsoft/xlm-align-base", local_files_only=False)

    # NOTE: overwriting the forward pass for the frozen teacher mode
    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']

        # import ipdb; ipdb.set_trace()
        embs = self.fusion_text(**data['tokens_xlm_align']) # NOTE: using the xlm tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']
        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output

    def forward_lang(self, data, lang='de', audio_fwd=False):
        output = {}
        embs = self.fusion_text(**data['{}_xlm_align'.format(lang)]) # NOTE: using the xlm tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        return output

class BaselineTokenFusionPerModModelDistill(BaselineTokenFusionPerModModelLaBSE):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        del self.fusion_text
        self.fusion_text = AutoModel.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=False)

    # NOTE: overwriting the forward pass for the frozen teacher mode
    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        

        # import ipdb; ipdb.set_trace()
        embs = self.fusion_text(**data['tokens_distill']) # NOTE: using the distill tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']

        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output

    def forward_lang(self, data, lang='de', audio_fwd=False):
        output = {}
        embs = self.fusion_text(**data['{}_distill'.format(lang)]) # NOTE: using the distill tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        return output

class BaselineTokenFusionPerModModelSimCSE(BaselineTokenFusionPerModModelLaBSE):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        del self.fusion_text
        self.fusion_text = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base", local_files_only=False)

    # NOTE: overwriting the forward pass for the frozen teacher mode
    def forward(self, data, task=None, lang1='en', lang2=None, mask_text=False):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        # audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']

        embs = self.fusion_text(**data['tokens_sim_cse']) # NOTE: using the sim_cse model
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang1]['attention_mask'])
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])
        output['video_output_tokens'] = self.video_proj(video['all_tokens'])
        # output['video_output_tokens'] = video['all_tokens']
        output['video_padding_mask'] = video_raw_embed['attention_mask']

        # audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        # output["audio_embed"] = self.audio_proj(audio['cls'])
        # output['audio_output_tokens'] = self.audio_proj(audio['all_tokens'])
        # output['audio_padding_mask'] = audio_raw_embed['attention_mask']

        output["audio_embed"] = output["video_embed"]
        output['audio_output_tokens'] = output['video_output_tokens']
        output['audio_padding_mask'] = output['video_padding_mask'] 

        return output

    def forward_lang(self, data, lang='de', audio_fwd=False): #NOTE: it will always be english input anyways
        output = {}
        embs = self.fusion_text(**data['{}_sim_cse'.format(lang)]) # NOTE: always just the english tokens
        text_cls = self.mean_pool(embs['last_hidden_state'], data[lang]['attention_mask'])
        output[lang] = self.text_proj(text_cls)
        return output
class BaselineTokenFusionPerModModelMCLIP(BaselineTokenFusionModel):
    def __init__(self,
                 video_params,
                 text_params,
                 audio_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 pre_projection='minimal',
                 projection='minimal',
                 cross_modal=True,
                 use_cls_tokens=True,
                 stategy_audio_pooling='avg_pool',
                 amd=False,
                 davenet_v2=False,
                 normalize_input=False,
                 add_masking_token_when_no_tokens=False,
                 n_masking_token_when_no_tokens=1,
                 add_masking_token_when_modality_is_missing=False,
                 mlfm_ratio=0,
                 mvm_ratio=0,
                 mam_ratio=0,
                 individual_projections=False,
                 normalize_before_avg_with_ind_proj=False,
                 use_positional_emb=True,
                 add_pos_embeds=True,
                 apply_projection=True,
                 use_norm_layers=False,
                 frozen_backbone=False,
                 use_norm_layers_before_proj=False
                 ):
        assert individual_projections
        assert apply_projection
        assert not normalize_before_avg_with_ind_proj
        assert not cross_modal
        super().__init__(video_params,
                         text_params,
                         audio_params,
                         fusion_params,
                         projection_dim=projection_dim,
                         load_checkpoint=load_checkpoint,
                         pre_projection=pre_projection,
                         projection=projection,
                         cross_modal=cross_modal,
                         use_cls_tokens=use_cls_tokens,
                         stategy_audio_pooling=stategy_audio_pooling,
                         amd=amd,
                         davenet_v2=davenet_v2,
                         normalize_input=normalize_input,
                         add_masking_token_when_no_tokens=add_masking_token_when_no_tokens,
                         n_masking_token_when_no_tokens=n_masking_token_when_no_tokens,
                         add_masking_token_when_modality_is_missing=add_masking_token_when_modality_is_missing,
                         mlfm_ratio=mlfm_ratio,
                         mvm_ratio=mvm_ratio,
                         mam_ratio=mam_ratio,
                         individual_projections=individual_projections,
                         normalize_before_avg_with_ind_proj=normalize_before_avg_with_ind_proj,
                         use_positional_emb=use_positional_emb,
                         add_pos_embeds=add_pos_embeds,
                         apply_projection=apply_projection,
                         use_norm_layers=use_norm_layers,
                         frozen_backbone=frozen_backbone,
                         use_norm_layers_before_proj=use_norm_layers_before_proj
                         )
        
        del self.fusion # clear the fusion text transformer
        self.fusion_text = AutoModel.from_pretrained("M-CLIP/M-BERT-Base-ViT-B", local_files_only=False)
        # self.fusion_text = AutoModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
        print("Adding projection layer")
        self.text_proj = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            get_projection(768, projection_dim, 'gated')
        )
        self.fusion_video = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.fusion_audio = FusionTransformer(audio_model_name=audio_params['model'], **fusion_params)
        self.fusion = self.fusion_video # hack to get the code to work while saving memory

    def forward(self, data, task=None):
        output = {}

        text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
        video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
        audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['nframes'])

        output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        embs = self.fusion_text(**data['tokens'])['last_hidden_state']
        att = data['tokens']['attention_mask'].squeeze(1)
        text_cls = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None] # mean pool
        # text_cls = embs[:, 0] # use first token
        output["text_embed"] = self.text_proj(text_cls)

        video = self.fusion_video(video=video_raw_embed)['video']
        output["video_embed"] = self.video_proj(video['cls'])

        audio = self.fusion_audio(audio=audio_raw_embed)['audio']
        output["audio_embed"] = self.audio_proj(audio['cls'])

        return output
