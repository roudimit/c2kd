import random

import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x


class Fused_Gated_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Fused_Gated_Unit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return torch.max(x, dim=1)[0]


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, modality_importance='none',
                 attention_ratio=1, act='softmax', temperature=1, qkv_per_mod=False, keys_per_mod=False,
                 there_is_cls_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              modality_importance=modality_importance, attention_ratio=attention_ratio,
                              act=act, temperature=temperature, qkv_per_mod=qkv_per_mod,
                              keys_per_mod=keys_per_mod,
                              there_is_cls_token=there_is_cls_token)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attention_mask=None, sizes=None, sizes_tva=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attention_mask, sizes, sizes_tva))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., modality_importance='none',
                 attention_ratio=1, act='softmax', debug=False, temperature=1, qkv_per_mod=False,
                 keys_per_mod=False, there_is_cls_token=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.hidden_dim = int(dim * attention_ratio)

        self.qkv_per_mod = qkv_per_mod
        self.keys_per_mod = keys_per_mod
        self.there_is_cls_token = there_is_cls_token
        if qkv_per_mod:
            if there_is_cls_token:
                self.qkv_cls = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)
            self.qkv_text = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)
            self.qkv_video = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)
            self.qkv_audio = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)
        elif keys_per_mod:
            print('there_is_cls_token', there_is_cls_token)
            self.qkv = nn.Linear(dim, self.hidden_dim * (5 + self.there_is_cls_token), bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, self.hidden_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        assert modality_importance in ['none', 'equal', 'per_token', 'per_modality']
        self.modality_importance = modality_importance
        self.act = act
        if self.act !='softmax':
            assert self.modality_importance == 'none'
        self.temperature = temperature

        self.debug = debug # TODO: remove

    def forward(self, x, attention_mask=None, sizes=None, sizes_tva=None):
        B, N, _ = x.shape
        if self.qkv_per_mod:
            offset = 0
            qkvs = []
            if self.there_is_cls_token:
                qkv_modules = [self.qkv_cls, self.qkv_text, self.qkv_video, self.qkv_audio]
            else:
                qkv_modules = [self.qkv_text, self.qkv_video, self.qkv_audio]

            for size, qkv_module in zip(sizes_tva, qkv_modules):
                if size != 0:
                    cur_x = x[:, offset:offset + size]
                    B, cur_N, _ = cur_x.shape
                    cur_qkv = qkv_module(cur_x).reshape(B, cur_N, 3, self.num_heads,
                                                        self.hidden_dim // self.num_heads).permute(2, 0, 3, 1, 4)
                    qkvs.append(cur_qkv)
                    offset += size
            qkv = torch.cat(qkvs, dim=3)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        elif self.keys_per_mod:
            all_qkv = self.qkv(x).reshape(B, N, (5 + self.there_is_cls_token), self.num_heads, self.hidden_dim // self.num_heads).permute(2, 0, 3, 1, 4)
            q, v, key_mudules = all_qkv[0], all_qkv[1], all_qkv[2:]
            offset = 0
            attn = []
            for size, k in zip(sizes_tva, key_mudules):
                if size != 0:
                    cur_q = q[:, :, offset:offset + size]
                    cur_attn = (cur_q @ k.transpose(-2, -1)) * self.scale
                    attn.append(cur_attn)
                    offset += size
            attn = torch.cat(attn, dim=2)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.hidden_dim // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.scale



        if attention_mask is not None:
            zero_attention_mask = (attention_mask == 0).view(B, 1, 1, N).expand_as(attn)  # (bs, n_heads, q_length, k_length)
        else:
            zero_attention_mask = None

        if self.debug:
            self.zero_attention_mask = zero_attention_mask
            self.attn_before = attn

        if self.modality_importance == 'none':
            if self.act == 'softmax':
                if zero_attention_mask is not None:
                    attn.masked_fill_(zero_attention_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)
                attn = attn.softmax(dim=-1)
            elif self.act == 'sigmoid':
                attn = attn / self.temperature
                attn = torch.sigmoid(attn)
                if zero_attention_mask is not None:
                    attention_mask = zero_attention_mask == 0
                    attn = attn * attention_mask  # (bs, n_heads, q_length, k_length)
                    attn = attn / attention_mask.sum(-1).unsqueeze(-1)
                else:
                    attn = attn / attn.shape[-1]
            else:
                raise NotImplementedError
        else:
            if sum(sizes) != x.shape[1]:
                raise ValueError()
            n_mod = len(sizes)

            if self.modality_importance == 'equal':
                softmax_multipliers = torch.tensor([1 / n_mod] * n_mod).to(x.device)
            elif self.modality_importance == 'per_modality':
                mean_attn = []
                offset = 0
                for size in sizes:
                    attn_per_mod = attn[:, :, :, offset: offset + size]
                    if zero_attention_mask is not None:
                        non_zero = zero_attention_mask[:, :, :, offset: offset + size] == 0
                        cur_mean_attn = (attn_per_mod * non_zero).sum(-1).sum(-1) / non_zero.sum(-1).sum(-1).clamp(min=1)
                        cur_mean_attn.masked_fill_(cur_mean_attn==0, -float("inf"))
                    else:
                        cur_mean_attn = attn_per_mod.mean(dim=-1).mean(dim=-1)
                    mean_attn.append(cur_mean_attn)
                    offset += size
                mean_attn = torch.stack(mean_attn, dim=0)
                mean_attn = mean_attn.softmax(dim=0)
                softmax_multipliers = [mean_attn[i].unsqueeze(-1).unsqueeze(-1) for i in range(n_mod)]
                if random.random() < 1 / 100:
                    print(sizes, mean_attn.permute(1, 2, 0)[0, :4])  # TODO: it's for debug -> remove
            elif self.modality_importance == 'per_token':
                mean_attn = []
                offset = 0
                for size in sizes:
                    mean_attn.append(attn[:, :, :, offset: offset + size].mean(dim=-1))
                    offset += size
                mean_attn = torch.stack(mean_attn, dim=0)
                mean_attn = mean_attn.softmax(dim=0)
                softmax_multipliers = [mean_attn[i].unsqueeze(-1) for i in range(n_mod)]
                if random.random() < 1 / 100:
                    print(sizes, mean_attn.permute(1, 2, 3, 0)[0, :4, 10])  # TODO: it's for debug -> remove
            else:
                raise NotImplementedError

            if self.debug:
                self.debug_softmax_multipliers = softmax_multipliers

            if zero_attention_mask is not None:
                attn.masked_fill_(zero_attention_mask, -float("inf"))

            attn_softmax = []
            offset = 0
            for size, softmax_multiplier in zip(sizes, softmax_multipliers):
                cur_attn_softmax = attn[:, :, :, offset: offset + size].softmax(dim=-1) * softmax_multiplier
                if zero_attention_mask is not None:
                    zero_mask = (softmax_multiplier == 0).expand_as(cur_attn_softmax) # (bs, n_heads, q_length)
                    cur_attn_softmax.masked_fill_(zero_mask, 0)

                attn_softmax.append(cur_attn_softmax)
                offset += size
            attn = torch.cat(attn_softmax, dim=-1)

        if self.debug:
            self.debug_attn = attn

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
