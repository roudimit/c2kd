import collections

from timm.models.vision_transformer import _init_vit_weights, named_apply, trunc_normal_
import torch.nn as nn
from functools import partial
import torch
from model.layers import Block
from model.loss import normalize_embeddings

class FusionTransformer(nn.Module):
    def __init__(self, num_modalities=3, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='',
                 audio_model_name=None,
                 type_embedding=True,
                 cls_token=False,
                 masking_token=False,
                 masking_token_per_modality=False,
                 cls_token_per_pair=False,
                 modality_importance='none',
                 attention_ratio=1,
                 apply_norm_layer=True,
                 use_attention_masks=True,
                 attention_activation='softmax',
                 attention_act_temperature=1,
                 qkv_per_mod=False,
                 keys_per_mod=False,
                 average_as_cls=False,
                 max_pooled_as_cls=False,
                 apply_blocks=True,
                 normalize_before_averaging_cls=False,
                 norm_layer_per_mod=False
                 ):
        super().__init__()
        assert num_modalities in [3, 5]

        # TODO: delete num_modalities from init in future
        self.num_modalities = num_modalities
        assert audio_model_name in [None, 'AudioViTTransformer', 'ASTModel']
        self.audio_model_name = audio_model_name
        self.embed_dim = embed_dim
        self.average_as_cls = average_as_cls
        self.max_pooled_as_cls = max_pooled_as_cls
        self.apply_blocks = apply_blocks
        self.normalize_before_averaging_cls = normalize_before_averaging_cls
        assert (self.average_as_cls is False) or (self.max_pooled_as_cls is False)

        if type_embedding:
            self.type_embed = nn.Parameter(torch.zeros(self.num_modalities, 1, 1, embed_dim))
        else:
            self.type_embed = None

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        if cls_token:
            assert not average_as_cls
            assert not cls_token_per_pair

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_token_per_pair = None
        elif cls_token_per_pair:
            assert not average_as_cls
            assert self.num_modalities == 3
            self.cls_token = None
            self.cls_token_per_pair = nn.Parameter(torch.zeros(1, 3, embed_dim))
        else:
            self.cls_token = None
            self.cls_token_per_pair = None

        self.masking_token_per_modality = masking_token_per_modality
        if masking_token:
            if masking_token_per_modality:
                self.masking_token = nn.Parameter(torch.zeros(num_modalities, embed_dim))
            else:
                self.masking_token = nn.Parameter(torch.zeros(embed_dim))

        else:
            self.masking_token = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        assert modality_importance in ['none', 'equal', 'per_token', 'per_modality']
        self.modality_importance = modality_importance

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                modality_importance=self.modality_importance, attention_ratio=attention_ratio,
                act=attention_activation, temperature=attention_act_temperature, qkv_per_mod=qkv_per_mod,
                keys_per_mod=keys_per_mod,
                there_is_cls_token=cls_token
            )
            for i in range(depth)])

        self.norm_layer_per_mod = norm_layer_per_mod
        if self.norm_layer_per_mod:
            assert num_modalities == 3
            self.norm_text = norm_layer(embed_dim)
            self.norm_video = norm_layer(embed_dim)
            self.norm_audio = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
        self.apply_norm_layer = apply_norm_layer
        self.use_attention_masks = use_attention_masks

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')

        assert not ('nlhb' in mode) # TODO: fix next line
        head_bias = 0
        # head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.

        if self.type_embed is not None:
            trunc_normal_(self.type_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=.02)
            if self.masking_token is not None:
                trunc_normal_(self.masking_token, std=.02)
            if self.cls_token_per_pair is not None:
                trunc_normal_(self.cls_token_per_pair, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def forward(self, text=None, video=None, audio=None, caption=None, image=None):
        if self.num_modalities == 3:
            data = [text, video, audio]
        else:
            data = [text, video, audio, caption, image]

        if self.type_embed is not None:
            tokens = [x['all_tokens'] + type_embed for x, type_embed in zip(data, self.type_embed) if x is not None]
        else:
            tokens = [x['all_tokens'] for x in data if x is not None]
        sizes = [x.shape[1] for x in tokens]
        sizes_tva = [(x['all_tokens'].shape[1] if x is not None else 0) for x in [text, video, audio]]

        tokens = torch.cat(tokens, dim=1)

        # concatenate cls token
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tokens = torch.cat((cls_token, tokens), dim=1)
            offset = 1
        elif self.cls_token_per_pair is not None:
            if (text is not None) and (video is not None):
                cls_token_idx = 0
            elif (text is not None) and (audio is not None):
                cls_token_idx = 1
            elif (video is not None) and (audio is not None):
                cls_token_idx = 2
            else:
                cls_token_idx = -1

            if cls_token_idx != -1:
                cls_token = self.cls_token_per_pair[:, [cls_token_idx]].expand(tokens.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                tokens = torch.cat((cls_token, tokens), dim=1)
                offset = 1
            else:
                offset = 0
        else:
            offset = 0

        if self.use_attention_masks:
            tokens_mask = [x['attention_mask'] for x in data if x is not None]
            tokens_mask = torch.cat(tokens_mask, dim=1)
            if offset != 0:
                cls_token_mask = torch.ones((1, offset)).to(tokens_mask.device).expand(tokens_mask.shape[0], -1)
                tokens_mask = torch.cat((cls_token_mask, tokens_mask), dim=1)
        else:
            tokens_mask = None

        if self.apply_blocks:
            for block in self.blocks:
                # if there is cls tokens, threat it as additional modality
                block_sizes = sizes if offset == 0 else [offset] + sizes
                sizes_tva = sizes_tva if offset == 0 else [offset] + sizes_tva
                tokens = block(tokens, attention_mask=tokens_mask, sizes=block_sizes, sizes_tva=sizes_tva)

        if self.apply_norm_layer:
            if not self.norm_layer_per_mod:
                tokens = self.norm(tokens)

        output = collections.OrderedDict()

        def _get_average(tokens, attention_mask):
            attention_mask = attention_mask.unsqueeze(2).expand_as(tokens)
            return (tokens * attention_mask).sum(1) / attention_mask.sum(1)

        def _get_max_pooled(tokens, attention_mask):
            attention_mask = attention_mask.unsqueeze(2).expand_as(tokens)
            tokens = tokens.clone()
            tokens.masked_fill_(attention_mask == 0, -1e3)  # (bs, n_tokens, dim)
            return torch.nn.functional.adaptive_max_pool1d(tokens.permute(0, 2, 1), output_size=1).squeeze(-1)

        def _get_cls_token(tokens, attention_mask, flag_ASTModel=False):
            if self.average_as_cls:
                return _get_average(tokens, attention_mask)
            elif self.max_pooled_as_cls:
                return _get_max_pooled(tokens, attention_mask)
            else:
                if flag_ASTModel:
                    return (tokens[:, 0] + tokens[:, 1]) / 2
                else:
                    return tokens[:, 0]

        if text is not None:
            n_tokens = text['all_tokens'].size(1)
            attention_mask = text['attention_mask']
            all_tokens = tokens[:, offset:offset + n_tokens]

            if self.apply_norm_layer and self.norm_layer_per_mod:
                all_tokens = self.norm_text(all_tokens)
            offset += n_tokens
            output['text'] = {
                "all_tokens": all_tokens,
                "cls": _get_cls_token(all_tokens, attention_mask)
            }

        if video is not None:
            n_tokens = video['all_tokens'].size(1)
            attention_mask = video['attention_mask']
            all_tokens = tokens[:, offset:offset + n_tokens]

            if self.apply_norm_layer and self.norm_layer_per_mod:
                all_tokens = self.norm_video(all_tokens)

            offset += n_tokens
            output['video'] = {
                "all_tokens": all_tokens,
                "cls": _get_cls_token(all_tokens, attention_mask)
            }

        if audio is not None:
            n_tokens = audio['all_tokens'].size(1)
            attention_mask = audio['attention_mask']
            all_tokens = tokens[:, offset: offset + n_tokens]

            if self.apply_norm_layer and self.norm_layer_per_mod:
                all_tokens = self.norm_audio(all_tokens)

            offset += n_tokens
            output['audio'] = {
                "all_tokens": all_tokens,
                "cls": _get_cls_token(all_tokens, attention_mask, flag_ASTModel=(self.audio_model_name == 'ASTModel'))
            }

        if caption is not None:
            n_tokens = caption['all_tokens'].size(1)
            attention_mask = caption['attention_mask']

            all_tokens = tokens[:, offset:offset + n_tokens]
            offset += n_tokens
            output['caption'] = {
                "all_tokens": all_tokens,
                "cls": _get_cls_token(all_tokens, attention_mask)
            }

        if image is not None:
            n_tokens = image['all_tokens'].size(1)
            attention_mask = image['attention_mask']

            all_tokens = tokens[:, offset:offset + n_tokens]
            offset += n_tokens
            output['image'] = {
                "all_tokens": all_tokens,
                "cls": _get_cls_token(all_tokens, attention_mask)
            }

        modalities = list(output.keys())

        if (self.cls_token is not None) or ((self.cls_token_per_pair is not None) and (cls_token_idx != -1)):
            modalities = '_'.join(modalities)
            if modalities not in output:
                output[modalities] = {}
            output[modalities]['cls'] = tokens[:, 0]
        else:
            for i, mod1 in enumerate(modalities):
                for mod2 in modalities[i + 1:]:
                    if self.normalize_before_averaging_cls:
                        output[f'{mod1}_{mod2}'] = {'cls': (normalize_embeddings(output[mod1]['cls']) +
                                                            normalize_embeddings(output[mod2]['cls'])) / 2}
                    else:
                        output[f'{mod1}_{mod2}'] = {'cls': (output[mod1]['cls'] + output[mod2]['cls']) / 2}

        return output
