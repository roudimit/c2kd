import torch.nn as nn
import torch
from model.loss import NormSoftmaxLoss, NormSoftmaxLoss2, MMS_Loss, sim_matrix
import random
import numpy as np


def average_embeddings(ids_arr, embed_arr, verbose=False):
    # check if ids are unique, if not average embedings with the same ids
    ids_arr = np.array(ids_arr)
    unique_ids, counts = np.unique(ids_arr, return_counts=True)

    if len(ids_arr) != len(unique_ids):
        # group and average https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        index = {id_: idx for idx, id_ in enumerate(unique_ids)}
        indexed_ids = [index[id_] for id_ in ids_arr]

        for name, embed in embed_arr.items():
            labels = torch.LongTensor(indexed_ids).to(embed.device)
            labels = labels.view(labels.size(0), 1).expand(-1, embed.size(1))
            unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

            res = torch.zeros_like(unique_labels, dtype=embed.dtype).scatter_add_(0, labels, embed)
            embed_arr[name] = res / labels_count.float().unsqueeze(1)

            # if there were items that we wanted to skip (id=-1):
            if '-1' in index:
                bad_label = index['-1']
                idx_bad_label = (unique_labels[:, 0] == bad_label).nonzero(as_tuple=True)[0] #https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
                if verbose:
                    print('Percent of skipped:', (labels == bad_label).sum() / (labels.shape[0] * labels.shape[1]) )
                embed = embed_arr[name]
                if idx_bad_label == 0:
                    embed = embed[1:]
                elif idx_bad_label == len(unique_labels) - 1:
                    embed = embed[:-1]
                else:
                    embed = torch.cat(
                        (embed[:idx_bad_label], embed[idx_bad_label + 1:]),
                        dim=0)
                embed_arr[name] = embed

    return embed_arr

class MultilingualTVLoss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.05):
        super().__init__()
        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
            self.flag_normalize_embeddings = True

    def forward(self, text, video):
        loss = self.contrastive_loss(sim_matrix(text, video, flag_normalize_embeddings=self.flag_normalize_embeddings))
        return loss
class MultimodalNormSoftmaxLoss(nn.Module):
    def __init__(self, contrastive_loss='NormSoftmax', temperature=0.05,
                 retrieval_weight=1, mlm_weight=1, mlfm_weight=1, mvm_weight=1, mam_weight=1,
                 tv_weight=0, ta_weight=0, va_weight=0,
                 t_va_weight=0, v_ta_weight=0, a_tv_weight=0,
                 t__v_a_weight=0, t__a_v_weight=0, v__t_a_weight=0, v__a_t_weight=0, a__t_v_weight=0, a__v_t_weight=0,
                 mlm_text_weight=0, mlm_tv_weight=0, mlm_ta_weight=0, mlm_tva_weight=0,
                 mlfm_text_weight=0, mlfm_tv_weight=0, mlfm_ta_weight=0, mlfm_tva_weight=0,
                 mvm_video_weight=0, mvm_tv_weight=0, mvm_va_weight=0, mvm_tva_weight=0,
                 mam_audio_weight=0, mam_ta_weight=0, mam_va_weight=0, mam_tva_weight=0,
                 cross_entropy_ignore_index=-1,
                 hero_contrastive_loss=False,
                 new_masking_contrastive_loss=False,
                 average_over_the_same_ids=False,
                 use_nonempty_mask=False,
                 flag_nonempty_mask_or=False):
        super().__init__()

        if contrastive_loss == 'NormSoftmax':
            self.contrastive_loss = NormSoftmaxLoss(temperature=temperature)
            self.flag_normalize_embeddings = True
        elif contrastive_loss == 'MMS':
            self.contrastive_loss = MMS_Loss()
            self.flag_normalize_embeddings = False
        elif contrastive_loss == 'MMS_normalized':
            self.contrastive_loss = MMS_Loss()
            self.flag_normalize_embeddings = True
        else:
            raise NotImplementedError()

        self.new_masking_contrastive_loss = new_masking_contrastive_loss
        if new_masking_contrastive_loss:
            self.contrastive_loss_for_masked = NormSoftmaxLoss(temperature=temperature)
        else:
            self.contrastive_loss_for_masked = NormSoftmaxLoss2(temperature=temperature)

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=cross_entropy_ignore_index)
        self.retrieval_weight = retrieval_weight
        self.mlm_weight = mlm_weight
        self.mlfm_weight = mlfm_weight
        self.mvm_weight = mvm_weight
        self.mam_weight = mam_weight

        self.tv_weight = tv_weight
        self.ta_weight = ta_weight
        self.va_weight = va_weight

        self.t_va_weight = t_va_weight
        self.v_ta_weight = v_ta_weight
        self.a_tv_weight = a_tv_weight

        self.t__v_a_weight = t__v_a_weight
        self.t__a_v_weight = t__a_v_weight
        self.v__t_a_weight = v__t_a_weight
        self.v__a_t_weight = v__a_t_weight
        self.a__t_v_weight = a__t_v_weight
        self.a__v_t_weight = a__v_t_weight

        self.mlm_text_weight = mlm_text_weight
        self.mlm_tv_weight = mlm_tv_weight
        self.mlm_ta_weight = mlm_ta_weight
        self.mlm_tva_weight = mlm_tva_weight

        self.mlfm_text_weight = mlfm_text_weight
        self.mlfm_tv_weight = mlfm_tv_weight
        self.mlfm_ta_weight = mlfm_ta_weight
        self.mlfm_tva_weight = mlfm_tva_weight

        self.mvm_video_weight = mvm_video_weight
        self.mvm_tv_weight = mvm_tv_weight
        self.mvm_va_weight = mvm_va_weight
        self.mvm_tva_weight = mvm_tva_weight

        self.mam_audio_weight = mam_audio_weight
        self.mam_ta_weight = mam_ta_weight
        self.mam_va_weight = mam_va_weight
        self.mam_tva_weight = mam_tva_weight

        self.hero_contrastive_loss = hero_contrastive_loss
        self.average_over_the_same_ids = average_over_the_same_ids
        self.use_nonempty_mask = use_nonempty_mask
        self.flag_nonempty_mask_or = flag_nonempty_mask_or

        self.temperature = temperature

    def forward(self, input_data, task=None, ids=None):
        if task is None:
            task = 'Retrieval'
        assert not self.average_over_the_same_ids or task == 'Retrieval'

        loss_info = {}
        final_loss = 0

        if 'Retrieval' in task:
            loss_sum = 0
            weight_sum = 0

            if self.average_over_the_same_ids:
                input_data = average_embeddings(ids, input_data)

            nonempty = {}
            if self.use_nonempty_mask:
                nonempty['tv'] = input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask']
                nonempty['ta'] = input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']
                nonempty['va'] = input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask']

                if self.flag_nonempty_mask_or:
                    nonempty['t_va'] = input_data['text_nonempty_input_mask'] & (input_data['video_nonempty_input_mask'] | input_data['audio_nonempty_input_mask'])
                    nonempty['v_ta'] = input_data['video_nonempty_input_mask'] & (input_data['text_nonempty_input_mask'] | input_data['audio_nonempty_input_mask'])
                    nonempty['a_tv'] = input_data['audio_nonempty_input_mask'] & (input_data['text_nonempty_input_mask'] | input_data['video_nonempty_input_mask'])
                else:
                    nonempty['t_va'] = input_data['text_nonempty_input_mask'] & (
                                input_data['video_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
                    nonempty['v_ta'] = input_data['video_nonempty_input_mask'] & (
                                input_data['text_nonempty_input_mask'] & input_data['audio_nonempty_input_mask'])
                    nonempty['a_tv'] = input_data['audio_nonempty_input_mask'] & (
                                input_data['text_nonempty_input_mask'] & input_data['video_nonempty_input_mask'])

                nonempty['t__v_a'] = nonempty['t__a_v'] = nonempty['t_va']
                nonempty['v__t_a'] = nonempty['v__a_t'] = nonempty['v_ta']
                nonempty['a__t_v'] = nonempty['a__v_t'] = nonempty['a_tv']

            for name, embed_name1, embed_name2, weight in [
                ('tv', 'text_embed', 'video_embed', self.tv_weight),
                ('ta', 'text_embed', 'audio_embed', self.ta_weight),
                ('va', 'video_embed', 'audio_embed', self.va_weight),

                ('t_va', 'text_embed', 'va_embed', self.t_va_weight),
                ('v_ta', 'video_embed', 'ta_embed', self.v_ta_weight),
                ('a_tv', 'audio_embed', 'tv_embed', self.a_tv_weight),

                ('t__v_a', 'text_embed', 'video_va_embed', self.t__v_a_weight),
                ('t__a_v', 'text_embed', 'audio_va_embed', self.t__a_v_weight),
                ('v__t_a', 'video_embed', 'text_ta_embed', self.v__t_a_weight),
                ('v__a_t', 'video_embed', 'audio_ta_embed', self.v__a_t_weight),
                ('a__t_v', 'audio_embed', 'text_tv_embed', self.a__t_v_weight),
                ('a__v_t', 'audio_embed', 'video_tv_embed', self.a__v_t_weight),
            ]:
                if (embed_name1 in input_data) and (embed_name2 in input_data) and (weight != 0):
                    if self.use_nonempty_mask:
                        nonempty_mask = nonempty[name]
                        embed1 = input_data[embed_name1][nonempty_mask]
                        embed2 = input_data[embed_name2][nonempty_mask]
                    else:
                        embed1 = input_data[embed_name1]
                        embed2 = input_data[embed_name2]

                    loss = self.contrastive_loss(sim_matrix(embed1, embed2,
                                                            flag_normalize_embeddings=self.flag_normalize_embeddings))
                    loss_info[name] = loss.item()
                    loss_sum += weight * loss
                    weight_sum += weight

            loss_info['Retrieval'] = (loss_sum / weight_sum).item()
            final_loss += self.retrieval_weight * (loss_sum / weight_sum)
        if 'MLM' in task and self.mlm_weight != 0:
            loss_sum = 0
            weight_sum = 0

            output_labels = input_data['text_output_labels']
            for loss_name, weight, pred_name in [
                ('mlm_text', self.mlm_text_weight, 'text_t_pred'),
                ('mlm_tv', self.mlm_tv_weight, 'text_tv_pred'),
                ('mlm_ta', self.mlm_ta_weight, 'text_ta_pred'),
                ('mlm_tva', self.mlm_tva_weight, 'text_tva_pred'),
            ]:
                if weight != 0:
                    pred = input_data[pred_name]
                    loss = self.cross_entropy(pred.view(-1, pred.size(-1)), output_labels.view(-1))
                    loss_info[loss_name] = loss.item()
                    loss_sum += weight * loss
                    weight_sum += weight

            loss_info['mlm'] = (loss_sum / weight_sum).item()
            final_loss += self.mlm_weight * (loss_sum / weight_sum)
        if 'MLFM' in task and self.mlfm_weight != 0:
            lossname_weight_predname = [
                ('mlfm_text', self.mlfm_text_weight, 'text_t_pred'),
                ('mlfm_tv', self.mlfm_tv_weight, 'text_tv_pred'),
                ('mlfm_ta', self.mlfm_ta_weight, 'text_ta_pred'),
                ('mlfm_tva', self.mlfm_tva_weight, 'text_tva_pred'),
            ]
            loss = _masked_contrastive_loss(input_data, 'text_masked', 'text_correct_tokens', 'text_special_tokens_mask',
                                            lossname_weight_predname=lossname_weight_predname,
                                            hero_contrastive_loss=self.hero_contrastive_loss,
                                            contrastive_loss_func=self.contrastive_loss_for_masked,
                                            loss_info=loss_info,
                                            flag_normalize_embeddings=self.flag_normalize_embeddings,
                                            use_only_pos_matrix=self.new_masking_contrastive_loss)
            loss_info['mlfm'] = loss.item()
            final_loss += self.mlfm_weight * loss
        if 'MVM' in task and self.mvm_weight != 0:
            lossname_weight_predname = [
                ('mvm_video', self.mvm_video_weight, 'video_v_pred'),
                ('mvm_tv', self.mvm_tv_weight, 'video_tv_pred'),
                ('mvm_va', self.mvm_va_weight, 'video_va_pred'),
                ('mvm_tva', self.mvm_tva_weight, 'video_tva_pred'),
            ]
            loss = _masked_contrastive_loss(input_data, 'video_masked', 'video_correct_tokens', 'video_special_tokens_mask',
                                            lossname_weight_predname=lossname_weight_predname,
                                            hero_contrastive_loss=self.hero_contrastive_loss,
                                            contrastive_loss_func=self.contrastive_loss_for_masked,
                                            loss_info=loss_info,
                                            flag_normalize_embeddings=self.flag_normalize_embeddings,
                                            use_only_pos_matrix=self.new_masking_contrastive_loss)
            loss_info['mvm'] = loss.item()
            final_loss += self.mvm_weight * loss
        if 'MAM' in task and self.mam_weight != 0:
            lossname_weight_predname = [
                ('mam_audio', self.mam_audio_weight, 'audio_a_pred'),
                ('mam_ta', self.mam_ta_weight, 'audio_ta_pred'),
                ('mam_va', self.mam_va_weight, 'audio_va_pred'),
                ('mam_tva', self.mam_tva_weight, 'audio_tva_pred'),
            ]
            loss = _masked_contrastive_loss(input_data, 'audio_masked', 'audio_correct_tokens', 'audio_special_tokens_mask',
                                            lossname_weight_predname=lossname_weight_predname,
                                            hero_contrastive_loss=self.hero_contrastive_loss,
                                            contrastive_loss_func=self.contrastive_loss_for_masked,
                                            loss_info=loss_info,
                                            flag_normalize_embeddings=self.flag_normalize_embeddings,
                                            use_only_pos_matrix=self.new_masking_contrastive_loss)
            loss_info['mam'] = loss.item()
            final_loss += self.mam_weight * loss

        return final_loss, loss_info


def _masked_contrastive_loss(input_data, masked_name, correct_tokens_name, special_tokens_mask_name,
                             lossname_weight_predname,
                             hero_contrastive_loss, contrastive_loss_func, loss_info, flag_normalize_embeddings,
                             use_only_pos_matrix):
    loss_sum = 0
    weight_sum = 0
    masked = input_data[masked_name]
    correct_tokens = input_data[correct_tokens_name]
    special_tokens_mask = input_data[special_tokens_mask_name]

    masked = masked.view(-1)
    correct_tokens = correct_tokens.detach().view(-1, correct_tokens.size(-1))
    for loss_name, weight, pred_name in lossname_weight_predname:
        if weight != 0:
            pred = input_data[pred_name]
            if hero_contrastive_loss:
                pred = pred.view(-1, pred.size(-1))
                pred_output = pred[masked]
                pos_output = correct_tokens[masked]
                neg_output = pred[(~masked) & (~special_tokens_mask.view(-1))]
            else:
                pred = pred.view(-1, pred.size(-1))
                pred_output = pred[masked]
                pos_output = correct_tokens[masked]
                neg_output = correct_tokens[(~masked) & (~special_tokens_mask.view(-1))]

            if use_only_pos_matrix:
                pos_score = sim_matrix(pred_output, pos_output, flag_normalize_embeddings=flag_normalize_embeddings)
                loss = contrastive_loss_func(pos_score)
            else:
                # TODO: not well debuged
                n_pos = len(pos_output)  # sample just a subset of negative values
                indice = random.sample(range(len(neg_output)), n_pos)
                indice = torch.tensor(indice)
                neg_output = neg_output[indice]
                pos_score = sim_matrix(pred_output, pos_output, flag_normalize_embeddings=flag_normalize_embeddings)
                neg_score = sim_matrix(pred_output, neg_output, flag_normalize_embeddings=flag_normalize_embeddings)
                score = torch.cat((pos_score, neg_score), dim=1)
                loss = contrastive_loss_func(score)
            loss_info[loss_name] = loss.item()

            loss_sum += weight * loss
            weight_sum += weight

    return loss_sum / weight_sum