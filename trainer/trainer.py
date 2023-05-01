import collections

import os
import numpy as np
import torch
import tqdm

from model.loss import sim_matrix, normalize_embeddings, normalize_embeddings_seq
from model.multimodal_loss import average_embeddings
from base import BaseTrainer
from utils import inf_loop
import itertools
from torch.cuda.amp import autocast, GradScaler
# import clip
import itertools

def _move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    else:
        return {key: val.to(device) for key, val in data.items()}


def _apply_clip_text_model(clip_text_model, data, device):
    with torch.no_grad():
        input_x = clip.tokenize(data['raw_text'], truncate=True).to(device)
        x = clip_text_model.token_embedding(input_x).type(
            clip_text_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + clip_text_model.positional_embedding.type(clip_text_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_text_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if clip_text_model.use_proj_for_text:
            x = clip_text_model.ln_final(x).type(clip_text_model.dtype)
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x @ clip_text_model.text_projection

        x = x.detach().cpu()
        input_x = input_x.cpu()

    batch_size, _, dim = x.shape
    prev_n_tokens = data['text'].shape[1]

    input_x = input_x[:, 1:]  # first token is a token of beginning of the sentence
    new_text = x[:, 1:]  # first token is a token of beginning of the sentence
    new_text_mask = torch.zeros(batch_size, prev_n_tokens)

    for i in range(len(x)):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        n_eot = input_x[i].argmax().item()
        new_text_mask[i, :n_eot] = 1

    new_text = x[:, :prev_n_tokens]
    data['text'] = new_text.type(data['text'].dtype)
    data['text_mask'] = new_text_mask.type(data['text_mask'].dtype)
    return data


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, teacher_models=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].effective_batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.effective_batch_size for x in self.data_loader])
        self.tokenizer = tokenizer

        cfg_trainer = config['trainer']
        self.max_samples_per_epoch = cfg_trainer['max_samples_per_epoch']
        self.max_text_token_length = cfg_trainer.get("max_text_token_length")
        self.mixed_precision = cfg_trainer.get("mixed_precision", False)
        self.scaler = GradScaler()
        self.tasks = cfg_trainer.get("tasks", ['Retrieval'])
        self.eval_task = cfg_trainer.get("eval_task")
        self.batch_size_per_task = cfg_trainer.get('batch_size_per_task')
        self.resample_dataset_each_epoch = cfg_trainer.get("resample_dataset_each_epoch", False)
        self.clip_grad = cfg_trainer.get("clip_grad")
        self.use_eval_mode_always = cfg_trainer.get("use_eval_mode_always", False)
        self.save_latest = cfg_trainer.get('save_latest', True)

        self.use_clip_text_model = cfg_trainer.get("use_clip_text_model", False)
        if self.use_clip_text_model:
            import clip
            device, device_ids = self._prepare_device(config['n_gpu'])
            self.clip_text_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_text_model.eval()
            use_clip_text_model_use_proj = cfg_trainer.get("use_clip_text_model_use_proj", False)
            self.clip_text_model.use_proj_for_text = use_clip_text_model_use_proj
            # if len(device_ids) > 1:
            #     self.clip_text_model = torch.nn.DataParallel(self.clip_text_model, device_ids=device_ids)
        else:
            self.clip_text_model = None

        if len_epoch is None:
            # epoch-based training
            # Original code:
            # # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = [inf_loop(x) for x in data_loader]
            self.len_epoch = len_epoch

        self.teachers = {}
        for name in teacher_models:
            teacher_model = teacher_models[name]
            if 'xlm' in name:
                if 'MSRVTT' in config.config['data_loader'][0]['args']['dataset_name']:
                    if 'msrvtt' in config.config['data_loader'][0]['args']['dataset_kwargs']['data_path']:
                        checkpoint_path = './weights/c2kd_teachers/msrvtt/xlm/latest_model.pth'
                    elif 'vatex' in config.config['data_loader'][0]['args']['dataset_kwargs']['data_path']: # vatex uses msr-vtt dataloader
                        checkpoint_path = './weights/c2kd_teachers/vatex/xlm/latest_model.pth'
                elif 'RUDDER' in config.config['data_loader'][0]['args']['dataset_name']:
                    checkpoint_path = './weights/c2kd_teachers/rudder/xlm/latest_model.pth'
                else:
                    checkpoint_path = './weights/c2kd_teachers/youcook2/xlm/latest_model.pth'
                self.model_teacher_xlm = self._load_checkpoint_teacher(model, teacher_model, checkpoint_path).to(self.device)
                self.teachers['xlm'] = self.model_teacher_xlm
            elif 'mbert' in name:
                if 'MSRVTT' in config.config['data_loader'][0]['args']['dataset_name']:
                    if 'msrvtt' in config.config['data_loader'][0]['args']['dataset_kwargs']['data_path']:
                        checkpoint_path = './weights/c2kd_teachers/msrvtt/mbert/latest_model.pth'
                    elif 'vatex' in config.config['data_loader'][0]['args']['dataset_kwargs']['data_path']: # vatex uses msr-vtt dataloader
                        checkpoint_path = './weights/c2kd_teachers/vatex/mbert/latest_model.pth'
                elif 'RUDDER' in config.config['data_loader'][0]['args']['dataset_name']:
                    checkpoint_path = './weights/c2kd_teachers/rudder/mbert/latest_model.pth' 
                else:
                    checkpoint_path = './weights/c2kd_teachers/youcook2/mbert/latest_model.pth'
                self.model_teacher_mbert = self._load_checkpoint_teacher(model, teacher_model, checkpoint_path).to(self.device)
                self.teachers['mbert'] = self.model_teacher_mbert
            elif 'distill' in name:
                if 'MSRVTT' in config.config['data_loader'][0]['args']['dataset_name']:
                    if 'msrvtt' in config.config['data_loader'][0]['args']['dataset_kwargs']['data_path']:
                        checkpoint_path = './weights/c2kd_teachers/msrvtt/distill/latest_model.pth'
                    elif 'vatex' in config.config['data_loader'][0]['args']['dataset_kwargs']['data_path']: # vatex uses msr-vtt dataloader
                        checkpoint_path = './weights/c2kd_teachers/vatex/distill/latest_model.pth'
                elif 'RUDDER' in config.config['data_loader'][0]['args']['dataset_name']:
                    checkpoint_path = './weights/c2kd_teachers/rudder/distill/latest_model.pth'
                else:
                    checkpoint_path = './weights/c2kd_teachers/youcook2/distill/latest_model.pth'
                self.model_teacher_distill = self._load_checkpoint_teacher(model, teacher_model, checkpoint_path).to(self.device)
                self.teachers['distill'] = self.model_teacher_distill

        # if self.config.args.teacher_sim_pool in ['learnable_multitask', 'learnable_poolers']:
        #     self.learnable_weights = torch.nn.parameter.Parameter(torch.tensor([0.2, 0.4, 0.4]))

    def _load_checkpoint_teacher(self, model, teacher_model, checkpoint_path):
        import copy
        self.model = teacher_model
        self._resume_checkpoint(checkpoint_path) # load the desired weights
        model_teacher = copy.deepcopy(self.model) # copy the loaded weights
        del self.model # delete the loaded weights and load random weights again
        self.model = model.to(self.device) # copy the original model with optimizer
        return model_teacher # return the model with loaded weights


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        langs =  ['de', 'fr', 'cs', 'zh', 'ru', 'vi', 'sw', 'es', 'en', 'ja'] # include en for training
        langs +=  ['hi', 'mr', 'ta', 'te', 'kn', 'ml'] # new RUDDER langs
        torch.cuda.empty_cache()

        if self.use_eval_mode_always:
            self.model.eval()
        else:
            self.model.train()

        if self.resample_dataset_each_epoch:
            for dataloader in self.data_loader:
                dataloader.resample_dataset()

        total_loss = [0] * len(self.data_loader)

        if self.config.args.stage == 'c2kd':
            print('Teachers: {}'.format(self.config.args.teachers))
            print('Contrastive weight: {}, distill loss temp: {}, teacher sim pooling: {}'.format(
                self.config.args.balance, self.config.args.distill_temp, self.config.args.teacher_sim_pool,
            ))

        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            for dl_idx, data in enumerate(data_li):
                task = np.random.choice(self.tasks)
                data = format_dataloader_output(data)

                if self.tokenizer is not None:
                    for field in ['text', 'caption']:
                        if field in data:
                            data[field] = self.tokenizer(data[field], return_tensors='pt', padding=True,
                                                          truncation=True, max_length=self.max_text_token_length,
                                                          return_special_tokens_mask=True)

                if self.clip_text_model is not None:
                    data = _apply_clip_text_model(self.clip_text_model, data, self.device)

                if (self.batch_size_per_task is not None) and task in self.batch_size_per_task:
                    bs = self.batch_size_per_task[task]
                    data['text'] = {key: val[:bs] for key, val in data['text'].items()}
                    data['video'] = data['video'][:bs]
                    if 'audio' in data:
                        data['audio'] = data['audio'][:bs]

                for field in ['text', 'text_mask', 'video', 'video_mask', 'audio', 'audio_mask', 'nframes', 'caption', 'image', \
                            'tokens'] + langs + [i + '_audio' for i in langs] + \
                            ['tokens_xlm', 'tokens_mbert', 'tokens_infoxlm', 'tokens_xlm_align', \
                            'tokens_distill', 'tokens_sim_cse'] + \
                            [i + '_xlm' for i in langs] + [i + '_mbert' for i in langs] + [i + '_infoxlm' for i in langs] + \
                            [i + '_xlm_align' for i in langs] + [i + '_distill' for i in langs] + \
                            [i + '_sim_cse' for i in langs]:
                    if field in data:
                        data[field] = _move_to_device(data[field], self.device)
                langs = [i for i in langs if i in data] # update langs for current dataset
                if self.mixed_precision:
                    self.optimizer.zero_grad()

                    if self.config.args.stage == 'baseline-zero-shot': # english-only training
                        with autocast():
                            output = self.model(data, task=task)
                            loss, loss_info = self.loss(output, task=task, ids=data['meta']['ids'])
                        self.scaler.scale(loss).backward()
                        if self.clip_grad:
                            self.scaler.unscale_(self.optimizer)
                            print(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    elif self.config.args.stage == 'baseline-translate-train': # translate-train
                        with autocast():
                            output = self.model(data, task=task)
                            loss, loss_info = self.loss(output, task=task, ids=data['meta']['ids'])
                            lang_outputs = {'en': output['text_embed']}
                            for lang in langs:
                                if lang != 'en':
                                    lang_outputs[lang] = self.model.forward_lang(data, lang)
                                    loss += self.loss_tv(lang_outputs[lang][lang], output['video_embed'])
                        self.scaler.scale(loss).backward()
                        if self.clip_grad:
                            self.scaler.unscale_(self.optimizer)
                            print(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    elif self.config.args.stage == 'c2kd': # proposed method
                        from model.metric import retrieval_metrics
                        self.contrastive_loss_weight = self.config.args.balance
                        self.distill_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
                        self.distill_temp = self.config.args.distill_temp
                        self.distill_loss_weight = 1 - self.config.args.balance

                        self.teachers = {}
                        if 'xlm' in self.config.args.teachers:
                            self.teachers['xlm'] = self.model_teacher_xlm
                        if 'mbert' in self.config.args.teachers:
                            self.teachers['mbert'] = self.model_teacher_mbert
                        if 'distill' in self.config.args.teachers:
                            self.teachers['distill'] = self.model_teacher_distill
                        if 'sim_cse' in self.config.args.teachers:
                            self.teachers['sim_cse'] = self.model_teacher_sim_cse
                        if 'labse' in self.config.args.teachers:
                            self.teachers['labse'] = self.model_teacher_labse
                        
                        self.teacher_sims = {i:0 for i in self.teachers}
                        with autocast():
                            output = self.model(data, task=task)
                            loss, loss_info = self.loss(output, task=task, ids=data['meta']['ids'])
                            loss = self.contrastive_loss_weight * loss
                            lang_outputs = {'en': {'en': output['text_embed']}}
                            with torch.no_grad():
                                for name, teacher in self.teachers.items():
                                    output_teacher = teacher(data, task=task)
                                    self.teacher_sims[name] = sim_matrix(output_teacher['text_embed'], output_teacher['video_embed'])
                                if self.config.args.teacher_sim_pool == 'max':
                                    en_sims = torch.min(torch.stack([sims for _, sims in self.teacher_sims.items()]), dim=0)[0]
                                elif self.config.args.teacher_sim_pool == 'min':
                                    en_sims = torch.max(torch.stack([sims for _, sims in self.teacher_sims.items()]), dim=0)[0]
                                elif self.config.args.teacher_sim_pool == 'mean':
                                    en_sims = torch.mean(torch.stack([sims for _, sims in self.teacher_sims.items()]), dim=0)
                                elif self.config.args.teacher_sim_pool in ['multitask', 'learnable_multitask']:
                                    pass
                                elif self.config.args.teacher_sim_pool == 'learnable_poolers':
                                    en_sims = [torch.min(torch.stack([sims for _, sims in self.teacher_sims.items()]), dim=0)[0],
                                            torch.max(torch.stack([sims for _, sims in self.teacher_sims.items()]), dim=0)[0],
                                            torch.mean(torch.stack([sims for _, sims in self.teacher_sims.items()]), dim=0)]
                                    self.teacher_sims = {'': i for i in en_sims}
                                else:
                                    raise ValueError
                            # print(retrieval_metrics(en_sims.cpu()))
                            for lang in langs:
                                if lang != 'en': 
                                    lang_outputs[lang] = self.model.forward_lang(data, lang)
                                # NOTE: un-indent is for including English in computation
                                loss += self.contrastive_loss_weight * self.loss_tv(lang_outputs[lang][lang], output['video_embed'])
                                t2v_sims = sim_matrix(lang_outputs[lang][lang], output['video_embed'])
                                if self.config.args.teachtext == True: # ablation study
                                    tt_loss = torch.nn.SmoothL1Loss(reduction="mean")
                                    loss += self.distill_loss_weight * tt_loss(t2v_sims, en_sims)
                                    continue # skip the distill loss
                                    # NOTE: ablation with softmax normalization
                                    # tt_loss = torch.nn.SmoothL1Loss(reduction="mean")
                                    # inputs = torch.softmax(t2v_sims / self.distill_temp, dim=1)
                                    # targets = torch.softmax(en_sims / self.distill_temp, dim=1)
                                    # loss += self.distill_loss_weight * tt_loss(inputs, targets)
                                    # inputs = torch.softmax(t2v_sims.t() / self.distill_temp, dim=1)
                                    # targets = torch.softmax(en_sims.t() / self.distill_temp, dim=1)
                                    # loss += self.distill_loss_weight * tt_loss(inputs, targets)
                                    # continue # skip the distill loss
                                if not self.config.args.teacher_sim_pool in ['multitask', 'learnable_multitask', 'learnable_poolers']:
                                    # distill loss row-wise
                                    inputs = torch.nn.functional.log_softmax(t2v_sims / self.distill_temp, dim=1)
                                    targets = torch.nn.functional.log_softmax(en_sims / self.distill_temp, dim=1)
                                    loss += self.distill_loss_weight * self.distill_loss(inputs, targets)
                                    # distill loss column-wise
                                    inputs = torch.nn.functional.log_softmax(t2v_sims.t() / self.distill_temp, dim=1)
                                    targets = torch.nn.functional.log_softmax(en_sims.t() / self.distill_temp, dim=1)
                                    loss += self.distill_loss_weight * self.distill_loss(inputs, targets)
                                elif self.config.args.teacher_sim_pool == 'multitask': # apply the distillation loss with each teacher
                                    for name, teacher_sim in self.teacher_sims.items():
                                        # distill loss row-wise
                                        inputs = torch.nn.functional.log_softmax(t2v_sims / self.distill_temp, dim=1)
                                        targets = torch.nn.functional.log_softmax(teacher_sim / self.distill_temp, dim=1)
                                        loss += self.distill_loss_weight * self.distill_loss(inputs, targets)
                                        # distill loss column-wise
                                        inputs = torch.nn.functional.log_softmax(t2v_sims.t() / self.distill_temp, dim=1)
                                        targets = torch.nn.functional.log_softmax(teacher_sim.t() / self.distill_temp, dim=1)
                                        loss += self.distill_loss_weight * self.distill_loss(inputs, targets)
                                elif self.config.args.teacher_sim_pool in ['learnable_multitask', 'learnable_poolers']:
                                    print('Learnable weights:', self.model.learnable_weights)
                                    for index, (name, teacher_sim) in enumerate(self.teacher_sims.items()):
                                        # distill loss row-wise
                                        inputs = torch.nn.functional.log_softmax(t2v_sims / self.distill_temp, dim=1)
                                        targets = torch.nn.functional.log_softmax(teacher_sim / self.distill_temp, dim=1)
                                        loss += torch.softmax(self.model.learnable_weights, 0)[index] * self.distill_loss(inputs, targets)
                                        # loss += torch.softmax(self.learnable_weights, 0)[index] * self.distill_loss(inputs, targets)
                                        # distill loss column-wise
                                        inputs = torch.nn.functional.log_softmax(t2v_sims.t() / self.distill_temp, dim=1)
                                        targets = torch.nn.functional.log_softmax(teacher_sim.t() / self.distill_temp, dim=1)
                                        loss += torch.softmax(self.model.learnable_weights, 0)[index] * self.distill_loss(inputs, targets)
                                        # loss += torch.softmax(self.learnable_weights, 0)[index] * self.distill_loss(inputs, targets)

                        self.scaler.scale(loss).backward()
                        if self.clip_grad:
                            self.scaler.unscale_(self.optimizer)
                            print(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else: 
                        raise NotImplementedError

                else:
                    self.optimizer.zero_grad()
                    output = self.model(data, task=task)
                    loss, loss_info = self.loss(output, task=task, ids=data['meta']['ids'])
                    loss.backward()
                    self.optimizer.step()

                if self.writer is not None:
                    for loss_name, loss_value in loss_info.items():
                        self.writer.log_scalar(f'loss_train_{loss_name}_{dl_idx}', loss_value, step=self.step)
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item(), step=self.step)

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))

                self.optimizer.zero_grad()
                del data, output, loss

                self.step += 1

            if batch_idx == self.len_epoch:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.save_latest:
            self._save_checkpoint(epoch, save_latest=True)

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        torch.cuda.empty_cache()
        self.model.eval()

        nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}
        val_loss = [0] * len(self.valid_data_loader)

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                dl_nested_metrics, dl_val_loss, dl_val_loss_detailed = eval(self.model, dl, self.tokenizer,
                                                                            self.max_text_token_length, self.device,
                                                                            self.metrics,
                                                                            self.loss.flag_normalize_embeddings,
                                                                            self.loss,
                                                                            self.eval_task,
                                                                            self.clip_text_model,
                                                                            epoch)

                val_loss[dl_idx] = dl_val_loss
                nested_metrics[dl_idx] = dl_nested_metrics

                if self.writer is not None:
                    self.writer.log_scalar(f'loss_val_{dl_idx}', dl_val_loss, step=epoch)
                    for loss_name, loss_value in dl_val_loss_detailed.items():
                        self.writer.log_scalar(f'loss_val_{loss_name}_{dl_idx}', loss_value, step=epoch)

                short_verbose(epoch=epoch, dl_nested_metrics=dl_nested_metrics, dataset_name=dl.dataset_name)
                for metric in self.metrics:
                    metric_name = metric.__name__
                    res = dl_nested_metrics[metric_name]
                    verbose(epoch=epoch, metrics=res, name=dl.dataset_name,
                            mode=metric_name)

                    if self.writer is not None:
                        to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                    name=self.valid_data_loader[dl_idx].dataset_name)
                        for key, val in to_write.items():
                            self.writer.log_scalar(key, val, step=epoch)

        res_dict = {f'val_loss_{dl_idx}': val_loss[dl_idx]
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].effective_batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def eval(model, dl, tokenizer, max_text_token_length, device, metrics, flag_normalize_embeddings, loss_func=None,
         eval_task=None, clip_text_model=None, epoch=None):
    total_val_loss = 0
    total_val_loss_detailed = collections.defaultdict(lambda: 0)
    meta_arr = []
    ids_arr = []
    embed_arr = collections.defaultdict(lambda: [])
    video_arr = collections.defaultdict(lambda: [])
    langs = ['de', 'fr', 'cs', 'zh', 'ru', 'vi', 'sw', 'es', 'en', 'ja']
    langs +=  ['hi', 'mr', 'ta', 'te', 'kn', 'ml'] # new RUDDER langs

    with torch.no_grad():
        for data in tqdm.tqdm(dl):
            data = format_dataloader_output(data)

            meta_arr.append(data['meta'])
            ids_arr.extend(data['meta']['ids'])
            if tokenizer is not None:
                for field in ['text', 'caption']:
                    if field in data:
                        data[field] = tokenizer(data[field], return_tensors='pt', padding=True,
                                                     truncation=True, max_length=max_text_token_length,
                                                     return_special_tokens_mask=True)
            if clip_text_model is not None:
                data = _apply_clip_text_model(clip_text_model, data, device)

            for field in ['text', 'text_mask', 'video', 'video_mask', 'audio', 'audio_mask', 'nframes', 'caption', 'image', \
                            'tokens'] + langs + [i + '_audio' for i in langs] + \
                            ['tokens_xlm', 'tokens_mbert', 'tokens_infoxlm', 'tokens_xlm_align', \
                            'tokens_distill', 'tokens_sim_cse'] + \
                            [i + '_xlm' for i in langs] + [i + '_mbert' for i in langs] + [i + '_infoxlm' for i in langs] + \
                            [i + '_xlm_align' for i in langs] + [i + '_distill' for i in langs] + \
                            [i + '_sim_cse' for i in langs]:
                if field in data:
                    data[field] = _move_to_device(data[field], device)
            embeds = model(data, task=eval_task)
            video_arr['vid_local'].append(normalize_embeddings_seq(embeds['video_output_tokens']).cpu())
            video_arr['vid_padding_mask'].append(((1 - embeds['video_padding_mask']) * -10000).cpu())
            for name, embed in embeds.items():
                if '_embed' in name:
                    embed_arr[name].append(embed.cpu())
            langs = [i for i in langs if i in data] # update langs for current dataset
            for lang in langs:
                lang_embed = model.forward_lang(data, lang, audio_fwd=False)
                embed_arr[lang].append(lang_embed[lang].cpu())
            if loss_func is not None:
                loss, loss_info = loss_func(embeds, task=eval_task, ids=data['meta']['ids'])
                for loss_name, loss_value in loss_info.items():
                    total_val_loss_detailed[loss_name] += loss_value
                total_val_loss += loss.item()
                del loss
            del data, embeds

    val_loss = total_val_loss / len(dl)
    val_loss_detailed = {loss_name: loss_value / len(dl) for loss_name, loss_value in total_val_loss_detailed.items()}

    # compute scores
    nested_metrics = {}

    for name, embed in embed_arr.items():
        embed_arr[name] = torch.cat(embed, dim=0)
        print(embed_arr[name].shape)
    for name, embed in video_arr.items():
        video_arr[name] = torch.cat(embed, dim=0)
        print(video_arr[name].shape)

    embed_arr = average_embeddings(ids_arr, embed_arr, verbose=True)

    if flag_normalize_embeddings:
        for name, embed in embed_arr.items():
            embed_arr[name] = normalize_embeddings(embed)

    sims = {}
    for name1 in ['text_embed', 'video_embed', 'audio_embed']:
        if name1 not in embed_arr:
            continue
        embed1 = embed_arr[name1]
        for name2, embed2 in embed_arr.items():
            name1 = name1.replace('_embed', '').replace('text', 't').replace('audio', 'a').replace('video', 'v')
            name2 = name2.replace('_embed', '').replace('text', 't').replace('audio', 'a').replace('video', 'v')
            if name1 == name2 or f'{name2}2{name1}' in sims:
                continue
            sims[f'{name1}2{name2}'] = sim_matrix(embed1, embed2, flag_normalize_embeddings=flag_normalize_embeddings).detach().cpu().numpy()
    for lang in langs:
        video_embed = embed_arr['video_embed']
        sims['t2v_{}'.format(lang)] = sim_matrix(embed_arr[lang], video_embed, flag_normalize_embeddings=flag_normalize_embeddings).detach().cpu().numpy()
        
    for metric in metrics:
        metric_name = metric.__name__
        if hasattr(dl.dataset, 'complete_dataset_size'):
            complete_dataset_size = dl.dataset.complete_dataset_size
        else:
            complete_dataset_size = None

        res = metric(sims, complete_dataset_size=complete_dataset_size)
        nested_metrics[metric_name] = res
    return nested_metrics, val_loss, val_loss_detailed


def short_verbose(epoch, dl_nested_metrics, dataset_name):
    for metric_set_name in ['t2v_metrics', 't2v+a_metrics', 't2va_metrics']:
        if metric_set_name in dl_nested_metrics:
            metrics = dl_nested_metrics[metric_set_name]
            if all((metric_name in metrics) for metric_name in ['R1', 'R5', 'R10', 'R50', 'MedR', 'MeanR']):
                r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]

                msg = f"[{metric_set_name}]{dataset_name:s} epoch {epoch}         {r1:.1f} {r5:.1f} {r10:.1f} {metrics['MedR']:g}"
                msg += f"           {r50:.1f} {metrics['MedR']:g} {metrics['MeanR']:.1f}"
                print(msg)


def verbose(epoch, metrics, mode, name="TEST"):
    if all((metric_name in metrics) for metric_name in ['R1', 'R5', 'R10', 'R50', 'MedR', 'MeanR']):
        r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
        msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
        msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
        msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
        print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res


def format_dataloader_output(data):
    if isinstance(data, list):
        # TODO: rewrite.. necessary for TAR dataloader
        dataset_name = data[0][0][0]
        if dataset_name in ['HowTo100M', 'YouCook2', 'YouCook2_Caption', 'WebVid2M', 'MSRVTT', 'MSRVTT_Caption']:
            new_data = {
                "meta": {
                    'dataset': data[0],
                    'paths': data[1],
                    "ids": data[2]
                },
                'video': data[3],
                "text": data[4],
                'unroll_fake_1st_dim': data[5]
            }
            if len(data) >= 7:
                new_data['audio'] = data[6]
            if len(data) >= 8:
                new_data['audio_mask'] = data[7]
            if len(data) >= 9:
                new_data['nframes'] = data[8]
            data = new_data

            if dataset_name in ['YouCook2_Caption', 'MSRVTT_Caption']:
                data['caption'] = data.pop('text')
        elif dataset_name in ['CC']:
            data = {
                "meta": {
                    'dataset': data[0],
                    'paths': data[1],
                    'ids': data[2],
                },
                'image': data[3],
                "caption": data[4],
            }
        else:
            raise NotImplementedError
    if data.get('unroll_fake_1st_dim', [False])[0]:
        # TODO: rewrite.. necessary for feature dataloader and TAR dataloader
        for field in ['text', 'caption', 'raw_text']:
            if field in data:
                if torch.is_tensor(data[field]):
                    data[field] = data[field].view(-1, *data[field].shape[2:])
                else:
                    data[field] = list(itertools.chain.from_iterable(zip(*data[field]))) # TODO!  THIS is very bad code

        data['meta'] = {
            'dataset': list(itertools.chain.from_iterable(zip(*data['meta']['dataset']))), # TODO!  THIS is very bad code
            'paths': list(itertools.chain.from_iterable(zip(*data['meta']['paths']))),
            'ids': list(itertools.chain.from_iterable(zip(*data['meta']['ids'])))
        }

        for field in ['video', 'video_mask', 'audio', 'audio_mask', 'text_mask', 'image', 'nframes', 'y_true']:
            if field in data:
                data[field] = data[field].view(-1, *data[field].shape[2:])
    return data
