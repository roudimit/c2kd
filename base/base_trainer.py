import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import os

from model.multimodal_loss import MultilingualTVLoss

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config, writer=None, init_val=False):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.init_val = init_val
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        self.model.device = self.device


        loss = loss.to(self.device)
        self.loss = loss
        print('Using temperature of {} for contrastive losses'.format(self.loss.temperature))
        self.loss_tv = MultilingualTVLoss(temperature=self.loss.temperature)
        self.metrics = metrics
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.resume_only_model = cfg_trainer.get('resume_only_model', False)
        self.resume_opt = cfg_trainer.get('resume_opt', False)
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.init_val = cfg_trainer.get('init_val', True)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.step = 0

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        #self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer = writer

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
            # self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        langs =  ['de', 'fr', 'cs', 'zh', 'ru', 'vi', 'sw', 'es', 'en', 'ja']
        langs +=  ['hi', 'mr', 'ta', 'te', 'kn', 'ml'] # new RUDDER langs
        if self.init_val:
            _ = self._valid_epoch(-1)

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'nested_val_metrics':
                    # NOTE: currently only supports two layers of nesting
                    for subkey, subval in value.items():
                        for subsubkey, subsubval in subval.items():
                            for subsubsubkey, subsubsubval in subsubval.items():
                                log[f"val_{subkey}_{subsubkey}_{subsubsubkey}"] = subsubsubval
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            
            # print average t2v stats for different langs
            lang_stats_t2v = {'R1': 0, 'R5': 0, 'R10': 0, 'MedR': 0, 'MeanR': 0, 'geometric_mean_R1-R5-R10': 0}
            num_langs = 0
            for lang in langs:
                if 't2v_{}_metrics'.format(lang) in result['nested_val_metrics'][0]:
                    num_langs += 1
                    for stat in result['nested_val_metrics'][0]['t2v_{}_metrics'.format(lang)]:
                        lang_stats_t2v[stat] += result['nested_val_metrics'][0]['t2v_{}_metrics'.format(lang)][stat]
            for stat in lang_stats_t2v:
                lang_stats_t2v[stat] /= num_langs
                print('avg t2v {} for {} langs : {}'.format(stat, num_langs, lang_stats_t2v[stat]))
            lang_stats_t2v_a = {'R1': 0, 'R5': 0, 'R10': 0, 'MedR': 0, 'MeanR': 0, 'geometric_mean_R1-R5-R10': 0}
            num_langs = 0
            for lang in langs:
                if 't2v+a_{}_metrics'.format(lang) in result['nested_val_metrics'][0]:
                    num_langs += 1
                    for stat in result['nested_val_metrics'][0]['t2v+a_{}_metrics'.format(lang)]:
                        lang_stats_t2v_a[stat] += result['nested_val_metrics'][0]['t2v+a_{}_metrics'.format(lang)][stat]
            if num_langs != 0: # not always using audio
                for stat in lang_stats_t2v_a:
                    lang_stats_t2v_a[stat] /= num_langs
                    print('avg t2v+a {} for {} langs : {}'.format(stat, num_langs, lang_stats_t2v_a[stat]))
            lang_stats_a2v = {'R1': 0, 'R5': 0, 'R10': 0, 'MedR': 0, 'MeanR': 0, 'geometric_mean_R1-R5-R10': 0}
            num_langs = 0
            for lang in langs:
                if 'a2v_{}_metrics'.format(lang) in result['nested_val_metrics'][0]:
                    num_langs += 1
                    for stat in result['nested_val_metrics'][0]['a2v_{}_metrics'.format(lang)]:
                        lang_stats_a2v[stat] += result['nested_val_metrics'][0]['a2v_{}_metrics'.format(lang)][stat]
            if num_langs != 0: # not always using audio
                for stat in lang_stats_a2v:
                    lang_stats_a2v[stat] /= num_langs
                    print('avg a2v {} for {} langs : {}'.format(stat, num_langs, lang_stats_a2v[stat]))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if best:
                self._save_checkpoint(epoch, save_best=best)
            if epoch % self.save_period == 0 :
                self._save_checkpoint(epoch)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, save_latest=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'step': self.step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_latest:
            # for safety
            tmp_best_path = str(self.checkpoint_dir / 'tmp.pth')
            torch.save(state, tmp_best_path)
            best_path = str(self.checkpoint_dir / 'latest_model.pth')
            os.rename(tmp_best_path, best_path)

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

        if not(save_best or save_latest):
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')

        if not self.resume_only_model:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        state_dict = checkpoint['state_dict']

        load_state_dict_keys = list(state_dict.keys())
        curr_state_dict_keys = list(self.model.state_dict().keys())
        redo_dp = False
        if not curr_state_dict_keys[0].startswith('module.') and load_state_dict_keys[0].startswith('module.'):
            undo_dp = True
        elif curr_state_dict_keys[0].startswith('module.') and not load_state_dict_keys[0].startswith('module.'):
            redo_dp = True
            undo_dp = False
        else:
            undo_dp = False

        if undo_dp:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
        elif redo_dp:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict

        self.model.load_state_dict(new_state_dict, strict=False) # don't load the audio weights

        if not self.resume_only_model or self.resume_opt:
            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        del checkpoint, new_state_dict
        torch.cuda.empty_cache()