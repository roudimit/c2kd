import argparse
import collections
import data_loader.data_loader as module_data
import model.multimodal_loss as module_loss
from model.metric import RetrievalMetric

import model.model as module_arch
import utils.visualizer as module_vis
from utils.util import replace_nested_dict_item
from parse_config import ConfigParser
from trainer import Trainer
from sacred import Experiment
import transformers
import os
import torch.optim as optim


ex = Experiment('train')


@ex.main
def run():
    logger = config.get_logger('train')
    logger.info(f"Config: {config['name']}")

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # TODO: improve Create identity (do nothing) visualiser?
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # build tokenizer
    # tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
    #                                                        TOKENIZERS_PARALLELISM=False)
    # TODO: remove, necessary for feature dataset
    if config['arch']['args'].get('text_params', {}).get('model', None) is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)
    else:
        tokenizer = None

    # setup data_loader instances
    data_loader = init_dataloader(config, module_data, data_loader='data_loader')
    valid_data_loader = init_dataloader(config, module_data, data_loader='val_data_loader')
    print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
    print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')


    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metric_break_ties = config.config.get('metric_break_ties', 'averaging')
    metrics = [RetrievalMetric(met, break_ties=metric_break_ties) for met in config['metrics']]
            

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer = config.initialize('optimizer', transformers, trainable_params)
    # TODO: remove, necessary for feature dataset
    if config['optimizer']['type'] == 'Adam':
        if config.args.teacher_sim_pool in ['learnable_multitask', 'learnable_poolers']:
            import torch
            trainable_params = [i for i in trainable_params] # don't include learnable weights in trainable_params
            model.learnable_weights = torch.nn.parameter.Parameter(torch.tensor([0.33, 0.33, 0.33]))
            optimizer = optim.Adam([{'params': trainable_params},
                                    {'params': model.learnable_weights, 'lr':1e-2}], # use a larger learning rate
                                    **config['optimizer']['args'])
        else:
            optimizer = optim.Adam(trainable_params, **config['optimizer']['args']) # NOTE: original learning rate
    else:
        optimizer = config.initialize('optimizer', transformers, trainable_params)

    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        if hasattr(transformers, config.config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        elif config['lr_scheduler']['type'] == 'ExponentialLR':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **config['lr_scheduler']['args'])
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    else:
        writer = None

    if config.args.teachers == '':
        # teachers = ['arch_teacher_xlm', 'arch_teacher_mbert', 'arch_teacher_distill', 'arch_teacher_labse', 'arch_teacher_sim_cse']
        teachers = ['arch_teacher_xlm', 'arch_teacher_mbert', 'arch_teacher_distill'] # NOTE: main 3 teachers
    elif config.args.teachers == 'None':
        teachers = []
    else:
        teachers = [i for i in config.args.teachers.split(',')]
    teacher_models = {}
    for name in teachers:
        teacher_models[name] = config.initialize(name, module_arch)
    

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      teacher_models=teacher_models)
    trainer.train()


def init_dataloader(config, module_data, data_loader):
    if "type" in config[data_loader] and "args" in config[data_loader]:
        return [config.initialize(data_loader, module_data)]
    elif isinstance(config[data_loader], list):
        return [config.initialize(data_loader, module_data, index=idx) for idx in
                range(len(config[data_loader]))]
    else:
        raise ValueError("Check data_loader config, not correct format.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--neptune', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('--balance', default=0.1, type=float,
                      help='Weight of contrastive loss (lambda) and distill loss (1-lambda)')
    args.add_argument('--distill_temp', default=0.1, type=float,
                      help='Temperature for distillation loss')
    args.add_argument('--teacher_sim_pool', default='min', type=str,
                      help='Pooling type for teacher sims')
    args.add_argument('--teachtext', default=0, type=int,
                      help='Use teachtext or not for ablation study')
    args.add_argument('--teachers', default='None', type=str,
                      help='Comma separated list of teachers to use')
    args.add_argument('--stage', '--stage', default='c2kd', type=str,
                      help='Which stage of training (baseline-zero-shot, baseline-translate-train, c2kd)')
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--sp', '--save_period'], type=int, target=('trainer', 'save_period')),
        # CustomArgs(['--ev', '--use_eval_mode_always'], action='store_true', target=('trainer', 'use_eval_mode_always'))
    ]
    config = ConfigParser(args, options)
    ex.add_config(config.config)

    if config['trainer']['neptune']:
        # from neptunecontrib.monitoring.sacred import NeptuneObserver
        import neptune
        from neptune.new.integrations.sacred import NeptuneObserver
        # New API (Note the different import for NeptuneObserver)
        neptune_run = neptune.init_run(project='', 
                api_token='')

        ex.observers.append(NeptuneObserver(run=neptune_run))
        ex.run()

    else:
        run()
