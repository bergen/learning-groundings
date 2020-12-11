#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trainval.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/05/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Training and evaulating the Neuro-Symbolic Concept Learner.
"""

import time
import os.path as osp
import csv

import torch
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
import TrainerEnv
from jactorch.utils.meta import as_float

from nscl.datasets import get_available_datasets, initialize_dataset, get_dataset_builder

logger = get_logger(__file__)

parser = JacArgumentParser(description=__doc__.strip())

parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')

# training_target and curriculum learning
parser.add_argument('--expr', default=None, metavar='DIR', help='experiment name')
parser.add_argument('--training-target', required=True, choices=['derender', 'parser', 'all'])
parser.add_argument('--training-visual-modules', default='all', choices=['none', 'object', 'relation', 'all'])
parser.add_argument('--curriculum', default='all', choices=['off', 'scene', 'program', 'all','restricted','accelerated','intermediate','simple_syntax','extended','restrict_syntax','no_complex_syntax','all_syntax','all_syntax_fast','all_syntax_objects','all_syntax_accelerated','nonrelation_first','nonrelation_first_v2','nonrelation_first_v3','nonrelation_first_v4','nonrelation_first_v5','nonrelation_first_v6'])
parser.add_argument('--question-transform', default='off', choices=['off', 'basic', 'parserv1-groundtruth', 'parserv1-candidates', 'parserv1-candidates-executed'])
parser.add_argument('--concept-quantization-json', default=None, metavar='FILE')

# running mode
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--evaluate', action='store_true', help='run the validation only; used with --resume')

# training hyperparameters
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of total epochs to run')
parser.add_argument('--enums-per-epoch', type=int, default=1, metavar='N', help='number of enumerations of the whole dataset per epoch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='N', help='weight decay')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset (default: 0)')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient (default: 1)')
parser.add_argument('--clip-grad', type=float, metavar='F', help='gradient clipping')
parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs) (default: 1)')

# finetuning and snapshot
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
parser.add_argument('--save-interval', type=int, default=2, metavar='N', help='model save interval (epochs) (default: 10)')

# data related
parser.add_argument('--dataset', required=True, choices=get_available_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data-trim', type=float, default=0, metavar='F', help='trim the dataset')
parser.add_argument('--data-split',type=float, default=0.75, metavar='F', help='fraction / numer of training samples')
parser.add_argument('--data-vocab-json', type='checked_file', metavar='FILE')
parser.add_argument('--data-scenes-json', type='checked_file', metavar='FILE')
parser.add_argument('--data-questions-json', type='checked_file', metavar='FILE', nargs='+')

parser.add_argument('--extra-data-dir', type='checked_dir', metavar='DIR', help='extra data directory for validation')
parser.add_argument('--extra-data-scenes-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra scene json file for validation')
parser.add_argument('--extra-data-questions-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra question json file for validation')

parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=False, metavar='B', help='use tensorboard or not')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')


#scene graph
parser.add_argument('--attention-type', default='cnn', choices=['cnn', 'naive-rnn', 'naive-rnn-batched',
                                                                'naive-rnn-global-batched','structured-rnn-batched',
                                                                'max-rnn-batched','low-dim-rnn-batched','monet',
                                                                'scene-graph-object-supervised',
                                                                'structured-subtractive-rnn-batched',
                                                                'transformer',
                                                                'monet-lite',
                                                                'transformer-cnn',
                                                                'transformer-cnn-object-inference',
                                                                'transformer-cnn-object-inference-ablate-scope',
                                                                'transformer-cnn-object-inference-ablate-initialization',
                                                                'transformer-cnn-object-inference-sequential',
                                                                'transformer-cnn-object-inference-recurrent'])

parser.add_argument('--attention-loss', type='bool', default=False)
parser.add_argument('--anneal-rnn', type='bool', default=False)
parser.add_argument('--adversarial-loss', type='bool', default=False)
parser.add_argument('--adversarial-lr', type=float, default=0.0002, metavar='N', help='initial learning rate')
parser.add_argument('--presupposition-semantics', type='bool', default=False)
parser.add_argument('--mutual-exclusive', type='bool', default=True)
parser.add_argument('--subtractive-rnn', type='bool', default=False)
parser.add_argument('--subtract-from-scene', type='bool', default=True)
parser.add_argument('--rnn-type', default='lstm', choices=['lstm','gru'])
parser.add_argument('--full-recurrence', type='bool', default=True)
parser.add_argument('--lr-cliff-epoch', type=int, default=200) #this is the epoch at which the lr will fall by factor of 0.1
parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'rmsprop','adabelief'])
parser.add_argument('--fine-tune-resnet-epoch', type=int, default=100)
parser.add_argument('--fine-tune-semantics-epoch', type=int, default=100)
parser.add_argument('--restrict-finetuning', type='bool', default=True)
parser.add_argument('--resnet-type', default='resnet34', choices=['resnet34', 'resnet101','cmc_resnet','simclr_resnet','resnet34_pytorch'])
parser.add_argument('--transformer-use-queries', type='bool', default=False)
parser.add_argument('--filter-ops', type='bool', default=False)
parser.add_argument('--filter-relate', type='bool', default=False)
parser.add_argument('--filter-disjunction', type='bool', default=False)
parser.add_argument('--filter-relate-epoch', type=int, default=0)
parser.add_argument('--object-dropout', type='bool', default=False)
parser.add_argument('--object-dropout-rate', type=float, default=0.03)
parser.add_argument('--normalize-objects',type='bool',default=True)
parser.add_argument('--filter-additive',type='bool',default=False)
parser.add_argument('--relate-rescale',type='bool',default=False)
parser.add_argument('--relate-max',type='bool',default=False)
parser.add_argument('--logit-semantics',type='bool',default=False)
parser.add_argument('--bilinear-relation',type='bool',default=False)
parser.add_argument('--coord-semantics',type='bool',default=False)
parser.add_argument('--infer-num-objects',type='bool',default=False)
parser.add_argument('--pretrained-resnet',type='bool',default=True)
parser.add_argument('--loss-curriculum',type='bool',default=False)
parser.add_argument('--initialization-scope',type='bool',default=False)
parser.add_argument('--threshold-normalize', type=float, default=1)
parser.add_argument('--num-resnet-layers', type=int, default=3)

args = parser.parse_args()

if args.data_vocab_json is None:
    args.data_vocab_json = osp.join(args.data_dir, 'vocab.json')

args.data_image_root = osp.join(args.data_dir, 'images')
if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')

if args.extra_data_dir is not None:
    args.extra_data_image_root = osp.join(args.extra_data_dir, 'images')
    if args.extra_data_scenes_json is None:
        args.extra_data_scenes_json = osp.join(args.extra_data_dir, 'scenes.json')
    if args.extra_data_questions_json is None:
        args.extra_data_questions_json = osp.join(args.extra_data_dir, 'questions.json')

# filenames
args.series_name = args.dataset
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

# directories

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)


if "deobjectified_train" in args.data_image_root:
    dataset_type = "deobjectified"
elif "deobjectified_simple_questions" in args.data_image_root:
    dataset_type = "deobjectified_simple"
elif "supercube_train" in args.data_image_root:
    dataset_type = "supercube" 
elif "supercube_simple_questions" in args.data_image_root:
    dataset_type = "supercube_simple" 
elif "supercube_morequestions_simple" in args.data_image_root:
    dataset_type = "supercube_morequestions_simple"
elif "conditional_simple_questions" in args.data_image_root:
    dataset_type = "conditional_simple_questions"
elif "simple" in args.data_image_root:
    dataset_type = "simple"
else:
    dataset_type = "full"

def initialize_adversary():
    from nscl.nn.adversarial import Adversary
    adversary = Adversary(configs.model.sg_dims)
    return adversary

def main():
    args.dump_dir = ensure_path(osp.join(
        'dumps', args.series_name, args.desc_name, (
            ('curriculum_' + args.curriculum) +
            ('-dataset_' + dataset_type) +
            ('-' + args.expr if args.expr is not None else '') +
            ('-lr_' + str(args.lr)) + 
            ('-batch_' + str(args.batch_size*args.acc_grad)) + 
            ('-attention_' + str(args.attention_type)) +
            ('-resnet_type_' + str(args.resnet_type)) +
            ('-clip_grad_' + str(args.clip_grad))+
            ('-optimizer_'+str(args.optimizer))+
            ('-weight_decay_'+str(args.weight_decay))+
            ('-num_resnet_layers_'+str(args.num_resnet_layers))
        )
    ))

    if not args.debug:
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
        args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')
        args.log_file = osp.join(args.meta_dir, args.run_name + '.log')
        args.meter_file = osp.join(args.meta_dir, args.run_name + '.meter.json')

        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir_root = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
            args.tb_dir = ensure_path(osp.join(args.tb_dir_root, args.run_name))

    initialize_dataset(args.dataset)
    build_dataset = get_dataset_builder(args.dataset)

    dataset = build_dataset(args, configs, args.data_image_root, args.data_scenes_json, args.data_questions_json)

    dataset_trim = int(len(dataset) * args.data_trim) if args.data_trim <= 1 else int(args.data_trim)
    if dataset_trim > 0:
        dataset = dataset.trim_length(dataset_trim)

    dataset_split = int(len(dataset) * args.data_split) if args.data_split <= 1 else int(args.data_split)
    train_dataset, validation_dataset = dataset.split_trainval(dataset_split)

    extra_dataset = None
    if args.extra_data_dir is not None:
        extra_dataset = build_dataset(args, configs, args.extra_data_image_root, args.extra_data_scenes_json, args.extra_data_questions_json)


    main_train(train_dataset, validation_dataset, extra_dataset)






def main_train(train_dataset, validation_dataset, extra_dataset=None):
    logger.critical('Building the model.')
    model = desc.make_model(args, train_dataset.unwrapped.vocab)
    if args.adversarial_loss:
        adversary = initialize_adversary()

    if args.use_gpu:
        model.cuda()
        if args.adversarial_loss:
            adversary.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # from jactorch.parallel import UserScatteredJacDataParallel as JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        from torch.optim import RMSprop
        from adabelief_pytorch import AdaBelief
        if args.optimizer =='adamw':
            optimizer_fn = AdamW
        elif args.optimizer=='rmsprop':
            optimizer_fn = RMSprop
        elif args.optimizer=='adabelief':
            optimizer_fn = AdaBelief

        trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        if args.optimizer!='adabelief':
            optimizer = optimizer_fn(trainable_parameters, args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = AdaBelief(trainable_parameters, lr=args.lr, eps=1e-12, betas=(0.9,0.999),weight_decay=args.weight_decay)

        if args.adversarial_loss:
            from nscl.nn.reasoning_v1.losses import AdversarialLoss
            adversarial_loss = AdversarialLoss()
            adversarial_parameters = filter(lambda x: x.requires_grad, adversary.parameters())
            adversarial_optimizer = AdamW(adversarial_parameters, args.adversarial_lr)
            adversarial_trainer = {'adversary':adversary,'adversarial_optimizer':adversarial_optimizer,'adversarial_loss':adversarial_loss}


    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))

    trainer = TrainerEnv.TrainerEnv(model, optimizer)

    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            pass
            #args.start_epoch = extra['epoch']
            #logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))



    if args.use_tb and not args.debug:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

    if args.clip_grad:
        logger.info('Registering the clip_grad hook: {}.'.format(args.clip_grad))
        def clip_grad(self, feed_dict, loss, monitors, output_dict):
            from torch.nn.utils import clip_grad_norm_
            clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad)
        trainer.register_event('backward:after', clip_grad)

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    if args.embed:
        from IPython import embed; embed()



    # assert args.curriculum == 'off', 'Unimplemented feature: curriculum mode {}.'.format(args.curriculum)
    remove_ops = None
    if args.filter_ops:
        remove_ops = ['relate_attribute_equal','query_attribute_equal']
        if args.filter_relate:
            remove_ops = remove_ops + ['relate']
        if args.filter_disjunction:
            remove_ops = remove_ops + ['union']
            

    if args.curriculum=='restricted':
        curriculum_strategy = [
            (0, 3, 4),
            (10, 3, 6),
            (20, 3, 8),
            (30, 4, 8),
            (30, 4, 12),
            (1e9, None, None)
        ]
        validation_restriction = (4,12)
    elif args.curriculum=='accelerated':
        curriculum_strategy = [
            (0, 3, 4),
            (5, 3, 8),
            (10, 4, 8),
            (15, 5, 12),
            (20, 6, 12),
            (25, 7, 16),
            (30, 8, 20),
            (35, 9, 22),
            (40, 10, 25),
            (1e9, None, None)
        ]
    elif args.curriculum=='restrict_syntax':
        curriculum_strategy = [
            (0, 10, 8),
            (1e9, None, None)
        ]
    elif args.curriculum=='no_complex_syntax':
        curriculum_strategy = [
            (0, 10, 12),
            (1e9, None, None)
        ]
    elif args.curriculum=='all_syntax_objects':
        curriculum_strategy = [
            (0, 10, 25),
            (1e9, None, None)
        ]
    elif args.curriculum=='intermediate':
        curriculum_strategy = [
            (0, 3, 4),
            (10, 3, 6),
            (20, 4, 8),
            (30, 5, 12),
            (40, 6, 12),
            (50, 7, 16),
            (60, 8, 20),
            (70, 9, 22),
            (80, 10, 25),
            (1e9, None, None)
        ]
    elif args.curriculum=='all_syntax':
        curriculum_strategy = [
            (0, 3, 20),
            (10, 4, 20),
            (20, 5, 20),
            (30, 6, 20),
            (40, 7, 20),
            (50, 8, 20),
            (60, 9, 20),
            (70, 10, 20),
            (1e9, None, None)
        ]
    elif args.curriculum=='all_syntax_fast':
        curriculum_strategy = [
            (0, 3, 20),
            (2, 4, 20),
            (4, 5, 20),
            (6, 6, 20),
            (8, 7, 20),
            (10, 8, 20),
            (12, 9, 20),
            (14, 10, 20),
            (1e9, None, None)
        ]
    elif args.curriculum=='all_syntax_accelerated':
        curriculum_strategy = [
            (0, 3, 20),
            (5, 4, 20),
            (10, 5, 20),
            (15, 6, 20),
            (20, 7, 20),
            (25, 8, 20),
            (30, 9, 20),
            (35, 10, 20),
            (1e9, None, None)
        ]
    elif args.curriculum=='extended':
        curriculum_strategy = [
            (0, 3, 4),
            (10, 3, 6),
            (20, 3, 8),
            (30, 4, 8),
            (40, 5, 12),
            (50, 6, 12),
            (60, 7, 16),
            (70, 8, 20),
            (80, 9, 22),
            (90, 10, 25),
            (1e9, None, None)
        ]
    elif args.curriculum=='nonrelation_first':
        remove_ops_nonrelation = ['relate_attribute_equal','query_attribute_equal','relate','union']
        remove_ops_relation = ['relate_attribute_equal','query_attribute_equal','union']
        curriculum_strategy = [
            (0, 3, 4,remove_ops_nonrelation,0.00004,0),
            (10, 3, 6,remove_ops_nonrelation),
            (20, 3, 8,remove_ops_nonrelation),
            (30, 4, 8,remove_ops_nonrelation),
            (40, 5, 12,remove_ops_nonrelation),
            (50, 6, 12,remove_ops_nonrelation),
            (60, 7, 16,remove_ops_nonrelation),
            (70, 8, 20,remove_ops_nonrelation),
            (80, 9, 22,remove_ops_nonrelation),
            (90, 10, 25,remove_ops_nonrelation),
            (100, 3, 10,remove_ops_relation,0.00002,0.03),
            (110, 4, 10,remove_ops_relation),
            (120, 5, 15,remove_ops_relation),
            (130, 6, 20,remove_ops_relation),
            (140, 7, 20,remove_ops_relation),
            (150, 8, 20,remove_ops_relation),
            (160, 9, 20,remove_ops_relation),
            (170, 10, 20,remove_ops_relation),
            (1e9, None, None)
        ]
    elif args.curriculum=='nonrelation_first_v2':
        remove_ops_nonrelation = ['relate_attribute_equal','query_attribute_equal','relate','union']
        remove_ops_relation = ['relate_attribute_equal','query_attribute_equal','union']
        curriculum_strategy = [
            (0, 3, 4,remove_ops_nonrelation, args.lr, 0),
            (10, 3, 6,remove_ops_nonrelation),
            (20, 3, 8,remove_ops_nonrelation),
            (30, 4, 8,remove_ops_nonrelation),
            (40, 5, 12,remove_ops_nonrelation),
            (50, 6, 12,remove_ops_nonrelation),
            (60, 7, 16,remove_ops_nonrelation,args.lr/10,0),
            (70, 8, 20,remove_ops_nonrelation),
            (80, 9, 22,remove_ops_nonrelation),
            (90, 10, 25,remove_ops_nonrelation),
            (110, 10, 12,remove_ops_relation,0.00001,0),
            (115, 3, 20,remove_ops_relation,0.00001,0),
            (125, 4, 20,remove_ops_relation),
            (135, 5, 20,remove_ops_relation,0.000001,0.03),
            (145, 6, 20,remove_ops_relation),
            (155, 7, 20,remove_ops_relation),
            (165, 8, 20,remove_ops_relation),
            (175, 9, 20,remove_ops_relation),
            (185, 10, 20,remove_ops_relation),
            (1e9, None, None)
        ]
    elif args.curriculum=='nonrelation_first_v3':
        remove_ops_nonrelation = ['relate_attribute_equal','query_attribute_equal','relate','union']
        remove_ops_relation = ['relate_attribute_equal','query_attribute_equal','union']
        curriculum_strategy = [
            (0, 3, 4,remove_ops_nonrelation, args.lr, 0),
            (10, 3, 6,remove_ops_nonrelation),
            (20, 3, 8,remove_ops_nonrelation),
            (30, 4, 8,remove_ops_nonrelation),
            (40, 5, 12,remove_ops_nonrelation),
            (50, 6, 12,remove_ops_nonrelation),
            (60, 7, 16,remove_ops_nonrelation,args.lr/10,0),
            (70, 8, 20,remove_ops_nonrelation),
            (80, 9, 22,remove_ops_nonrelation),
            (90, 10, 25,remove_ops_nonrelation),
            (110, 10, 12,remove_ops_relation,0.00001,0.03),
            (120, 10, 20,remove_ops_relation),
            (130, 10, 21,remove_ops_relation),
            (140, 10, 22,remove_ops_relation),
            (150, 10, 24,remove_ops_relation),
            (160, 10, 25,remove_ops_relation),
            (1e9, None, None)
        ]
    elif args.curriculum=='nonrelation_first_v4':
        remove_ops_nonrelation = ['relate_attribute_equal','query_attribute_equal','relate','union']
        remove_ops_relation = ['relate_attribute_equal','query_attribute_equal','union']
        curriculum_strategy = [
            (0, 3, 4,remove_ops_nonrelation, args.lr, 0),
            (10, 3, 6,remove_ops_nonrelation),
            (20, 3, 8,remove_ops_nonrelation),
            (30, 4, 8,remove_ops_nonrelation),
            (40, 5, 12,remove_ops_nonrelation),
            (50, 6, 12,remove_ops_nonrelation),
            (60, 7, 16,remove_ops_nonrelation,args.lr/10,0),
            (70, 8, 20,remove_ops_nonrelation),
            (80, 9, 22,remove_ops_nonrelation),
            (90, 10, 25,remove_ops_nonrelation),
            (110, 10, 12,remove_ops_relation,0.00001,0),
            (120, 10, 20,remove_ops_relation),
            (130, 10, 21,remove_ops_relation),
            (140, 10, 22,remove_ops_relation),
            (150, 10, 24,remove_ops_relation),
            (160, 10, 25,remove_ops_relation),
            (1e9, None, None)
        ]
    elif args.curriculum=='nonrelation_first_v5':
        remove_ops_nonrelation = ['relate_attribute_equal','query_attribute_equal','relate','union']
        remove_ops_relation = ['relate_attribute_equal','query_attribute_equal','union']
        curriculum_strategy = [
            (0, 3, 4,remove_ops_nonrelation, args.lr, 0),
            (10, 3, 6,remove_ops_nonrelation),
            (20, 3, 8,remove_ops_nonrelation),
            (30, 4, 8,remove_ops_nonrelation),
            (40, 5, 12,remove_ops_nonrelation),
            (50, 6, 12,remove_ops_nonrelation),
            (60, 7, 16,remove_ops_nonrelation),
            (70, 8, 20,remove_ops_nonrelation),
            (80, 9, 22,remove_ops_nonrelation),
            (90, 10, 25,remove_ops_nonrelation),
            (110, 10, 12,remove_ops_relation,args.lr/4,args.object_dropout_rate),
            (120, 3, 20,remove_ops_relation),
            (130, 4, 20,remove_ops_relation),
            (140, 5, 20,remove_ops_relation),
            (150, 6, 20,remove_ops_relation),
            (160, 7, 20,remove_ops_relation),
            (170, 8, 20,remove_ops_relation),
            (180, 9, 20,remove_ops_relation),
            (190, 10, 20,remove_ops_relation),
            (200, 10, 21,remove_ops_relation),
            (210, 10, 22,remove_ops_relation),
            (220, 10, 25,remove_ops_relation),
            (230, 10, 25,remove_ops_relation),
            (220, 10, 25,remove_ops_relation),
            (240, 10, 25,remove_ops_relation),
            (250, 10, 25,remove_ops_relation),
            (260, 10, 25,remove_ops_relation),
            (1e9, None, None)
        ]
    elif args.curriculum=='nonrelation_first_v6':
        remove_ops_nonrelation = ['relate_attribute_equal','query_attribute_equal','relate','union']
        remove_ops_includerelation_union = ['relate_attribute_equal','query_attribute_equal']
        curriculum_strategy = [
            (0, 3, 4,remove_ops_nonrelation, args.lr, 0),
            (10, 3, 6,remove_ops_nonrelation),
            (20, 3, 8,remove_ops_nonrelation),
            (30, 4, 8,remove_ops_nonrelation),
            (40, 5, 12,remove_ops_nonrelation),
            (50, 6, 12,remove_ops_includerelation_union),
            (60, 6, 16,remove_ops_includerelation_union),
            (70, 6, 20,remove_ops_includerelation_union),
            (80, 7, 20,remove_ops_includerelation_union),
            (90, 8, 25,remove_ops_includerelation_union),
            (110, 9, 25,remove_ops_includerelation_union),
            (210, 10, 22,remove_ops_includerelation_union),
            (220, 10, 25,remove_ops_includerelation_union),
            (230, 10, 25,remove_ops_includerelation_union),
            (220, 10, 25,remove_ops_includerelation_union),
            (240, 10, 25,remove_ops_includerelation_union),
            (250, 10, 25,remove_ops_includerelation_union),
            (260, 10, 25,remove_ops_includerelation_union),
            (1e9, None, None)
        ]
    elif args.curriculum=='simple_syntax':
        curriculum_strategy = [
            (0, 3, 4),
            (10, 3, 6),
            (25, 4, 6),
            (35, 5, 6),
            (45, 6, 6),
            (55, 7, 6),
            (65, 8, 6),
            (75, 9, 6),
            (85, 10, 6),
            (95, 10, 25),
            (1e9, None, None)
        ]
    else:
        curriculum_strategy = [
            (0, 3, 4),
            (5, 3, 6),
            (10, 3, 8),
            (15, 4, 8),
            (25, 4, 12),
            (35, 5, 12),
            (45, 6, 12),
            (55, 7, 16),
            (65, 8, 20),
            (75, 9, 22),
            (90, 10, 25),
            (1e9, None, None)
        ]

    trainer.register_event('backward:after', backward_check_nan)

    if args.curriculum == 'restricted':
            max_validation_scene_size, max_validation_program_size = validation_restriction
            validation_dataset = validation_dataset.filter_scene_size(max_validation_scene_size)
            validation_dataset = validation_dataset.filter_program_size_raw(max_validation_program_size)


    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)
    if extra_dataset is not None:
        extra_dataloader = extra_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    if args.evaluate:
        meters.reset()
        model.eval()
        validate_epoch(0, trainer, validation_dataloader, meters)
        if extra_dataset is not None:
            validate_epoch(0, trainer, extra_dataloader, meters, meter_prefix='validation_extra')
        logger.critical(meters.format_simple('Validation', {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))
        return meters

    gradient_magnitudes = {}

    if args.loss_curriculum:
        curriculum_index = 0
        curriculum_updated = True


    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        meters.reset()

        model.train()
        if args.adversarial_loss:
            adversary.train()

        this_train_dataset = train_dataset
        if args.curriculum != 'off':
            if args.loss_curriculum:
                try:
                    strategy = curriculum_strategy[curriculum_index]
                    loss_threshold = 0.2*((2/3)**curriculum_index) + 0.03
                    if args.object_dropout_rate<0.01:
                        loss_threshold = loss_threshold - 0.01
                except Exception as e:
                    curriculum_index = curriculum_index - 1
                    strategy = curriculum_strategy[curriculum_index]
                #max_scene_size, max_program_size = strategy[1],strategy[2]

            else:
                for si, s in enumerate(curriculum_strategy):
                    if curriculum_strategy[si][0] < epoch <= curriculum_strategy[si + 1][0]:
                        strategy = curriculum_strategy[si]


            max_scene_size, max_program_size = strategy[1],strategy[2]
            if 'nonrelation_first' in args.curriculum:
                remove_ops = strategy[3]

                reinit_optimizer = (epoch==strategy[0]+1 and not args.loss_curriculum) or (args.loss_curriculum and curriculum_updated)
                if len(strategy)>=5 and reinit_optimizer:
                    trainer.set_learning_rate(strategy[4])
                    #trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
                    #optimizer = optimizer_fn(trainable_parameters, strategy[4], weight_decay=configs.train.weight_decay)
                    #if args.acc_grad > 1:
                    #    optimizer = AccumGrad(optimizer, args.acc_grad)
                    #trainer = TrainerEnv.TrainerEnv(model, optimizer)
                if len(strategy)>=6:
                    
                    args.object_dropout_rate = strategy[5]

                                
            this_train_dataset = this_train_dataset.filter_scene_size(max_scene_size)
            this_train_dataset = this_train_dataset.filter_program_size_raw(max_program_size)
            logger.critical('Building the data loader. Curriculum = {}/{}, length = {}.'.format(max_scene_size, max_program_size, len(this_train_dataset)))

        if remove_ops is not None:
            this_train_dataset = this_train_dataset.filter_question_type(disallowed=remove_ops)
            #if epoch < args.filter_relate_epoch:
            #    this_train_dataset = this_train_dataset.filter_question_type(disallowed=['relate'])

        train_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=True, drop_last=True, nr_workers=args.data_workers)

        for enum_id in range(args.enums_per_epoch):
            if args.adversarial_loss:
                train_epoch(epoch, trainer, train_dataloader, meters, model, adversarial_trainer,gradient_magnitudes=gradient_magnitudes)
            else:
                train_epoch(epoch, trainer, train_dataloader, meters, model, args,gradient_magnitudes=gradient_magnitudes)


        if args.loss_curriculum:
            try:
                if meters.avg['loss'] < loss_threshold:
                    model.eval()
                    validate_epoch(epoch, trainer, validation_dataloader, meters)
                    logger.critical(meters.format_simple(
                    'Epoch = {}'.format(epoch),
                    {k: v for k, v in meters.avg.items() if k.startswith('validation')}, compressed=False ))

                    fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
                    
                    trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))
                csv_name = osp.join(args.meta_dir, 'gradient_magnitudes_epoch_{}.csv'.format(epoch))
                write_dict_to_csv(csv_name,gradient_magnitudes)
            except Exception as e:
                pass
        elif epoch % args.validation_interval == 0:
            model.eval()
            validate_epoch(epoch, trainer, validation_dataloader, meters)
            logger.critical(meters.format_simple(
                'Epoch = {}'.format(epoch),
                {k: v for k, v in meters.avg.items() if k.startswith('validation')}, compressed=False ))

        if not args.debug:
            meters.dump(args.meter_file)

        logger.critical(meters.format_simple(
            'Epoch = {}'.format(epoch),
            {k: v for k, v in meters.avg.items() if not k.startswith('validation')},
            compressed=False
        ))

        if not args.loss_curriculum:
            if epoch % args.save_interval == 0 and not args.debug:
                fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
                csv_name = osp.join(args.meta_dir, 'gradient_magnitudes_epoch_{}.csv'.format(epoch))
                trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))
                write_dict_to_csv(csv_name,gradient_magnitudes)

        if epoch > int(args.lr_cliff_epoch):
            trainer.set_learning_rate(args.lr * 0.1)

        try:
            if meters.avg['loss'] < loss_threshold:
                curriculum_index = curriculum_index + 1
                curriculum_updated = True
                epochs_since_last_update = 0
            else:
                curriculum_updated = False
                epochs_since_last_update += 1
        except Exception as e:
            pass

def write_dict_to_csv(file,d):
    #d is a dict of lists
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        for k in d.keys():
            l = d[k]
            for n in l:
                row = [k,n]
                writer.writerow(row)

def backward_check_nan(self, feed_dict, loss, monitors, output_dict):
    import torch
    caught_nan = False
    for name, param in self.model.named_parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad.data).any().item():
            caught_nan = True
            print('Caught NAN in gradient.', name)
    if caught_nan:
        print(loss)
        for k in feed_dict.keys():
            print(k)
        print(self.model.scene_graph.object_representations.grad)
        print(self.model.scene_graph.object_representations_batched.grad)


def get_model_gradient_magnitude(model):
    grads = []
    for param in model.parameters():
        try:
            grads.append(param.grad.view(-1))
        except Exception as e:
            continue
    grads = torch.cat(grads)
    n = torch.norm(grads, p=2)
    return n.item()

def train_epoch(epoch, trainer, train_dataloader, meters,model,args,adversarial_trainer=None,gradient_magnitudes=None):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)


    if adversarial_trainer is not None:
        adversary = adversarial_trainer['adversary']
        adversarial_optimizer = adversarial_trainer['adversarial_optimizer']
        adversarial_loss = adversarial_trainer['adversarial_loss']

    meters.update(epoch=epoch)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)

    end = time.time()

    gradient_magnitudes[epoch] = []

    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)

            feed_dict['epoch'] = epoch
            feed_dict['args'] = args

            if adversarial_trainer is not None:
                feed_dict['adversary'] = adversary


            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict, cast_tensor=False)
            

            try:
                gradient_magnitude = get_model_gradient_magnitude(model)
                gradient_magnitudes[epoch].append(gradient_magnitude)
            except Exception as e:
                pass


            if adversarial_trainer is not None:
                num_adversarial_steps = 1
                for l in range(num_adversarial_steps):
                    adversary.zero_grad()
                    f_sng = output_dict['scene_graph']
                    f_sng = [[x[0],x[1].detach(),x[2]] for x in f_sng]
                    l = -1 * adversarial_loss(f_sng,adversary)
                    print(l)
                    l.backward()
                    adversarial_optimizer.step()


                

            step_time = time.time() - end; end = time.time()

            n = feed_dict['image'].size(0)
            meters.update(loss=loss, n=n)
            meters.update(monitors, n=n)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                {k: v for k, v in meters.val.items() if not k.startswith('validation') and k != 'epoch' and k.count('/') <= 1},
                compressed=True
            ))
            pbar.update()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)


def validate_epoch(epoch, trainer, val_dataloader, meters, meter_prefix='validation'):
    end = time.time()
    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            feed_dict['epoch'] = epoch
            feed_dict['args'] = args

            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            output_dict, extra_info = trainer.evaluate(feed_dict, cast_tensor=False)
            monitors = {meter_prefix + '/' + k: v for k, v in as_float(output_dict['monitors']).items()}
            step_time = time.time() - end; end = time.time()

            n = feed_dict['image'].size(0)
            meters.update(monitors, n=n)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                compressed=True
            ))
            pbar.update()

            end = time.time()


if __name__ == '__main__':
    main()