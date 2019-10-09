#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize-dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/18/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os.path as osp
from PIL import Image

import torch.backends.cudnn as cudnn

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jacinle.random as random
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.utils.tqdm import tqdm
from jaclearn.visualize.box import vis_bboxes
from jacinle.utils.imp import load_source
from jactorch.train import TrainerEnv
from jaclearn.visualize.html_table import HTMLTableVisualizer, HTMLTableColumnDesc
from nscl.datasets import get_available_symbolic_datasets, initialize_dataset, get_symbolic_dataset_builder, get_dataset_builder
from nscl.datasets.common.vocab import Vocab

from torch import nn
import torch
from torchvision import transforms

logger = get_logger(__file__)


parser = JacArgumentParser()
parser.add_argument('--dataset', required=True, choices=get_available_symbolic_datasets(), help='dataset')
parser.add_argument('--data-dir', required=True)
parser.add_argument('--data-scenes-json', type='checked_file')
parser.add_argument('--data-questions-json', type='checked_file')
parser.add_argument('--data-vocab-json', type='checked_file')
parser.add_argument('-n', '--nr-vis', type=int, help='number of visualized questions')
parser.add_argument('--random', type='bool', default=False, help='random choose the questions')
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')

parser.add_argument('--visualize-attention', type='bool', default=False)
parser.add_argument('--desc', type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint (default: none)')

args = parser.parse_args()

args.data_image_root = osp.join(args.data_dir, 'images')
args.data_vis_dir = osp.join(args.data_dir, 'visualize')
if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')
if args.data_vocab_json is None:
    args.data_vocab_json = osp.join(args.data_dir, 'vocab.json')
vocab = Vocab.from_json(args.data_vocab_json)


desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)

def make_model():
    model = desc.make_model(args, vocab)
    model.cuda()
        # Use the customized data parallel if applicable.
    cudnn.benchmark = False
    
    from jactorch.optim import AdamW
    trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = AdamW(trainable_parameters, 0.1, weight_decay=configs.train.weight_decay)

    trainer = TrainerEnv(model, optimizer)

    _ = trainer.load_checkpoint(args.resume)

    return trainer._model

def get_data():
    initialize_dataset(args.dataset)
    build_dataset = get_dataset_builder(args.dataset)

    dataset = build_dataset(args, configs, args.data_image_root, args.data_scenes_json, args.data_questions_json)

    dataset_size = 100
    dataset = dataset.trim_length(dataset_size)

    dataloader = dataset.make_dataloader(batch_size=1, shuffle=False, drop_last=False, nr_workers=1)
    train_iter = iter(dataloader)
    return train_iter, dataset_size


def get_attention(model,feed_dict):
    scene_representation = model.resnet(feed_dict['image'].cuda())
    attention = model.scene_graph.compute_attention(scene_representation,feed_dict['objects'].cuda(), feed_dict['objects_length'])
    return attention

def visualize_attention():
    train_iter, dataset_size = get_data()

    model = make_model()
    
    images = []

    feed_dict = next(train_iter)
    for i in range(dataset_size):
        prev_image_name = feed_dict['image_filename'][0]
        try:
            feed_dict = next(train_iter)
        except:
            break
        new_image_name = feed_dict['image_filename'][0]
        if prev_image_name == new_image_name:
            continue
        attention = get_attention(model, feed_dict) 
        image = feed_dict['image']
        print(feed_dict['image_filename'])

        image_filename = osp.join(args.data_image_root, new_image_name)
        pil_image = Image.open(image_filename)
        torch_image = transforms.ToTensor()(pil_image)
        print(torch_image.size())
        #scene_pil_image = transforms.ToPILImage()(torch.squeeze(image,dim=0))
        for j in range(feed_dict['objects_length']):
            object_attention = torch.unsqueeze(torch.unsqueeze(attention[j,:].cpu(),dim=0),dim=0)
            upsampled_attention = torch.squeeze(nn.functional.interpolate(object_attention,size=(320,480)))
            image_filtered = torch_image*upsampled_attention

            mask=torch.zeros(image_filtered.size())
            image_filtered = torch.where(image_filtered>0.1,mask,torch_image)
            #attention_image = transforms.ToPILImage()(upsampled_attention)
            #pil_image.paste(attention_image,(0,0))
            object_image = transforms.ToPILImage()(image_filtered)
            images.append((pil_image,object_image))

    save_images(images)

def save_images(images):
    #images is a list of pairs of images
    vis = HTMLTableVisualizer(args.data_vis_dir, 'Dataset: ' + args.dataset.upper())
    vis.begin_html()

    indices = len(images)

    with vis.table('Visualize', [
        HTMLTableColumnDesc('scene', 'QA', 'figure', {'width': '50%'},None),
        HTMLTableColumnDesc('object', 'QA', 'figure', {'width': '50%'},None),
    ]):
        for i in tqdm(indices):
            scene_image = images[i][0]
            object_image = images[i][1]

            scene_fig, ax = vis_bboxes(scene_image, [], 'object', add_text=False)
            object_fig, ax = vis_bboxes(object_image, [], 'object', add_text=False)

            vis.row(scene=scene_fig, object=object_fig)
            plt.close()
    vis.end_html()

def main():
    initialize_dataset(args.dataset)
    build_symbolic_dataset = get_symbolic_dataset_builder(args.dataset)
    dataset = build_symbolic_dataset(args)

    if args.nr_vis is None:
        args.nr_vis = min(100, len(dataset))

    if args.random:
        indices = random.choice(len(dataset), size=args.nr_vis, replace=False)
    else:
        indices = list(range(args.nr_vis))

    vis = HTMLTableVisualizer(args.data_vis_dir, 'Dataset: ' + args.dataset.upper())
    vis.begin_html()
    with vis.table('Metainfo', [
        HTMLTableColumnDesc('k', 'Key', 'text', {},None),
        HTMLTableColumnDesc('v', 'Value', 'code', {},None)
    ]):
        for k, v in args.__dict__.items():
            vis.row(k=k, v=v)

    with vis.table('Visualize', [
        HTMLTableColumnDesc('id', 'QuestionID', 'text', {},None),
        HTMLTableColumnDesc('image', 'QA', 'figure', {'width': '100%'},None),
        HTMLTableColumnDesc('qa', 'QA', 'text', css=None,td_css={'width': '30%'}),
        HTMLTableColumnDesc('p', 'Program', 'code', css=None,td_css={'width': '30%'})
    ]):
        for i in tqdm(indices):
            feed_dict = GView(dataset[i])
            image_filename = osp.join(args.data_image_root, feed_dict.image_filename)
            image = Image.open(image_filename)

            if 'objects' in feed_dict:
                fig, ax = vis_bboxes(image, feed_dict.objects, 'object', add_text=False)
            else:
                fig, ax = vis_bboxes(image, [], 'object', add_text=False)
            _ = ax.set_title('object bounding box annotations')

            QA_string = """
                <p><b>Q</b>: {}</p>
                <p><b>A</b>: {}</p>
            """.format(feed_dict.question_raw, feed_dict.answer)
            P_string = '\n'.join([repr(x) for x in feed_dict.program_seq])

            vis.row(id=i, image=fig, qa=QA_string, p=P_string)
            plt.close()
    vis.end_html()

    logger.info('Happy Holiday! You can find your result at "http://monday.csail.mit.edu/xiuming' + osp.realpath(args.data_vis_dir) + '".')


if __name__ == '__main__':
    visualize_attention()

