import numpy as np
from scipy.optimize import linear_sum_assignment
from statistics import mean


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

import csv

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

parser.add_argument('--extra-data-dir', type='checked_dir', metavar='DIR', help='extra data directory for validation')
parser.add_argument('--extra-data-scenes-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra scene json file for validation')
parser.add_argument('--extra-data-questions-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra question json file for validation')

parser.add_argument('--visualize-attention', type='bool', default=False)
parser.add_argument('--desc', type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint (default: none)')


parser.add_argument('--attention-type', default='cnn', choices=['cnn', 'naive-rnn', 'naive-rnn-batched',
                                                                'naive-rnn-global-batched','structured-rnn-batched','max-rnn-batched'])

parser.add_argument('--attention-loss', type='bool', default=False)
parser.add_argument('--anneal-rnn', type='bool', default=False)
parser.add_argument('--adversarial-loss', type='bool', default=False)
parser.add_argument('--adversarial-lr', type=float, default=0.0002, metavar='N', help='initial learning rate')
parser.add_argument('--presupposition-semantics', type='bool', default=False)
parser.add_argument('--subtractive-rnn', type='bool', default=False)

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


#this is information for the validation set
if args.extra_data_dir is not None:
    args.extra_data_image_root = osp.join(args.extra_data_dir, 'images')
    if args.extra_data_scenes_json is None:
        args.extra_data_scenes_json = osp.join(args.extra_data_dir, 'scenes.json')
    if args.extra_data_questions_json is None:
        args.extra_data_questions_json = osp.join(args.extra_data_dir, 'questions.json')


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

    model = trainer.model
    model.eval()

    return model

def get_scene_graph(model,feed_dict):
    feed_dict['image'] = feed_dict['image'].cuda()
    sng = model.get_sng(feed_dict)
    return sng[0]

def get_object_graph(model,features,one_hot):
    d = {}

    q_color = ['color']
    q_size = ['size']
    q_material = ['material']
    q_shape = ['shape']

    a_color = model.reasoning.inference_query(features,one_hot,q_color)
    a_size = model.reasoning.inference_query(features,one_hot,q_size)
    a_material = model.reasoning.inference_query(features,one_hot,q_material)
    a_shape = model.reasoning.inference_query(features,one_hot,q_shape)

    d['color'] = a_color
    d['size'] = a_size
    d['material'] = a_material
    d['shape'] = a_shape

    return d


def ground_truth_graph(scene_dict):
    objects = scene_dict['scene'][0]['objects']

    attributes = ['color','shape','material','size']

    scene = []

    for obj in objects:
        obj_rep = {}
        for a in attributes:
            obj_rep[a] = obj[a]

        scene.append(obj_rep)

    return scene

def inferred_graph(model, feed_dict):
    features = get_scene_graph(model,feed_dict)

    num_objects = features[1].size(0)

    scene = []

    for j in range(num_objects):
        one_hot = torch.zeros(num_objects, dtype=torch.float, device=features[1].device)
        one_hot[j] = 1
        object_graph = get_object_graph(model,features,one_hot)
        scene.append(object_graph)

    return scene



def dist_objects(obj1,obj2):
    #assumes obj1, obj2 are dicts from attribute->val

    keys = obj1.keys()

    diff = len([x for x in keys if obj1[x]!=obj2[x]])

    return diff


def scene_dist_matrix(scene1,scene2):
    #scene1, scene2 are lists of objects

    num_objs = len(scene1)

    matrix = np.zeros((num_objs,num_objs))

    for i in range(num_objs):
        for j in range(num_objs):
            matrix[i,j] = dist_objects(scene1[i],scene2[j])

    return matrix


def optimal_assignment(scene1, scene2):
    dist_matrix = scene_dist_matrix(scene1,scene2)

    assignment = {}
    row_ind,col_ind = linear_sum_assignment(dist_matrix)

    for i in range(len(row_ind)):
        assignment[i] = col_ind[i]

    return assignment


def get_data(batch_size=1, dataset_size=500):
    initialize_dataset(args.dataset)
    build_dataset = get_dataset_builder(args.dataset)

    #use validation set
    dataset = build_dataset(args, configs, args.extra_data_image_root, args.extra_data_scenes_json, args.extra_data_questions_json)

    if dataset_size is not None:
        dataset = dataset.trim_length(dataset_size)

    dataloader = dataset.make_dataloader(batch_size=batch_size, shuffle=False, drop_last=False, nr_workers=1)
    train_iter = iter(dataloader)
    return train_iter, dataset_size


def graph_accuracy(model, feed_dict):
    ground_truth = ground_truth_graph(feed_dict)
    inferred = inferred_graph(model, feed_dict)

    assignment = optimal_assignment(ground_truth,inferred)

    num_objects = len(ground_truth)

    accuracies = []

    for i in range(num_objects):
        gt_obj = ground_truth[i]
        inf_obj = inferred[assignment[i]]
        proportion_correct = len([k for k in gt_obj.keys() if gt_obj[k]==inf_obj[k]])/len(gt_obj.keys())

        print(len([k for k in gt_obj.keys() if gt_obj[k]==inf_obj[k]]))


        accuracies.append(proportion_correct)


    return mean(accuracies)



def main():
    accuracies = []

    validation_iter, _ = get_data(batch_size=1,dataset_size=None)
    model = make_model()
    for i in range(len(validation_iter)):
        feed_dict = next(validation_iter)
        new_image_name = feed_dict['image_filename'][0]
        if i==0:
            old_image_name = feed_dict['image_filename'][0]
        elif old_image_name == new_image_name:
            continue


        old_image_name = new_image_name

        accuracy = graph_accuracy(model, feed_dict)

        accuracies.append(accuracy)

    print(mean(accuracies))


if __name__=="__main__":
    main()
        

        




