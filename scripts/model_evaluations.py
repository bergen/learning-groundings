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
from nscl.datasets import get_available_symbolic_datasets, initialize_dataset, get_symbolic_dataset_builder, get_dataset_builder,get_available_datasets
from nscl.datasets.common.vocab import Vocab

from torch import nn
import torch
from torchvision import transforms

import csv

logger = get_logger(__file__)

parser = JacArgumentParser()

parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')

# training_target and curriculum learning
parser.add_argument('--expr', default=None, metavar='DIR', help='experiment name')
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
    feed_dict['args'] = args
    feed_dict['image'] = feed_dict['image'].cuda()
    sng = model.get_sng(feed_dict)
    return sng[0]

def get_object_graph(model,features,one_hot):
    d = {}

    q_color = ['color']
    q_size = ['size']
    q_material = ['material']
    q_shape = ['shape']

    a_color = model.reasoning.inference_query(features,one_hot,q_color,args)
    a_size = model.reasoning.inference_query(features,one_hot,q_size,args)
    a_material = model.reasoning.inference_query(features,one_hot,q_material,args)
    a_shape = model.reasoning.inference_query(features,one_hot,q_shape,args)

    d['color'] = a_color
    d['size'] = a_size
    d['material'] = a_material
    d['shape'] = a_shape

    return d


def get_relations(model,features):
    relations = ['front','behind','left','right']
    d={}

    for r in relations:
        d[r] = model.reasoning.inference_relate(features,r,args)

    return d

def ground_truth_graph(scene_dict):
    objects = scene_dict['scene'][0]['objects']
    relations = scene_dict['scene'][0]['relationships']


    attributes = ['color','shape','material','size']

    scene = []

    for obj in objects:
        obj_rep = {}
        for a in attributes:
            obj_rep[a] = obj[a]

        scene.append(obj_rep)


    return scene,relations

def select_rows_cols(obj_indices,relation_matrix):
    obj_indices = torch.tensor(obj_indices,device=relation_matrix.device,dtype=torch.long)
    relation_matrix = torch.index_select(relation_matrix,0,obj_indices)
    relation_matrix = torch.index_select(relation_matrix,1,obj_indices)
    relation_matrix = torch.round(torch.exp(relation_matrix))


    return relation_matrix


def inferred_graph(model, feed_dict):
    features = get_scene_graph(model,feed_dict)

    num_objects = features[1].size(0)

    scene = []
    obj_indices = []

    for j in range(num_objects):
        one_hot = -100*torch.ones(num_objects, dtype=torch.float, device=features[1].device)
        one_hot[j] = 0
        object_graph = get_object_graph(model,features,one_hot)
        obj_weight = float(features[3][j])
        if obj_weight > -0.3:
            scene.append(object_graph)
            obj_indices.append(j)

    relation_dict = get_relations(model,features)
    transformed_relation_dict = {}
    for r in relation_dict:
        matrix = relation_dict[r]
        transformed_relation_dict[r] = select_rows_cols(obj_indices,matrix)

    return scene, transformed_relation_dict



def dist_objects(obj1,obj2,relation_dict1,relation_dict2,index1,index2):
    #assumes obj1, obj2 are dicts from attribute->val

    keys = obj1.keys()

    diff = len([x for x in keys if obj1[x]!=obj2[x]])


    total_diff_r = 0

    for r in relation_dict1:
        num_relations_obj1 = len(relation_dict1[r][index1])
        num_relations_obj2 = float(torch.sum(relation_dict2[r][index2,:]))

        diff_r = abs(num_relations_obj1-num_relations_obj2)

        total_diff_r += diff_r

    diff = diff + 0.1*total_diff_r

    return diff


def scene_dist_matrix(scene1,scene2,relation_dict1,relation_dict2):
    #scene1, scene2 are lists of objects

    num_gt_objs = len(scene1)
    num_inferred_objs = len(scene2)

    matrix = np.zeros((num_gt_objs,num_inferred_objs))

    for i in range(num_gt_objs):
        for j in range(num_inferred_objs):
            matrix[i,j] = dist_objects(scene1[i],scene2[j],relation_dict1,relation_dict2,i,j)

    return matrix


def optimal_assignment(scene1, scene2,relation_dict1,relation_dict2):
    dist_matrix = scene_dist_matrix(scene1,scene2,relation_dict1,relation_dict2)

    assignment = {}
    row_ind,col_ind = linear_sum_assignment(dist_matrix)

    for i in range(len(row_ind)):
        assignment[row_ind[i]] = col_ind[i]

    return assignment


def get_data(batch_size=1, dataset_size=500):
    initialize_dataset(args.dataset)
    build_dataset = get_dataset_builder(args.dataset)

    #use validation set
    dataset = build_dataset(args, configs, args.data_image_root, args.data_scenes_json, args.data_questions_json)

    if dataset_size is not None:
        dataset = dataset.trim_length(dataset_size)

    dataloader = dataset.make_dataloader(batch_size=batch_size, shuffle=False, drop_last=False, nr_workers=1)
    train_iter = iter(dataloader)
    return train_iter, dataset_size


def graph_accuracy(model, feed_dict):
    ground_truth_attributes, ground_truth_relations = ground_truth_graph(feed_dict)
    inferred_attributes,inferred_relations = inferred_graph(model, feed_dict)

    assignment = optimal_assignment(ground_truth_attributes,inferred_attributes,ground_truth_relations,inferred_relations)



    num_objects = len(ground_truth_attributes)

    num_inferred_objects = len(inferred_attributes)

    accuracies = []
    relation_accuracies = []

    relationships = ground_truth_relations.keys()

    for i in assignment:
        gt_obj = ground_truth_attributes[i]
        inf_obj = inferred_attributes[assignment[i]]

        gt_relations = []

        proportion_correct = len([k for k in gt_obj.keys() if gt_obj[k]==inf_obj[k]])/len(gt_obj.keys())



        accuracies.append(proportion_correct)

        for r in ground_truth_relations:
            rel = ground_truth_relations[r]
            gt_obj_relations = rel[i]
            inferred_obj_relations = inferred_relations[r][assignment[i],:].tolist()

            transformed_gt_obj_relations = [assignment[j] for j in gt_obj_relations if j in assignment]
            z = [0.0]*(max(assignment.values())+1)
            for j in transformed_gt_obj_relations:
                z[j] = 1.0
            num_relation_correct = len([k for k in range(len(assignment)) if z[k]==inferred_obj_relations[k]])

            relation_accuracies.append(num_relation_correct)




    precision = sum(accuracies)/num_inferred_objects
    recall = sum(accuracies)/num_objects

    relation_precision = mean(relation_accuracies) / num_inferred_objects
    relation_recall = mean(relation_accuracies) / num_objects




    return precision,recall, relation_precision, relation_recall



def main():
    precisions = []
    recalls = []
    relation_precisions = []
    relation_recalls = []

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

        precision,recall, relation_precision, relation_recall = graph_accuracy(model, feed_dict)

        precisions.append(precision)
        recalls.append(recall)
        relation_precisions.append(relation_precision)
        relation_recalls.append(relation_recall)

    print(mean(precisions))
    print(mean(recalls))
    print(mean(relation_precisions))
    print(mean(relation_recalls))


if __name__=="__main__":
    main()
        

        




