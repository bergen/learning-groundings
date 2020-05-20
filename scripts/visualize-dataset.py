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
                                                                'naive-rnn-global-batched','structured-rnn-batched',
                                                                'max-rnn-batched','low-dim-rnn-batched','monet',
                                                                'scene-graph-object-supervised',
                                                                'structured-subtractive-rnn-batched',
                                                                'transformer',
                                                                'monet-lite'])

parser.add_argument('--attention-loss', type='bool', default=False)
parser.add_argument('--anneal-rnn', type='bool', default=False)
parser.add_argument('--adversarial-loss', type='bool', default=False)
parser.add_argument('--adversarial-lr', type=float, default=0.0002, metavar='N', help='initial learning rate')
parser.add_argument('--presupposition-semantics', type='bool', default=False)
parser.add_argument('--subtractive-rnn', type='bool', default=False)
parser.add_argument('--subtract-from-scene', type='bool', default=True)
parser.add_argument('--rnn-type', default='lstm', choices=['lstm','gru'])
parser.add_argument('--full-recurrence', type='bool', default=True)
parser.add_argument('--lr-cliff-epoch', type=int, default=50) #this is the epoch at which the lr will fall by factor of 0.1
parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'rmsprop'])
parser.add_argument('--fine-tune-resnet-epoch', type=int, default=100)
parser.add_argument('--restrict-finetuning', type='bool', default=True)
parser.add_argument('--resnet-type', default='resnet34', choices=['resnet34', 'resnet101','cmc_resnet','simclr_resnet'])
parser.add_argument('--transformer-use-queries', type='bool', default=False)
parser.add_argument('--filter-ops', type='bool', default=False)


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

def write_to_csv(r,file_name):
    with open(file_name,'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(['filename','question','accurate'])
        for row in r:
            wr.writerow(row)

def load_csv(file_name):
    with open(file_name,'r') as f:
        r = csv.reader(f)
        rows = [s for s in r]
    return rows

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


def get_attention(model,feed_dict):
    feed_dict['image'] = feed_dict['image'].cuda()
    attention = model.get_attention(feed_dict)
    return attention

def model_forward(model,feed_dict):
    feed_dict['image'] = feed_dict['image'].cuda()
    outputs = model(feed_dict)


    return outputs

def check_if_relational(feed_dict):
    program_seq = feed_dict['program_seq'][0]
    ops = [d['op'] for d in program_seq]
    return 'relate' in ops

def normalize_answer(a):
    if a in [True,False]:
        a = int(a)
    return a

def get_validation_results(filename):
    validation_iter, _ = get_data(batch_size=64,dataset_size=None)

    model = make_model()
    
    all_filenames = []
    all_questions = []
    all_accuracies = []

    for i in range(len(validation_iter)):
        d = {}

        feed_dict = next(validation_iter)
        

        outputs = model_forward(model, feed_dict)


        filenames = feed_dict['image_filename']
        questions = feed_dict['question_raw']
        correct_answers = feed_dict['answer']
        correct_answers = list(map(normalize_answer,correct_answers))
        answers = outputs['answer']

        accuracies = [answers[i]==correct_answers[i] for i in range(len(answers))]

        all_filenames += filenames
        all_questions += questions
        all_accuracies += accuracies

    results = zip(all_filenames,all_questions,all_accuracies)
    
    write_to_csv(results, filename)


def get_wrong_ids():
    wrong_ids = load_csv('/home/lbergen/NeuralDRS/NSCL-PyTorch-Release/data_vis_dir/simplest_wrong.csv')
    return [int(r[0]) for r in wrong_ids]

def visualize_sum_attentions(filter_for_ids=False):
    processed = []

    validation_iter, _ = get_data(batch_size=1,dataset_size=500)
    model = make_model()
    for i in range(len(validation_iter)):
        feed_dict = next(validation_iter)
        new_image_name = feed_dict['image_filename'][0]
        if i==0:
            old_image_name = feed_dict['image_filename'][0]
        elif old_image_name == new_image_name:
            continue

        #test_scene_graph(model,feed_dict)

        old_image_name = new_image_name

        attention = get_attention(model, feed_dict) 
        #model_forward(model, feed_dict)
        image = feed_dict['image']
        print(feed_dict['image_filename'])

        image_filename = osp.join(args.data_image_root, new_image_name)
        pil_image = Image.open(image_filename)
        
        torch_image = transforms.ToTensor()(pil_image)
        
        #scene_pil_image = transforms.ToPILImage()(torch.squeeze(image,dim=0))


        
        d={}
        
        #if prev_image_name == new_image_name:
        #    continue

        #scene_pil_image = transforms.ToPILImage()(torch.squeeze(image,dim=0))
        for j in range(feed_dict['objects_length']):
            object_attention = torch.unsqueeze(torch.unsqueeze(attention[0,j,:].cpu(),dim=0),dim=0)
            upsampled_attention = torch.squeeze(nn.functional.interpolate(object_attention,size=(320,480)))
            
            if j==0:
                total_attention = upsampled_attention*upsampled_attention
            else:
                total_attention +=upsampled_attention*upsampled_attention

            #image_filtered = torch_image*upsampled_attention

        mask=torch.zeros(total_attention.size())
        image_filtered = total_attention*torch_image
        #image_filtered = torch.where(total_attention>0.1,mask,torch_image)
        #attention_image = transforms.ToPILImage()(upsampled_attention)
        #pil_image.paste(attention_image,(0,0))
        object_image = transforms.ToPILImage()(image_filtered)

        d['original_image'] = pil_image
        d['attention_image'] = object_image
        processed.append(d)

    #processed.sort(key=lambda x: (x['correct'],not x['relational']))
    save_images(processed)

def visualize_attention_per_object():
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
        #model_forward(model, feed_dict)
        image = feed_dict['image']
        print(feed_dict['image_filename'])

        image_filename = osp.join(args.data_image_root, new_image_name)
        pil_image = Image.open(image_filename)
        torch_image = transforms.ToTensor()(pil_image)
        
        #scene_pil_image = transforms.ToPILImage()(torch.squeeze(image,dim=0))
        for j in range(feed_dict['objects_length']):
            object_attention = torch.unsqueeze(torch.unsqueeze(attention[j,:].cpu(),dim=0),dim=0)
            upsampled_attention = torch.squeeze(nn.functional.interpolate(object_attention,size=(320,480)))
            mask=torch.zeros(upsampled_attention.size())

            #image_filtered = torch_image*upsampled_attention

            
            image_filtered = torch.where(upsampled_attention>0.2,mask,torch_image)
            #attention_image = transforms.ToPILImage()(upsampled_attention)
            #pil_image.paste(attention_image,(0,0))
            object_image = transforms.ToPILImage()(image_filtered)
            images.append((pil_image,object_image))

    save_images(images)

def save_images(processed):
    #images is a list of pairs of images
    vis = HTMLTableVisualizer(args.data_vis_dir, 'Dataset: ' + args.dataset.upper())
    vis.begin_html()

    indices = len(processed)

    # if qa is None:
    #     with vis.table('Visualize', [
    #         HTMLTableColumnDesc('scene', 'Scene', 'figure', {'width': '50%'},None),
    #         HTMLTableColumnDesc('object', 'Attention', 'figure', {'width': '50%'},None),
    #     ]):
    #         for i in tqdm(indices):
    #             scene_image = images[i][0]
    #             object_image = images[i][1]

    #             scene_fig, ax = vis_bboxes(scene_image, [], 'object', add_text=False)
    #             object_fig, ax = vis_bboxes(object_image, [], 'object', add_text=False)

    #             vis.row(scene=scene_fig, object=object_fig)
    #             plt.close()
    #     vis.end_html()

    # else:
    with vis.table('Visualize', [
        HTMLTableColumnDesc('scene', 'Scene', 'figure', {'width': '80%'},None),
        HTMLTableColumnDesc('object', 'Attention', 'figure', {'width': '80%'},None)
    ]):
        for i in tqdm(indices):
                d = processed[i]
                scene_image = d['original_image']
                object_image = d['attention_image']
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


def view_feed_dict():
    validation_iter, _ = get_data(batch_size=1,dataset_size=20)
    model = make_model()

    for i in range(len(validation_iter)):
        feed_dict = next(validation_iter)
        feed_dict['image'] = feed_dict['image'].cuda()
        sng = model.get_sng(feed_dict)
        #outputs = model_forward(model,feed_dict)
        model.reasoning.inference_query(sng)

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

def get_scene_graph(model,feed_dict):
    feed_dict['image'] = feed_dict['image'].cuda()
    sng = model.get_sng(feed_dict)
    return sng[0]

def test_scene_graph(model,feed_dict):
    feed_dict['image'] = feed_dict['image'].cuda()
    batched_sng = model.get_sng(feed_dict)

    scene_representation = model.resnet(feed_dict['image'])
    unbatched_sng = model.scene_graph.test_batching(scene_representation,feed_dict['objects'].cuda(), feed_dict['objects_length'])

    print(batched_sng)
    print(unbatched_sng)
    

def visualize_scene_graph():
    data = []

    validation_iter, _ = get_data(batch_size=1,dataset_size=500)
    model = make_model()
    for i in range(len(validation_iter)):
        feed_dict = next(validation_iter)
        new_image_name = feed_dict['image_filename'][0]
        if i==0:
            old_image_name = feed_dict['image_filename'][0]
        elif old_image_name == new_image_name:
            continue

        #test_scene_graph(model,feed_dict)

        old_image_name = new_image_name

        attention = get_attention(model, feed_dict) 
        #model_forward(model, feed_dict)
        image = feed_dict['image']
        print(feed_dict['image_filename'])

        image_filename = osp.join(args.data_image_root, new_image_name)
        pil_image = Image.open(image_filename)
        
        torch_image = transforms.ToTensor()(pil_image)
        
        #scene_pil_image = transforms.ToPILImage()(torch.squeeze(image,dim=0))

        features = get_scene_graph(model,feed_dict)

        

        num_objects = features[1].size(0)

        for j in range(num_objects):
            #get the graph for the current object
            one_hot = torch.zeros(num_objects, dtype=torch.float, device=features[1].device)
            one_hot[j] = 1
            object_graph = get_object_graph(model,features,one_hot)


            #get the attentions
            object_attention = torch.unsqueeze(torch.unsqueeze(attention[0,j,:].cpu(),dim=0),dim=0)

            upsampled_attention = torch.squeeze(nn.functional.interpolate(object_attention,size=(320,480)))
            #print(upsampled_attention)
            
            mask=torch.zeros(upsampled_attention.size())
            #image_filtered = torch.where(upsampled_attention>0.95,mask,torch_image)
            if j==0:
                image_filtered = upsampled_attention*upsampled_attention*upsampled_attention*torch_image 
            else:
                image_filtered = image_filtered + upsampled_attention*upsampled_attention*upsampled_attention*torch_image 

            #image_filtered = torch_image
        
            object_image = transforms.ToPILImage()(image_filtered)


            d = {}
            d['image'] = pil_image
            d['object_attention'] = object_image
            d['object_graph'] = object_graph
            data.append(d)


        old_image_name = new_image_name

    save_scene_graph(data)


def save_scene_graph(processed):
    #processed is a dict
    vis = HTMLTableVisualizer(args.data_vis_dir, 'Dataset: ' + args.dataset.upper() + "_scenegraph")
    vis.begin_html()

    indices = len(processed)

    with vis.table('Visualize', [
        HTMLTableColumnDesc('scene', 'Scene', 'figure', {'width': '80%'},None),
        HTMLTableColumnDesc('attention', 'Attention', 'figure', {'width': '80%'},None),
        HTMLTableColumnDesc('graph', 'Object representation', 'text', css=None,td_css={'width': '30%'}),
    ]):
        for i in tqdm(indices):
                d = processed[i]
                scene_image = d['image']
                attention_image = d['object_attention']
                graph = d['object_graph']


                scene_fig, ax = vis_bboxes(scene_image, [], 'object', add_text=False)

                attention_fig, ax = vis_bboxes(attention_image, [], 'object', add_text=False)


                graph_string = """
                    <p><b>Color</b>: {}</p>
                    <p><b>Material</b>: {}</p>
                    <p><b>Size</b>: {}</p>
                    <p><b>Shape</b>: {}</p>
                """.format(graph['color'], graph['material'], graph['size'], graph['shape'])



                vis.row(scene=scene_fig,attention=attention_fig,graph=graph_string)
                plt.close()
    vis.end_html()


if __name__ == '__main__':
    #visualize_sum_attentions(filter_for_ids=False)
    #get_validation_results('validation_results.csv')
    visualize_scene_graph()

