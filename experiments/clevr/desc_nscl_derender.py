#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_nscl_derender.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/10/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

Note that, in order to train this model, one must use the curriculum learning.
"""

import torch
from jacinle.utils.container import GView
from nscl.models.reasoning_v1 import make_reasoning_v1_configs, ReasoningV1Model
from nscl.models.utils import canonize_monitors, update_from_loss_module

configs = make_reasoning_v1_configs()
#configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True


class Model(ReasoningV1Model):
    def __init__(self, args, vocab):
        super().__init__(args, vocab, configs)

        try:
            self.attention_loss = args.attention_loss
        except Exception as e:
            pass

        try:
            self.use_adversarial_loss = args.adversarial_loss
        except Exception as e:
            pass

        try:
            self.anneal_rnn = args.anneal_rnn
        except Exception as e:
            pass

        self.attention_type = args.attention_type

        self.fine_tune_resnet_epoch = args.fine_tune_resnet_epoch
        self.fine_tune_semantics_epoch = args.fine_tune_semantics_epoch
        self.normalize_objects = args.normalize_objects


    def get_object_lengths(self,feed_dict):
        object_lengths = []
        scene = feed_dict['scene']
        for d in scene:
            num_objects = len(d['objects'])
            object_lengths.append(num_objects)

        return object_lengths

    def forward(self, feed_dict):
        object_lengths = self.get_object_lengths(feed_dict)

        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        args = feed_dict.args


        if self.attention_type=='monet':
            f_scene = feed_dict.image     
        else:
            if feed_dict.epoch < self.fine_tune_resnet_epoch:
                f_scene = self.resnet(feed_dict.image)
            else:
                with torch.no_grad():
                    f_scene = self.resnet(feed_dict.image)

        if self.attention_type=='structured-rnn-batched' or self.attention_type=='structured-subtractive-rnn-batched':
            f_sng = self.scene_graph(f_scene, feed_dict.objects, object_lengths,feed_dict.epoch)
        elif self.attention_type=='scene-graph-object-supervised':
            f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)
        else:
            f_sng = self.scene_graph(f_scene, feed_dict.objects, object_lengths,args)
        
        

        programs = feed_dict.program_qsseq

        if feed_dict.epoch >= self.fine_tune_semantics_epoch:
            for p in self.reasoning.parameters():
                p.requires_grad = False
            #print(self.reasoning.embedding_attribute.get_concept('blue').embedding)
        
        programs, buffers, answers = self.reasoning(f_sng, programs, fd=feed_dict)
        outputs['buffers'] = buffers
        outputs['answer'] = answers

        #update_from_loss_module(monitors, outputs, self.scene_loss(
        #    feed_dict, f_sng,
        #    self.reasoning.embedding_attribute, self.reasoning.embedding_relation
        #))
        update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))

        canonize_monitors(monitors)

        if self.training:
            loss = monitors['loss/qa']

            if self.attention_loss:
                attention = self.scene_graph.compute_attention(f_scene, feed_dict.objects, object_lengths)
                loss += self.compute_attention_loss(attention)
            #if configs.train.scene_add_supervision:
            #    loss = loss + monitors['loss/scene']
            if self.use_adversarial_loss:
                w=0.01
                loss += w*self.adversarial_loss(f_sng,feed_dict.adversary)
                outputs['scene_graph'] = f_sng
            if not self.normalize_objects: #penalize large object representations
                w=0.001
                loss = loss + w*self.regularize_object_magnitude(f_sng)
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            outputs['buffers'] = buffers
            return outputs

    def get_sng(self, feed_dict):

        object_lengths = self.get_object_lengths(feed_dict)

        feed_dict = GView(feed_dict)

        f_scene = self.resnet(feed_dict.image)


        if self.anneal_rnn:
            f_sng = self.scene_graph(f_scene, feed_dict.objects, object_lengths,60)
        else:
            f_sng = self.scene_graph(f_scene, feed_dict.objects, object_lengths)
        
        
        return f_sng

    def get_attention(self,feed_dict,visualize_foreground=False):
        object_lengths = self.get_object_lengths(feed_dict)

        feed_dict = GView(feed_dict)

        f_scene = self.resnet(feed_dict.image)
        if self.anneal_rnn:
            attention = self.scene_graph.compute_attention(f_scene,feed_dict.objects, object_lengths,60)
        else:
            attention = self.scene_graph.compute_attention(f_scene,feed_dict.objects, object_lengths,visualize_foreground)

        return attention

    def compute_attention_loss(self,attention):
        w = 0.01

        width = attention.size(2)
        height = attention.size(3)
        loss = torch.tensor(0,dtype=torch.float,device=attention.device)

        for i in range(width):
            for j in range(height):
                if i<width-1:
                    diff = attention[:,:,i,j] - attention[:,:,i+1,j]
                    sq_diff = torch.pow(diff,2)
                    loss+=torch.sum(sq_diff)
                if j<height-1:
                    diff = attention[:,:,i,j] - attention[:,:,i,j+1]
                    sq_diff = torch.pow(diff,2)
                    loss+=torch.sum(sq_diff)
        
        return w*loss

    def regularize_object_magnitude(self,f_sng):
        total_loss = torch.tensor(0,dtype=torch.float,device=f_sng[0][1].device)

        for scene in f_sng:
            objects = scene[1]
            objects = objects.view(-1)

            total_loss = total_loss + torch.norm(objects)
        return total_loss


def make_model(args, vocab):
    return Model(args, vocab)
