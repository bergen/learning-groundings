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
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True


class Model(ReasoningV1Model):
    def __init__(self, args, vocab):
        super().__init__(args, vocab, configs)

        self.ising_matrix = initialize_ising_matrix()
        self.attention_loss = args.attention_loss

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


        f_scene = self.resnet(feed_dict.image)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, object_lengths)
        
        

        programs = feed_dict.program_qsseq
        programs, buffers, answers = self.reasoning(f_sng, programs, fd=feed_dict)
        outputs['buffers'] = buffers
        outputs['answer'] = answers

        update_from_loss_module(monitors, outputs, self.scene_loss(
            feed_dict, f_sng,
            self.reasoning.embedding_attribute, self.reasoning.embedding_relation
        ))
        update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))

        canonize_monitors(monitors)

        if self.training:
            loss = monitors['loss/qa']
            if self.attention_loss:
                attention = self.scene_graph.compute_attention(f_scene, feed_dict.objects, object_lengths)
                loss += self.compute_attention_loss(attention)
            if configs.train.scene_add_supervision:
                loss = loss + monitors['loss/scene']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            outputs['buffers'] = buffers
            return outputs

    def get_sng(self, feed_dict):

        object_lengths = self.get_object_lengths(feed_dict)

        feed_dict = GView(feed_dict)

        f_scene = self.resnet(feed_dict.image)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, object_lengths)
        
        
        return f_sng

    def get_attention(self,feed_dict):
        object_lengths = self.get_object_lengths(feed_dict)

        feed_dict = GView(feed_dict)

        f_scene = self.resnet(feed_dict.image)
        attention = self.scene_graph.compute_attention(f_scene,feed_dict.objects, object_lengths)

        return attention

    def compute_attention_loss(self,attention):
        device = attention.device
        ising_matrix = self.ising_matrix.to(device)

        w = 0.1

        energy = torch.einsum('bcij,ijkl,bckl->',attention,ising_matrix,attention)

        return w*energy


def initialize_ising_matrix():
    width = 16
    height = 24
    ising_matrix = torch.zeros(width,height,width,height)
    for i in range(width):
        for j in range(height):
            if i<width-1:
                ising_matrix[i,j,i+1,j]=-1
            if j<height-1:
                ising_matrix[i,j,i,j+1]=-1

    return ising_matrix

def make_model(args, vocab):
    return Model(args, vocab)
