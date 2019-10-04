#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/19/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Scene Graph generation.
"""

import os

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn

from . import functional

DEBUG = bool(int(os.getenv('DEBUG_SCENE_GRAPH', 0)))

__all__ = ['SceneGraph']


class SceneGraph(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True):
        super().__init__()
        self.pool_size = 7
        self.feature_dim = feature_dim
        self.output_dims = output_dims
        self.downsample_rate = downsample_rate

        self.object_supervision = object_supervision
        self.concatenative_pair_representation = concatenative_pair_representation

        if self.object_supervision:
            self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
            self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
            self.relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

            if not DEBUG:
                self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
                self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)

                self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
                self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)

                self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))
                self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))

                self.obj1_linear = nn.Linear(output_dims[1],output_dims[1])
                self.obj2_linear = nn.Linear(output_dims[1],output_dims[1])

                self.reset_parameters()
            else:
                def gen_replicate(n):
                    def rep(x):
                        return torch.cat([x for _ in range(n)], dim=1)
                    return rep

                self.pool_size = 32
                self.object_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
                self.context_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
                self.relation_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
                self.context_feature_extract = gen_replicate(2)
                self.relation_feature_extract = gen_replicate(3)
                self.object_feature_fuse = jacnn.Identity()
                self.relation_feature_fuse = jacnn.Identity()

        else:
            self.num_objects_upperbound = 11
            self.object_coord_fuse = nn.Sequential(nn.Conv2d(feature_dim+2,feature_dim,kernel_size=1),nn.ReLU(True))
            self.query = nn.Parameter(torch.randn(self.num_objects_upperbound, feature_dim))
            self.object_features_layer = nn.Sequential(nn.Linear(feature_dim,output_dims[1]),nn.ReLU(True))
            self.obj1_linear = nn.Linear(output_dims[1],output_dims[1])
            self.obj2_linear = nn.Linear(output_dims[1],output_dims[1])
            self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input, objects, objects_length):
        object_features = input
        

        if self.object_supervision and self.concatenative_pair_representation:
            context_features = self.context_feature_extract(input)
            outputs = list()
            objects_index = 0
            for i in range(input.size(0)):
                box = objects[objects_index:objects_index + objects_length[i].item()]
                #box is a list of object boundaries for the image

                objects_index += objects_length[i].item()

                with torch.no_grad():
                    batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)

                    # generate a "full-image" bounding box
                    image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate

                    image_box = torch.cat([
                        torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                        torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                        image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                        image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
                    ], dim=-1)

                    # intersection maps
                    box_context_imap = functional.generate_intersection_map(box, image_box, self.pool_size)

                this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
                x, y = this_context_features.chunk(2, dim=1)
                this_object_features = self.object_feature_fuse(torch.cat([
                    self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1)),
                    x, y * box_context_imap
                ], dim=1))

                object_representations = self._norm(self.object_feature_fc(this_object_features.view(box.size(0), -1)))

                object_pair_representations = self.objects_to_pair_representations(object_representations)


                if DEBUG:
                    outputs.append([
                        None,
                        this_object_features,
                        None
                    ])
                else:
                    outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])

        elif self.object_supervision and not self.concatenative_pair_representation:
            object_features = input
            context_features = self.context_feature_extract(input)
            relation_features = self.relation_feature_extract(input)

            outputs = list()
            objects_index = 0
            for i in range(input.size(0)):
                box = objects[objects_index:objects_index + objects_length[i].item()]
                objects_index += objects_length[i].item()

                with torch.no_grad():
                    batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)

                    # generate a "full-image" bounding box
                    image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                    image_box = torch.cat([
                        torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                        torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                        image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                        image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
                    ], dim=-1)

                    # meshgrid to obtain the subject and object bounding boxes
                    sub_id, obj_id = jactorch.meshgrid(torch.arange(box.size(0), dtype=torch.int64, device=box.device), dim=0)
                    sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)
                    sub_box, obj_box = jactorch.meshgrid(box, dim=0)
                    sub_box = sub_box.contiguous().view(box.size(0) ** 2, 4)
                    obj_box = obj_box.contiguous().view(box.size(0) ** 2, 4)

                    # union box
                    union_box = functional.generate_union_box(sub_box, obj_box)
                    rel_batch_ind = i + torch.zeros(union_box.size(0), 1, dtype=box.dtype, device=box.device)

                    # intersection maps
                    box_context_imap = functional.generate_intersection_map(box, image_box, self.pool_size)
                    sub_union_imap = functional.generate_intersection_map(sub_box, union_box, self.pool_size)
                    obj_union_imap = functional.generate_intersection_map(obj_box, union_box, self.pool_size)

                this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
                x, y = this_context_features.chunk(2, dim=1)
                this_object_features = self.object_feature_fuse(torch.cat([
                    self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1)),
                    x, y * box_context_imap
                ], dim=1))

                this_relation_features = self.relation_roi_pool(relation_features, torch.cat([rel_batch_ind, union_box], dim=-1))
                x, y, z = this_relation_features.chunk(3, dim=1)
                this_relation_features = self.relation_feature_fuse(torch.cat([
                    this_object_features[sub_id], this_object_features[obj_id],
                    x, y * sub_union_imap, z * obj_union_imap
                ], dim=1))

                print(self._norm(self.relation_feature_fc(this_relation_features.view(box.size(0) * box.size(0), -1)).view(box.size(0), box.size(0), -1)).size())
                if DEBUG:
                    outputs.append([
                        None,
                        this_object_features,
                        this_relation_features
                    ])
                else:
                    outputs.append([
                        None,
                        self._norm(self.object_feature_fc(this_object_features.view(box.size(0), -1))),
                        self._norm(self.relation_feature_fc(this_relation_features.view(box.size(0) * box.size(0), -1)).view(box.size(0), box.size(0), -1))
                    ])

        elif not self.object_supervision and self.concatenative_pair_representation:
            outputs = list()
            #object_features has shape batch_size x 256 x 16 x 24
            obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),self.query.device)
            
            for i in range(input.size(0)):
                single_scene_object_features =  torch.squeeze(object_features[i,:],dim=0) #dim=256 x 16 x 24
                scene_object_coords = torch.unsqueeze(torch.cat((single_scene_object_features,obj_coord_map),dim=0),dim=0)

                fused_object_coords = torch.squeeze(self.object_coord_fuse(scene_object_coords),dim=0) #dim=256 x Z x Y


                num_objects = objects_length[i].item()
                relevant_queries = self.query[0:num_objects,:] #num_objects x feature_dim

                attention_map = torch.einsum("ij,jkl -> ikl", relevant_queries,fused_object_coords) #dim=num_objects x Z x Y
                attention_map = nn.Softmax(1)(attention_map.view(num_objects,-1)).view_as(attention_map)
                object_values = torch.einsum("ijk,ljk -> il", attention_map, fused_object_coords) #dim=num_objects x 256

                object_representations = self._norm(self.object_features_layer(object_values))

                object_pair_representations = self.objects_to_pair_representations(object_representations)

                outputs.append([
                            None,
                            object_representations,
                            object_pair_representations
                        ])

        return outputs

    def objects_to_pair_representations(self, object_representations):
        num_objects = object_representations.size(0)

        obj1_representations = self.obj1_linear(object_representations)
        obj2_representations = self.obj2_linear(object_representations)

        obj1_representations.unsqueeze_(-1)
        obj2_representations.unsqueeze_(-1)

        obj1_representations = obj1_representations.transpose(1,2)
        obj2_representations = obj2_representations.transpose(1,2).transpose(0,1)

        obj1_representations = obj1_representations.repeat(1,num_objects,1)
        obj2_representations = obj2_representations.repeat(num_objects,1,1)

        object_pair_representations = obj1_representations+obj2_representations

        return object_pair_representations

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)

def coord_map(shape,device, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n).to(device)
    y_coord_row = torch.linspace(start, end, steps=m).to(device)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return torch.cat([x_coords, y_coords], 0)

