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

__all__ = ['SceneGraph','NaiveRNNSceneGraph','AttentionCNNSceneGraph']


class SceneGraph(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True):
        super().__init__()
        self.pool_size = 7
        self.feature_dim = feature_dim
        self.output_dims = output_dims
        self.downsample_rate = downsample_rate

        self.object_supervision = object_supervision
        self.concatenative_pair_representation = concatenative_pair_representation



        self.object_coord_fuse = nn.Sequential(nn.Conv2d(feature_dim+2,feature_dim,kernel_size=1),nn.ReLU(True))
        
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
        


       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device)
        
        for i in range(input.size(0)):
            single_scene_object_features =  torch.squeeze(object_features[i,:],dim=0) #dim=256 x 16 x 24
            scene_object_coords = torch.unsqueeze(torch.cat((single_scene_object_features,obj_coord_map),dim=0),dim=0)

            fused_object_coords = torch.squeeze(self.object_coord_fuse(scene_object_coords),dim=0) #dim=256 x Z x Y


            num_objects = objects_length[i].item()

            queries = self.get_queries(fused_object_coords,num_objects)



            attention_map = torch.einsum("ij,jkl -> ikl", queries,fused_object_coords) #dim=num_objects x Z x Y
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

    def get_queries(self,fused_object_coords,num_objects):
        pass

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)


    def compute_attention(self,input,objects,objects_length):
        object_features = torch.squeeze(input,dim=0)
        obj_coord_map = coord_map((object_features.size(1),object_features.size(2)),self.query.device)

        scene_object_coords = torch.unsqueeze(torch.cat((object_features,obj_coord_map),dim=0),dim=0)

        fused_object_coords = torch.squeeze(self.object_coord_fuse(scene_object_coords),dim=0) #dim=256 x Z x Y


        num_objects = objects_length[0]
        relevant_queries = self.query[0:num_objects,:] #num_objects x feature_dim

        attention_map = self.temperature*torch.einsum("ij,jkl -> ikl", relevant_queries,fused_object_coords) #dim=num_objects x Z x Y
        attention_map = nn.Softmax(1)(attention_map.view(num_objects,-1)).view_as(attention_map)

        return attention_map


class NaiveRNNSceneGraph(SceneGraph):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True):
        super().__init__(feature_dim, output_dims, downsample_rate)

        self.attention_rnn = nn.LSTM(feature_dim*16*24, feature_dim,batch_first=True)

    

    def get_queries(self,fused_object_coords,num_objects):
        rnn_input = fused_object_coords.view(-1,self.feature_dim*16*24).expand(num_objects,-1)
        rnn_input = torch.unsqueeze(rnn_input,dim=0)
        queries,_ = self.attention_rnn(rnn_input)
        queries = torch.squeeze(queries,dim=0)
        return queries

class NaiveRNNSceneGraphBatched(SceneGraph):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True):
        super().__init__(feature_dim, output_dims, downsample_rate)

        self.attention_rnn = nn.LSTM(feature_dim*16*24, feature_dim,batch_first=True)

    def forward(self, input, objects, objects_length):
        object_features = input
        

        batch_size = input.size(0)
        max_num_objects = max(objects_length)
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = torch.unsqueeze(coord_map((object_features.size(2),object_features.size(3)),object_features.device),dim=0)

        obj_coord_map_batched = obj_coord_map.repeat(batch_size,1,1,1)

        scene_object_coords_batched = torch.cat((object_features,obj_coord_map_batched), dim=1)

        fused_object_coords_batched = self.object_coord_fuse(scene_object_coords_batched)

        queries = self.get_queries_batched(fused_object_coords_batched, batch_size, max_num_objects)

        attention_map_batched = torch.einsum("bij,bjkl -> bikl", queries,fused_object_coords_batched)
        attention_map_batched = nn.Softmax(2)(attention_map_batched.reshape(batch_size,max_num_objects,-1)).view_as(attention_map_batched)
        object_values_batched = torch.einsum("bijk,bljk -> bil", attention_map_batched, fused_object_coords_batched) 
        object_representations_batched = self._norm(self.object_features_layer(object_values_batched))

        object_pair_representations_batched = self.objects_to_pair_representations_batched(object_representations_batched)


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i].item()
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0).contiguous()
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs


    def objects_to_pair_representations_batched(self, object_representations_batched):
        num_objects = object_representations_batched.size(1)

        obj1_representations = self.obj1_linear(object_representations_batched)
        obj2_representations = self.obj2_linear(object_representations_batched)

        obj1_representations.unsqueeze_(-1)#now batch_size x num_objects x feature_dim x 1
        obj2_representations.unsqueeze_(-1)

        obj1_representations = obj1_representations.transpose(2,3)
        obj2_representations = obj2_representations.transpose(2,3).transpose(1,2)

        obj1_representations = obj1_representations.repeat(1,1,num_objects,1)
        obj2_representations = obj2_representations.repeat(1,num_objects,1,1)

        object_pair_representations = obj1_representations+obj2_representations
        object_pair_representations = object_pair_representations

        return object_pair_representations

    def get_queries_batched(self,fused_object_coords,batch_size,max_num_objects):
        rnn_input = fused_object_coords.reshape(batch_size,-1,self.feature_dim*16*24).expand(-1,max_num_objects,-1)
        queries,_ = self.attention_rnn(rnn_input)
        return queries

class AttentionCNNSceneGraph(SceneGraph):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True):
        super().__init__(feature_dim, output_dims, downsample_rate)

        self.attention_cnn = nn.Sequential(nn.Conv2d(self.feature_dim+1,self.feature_dim,kernel_size=1),nn.ReLU())
        self.attention_fc = nn.Sequential(nn.Linear(self.feature_dim*16*24,self.feature_dim), nn.ReLU())


    def forward(self, input, objects, objects_length):
        object_features = input
        
        device = object_features.device

        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),device)
        
        for i in range(input.size(0)):
            single_scene_object_features =  torch.squeeze(object_features[i,:],dim=0) #dim=256 x 16 x 24
            scene_object_coords = torch.unsqueeze(torch.cat((single_scene_object_features,obj_coord_map),dim=0),dim=0)

            fused_object_coords = torch.squeeze(self.object_coord_fuse(scene_object_coords),dim=0) #dim=256 x Z x Y


            num_objects = objects_length[i].item()


            attention_map_list = []
            max_attention = torch.zeros(fused_object_coords.size(1),fused_object_coords.size(2)).to(device)
            for j in range(num_objects):
                h = self.attention_cnn(torch.unsqueeze(torch.cat((torch.unsqueeze(max_attention,dim=0),fused_object_coords),dim=0), dim=0))
                query = torch.squeeze(self.attention_fc(h.view(1,-1)),dim=0)
            

                obj_attention_weights = torch.einsum("i,ijk -> jk",query,fused_object_coords)
                obj_attention = nn.Softmax(1)(obj_attention_weights.view(1,-1)).view_as(obj_attention_weights)
                attention_map_list.append(obj_attention)
                max_attention = torch.max(obj_attention,max_attention)

            attention_map = torch.stack(attention_map_list)


            #attention_map = torch.einsum("ij,jkl -> ikl", queries,fused_object_coords) #dim=num_objects x Z x Y
            #attention_map = nn.Softmax(1)(attention_map.view(num_objects,-1)).view_as(attention_map)
            object_values = torch.einsum("ijk,ljk -> il", attention_map, fused_object_coords) #dim=num_objects x 256

            object_representations = self._norm(self.object_features_layer(object_values))

            object_pair_representations = self.objects_to_pair_representations(object_representations)

            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])

        return outputs


        

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

