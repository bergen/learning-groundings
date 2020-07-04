#! /usr/bin/env python3
# Email  : maojiayuan@gmail.com
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Jiayuan Mao
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
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn
import math
import torchvision
import random



from . import functional

DEBUG = bool(int(os.getenv('DEBUG_SCENE_GRAPH', 0)))

__all__ = ['SceneGraph','NaiveRNNSceneGraph','AttentionCNNSceneGraph']


class SceneGraph(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__()
        self.pool_size = 7

        if args.resnet_type=='cmc_resnet':
            self.feature_dim = feature_dim
            self.img_channels = 8192
            self.collapse_image = True
        else:
            self.feature_dim = feature_dim
            self.collapse_image = False
        self.output_dims = output_dims
        self.downsample_rate = downsample_rate

        self.object_supervision = object_supervision
        self.concatenative_pair_representation = concatenative_pair_representation


        #self.object_coord_fuse = nn.Sequential(nn.Conv2d(feature_dim+2,feature_dim,kernel_size=1), nn.BatchNorm2d(feature_dim), nn.ReLU())
        if self.collapse_image:
            self.object_coord_fuse = nn.Sequential(nn.Conv2d(self.img_channels+2,self.feature_dim,kernel_size=1), nn.ReLU())
        else:
            self.object_coord_fuse = nn.Sequential(nn.Conv2d(feature_dim+2,feature_dim,kernel_size=1), nn.ReLU())
        
        self.object_features_layer = nn.Sequential(nn.Linear(feature_dim,output_dims[1]),nn.ReLU())
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


            num_objects = objects_length[i]

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
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=(16,24))

        self.attention_rnn = nn.LSTM(feature_dim*img_input_dim[0]*img_input_dim[1], feature_dim,batch_first=True)

    

    def get_queries(self,fused_object_coords,num_objects):
        rnn_input = fused_object_coords.view(-1,self.feature_dim*16*24).expand(num_objects,-1)
        rnn_input = torch.unsqueeze(rnn_input,dim=0)
        queries,_ = self.attention_rnn(rnn_input)
        queries = torch.squeeze(queries,dim=0)
        return queries



class NaiveRNNSceneGraphBatchedBase(NaiveRNNSceneGraph):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=(16,24))


    def objects_to_pair_representations(self, object_representations_batched):
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

    def get_queries(self,fused_object_coords,batch_size,max_num_objects):
        rnn_input = fused_object_coords.reshape(batch_size,-1,self.feature_dim*16*24).expand(-1,max_num_objects,-1)
        queries,_ = self.attention_rnn(rnn_input)
        return queries

class NaiveRNNSceneGraphBatched(NaiveRNNSceneGraphBatchedBase):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=(16,24))

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

        queries = self.get_queries(fused_object_coords_batched, batch_size, max_num_objects)

        attention_map_batched = torch.einsum("bij,bjkl -> bikl", queries,fused_object_coords_batched)
        attention_map_batched = nn.Softmax(2)(attention_map_batched.reshape(batch_size,max_num_objects,-1)).view_as(attention_map_batched)
        object_values_batched = torch.einsum("bijk,bljk -> bil", attention_map_batched, fused_object_coords_batched) 
        object_representations_batched = self._norm(self.object_features_layer(object_values_batched))

        object_pair_representations_batched = self.objects_to_pair_representations(object_representations_batched)


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs

    def compute_attention(self,input, objects, objects_length):
        object_features = input
        

        batch_size = input.size(0)
        max_num_objects = max(objects_length)
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = torch.unsqueeze(coord_map((object_features.size(2),object_features.size(3)),object_features.device),dim=0)

        obj_coord_map_batched = obj_coord_map.repeat(batch_size,1,1,1)

        scene_object_coords_batched = torch.cat((object_features,obj_coord_map_batched), dim=1)

        fused_object_coords_batched = self.object_coord_fuse(scene_object_coords_batched)

        queries = self.get_queries(fused_object_coords_batched, batch_size, max_num_objects)

        attention_map_batched = torch.einsum("bij,bjkl -> bikl", queries,fused_object_coords_batched)
        attention_map_batched = nn.Softmax(2)(attention_map_batched.reshape(batch_size,max_num_objects,-1)).view_as(attention_map_batched)

        return attention_map_batched


class MaxRNNSceneGraphBatched(NaiveRNNSceneGraphBatched):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True, args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=img_input_dim)

        try:
            self.rnn_type =  args.rnn_type
        except Exception as e:
            self.rnn_type = 'lstm'

        if self.rnn_type=='lstm':
            self.attention_rnn = nn.LSTM(feature_dim, feature_dim,batch_first=True)
        elif self.rnn_type=='gru':
            self.attention_rnn = nn.GRU(feature_dim, feature_dim,batch_first=True)

        self.maxpool = nn.MaxPool2d(img_input_dim)

        try:
            self.subtractive_rnn = args.subtractive_rnn
        except Exception as e:
            self.subtractive_rnn = False

        try:
            self.full_recurrence = args.full_recurrence
        except Exception as e:
            self.full_recurrence = True

        try:
            self.subtract_from_scene = args.subtract_from_scene
        except Exception as e:
            self.subtract_from_scene = True
            
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

        queries = self.get_queries(fused_object_coords_batched, batch_size, max_num_objects)

        if not self.subtractive_rnn:
            attention_map_batched = torch.einsum("bij,bjkl -> bikl", queries,fused_object_coords_batched)
            attention_map_batched = nn.Softmax(2)(attention_map_batched.reshape(batch_size,max_num_objects,-1)).view_as(attention_map_batched)
            object_values_batched = torch.einsum("bijk,bljk -> bil", attention_map_batched, fused_object_coords_batched) 
            

        else:
            object_representations = []
            remaining_scene = fused_object_coords_batched

            
            for query in queries:

                if not self.full_recurrence:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,fused_object_coords_batched)
                else:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,remaining_scene)
                attention_map_batched = nn.Softmax(1)(attention_map_batched.reshape(batch_size,-1)).view_as(attention_map_batched)

                if self.subtract_from_scene:
                    object_values = torch.einsum("bjk,bljk -> bl", attention_map_batched, remaining_scene) 
                else:
                    object_values = torch.einsum("bjk,bljk -> bl", attention_map_batched, fused_object_coords_batched) 
                object_representations.append(object_values)

                weighted_scene = torch.einsum("bjk,bljk -> bljk", attention_map_batched, remaining_scene) 

                remaining_scene = remaining_scene - weighted_scene

            object_values_batched = torch.stack(object_representations,dim=1)

        object_representations_batched = self._norm(self.object_features_layer(object_values_batched))
        object_pair_representations_batched = self.objects_to_pair_representations(object_representations_batched)


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs

    def get_queries(self,fused_object_coords,batch_size,max_num_objects):

        if not self.subtractive_rnn:
            rnn_input = self.maxpool(fused_object_coords).squeeze(-1).squeeze(-1)
            rnn_input = rnn_input.unsqueeze(1).expand(-1,max_num_objects,-1)
            queries,_ = self.attention_rnn(rnn_input)
            return queries

        else:
            device = fused_object_coords.device

            
            if self.rnn_type == 'lstm':
                h = torch.zeros(1,batch_size,self.feature_dim).to(device), torch.zeros(1,batch_size,self.feature_dim).to(device)
            else:
                h = torch.zeros(1,batch_size,self.feature_dim).to(device)

            query_list = []

            remaining_scene = fused_object_coords

            for i in range(max_num_objects):

                scene_representation = self.maxpool(remaining_scene).squeeze(-1).squeeze(-1).unsqueeze(1)
                
                output, h = self.attention_rnn(scene_representation,h)

                query = output.view(batch_size,-1)

                if not self.full_recurrence:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,fused_object_coords)
                else:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,remaining_scene)

                attention_map_batched = nn.Softmax(1)(attention_map_batched.reshape(batch_size,-1)).view_as(attention_map_batched)

                weighted_scene = torch.einsum("bjk,bljk -> bljk", attention_map_batched, remaining_scene) 

                remaining_scene = remaining_scene - weighted_scene


                query_list.append(query)



            #queries = torch.stack(query_list,dim=1)

            return query_list


    def test_batching(self,input, objects, objects_length):

        def get_queries(fused_object_coords,num_objects):
            rnn_input = self.maxpool(fused_object_coords.unsqueeze(0)).squeeze(-1).squeeze(-1)
            rnn_input = rnn_input.unsqueeze(1).expand(-1,num_objects,-1)
            queries,_ = self.attention_rnn(rnn_input)
            queries = torch.squeeze(queries,dim=0)
            return queries

        def objects_to_pair_representations(object_representations):
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

        object_features = input
        


       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device)
        
        for i in range(input.size(0)):
            single_scene_object_features =  torch.squeeze(object_features[i,:],dim=0) #dim=256 x 16 x 24
            scene_object_coords = torch.unsqueeze(torch.cat((single_scene_object_features,obj_coord_map),dim=0),dim=0)

            fused_object_coords = torch.squeeze(self.object_coord_fuse(scene_object_coords),dim=0) #dim=256 x Z x Y


            num_objects = objects_length[i]

            queries = get_queries(fused_object_coords,num_objects)



            attention_map = torch.einsum("ij,jkl -> ikl", queries,fused_object_coords) #dim=num_objects x Z x Y
            attention_map = nn.Softmax(1)(attention_map.view(num_objects,-1)).view_as(attention_map)
            object_values = torch.einsum("ijk,ljk -> il", attention_map, fused_object_coords) #dim=num_objects x 256

            object_representations = self._norm(self.object_features_layer(object_values))

            object_pair_representations = objects_to_pair_representations(object_representations)


            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])

        return outputs

class PositionalEncodingDecoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncodingDecoder, self).__init__()
        self.div_term = torch.exp(torch.arange(0, d_model/2, 2).float() * (-math.log(10000.0) / (d_model/2)))
        self.dropout = nn.Dropout(p=dropout)

        self.max_len = max_len
        self.d_model = d_model

        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        
        #self.register_buffer('pe', pe)

    def forward(self, x):
        device = x.device

        # 

        pe_x = torch.zeros(self.max_len, (int(self.d_model/2)))
        position_x = torch.randint(0, 24,(self.max_len,), dtype=torch.float).unsqueeze(1)
        pe_x[:, 0::2] = torch.sin(position_x * self.div_term)
        pe_x[:, 1::2] = torch.cos(position_x * self.div_term)
        pe_x = pe_x.unsqueeze(0).transpose(0, 1).to(device)

        pe_y = torch.zeros(self.max_len,  (int(self.d_model/2)))
        position_y = torch.randint(0, 16,(self.max_len,), dtype=torch.float).unsqueeze(1)
        pe_y[:, 0::2] = torch.sin(position_y * self.div_term)
        pe_y[:, 1::2] = torch.cos(position_y * self.div_term)
        pe_y = pe_y.unsqueeze(0).transpose(0, 1).to(device)


        pe = torch.cat((pe_x,pe_y),dim=2)


        x = x + pe[:x.size(0), :]
        return x


class PositionalEncodingEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncodingEncoder, self).__init__()
        self.div_term = torch.exp(torch.arange(0, d_model/2, 2).float() * (-math.log(10000.0) / (d_model/2)))
        self.dropout = nn.Dropout(p=dropout)

        self.max_len = max_len
        self.feature_dim = int(d_model/2)


        pe_x = torch.zeros(24, self.feature_dim)
        position_x = torch.arange(0, 24, dtype=torch.float).unsqueeze(1)
        pe_x[:, 0::2] = torch.sin(position_x * self.div_term)
        pe_x[:, 1::2] = torch.cos(position_x * self.div_term)
        #pe_x is now (16,d_model/2)
        pe_x = pe_x.unsqueeze(0).expand(torch.Size((16,24,self.feature_dim))).unsqueeze(0)

        

        pe_y = torch.zeros(16, self.feature_dim)
        position_y= torch.arange(0, 16, dtype=torch.float).unsqueeze(1)
        pe_y[:, 0::2] = torch.sin(position_y * self.div_term)
        pe_y[:, 1::2] = torch.cos(position_y * self.div_term)
        #pe_y is now (24,d_model/2)
        pe_y = pe_y.unsqueeze(1).expand(torch.Size((16,24,self.feature_dim))).unsqueeze(0)

        pe = torch.cat((pe_x,pe_y),dim=3)
        pe = pe.permute(0,3,1,2)

        self.register_buffer('pe', pe)

    def forward(self, x):



        x = x + self.pe
        return x


class TransformerSceneGraph(NaiveRNNSceneGraphBatchedBase):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True, args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=img_input_dim)

        self.use_queries = args.transformer_use_queries

        self.positional_encoding_decoder = PositionalEncodingDecoder(d_model=feature_dim)
        self.positional_encoding_encoder = PositionalEncodingEncoder(d_model=feature_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8,dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8,dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.maxpool = nn.MaxPool2d(img_input_dim)
            
    def forward(self, input, objects, objects_length):
        object_features = input
        

        batch_size = input.size(0)
        
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        

        




        object_values_batched  = self.get_objects(object_features, batch_size, objects_length)



        object_representations_batched = self._norm(self.object_features_layer(object_values_batched))
        object_pair_representations_batched = self.objects_to_pair_representations(object_representations_batched)


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs

    def get_objects(self,object_features,batch_size,objects_length):
        max_num_objects = max(objects_length)
        object_mask = torch.zeros(batch_size,max_num_objects)
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_mask[i,num_objects:] = 1
        object_mask = object_mask.bool().to(object_features.device)


        fused_object_coords = self.positional_encoding_encoder(object_features)

        transformer_memory = fused_object_coords.reshape(fused_object_coords.size(0),fused_object_coords.size(1),-1)

        #permute to (row x col) x batch x feature
        transformer_memory = transformer_memory.permute(2,0,1)

        transformer_memory = self.transformer_encoder(transformer_memory)

        # if self.roi_pool:
        #     boxes = []
        #     for i in range(batch_size):
        #         for j in range(max_num_objects):
        #             x_left
        #             boxes.append([i,random.randint(0,15),random.randint(0,23)])
        # else:
        transformer_input = self.maxpool(fused_object_coords).squeeze(-1).squeeze(-1)
        transformer_input = transformer_input.unsqueeze(0).expand(max_num_objects,-1,-1)
        transformer_input = self.positional_encoding_decoder(transformer_input)
        transformer_output = self.transformer_decoder(transformer_input,transformer_memory,tgt_key_padding_mask=object_mask)


        if self.use_queries:
            queries = transformer_output.permute(1,0,2)
            attention_map_batched = torch.einsum("bij,bjkl -> bikl", queries,fused_object_coords)
            attention_map_batched = nn.Softmax(2)(attention_map_batched.reshape(batch_size,max_num_objects,-1)).view_as(attention_map_batched)
            objects = torch.einsum("bijk,bljk -> bil", attention_map_batched, fused_object_coords) 
        #permute back to batch x max_num_objects x feature
        else:
            objects = transformer_output.permute(1,0,2)

        return objects

        




class StructuredRNNSceneGraphBatched(NaiveRNNSceneGraphBatchedBase):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=img_input_dim)

        self.attention_rnn = nn.LSTM(2*feature_dim, feature_dim,batch_first=True)
        self.maxpool = nn.MaxPool2d(img_input_dim)


    def forward(self, input, objects, objects_length, epoch):
        object_features = input
        

        batch_size = input.size(0)
        max_num_objects = max(objects_length)
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = torch.unsqueeze(coord_map((object_features.size(2),object_features.size(3)),object_features.device),dim=0)

        obj_coord_map_batched = obj_coord_map.repeat(batch_size,1,1,1)

        scene_object_coords_batched = torch.cat((object_features,obj_coord_map_batched), dim=1)

        fused_object_coords_batched = self.object_coord_fuse(scene_object_coords_batched)

        object_values_batched = self.get_queries(fused_object_coords_batched, batch_size, max_num_objects,epoch)

        object_representations_batched = self._norm(self.object_features_layer(object_values_batched))

        object_pair_representations_batched = self.objects_to_pair_representations(object_representations_batched)


        outputs = []
        for i in range(batch_size):
            #get rid of any excess objects
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs


    def get_queries(self,fused_object_coords,batch_size,max_num_objects,epoch):
        device = fused_object_coords.device

        scene_representation = self.maxpool(fused_object_coords).squeeze(-1).squeeze(-1)

        object_representation = torch.zeros(batch_size,self.feature_dim).to(device)
        h,c = torch.zeros(1,batch_size,self.feature_dim).to(device), torch.zeros(1,batch_size,self.feature_dim).to(device)

        query_list = []
        object_list = []

        for i in range(max_num_objects):
            rnn_input = torch.cat((scene_representation,object_representation),dim=1).unsqueeze(1)
            output, (h,c) = self.attention_rnn(rnn_input,(h,c))

            query = output.view(batch_size,-1)
            attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,fused_object_coords)

            #reordered_object_coords = fused_object_coords.permute(0,2,3,1)
            #query_for_mul = query.unsqueeze(-1).unsqueeze(1)
            #attention_map_batched = torch.matmul(reordered_object_coords,query_for_mul).squeeze(-1)
            
            attention_map_batched = nn.Softmax(1)(attention_map_batched.reshape(batch_size,-1)).view_as(attention_map_batched)


            #attention_map_batched_reshape = attention_map_batched.view(batch_size,-1).unsqueeze(-1)
            #fused_object_coords_reshape = fused_object_coords.view(batch_size,self.feature_dim,-1)
            #object_representation = torch.matmul(fused_object_coords_reshape,attention_map_batched_reshape).squeeze(-1)
            
            object_representation = torch.einsum("bjk,bljk -> bl", attention_map_batched, fused_object_coords) 

            query_list.append(query)
            object_list.append(object_representation)



        queries = torch.stack(query_list,dim=1)
        objects = torch.stack(object_list,dim=1)

        #return queries

        return objects

    def compute_attention(self,input, objects, objects_length, epoch):
        object_features = input
        

        batch_size = input.size(0)
        max_num_objects = max(objects_length)
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        obj_coord_map = torch.unsqueeze(coord_map((object_features.size(2),object_features.size(3)),object_features.device),dim=0)

        obj_coord_map_batched = obj_coord_map.repeat(batch_size,1,1,1)

        scene_object_coords_batched = torch.cat((object_features,obj_coord_map_batched), dim=1)

        fused_object_coords_batched = self.object_coord_fuse(scene_object_coords_batched)

        queries = self.get_queries(fused_object_coords_batched, batch_size, max_num_objects, epoch)

        attention_map_batched = torch.einsum("bij,bjkl -> bikl", queries,fused_object_coords_batched)
        attention_map_batched = nn.Softmax(2)(attention_map_batched.reshape(batch_size,max_num_objects,-1)).view_as(attention_map_batched)

        return attention_map_batched


class StructuredSubtractiveRNNSceneGraphBatched(StructuredRNNSceneGraphBatched):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=img_input_dim)


        def get_queries(self,fused_object_coords,batch_size,max_num_objects,epoch):
            device = fused_object_coords.device

            scene_representation = self.maxpool(fused_object_coords).squeeze(-1).squeeze(-1)

            remaining_scene_representation = scene_representation
            identity_attention = torch.ones(batch_size,fused_object_coords.size(2),fused_object_coords.size(3)).to(device)
            remaining_attention = torch.ones(batch_size,fused_object_coords.size(2),fused_object_coords.size(3)).to(device)

            h,c = torch.zeros(1,batch_size,self.feature_dim).to(device), torch.zeros(1,batch_size,self.feature_dim).to(device)

            query_list = []
            object_list = []

            for i in range(max_num_objects):
                rnn_input = torch.cat((scene_representation,remaining_scene_representation),dim=1).unsqueeze(1)
                output, (h,c) = self.attention_rnn(rnn_input,(h,c))

                query = output.view(batch_size,-1)
                attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,fused_object_coords)
                

                #reordered_object_coords = fused_object_coords.permute(0,2,3,1)
                #query_for_mul = query.unsqueeze(-1).unsqueeze(1)
                #attention_map_batched = torch.matmul(reordered_object_coords,query_for_mul).squeeze(-1)
                
                attention_map_batched = nn.Softmax(1)(attention_map_batched.reshape(batch_size,-1)).view_as(attention_map_batched)
                

                remaining_attention = remaining_attention*(identity_attention-attention_map_batched)
                weighted_scene  = torch.einsum("bjkl,bkl -> bjkl",fused_object_coords,remaining_attention)
                remaining_scene_representation = self.maxpool(weighted_scene).squeeze(-1).squeeze(-1)



                #attention_map_batched_reshape = attention_map_batched.view(batch_size,-1).unsqueeze(-1)
                #fused_object_coords_reshape = fused_object_coords.view(batch_size,self.feature_dim,-1)
                #object_representation = torch.matmul(fused_object_coords_reshape,attention_map_batched_reshape).squeeze(-1)
                
                object_representation = torch.einsum("bjk,bljk -> bl", attention_map_batched, fused_object_coords) 

                query_list.append(query)
                object_list.append(object_representation)



            queries = torch.stack(query_list,dim=1)
            objects = torch.stack(object_list,dim=1)

            #return queries

            return objects


class SceneGraphObjectSupervision(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=True,concatenative_pair_representation=True,args=None,img_input_dim=(16,24)):
        super().__init__()
        self.pool_size = 7
        self.feature_dim = feature_dim
        self.output_dims = output_dims
        self.downsample_rate = downsample_rate

        self.object_supervision = object_supervision
        self.concatenative_pair_representation = concatenative_pair_representation

        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
        self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)

        self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
        self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)

        self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))
        self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))

        self.obj1_linear = nn.Linear(output_dims[1],int(output_dims[1]))
        self.obj2_linear = nn.Linear(output_dims[1],int(output_dims[1]))


        #self.combine_objects = nn.Linear(2*output_dims[1],output_dims[1])

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

            object_pair_representations = self._norm(self.objects_to_pair_representations(object_representations))


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

        object_pair_representations = obj1_representations + obj2_representations
        #object_pair_representations = self.combine_objects(object_pair_representations)


        return object_pair_representations

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim,padding=2,kernel_size=5,pool=False):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        if pool:
            self.maxpool = nn.MaxPool2d((16,24))
        else:
            self.maxpool = nn.Identity()
            self.globalpool = nn.Identity()

        self.residual_conv = nn.Conv2d(inp_dim, out_dim, padding=0, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(inp_dim, inp_dim, padding=padding, kernel_size=kernel_size, bias=True)
        self.conv2 = nn.Conv2d(inp_dim,inp_dim, padding=padding, kernel_size=kernel_size, bias=True)
        self.conv3 = nn.Conv2d(inp_dim, inp_dim, padding=padding, kernel_size=kernel_size, bias=True)
        self.conv4 = nn.Conv2d(inp_dim, out_dim, padding=padding, kernel_size=kernel_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = out + residual
        out = self.maxpool(out)

        
        return out 

class AttentionNet(nn.Module):
    def __init__(self, inp_dim):
        super(AttentionNet, self).__init__()
        self.fc_attention = nn.Sequential(nn.Linear(inp_dim, inp_dim),nn.ReLU(),nn.Linear(inp_dim,4))

        self.X = 16
        self.Y = 24

        self.epsilon=1e-5
    
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        device = x.device

        x = x.reshape(x.size(0),-1)
        params = self.fc_attention(x)
        gx_, gy_, log_sigma_x, log_sigma_y = params.split(1, 1)

        gx = torch.sigmoid(gx_)
        gy = torch.sigmoid(gy_)

        #gx = gx_
        #gy = gy_
        print(gx[0:10,:])
        

        sigma = 0.2*torch.sigmoid(log_sigma_x/2)
        #sigma_y = 0.2*torch.sigmoid(log_sigma_y/2)

        a = torch.linspace(0.0, 1.0, steps=self.X, device=device).view(1, -1)
        b = torch.linspace(0.0, 1.0, steps=self.Y, device=device).view(1, -1)

        Fx = torch.exp(-torch.pow(a - gx, 2) / sigma) #should be batchx16
        Fy = torch.exp(-torch.pow(b - gy, 2) / sigma) #should be batchx24


        
        
        Fx = Fx / (Fx.sum(1, True).expand_as(Fx) + self.epsilon)
 
        Fy = Fy / (Fy.sum(1, True).expand_as(Fy) + self.epsilon)

        attention = torch.einsum("bx,by -> bxy",Fx,Fy)

        




        return attention 



class LocalAttentionNet(nn.Module):
    def __init__(self, inp_dim, out_dim,padding=1,kernel_size=3,pool=False):
        super(LocalAttentionNet, self).__init__()
        self.relu = nn.ReLU()
        if pool:
            self.maxpool = nn.MaxPool2d((16,24))
        else:
            self.maxpool = nn.Identity()
            self.globalpool = nn.Identity()

        self.residual_conv = nn.Conv2d(inp_dim, out_dim, padding=0, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(inp_dim, inp_dim, padding=padding, kernel_size=kernel_size, bias=True)
        #self.norm = nn.InstanceNorm2d(out_dim,affine=True)
        self.conv2 = nn.Conv2d(inp_dim,inp_dim, padding=padding, kernel_size=kernel_size, bias=True)
        self.conv3 = nn.Conv2d(inp_dim,out_dim, padding=padding, kernel_size=kernel_size, bias=True)

        self.last_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                #m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                #m.bias.data.zero_()

        self.last_conv.bias.data.fill_(-2.19)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        #out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out + residual
        out = self.last_conv(out)
        #
        
        return out 

class TransformerCNN(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True, args=None,img_input_dim=(16,24)):
        super().__init__()
        self.object_dropout = args.object_dropout
        self.dropout_rate = args.object_dropout_rate
        self.normalize_objects = args.normalize_objects


        self.feature_dim = feature_dim
        self.output_dims = output_dims
        num_heads = 1
        self.attention_net_1 = LocalAttentionNet(self.feature_dim+2,num_heads,padding=2,kernel_size=5)
        self.attention_net_2 = LocalAttentionNet(self.feature_dim+1+2*num_heads,num_heads, padding=2, kernel_size=5)
        self.attention_net_3 = LocalAttentionNet(self.feature_dim+1+2*num_heads,num_heads, padding=2, kernel_size=5)
        
        #self.attention_net_4 = LocalAttentionNet(self.feature_dim+1+2*num_heads,1, padding=2, kernel_size=5)

        self.foreground_detector = LocalAttentionNet(self.feature_dim,1, padding=2, kernel_size=5)

        #self.object_net = Residual(self.feature_dim+3,self.feature_dim,padding=0,kernel_size=1,pool=True)
        
        self.maxpool = nn.MaxPool2d(3,padding=1,stride=1)
        #self.shared_feature_net = nn.Sequential(nn.Conv2d(feature_dim,feature_dim,kernel_size=1), nn.ReLU(),
        #    nn.Conv2d(feature_dim,feature_dim,kernel_size=1), nn.ReLU())

        #self.feature_net = Residual(feature_dim, feature_dim, padding=0, kernel_size=1)

        #self.object_features_layer = nn.Sequential(nn.Linear(feature_dim,output_dims[1]),nn.ReLU())
        self.obj1_linear = nn.Linear(output_dims[1],int(output_dims[1]))
        self.obj2_linear = nn.Linear(output_dims[1],int(output_dims[1]))
        #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                #m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                #m.bias.data.zero_()

        #self.attention_net_3.conv2.bias.data.fill_(-2.19)

    def sample_init(self,objects_length):
        x_pos = []
        y_pos = []
        max_length = max(objects_length)
        dist = 4
        x_pos.append(random.randint(0,16))
        y_pos.append(random.randint(0,24))
        for i in range(max_length-1):
            condition = lambda a,b: False
            while not all(map(condition,x_pos,y_pos)):
                x = random.randint(0,16)
                y = random.randint(0,24)
                condition = lambda a,b: (x-a)^2+(y-b)^2 <= dist^2
            x_pos.append(x)
            y_pos.append(y)

        return x_pos, y_pos




    def local_max(self,attention_map,objects_length):
        batch_size = attention_map.size(0)
        k = max(objects_length)
        objects_length = torch.tensor(objects_length)

        map_local_max = self.maxpool(attention_map)
        map_local_max = torch.eq(attention_map,map_local_max)
        map_local_max = attention_map * map_local_max.int().float()


        top_k_indices = torch.topk(map_local_max.view(batch_size,-1),k)[1]

        m_x, m_y = torch.meshgrid(torch.arange(16),torch.arange(24))
        m_x = m_x.to(attention_map.device).float()
        m_y = m_y.to(attention_map.device).float()
        

        

        #print(objects_length)
        
        
        sigma = 2

        indicator_maps = []

        #x_pos_all, y_pos_all = self.sample_init(objects_length)
        #print(top_k_indices)
        for i in range(k):
            #print(i)
            if True:
                indicator_map = torch.zeros(attention_map.size()).view(batch_size,-1).to(attention_map.device)
                indices = top_k_indices[:,i].unsqueeze(1)
                indicator_map = indicator_map.scatter_(1,indices,1).view_as(attention_map)

                x_pos = torch.einsum("bijk,jk -> b",indicator_map,m_x).view(batch_size,1,1,1)
                y_pos = torch.einsum("bijk,jk -> b",indicator_map,m_y).view(batch_size,1,1,1)

                #m_x = m_x.view(1,1,16,24)
                #m_y = m_y.view(1,1,16,24)

            else:
                x_pos = torch.tensor(x_pos_all[i], dtype=torch.float).expand(batch_size,1,1,1).to(attention_map.device)
                y_pos = torch.tensor(y_pos_all[i], dtype=torch.float).expand(batch_size,1,1,1).to(attention_map.device)

            #print(x_pos)

            Fx = -torch.pow(x_pos - m_x, 2) / sigma 
            Fy = -torch.pow(y_pos - m_y, 2) / sigma

            probs = Fx+Fy
            probs = probs - probs.logsumexp(dim=(2,3),keepdim=True)

            #print(probs)



            indicator_maps.append(probs)
        
        return indicator_maps



            
    def forward(self, input, objects, objects_length):
        object_features = input
        

        batch_size = input.size(0)
        
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        

    



        object_values_batched  = self.get_objects(object_features, batch_size, objects_length)

        if self.normalize_objects:
            object_representations_batched = self._norm(object_values_batched)
        else:
            object_representations_batched = object_values_batched
        #object_representations_batched = self._norm(self.object_features_layer(object_values_batched))
        object_pair_representations_batched = self._norm(self.objects_to_pair_representations(object_representations_batched))
        


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0)

            if self.training:
                if self.object_dropout:
                    #if random.random()<self.dropout_rate:
                    #    index = random.randrange(num_objects)
                    #    object_representations[index,:]=0
                    for j in range(num_objects):
                        if random.random()<self.dropout_rate:
                            object_representations = object_representations.index_fill(0,torch.tensor(j).to(object_representations.device),0)
                            #object_pair_representations[j,:,:]=0
                            #object_pair_representations[:,j,:]=0

            #object_pair_representations = self._norm(self.objects_to_pair_representations(object_representations))
            
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs

    def transformer_layer_start(self,feature_map,foreground_map, indicators):
        attentions = []
        foreground_map = F.logsigmoid(foreground_map)

        for indicator_map in indicators:
            filtered_foreground = indicator_map
            rep = torch.cat((feature_map,foreground_map,filtered_foreground),dim=1)
            attention = self.attention_net_1(rep)
            attentions.append(attention)
        return attentions

    def transformer_layer(self,feature_map,foreground_map, attentions,attention_net, objects_length):
        max_len = max(objects_length)
        objects_length = torch.tensor(objects_length)
        mask = (torch.arange(max_len).expand(len(objects_length), max_len) < objects_length.unsqueeze(1)).to(feature_map.device)

        foreground_map = F.logsigmoid(foreground_map)

        batch_size = feature_map.size(0)
        sum_scope = torch.zeros(batch_size,1,1,1).to(feature_map.device)

        for i in range(len(attentions)):
            attention = attentions[i]
            log_probs = F.logsigmoid(-attention)
            log_probs = torch.einsum("bljk,b -> bljk",log_probs,mask[:,i])
            sum_scope = sum_scope + log_probs

        new_attentions = []
        for attention in attentions:
            scope = sum_scope - F.logsigmoid(-attention)
            rep = torch.cat((feature_map,foreground_map,attention,scope),dim=1)
            new_attention = attention_net(rep)
            new_attentions.append(new_attention)

        return new_attentions




    def get_objects(self,object_features,batch_size,objects_length):
        max_num_objects = max(objects_length)
        #obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device).unsqueeze(0)
        #obj_coord_map = obj_coord_map.repeat(batch_size,1,1,1)

        #object_coord_cat = torch.cat((object_features,obj_coord_map),dim=1)
        foreground_map = self.foreground_detector(object_features)

        #foreground_map = foreground_features_fused[:,0,:,:].unsqueeze(1)
        #collapsed_features = foreground_features_fused[:,1:,:,:]
        foreground_attention = torch.sigmoid(foreground_map).squeeze(1)
        foreground = torch.einsum("bjk,bljk -> bljk", foreground_attention, object_features)


        


        #init_scope = torch.zeros((1, 1, 16, 24)).to(object_features.device)
        #log_scope = init_scope.expand(batch_size, -1, -1, -1)
        #log_scope = F.logsigmoid(foreground_map)

        object_representations = []

        indicators = self.local_max(foreground_map,objects_length)

        if True:
            attentions = self.transformer_layer_start(object_features,foreground_map,indicators)
        else:
            attentions = torch.normal(-1,1,size=(batch_size,1,16,24)).to(object_features.device)
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_2,objects_length)
        #
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_3,objects_length)
        #attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_4)
        #print(self.attention_net_4.conv1.weight.data)

        for slot in range(max_num_objects):
            
            attention = F.sigmoid(attentions[slot])
            attention = attention.squeeze(1)
            #else:
            #    attention = torch.exp(log_scope).squeeze(1)

            #attention = attention.squeeze(1)
            objects = torch.einsum("bjk,bljk -> bl", attention, foreground)
            #objects = self.maxpool(obj_cols_weighted).squeeze(-1).squeeze(-1)
            
            object_representations.append(objects)

        object_representations = torch.stack(object_representations,dim=1)
        return object_representations

    def compute_attention(self,object_features,objects,objects_length,visualize_foreground=False):
        max_num_objects = max(objects_length)
        #obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device).unsqueeze(0)
        #obj_coord_map = obj_coord_map.repeat(batch_size,1,1,1)

        #object_coord_cat = torch.cat((object_features,obj_coord_map),dim=1)
        foreground_map = self.foreground_detector(object_features)

        #foreground_map = foreground_features_fused[:,0,:,:].unsqueeze(1)
        #collapsed_features = foreground_features_fused[:,1:,:,:]
        foreground_attention = torch.sigmoid(foreground_map).squeeze(1)
        foreground = torch.einsum("bjk,bljk -> bljk", foreground_attention, object_features)


        


        #init_scope = torch.zeros((1, 1, 16, 24)).to(object_features.device)
        #log_scope = init_scope.expand(batch_size, -1, -1, -1)
        #log_scope = F.logsigmoid(foreground_map)

        attention_list = []

        indicators = self.local_max(foreground_map,objects_length)

        if True:
            attentions = self.transformer_layer_start(object_features,foreground_map,indicators)
        else:
            attentions = torch.normal(-1,1,size=(batch_size,1,16,24)).to(object_features.device)
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_2,objects_length)
        #
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_3,objects_length)
        #attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_4)
        #print(self.attention_net_4.conv1.weight.data)

        for slot in range(max_num_objects):
            
            attention = F.sigmoid(attentions[slot])
            attention = attention.squeeze(1)
            #else:
            #    attention = torch.exp(log_scope).squeeze(1)

            #attention = attention.squeeze(1)
            attention_list.append(attention)

        attention_list = torch.stack(attention_list,dim=1)
        return attention_list


        

    def objects_to_pair_representations(self, object_representations_batched):
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
        #object_pair_representations = object_pair_representations

        return object_pair_representations
    

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)

     
class MonetLiteSceneGraph(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True, args=None,img_input_dim=(16,24)):
        super().__init__()
        self.object_dropout = args.object_dropout
        self.feature_dim = feature_dim
        self.output_dims = output_dims
        self.collapsed_dim = 16
        self.attention_net = Residual(self.feature_dim+3,1,padding=2,kernel_size=5)
        #self.attention_net = AttentionNet(int(feature_dim/64)*2)

        self.foreground_detector = Residual(self.feature_dim+2,1, padding=1, kernel_size=3)

        #self.object_net = Residual(self.feature_dim+3,self.feature_dim,padding=0,kernel_size=1,pool=True)
        
        #self.maxpool = nn.MaxPool2d((16,24))
        #self.shared_feature_net = nn.Sequential(nn.Conv2d(feature_dim,feature_dim,kernel_size=1), nn.ReLU(),
        #    nn.Conv2d(feature_dim,feature_dim,kernel_size=1), nn.ReLU())

        #self.feature_net = Residual(feature_dim, feature_dim, padding=0, kernel_size=1)

        #self.object_features_layer = nn.Sequential(nn.Linear(feature_dim,output_dims[1]),nn.ReLU())
        self.obj1_linear = nn.Linear(output_dims[1],int(output_dims[1]/2))
        self.obj2_linear = nn.Linear(output_dims[1],int(output_dims[1]/2))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

        self.attention_net.conv4.bias.data.fill_(-2.19)
            
    def forward(self, input, objects, objects_length):
        object_features = input
        

        batch_size = input.size(0)
        
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        

    



        object_values_batched  = self.get_objects(object_features, batch_size, objects_length)


        object_representations_batched = self._norm(object_values_batched)
        #object_representations_batched = self._norm(self.object_features_layer(object_values_batched))
        object_pair_representations_batched = self._norm(self.objects_to_pair_representations(object_representations_batched))


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)

            if self.training:
                if self.object_dropout:
                    if random.random()<0.1:
                        index = random.randrange(num_objects)
                        object_representations[index,:]=0

            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs

    def get_objects(self,object_features,batch_size,objects_length):
        max_num_objects = max(objects_length)
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device).unsqueeze(0)
        obj_coord_map = obj_coord_map.repeat(batch_size,1,1,1)

        object_coord_cat = torch.cat((object_features,obj_coord_map),dim=1)
        foreground_map = self.foreground_detector(object_coord_cat)

        #foreground_map = foreground_features_fused[:,0,:,:].unsqueeze(1)
        #collapsed_features = foreground_features_fused[:,1:,:,:]
        foreground_attention = torch.sigmoid(foreground_map).squeeze(1)
        foreground = torch.einsum("bjk,bljk -> bljk", foreground_attention, object_features)


        


        #init_scope = torch.zeros((1, 1, 16, 24)).to(object_features.device)
        #log_scope = init_scope.expand(batch_size, -1, -1, -1)
        log_scope = F.logsigmoid(foreground_map)

        object_representations = []

        for slot in range(max_num_objects):
            #if slot < max_num_objects - 1:
            log_attention = self.attention_net(torch.cat((foreground,log_scope,obj_coord_map),dim=1))
            log_scope = log_scope + F.logsigmoid(-log_attention)
            attention = F.sigmoid(log_attention).squeeze(1)
            #else:
            #    attention = torch.exp(log_scope).squeeze(1)

            #attention = attention.squeeze(1)
            objects = torch.einsum("bjk,bljk -> bl", attention, foreground)
            #objects = self.maxpool(obj_cols_weighted).squeeze(-1).squeeze(-1)
            
            object_representations.append(objects)

        object_representations = torch.stack(object_representations,dim=1)
        return object_representations

    def compute_attention(self,object_features,objects,objects_length,visualize_foreground=False):
        max_num_objects = max(objects_length)
        batch_size = object_features.size(0)

        foreground_map = self.foreground_detector(object_features)
        foreground_attention = F.sigmoid(foreground_map).squeeze(1)
        foreground = torch.einsum("bjk,bljk -> bljk", foreground_attention, object_features)


        


        #init_scope = torch.zeros((1, 1, 16, 24)).to(object_features.device)
        #log_scope = init_scope.expand(batch_size, -1, -1, -1)
        log_scope = foreground_map

        attentions = []

        for slot in range(max_num_objects):
            #if slot < max_num_objects - 1:
            x = torch.cat((foreground,log_scope),dim=1)
            log_attention = self.attention_net(x)

            log_scope = log_scope + F.logsigmoid(-log_attention)
            #else:
            #    log_mask = log_scope

            attention = F.sigmoid(log_attention).squeeze(1)

            if visualize_foreground:
                attentions.append(foreground_attention)
            else:
                attentions.append(attention)

        attentions = torch.stack(attentions,dim=1)
        return attentions


        

    def objects_to_pair_representations(self, object_representations_batched):
        num_objects = object_representations_batched.size(1)

        obj1_representations = self.obj1_linear(object_representations_batched)
        obj2_representations = self.obj2_linear(object_representations_batched)


        obj1_representations.unsqueeze_(-1)#now batch_size x num_objects x feature_dim x 1
        obj2_representations.unsqueeze_(-1)

        obj1_representations = obj1_representations.transpose(2,3)
        obj2_representations = obj2_representations.transpose(2,3).transpose(1,2)

        obj1_representations = obj1_representations.repeat(1,1,num_objects,1)  
        obj2_representations = obj2_representations.repeat(1,num_objects,1,1)

        object_pair_representations = torch.cat((obj1_representations,obj2_representations),dim=-1)
        #object_pair_representations = object_pair_representations

        return object_pair_representations

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)


def coord_map(shape,device, start=0, end=1):
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

