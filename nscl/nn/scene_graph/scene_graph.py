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
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn
import math
import torchvision



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

        


class LowDimensionalRNNBatched(NaiveRNNSceneGraphBatched):
    #first maps the feature map into a low dimensional space, and then computes attention on this
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True, args=None):
        super().__init__(feature_dim, output_dims, downsample_rate)

        try:
            self.rnn_type =  args.rnn_type
        except Exception as e:
            self.rnn_type = 'lstm'

        self.projection_dim = 5
        self.attention_cnn = nn.Sequential(nn.Conv2d(feature_dim,self.projection_dim,kernel_size=1), nn.ReLU())


        if self.rnn_type=='lstm':
            self.attention_rnn = nn.LSTM(self.projection_dim, self.projection_dim,batch_first=True)
        elif self.rnn_type=='gru':
            self.attention_rnn = nn.GRU(self.projection_dim, self.projection_dim,batch_first=True)

        self.maxpool = nn.MaxPool2d((16,24))

        try:
            self.subtractive_rnn = args.subtractive_rnn
        except Exception as e:
            self.subtractive_rnn = False

        try:
            self.full_recurrence = args.full_recurrence
        except Exception as e:
            self.full_recurrence = True

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

            low_dim_scene = self.attention_cnn(fused_object_coords_batched)
            remaining_scene = low_dim_scene

            
            for query in queries:

                if not self.full_recurrence:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,low_dim_scene)
                else:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,remaining_scene)
                attention_map_batched = nn.Softmax(1)(attention_map_batched.reshape(batch_size,-1)).view_as(attention_map_batched)
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
                h = torch.zeros(1,batch_size,self.projection_dim).to(device), torch.zeros(1,batch_size,self.projection_dim).to(device)
            else:
                h = torch.zeros(1,batch_size,self.projection_dim).to(device)

            query_list = []

            low_dim_scene = self.attention_cnn(fused_object_coords)


            remaining_scene = low_dim_scene

            for i in range(max_num_objects):

                scene_representation = self.maxpool(remaining_scene).squeeze(-1).squeeze(-1).unsqueeze(1)
                
                output, h = self.attention_rnn(scene_representation,h)

                query = output.view(batch_size,-1)

                if not self.full_recurrence:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,low_dim_scene)
                else:
                    attention_map_batched = torch.einsum("bj,bjkl -> bkl", query,remaining_scene)

                attention_map_batched = nn.Softmax(1)(attention_map_batched.reshape(batch_size,-1)).view_as(attention_map_batched)

                weighted_scene = torch.einsum("bjk,bljk -> bljk", attention_map_batched, remaining_scene) 

                remaining_scene = remaining_scene - weighted_scene


                query_list.append(query)



            #queries = torch.stack(query_list,dim=1)

            return query_list



class NaiveRNNSceneGraphGlobalBatched(NaiveRNNSceneGraphBatchedBase):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True):
        super().__init__(feature_dim, output_dims, downsample_rate)

        self.maxpool = nn.MaxPool2d((16,24))

        self.object_fc = nn.Sequential(nn.Linear(2*self.feature_dim,self.output_dims[1]),nn.ReLU())
        

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

        global_context = self.maxpool(fused_object_coords_batched).squeeze(-1).squeeze(-1)
        global_context = global_context.unsqueeze(1).repeat(1,max_num_objects,1)
        object_global_cat = torch.cat((object_values_batched,global_context),dim=2)

        object_representations_batched = self._norm(self.object_fc(object_global_cat))


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


            num_objects = objects_length[i]


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


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv2d(inp_dim, 1, padding=2, kernel_size=5, bias=True)
        self.conv1 = nn.Conv2d(inp_dim, int(out_dim/2), padding=3, kernel_size=7, bias=True)
        self.conv2 = nn.Conv2d(int(out_dim/2), int(out_dim/2), padding=3, kernel_size=7, bias=True)
        self.conv4 = nn.Conv2d(int(out_dim/2), 1, padding=2, kernel_size=5, bias=True)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.relu(out)
        #out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = out + residual
        return out 

        
class MonetLiteSceneGraph(NaiveRNNSceneGraphBatchedBase):
    def __init__(self, feature_dim, output_dims, downsample_rate, object_supervision=False,concatenative_pair_representation=True, args=None,img_input_dim=(16,24)):
        super().__init__(feature_dim, output_dims, downsample_rate,args=args,img_input_dim=img_input_dim)

        self.attention_net = Residual(feature_dim+1,feature_dim+1)
        self.attention_net.conv4.bias.data.fill_(-2.19)

        self.feature_net = nn.Sequential(nn.Conv2d(feature_dim,feature_dim,kernel_size=1), nn.ReLU())
            
    def forward(self, input, objects, objects_length):
        object_features = self.feature_net(input)
        

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


        init_scope = torch.zeros((1, 1, 16, 24)).to(object_features.device)
        log_scope = init_scope.expand(batch_size, -1, -1, -1)

        object_representations = []

        for slot in range(max_num_objects):
            #if slot < max_num_objects - 1:
            x = torch.cat((object_features,log_scope),dim=1)
            log_attention = self.attention_net(x)

            log_scope = log_scope + F.logsigmoid(-log_attention)
            #else:
            #    log_mask = log_scope

            attention = F.sigmoid(log_attention).squeeze(1)

            objects = torch.einsum("bjk,bljk -> bl", attention, object_features)
            object_representations.append(objects)

        object_representations = torch.stack(object_representations,dim=1)
        return object_representations

    def compute_attention(self,input,objects,objects_length):
        object_features = self.feature_net(input)
        

        batch_size = input.size(0)

        max_num_objects = max(objects_length)
        object_mask = torch.zeros(batch_size,max_num_objects)


        init_scope = torch.zeros((1, 1, 16, 24)).to(object_features.device)
        log_scope = init_scope.expand(batch_size, -1, -1, -1)

        attentions = []

        for slot in range(max_num_objects):
            #if slot < max_num_objects - 1:
            x = torch.cat((object_features,log_scope),dim=1)
            log_attention = self.attention_net(x)

            log_scope = log_scope + F.logsigmoid(-log_attention)
            #else:
            #    log_mask = log_scope

            attention = F.sigmoid(log_attention).squeeze(1)

            attentions.append(attention)

        attentions = torch.stack(attentions,dim=1)
        return attentions



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

