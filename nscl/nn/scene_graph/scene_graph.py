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
        

    



        object_values_batched, spatial_representations_batched  = self.get_objects(object_features, batch_size, objects_length)

        if self.normalize_objects:
            object_representations_batched = self._norm(object_values_batched)
        else:
            object_representations_batched = object_values_batched
        #object_representations_batched = self._norm(self.object_features_layer(object_values_batched))
        object_pair_representations_batched = self._norm(self.objects_to_pair_representations(object_representations_batched))
        
        spatial_pair_representations_batched = self.spatial_to_pair_representations(spatial_representations_batched)

        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0)
            spatial_pair_representations = torch.squeeze(spatial_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0)

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
                        object_pair_representations,
                        spatial_pair_representations
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
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device)


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

        spatial_representations = []

        for slot in range(max_num_objects):
            
            attention = F.sigmoid(attentions[slot])
            attention = attention.squeeze(1)
            #else:
            #    attention = torch.exp(log_scope).squeeze(1)

            #attention = attention.squeeze(1)
            objects = torch.einsum("bjk,bljk -> bl", attention, foreground)
            #objects = self.maxpool(obj_cols_weighted).squeeze(-1).squeeze(-1)
            
            object_representations.append(objects)

            spatial_rep = torch.einsum("bjk,cjk -> bc", attention, obj_coord_map)
            spatial_representations.append(spatial_rep)

        object_representations = torch.stack(object_representations,dim=1)
        spatial_representations = torch.stack(spatial_representations,dim=1)

        return object_representations,spatial_representations

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
    


    def spatial_to_pair_representations(self, spatial_representations_batched):
        num_objects = spatial_representations_batched.size(1)

        obj1_representations = spatial_representations_batched
        obj2_representations = spatial_representations_batched


        obj1_representations = obj1_representations.unsqueeze(-1)#now batch_size x num_objects x 2 x 1
        obj2_representations = obj2_representations.unsqueeze(-1)


        obj1_representations = obj1_representations.transpose(2,3)
        obj2_representations = obj2_representations.transpose(2,3).transpose(1,2)

        obj1_representations = obj1_representations.repeat(1,1,num_objects,1)  
        obj2_representations = obj2_representations.repeat(1,num_objects,1,1)

        object_pair_representations = torch.cat((obj1_representations,obj2_representations),dim=3)
        #object_pair_representations = object_pair_representations

        return object_pair_representations
    

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)



class ObjectClassifier(nn.Module):
    def __init__(self, inp_dim):
        super(ObjectClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(inp_dim, inp_dim, padding=(4,0), kernel_size=5, stride=2, bias=True)
        #self.norm = nn.InstanceNorm2d(out_dim,affine=True)
        self.conv2 = nn.Conv2d(inp_dim,inp_dim, kernel_size=3, stride=2, bias=True)
        self.conv3 = nn.Conv2d(inp_dim,1, kernel_size=3, stride=2, bias=True)


        #self.reset_parameters()

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
        out = self.conv1(x)
        #out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        #
        
        return out 


class TransformerCNNObjectInference(nn.Module):
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

        #self.object_detector_rep = LocalAttentionNet(self.feature_dim,self.feature_dim,padding=1,kernel_size=3)
        self.object_classifier = ObjectClassifier(self.feature_dim+1+2*num_heads)
        #self.object_classifier = nn.Sequential(nn.Linear(self.feature_dim,1),nn.Sigmoid())


        self.object_indices = {}
        for i in range(64):
            for j in range(10):
                self.object_indices[(i,j)] = torch.tensor([[i,j,k] for k in range(self.feature_dim)])
        self.object_substitute_val = torch.ones(self.object_indices[(0,0)].shape[0])


        self.object_pair_indices_1 = {}
        for i in range(64):
            for j in range(10):
                self.object_pair_indices_1[(i,j)] = torch.tensor([[i,j,k,l] for k in range(10) for l in range(self.feature_dim)])
        self.object_pair_substitute_val = torch.ones(self.object_pair_indices_1[(0,0)].shape[0])

        self.object_pair_indices_2 = {}
        for i in range(64):
            for j in range(10):
                self.object_pair_indices_2[(i,j)] = torch.tensor([[i,k,j,l] for k in range(10) for l in range(self.feature_dim)])

    def forward(self, input, objects, objects_length):
        object_features = input
        

        batch_size = input.size(0)
        
       
        outputs = list()
        #object_features has shape batch_size x 256 x 16 x 24
        

    



        object_values_batched, object_weights  = self.get_objects(object_features, batch_size)

        if self.normalize_objects:
            object_representations_batched = self._norm(object_values_batched)
        else:
            object_representations_batched = object_values_batched
        #object_representations_batched = self._norm(self.object_features_layer(object_values_batched))
        object_pair_representations_batched = self._norm(self.objects_to_pair_representations(object_representations_batched))
        
        outputs = []
        for i in range(batch_size):
            num_objects = 10
            #object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            #object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0)
            #object_weights_scene = torch.squeeze(object_weights[i,:],dim=0)

            if self.training:
                if self.object_dropout:
                    #if random.random()<self.dropout_rate:
                    #    index = random.randrange(num_objects)
                    #    object_representations[index,:]=0

                    #epsilon = 0.0000001
                    #zeros = torch.zeros(object_weights_scene.size()).to(object_weights_scene.device)+epsilon
                    #ones = torch.ones(object_weights_scene.size()).to(object_weights_scene.device)
                    
                    #    object_weights_scene = torch.where(object_weights_scene>0.5,ones,zeros)
                    if random.random()<self.dropout_rate:
                        j = random.randrange(num_objects)
                        
                        object_representations_batched = object_representations_batched.index_put(tuple(self.object_indices[(i,j)].t().to(object_values_batched.device)),self.object_substitute_val.to(object_values_batched.device))
                        object_pair_representations_batched = object_pair_representations_batched.index_put(tuple(self.object_pair_indices_1[(i,j)].t().to(object_values_batched.device)),self.object_pair_substitute_val.to(object_values_batched.device))
                        object_pair_representations_batched = object_pair_representations_batched.index_put(tuple(self.object_pair_indices_1[(i,j)].t().to(object_values_batched.device)),self.object_pair_substitute_val.to(object_values_batched.device))
                            #object_pair_representations[j,:,:]=0
                            #object_pair_representations[:,j,:]=0

            else:
                if True:
                    threshold = 0.8
                    epsilon = 0.000000000001
                    #zeros = torch.zeros(object_weights[0].size()).to(object_weights.device)+epsilon
                    #ones = torch.ones(object_weights[0].size()).to(object_weights.device)

                    #object_weights_scene = torch.where(object_weights[i]>threshold,ones,zeros)
                    for j in range(num_objects):
                        if object_weights[i,j]<threshold:
                            object_representations_batched = object_representations_batched.index_put(tuple(self.object_indices[(i,j)].t().to(object_values_batched.device)),self.object_substitute_val.to(object_values_batched.device))
                            object_pair_representations_batched = object_pair_representations_batched.index_put(tuple(self.object_pair_indices_1[(i,j)].t().to(object_values_batched.device)),self.object_pair_substitute_val.to(object_values_batched.device))
                            object_pair_representations_batched = object_pair_representations_batched.index_put(tuple(self.object_pair_indices_1[(i,j)].t().to(object_values_batched.device)),self.object_pair_substitute_val.to(object_values_batched.device))

            #object_pair_representations = self._norm(self.objects_to_pair_representations(object_representations))
            
            
            #outputs.append([
            #            None,
            #            object_representations,
            #            object_pair_representations,
            #            object_weights_scene
            #        ])

        outputs = [None,object_representations_batched,object_pair_representations_batched,object_weights]


        return outputs


    def local_max(self,attention_map,max_num_objects):
        batch_size = attention_map.size(0)
        k = max_num_objects

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

            indicator_maps.append(probs)

        return indicator_maps

    def transformer_layer_start(self,feature_map,foreground_map, indicators):
        attentions = []
        foreground_map = F.logsigmoid(foreground_map)

        for indicator_map in indicators:
            filtered_foreground = indicator_map
            rep = torch.cat((feature_map,foreground_map,filtered_foreground),dim=1)
            attention = self.attention_net_1(rep)
            attentions.append(attention)
        return attentions

    def transformer_layer(self,feature_map,foreground_map, attentions,attention_net, max_num_objects):
        max_len = max_num_objects

        foreground_map = F.logsigmoid(foreground_map)

        batch_size = feature_map.size(0)
        sum_scope = torch.zeros(batch_size,1,1,1).to(feature_map.device)

        for i in range(len(attentions)):
            attention = attentions[i]
            log_probs = F.logsigmoid(-attention)
            sum_scope = sum_scope + log_probs

        new_attentions = []
        for attention in attentions:
            scope = sum_scope - F.logsigmoid(-attention)
            rep = torch.cat((feature_map,foreground_map,attention,scope),dim=1)
            new_attention = attention_net(rep)
            new_attentions.append(new_attention)

        return new_attentions


    def detect_objects(self,feature_map,foreground_map, attentions, max_num_objects):
        max_len = max_num_objects

        foreground_map = F.logsigmoid(foreground_map)

        batch_size = feature_map.size(0)
        sum_scope = torch.zeros(batch_size,1,1,1).to(feature_map.device)

        for i in range(len(attentions)):
            attention = attentions[i]
            log_probs = F.logsigmoid(-attention)
            sum_scope = sum_scope + log_probs

        object_probs = []
        for attention in attentions:
            scope = sum_scope - F.logsigmoid(-attention)
            rep = torch.cat((feature_map,foreground_map,attention,scope),dim=1)
            object_prob = self.object_classifier(rep)
            object_prob = object_prob.squeeze(-1).squeeze(-1).squeeze(-1)
            object_prob = torch.sigmoid(object_prob)
            object_probs.append(object_prob)

        return object_probs



    def get_objects(self,object_features,batch_size):
        max_num_objects = 10
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device)


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

        indicators = self.local_max(foreground_map,max_num_objects)

        attentions = self.transformer_layer_start(object_features,foreground_map,indicators)
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_2,max_num_objects)
        #
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_3,max_num_objects)
        #attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_4,max_num_objects)
        #print(self.attention_net_4.conv1.weight.data)
        #object_weights = []
        #object_detection_representation = self.object_detector_rep(object_features)
        object_weights = self.detect_objects(object_features,foreground_map, attentions,max_num_objects)

        for slot in range(max_num_objects):
            
            attention = F.sigmoid(attentions[slot])
            attention = attention.squeeze(1)
            #else:
            #    attention = torch.exp(log_scope).squeeze(1)

            #attention = attention.squeeze(1)
            objects = torch.einsum("bjk,bljk -> bl", attention, foreground)
            #objects = self.maxpool(obj_cols_weighted).squeeze(-1).squeeze(-1)
            
            object_representations.append(objects)

            if False:
                log_foreground_attention = torch.log(foreground_attention)
                log_attention = torch.log(attention)
                attention_product = (log_foreground_attention+log_attention).view(batch_size,-1)
                weight = self.object_detector_foreground(attention_product).squeeze(1)
                object_weights.append(weight)
            elif False:
                objects_normalized = self._norm(objects)
                weight = self.object_detector(objects_normalized)
                weight = weight.squeeze(1)
                object_weights.append(weight)
            elif False:
                normalized_attention = attention / torch.sum(attention,dim=(1,2)).view(batch_size,1,1)
                weight = torch.einsum("bjk,bjk -> b",normalized_attention,foreground_attention)
                object_weights.append(weight)
            elif False:
                current_object_detection_representation = torch.einsum("bjk,bljk -> bl",attention,object_detection_representation)
                current_object_detection_representation = self._norm(current_object_detection_representation)
                weight = self.object_classifier(current_object_detection_representation).squeeze(1)
                object_weights.append(weight)

            

        object_representations = torch.stack(object_representations,dim=1)
        object_weights = torch.stack(object_weights,dim=1)

        return object_representations,object_weights

    def compute_attention(self,object_features,objects,objects_length,visualize_foreground=False):
        max_num_objects = 10
        obj_coord_map = coord_map((object_features.size(2),object_features.size(3)),object_features.device)


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

        indicators = self.local_max(foreground_map,max_num_objects)

        attentions = self.transformer_layer_start(object_features,foreground_map,indicators)
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_2,max_num_objects)
        #
        attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_3,max_num_objects)
        #attentions = self.transformer_layer(object_features,foreground_map, attentions, self.attention_net_4,max_num_objects)
        #print(self.attention_net_4.conv1.weight.data)

        object_weights = self.detect_objects(object_features,foreground_map, attentions,max_num_objects)
        object_weights = [float(w.squeeze(0)) for w in object_weights]

        for slot in range(max_num_objects):
            
            attention = F.sigmoid(attentions[slot])
            attention = attention.squeeze(1)

            attention_list.append(attention)
            #else:
            #    attention = torch.exp(log_scope).squeeze(1)

            

            if False:
                log_foreground_attention = torch.log(foreground_attention)
                log_attention = torch.log(attention)
                attention_product = (log_foreground_attention+log_attention).view(batch_size,-1)
                weight = self.object_detector_foreground(attention_product).squeeze(1)
                object_weights.append(weight)
            elif False:
                objects_normalized = self._norm(objects)
                weight = self.object_detector(objects_normalized)
                weight = weight.squeeze(1)
                object_weights.append(weight)
            elif False:
                normalized_attention = attention / torch.sum(attention,dim=(1,2)).view(batch_size,1,1)
                weight = torch.einsum("bjk,bjk -> b",normalized_attention,foreground_attention)
                object_weights.append(weight)

            
        print(object_weights)
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

