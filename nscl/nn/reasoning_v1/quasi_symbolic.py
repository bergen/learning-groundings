#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quasi_symbolic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


"""
Quasi-Symbolic Reasoning.
"""

import six
import math

import torch
import torch.nn as nn

import jactorch.nn.functional as jacf

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from nscl.datasets.common.program_executor import ParameterResolutionMode
from nscl.datasets.definition import gdef
from . import concept_embedding
from collections import defaultdict

logger = get_logger(__file__)

__all__ = ['ConceptQuantizationContext', 'ProgramExecutorContext', 'DifferentiableReasoning', 'set_apply_self_mask']


_apply_self_mask = {'relate': True, 'relate_ae': True}


def set_apply_self_mask(key, value):
    logger.warning('Set {}.apply_self_mask[{}] to {}.'.format(set_apply_self_mask.__module__, key, value))
    assert key in _apply_self_mask, key
    _apply_self_mask[key] = value


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    
    return m * (1 - self_mask) + (-10) * self_mask


class InferenceQuantizationMethod(JacEnum):
    NONE = 0
    STANDARD = 1
    EVERYTHING = 2


_test_quantize = InferenceQuantizationMethod.STANDARD


def set_test_quantize(mode):
    global _test_quantize
    _test_quantize = InferenceQuantizationMethod.from_string(mode)




class ProgramExecutorContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, features, parameter_resolution,training,args):
        super().__init__()

        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

        # None, attributes, relations
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy]
        self._concept_groups_masks = defaultdict(lambda: None)

        self._attribute_groups_masks = defaultdict(lambda: None)
        self._attribute_query_masks = defaultdict(lambda: None)

        self.presupposition_semantics = args.presupposition_semantics   
        self.mutual_exclusive = args.mutual_exclusive 
        self.filter_additive = args.filter_additive
        self.relate_rescale = args.relate_rescale
        self.logit_semantics = args.logit_semantics
        self.relate_max = args.relate_max

        self.args = args

        self.train(training)

    def filter(self, selected, group, concept_groups,index):
        concept_group = concept_groups[group]
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_group, 1,index)

        mask = torch.min(selected, mask)
        #mask is a list of num_objects log-probabilities. each term is the log-probability that the object has the concept


        return mask


    def relate(self, selected, group, concept_groups,index):
        #selected = torch.log(selected)
        #selected is a log probability distribution over objects. Exactly one object is assumed to satisfy the conditions that produced selected 
        #concept groups is a list of relational concepts (e.g. ['front','left','right'])
        #group is an int, which indicates which concept from concept_groups is being used

        concept_group = concept_groups[group]
        mask = self._get_concept_groups_masks(concept_group, 2,index)
        #mask is a num_concept_groups x num_objects x num_objects tensor. it contains, for each object pair, the log probability the object pair satisfies a certain relational concept. 
        #the first dimension indexes the specific concept being used (e.g. 'front')


        mask = (mask + selected.unsqueeze(-1))

        mask = torch.logsumexp(mask,dim=-2) #need to verify that this logsumexp is over correct dimension

        if self.args.infer_num_objects:
            mask = torch.min(mask,torch.log(self.features[3][index]))


        return mask

    def relate_ae(self, selected, group, attribute_groups,index):
        #selected = torch.log(selected)
        #selected is a log probability distribution over objects. Exactly one object is assumed to satisfy the conditions that produced selected 
        #attribute_groups is a list of attributes (e.g. ['color', 'shape'])
        #group is an int, which indicates which attribute from attribute_groups is being used

        attribute = attribute_groups[group]
        mask = self._get_attribute_groups_masks(attribute,index)
        #mask is a num_attribute_groups x num_objects x num_objects tensor. it contains, for each object pair, the log probability the object pair has the same attribute value (for example, the same color). 
        #the first dimension indexes the specific attribute being used (e.g. 'color')
        mask = (mask + selected.unsqueeze(-1))
        mask = torch.logsumexp(mask,dim=-2) #need to verify that this logsumexp is over correct dimension

        return mask

    def unique(self, selected):
        #this is applied to transform type object_set to objects
        #needs to be performed before a query
        return nn.LogSoftmax(dim=0)(selected)

    def intersect(self, selected1, selected2):
        if self.filter_additive:
            selected = selected1 + selected2
        else:
            selected = torch.min(selected1, selected2)

        return selected

    def union(self, selected1, selected2):
        if self.filter_additive:
            selected = torch.log(1 - (1-torch.exp(selected1))*(1-torch.exp(selected2)))
        else:
            selected = torch.max(selected1, selected2)

        return selected

    def exist(self, selected):
        if self.logit_semantics:
            selected = nn.LogSigmoid()(selected)

        return selected.max(dim=-1)[0]


    def count(self, selected):
        if self.logit_semantics:
            selected = nn.LogSigmoid()(selected)
        if self.training:
            return torch.exp(selected).sum(dim=-1)
        else:
            return (selected > math.log(0.5)).float().sum()
            #return torch.exp(selected).sum(dim=-1).round()

    _count_margin = -0.25
    _count_tau = 0.1
    _count_equal_margin = 0.25
    _count_equal_tau = 0.1

    def count_greater(self, selected1, selected2):
        #selected1 and selected2 are tensors of length num_objects. Each is a tensor of log probabilities (not a distribution). Each number is the log probability that an object satisfies a given property
        
        if self.logit_semantics:
            selected1 = nn.LogSigmoid()(selected1)
            selected2 = nn.LogSigmoid()(selected2)

        a = torch.exp(selected1).sum(dim=-1)
        b = torch.exp(selected2).sum(dim=-1)

        #a = torch.logsumexp(selected1,dim=0)
        #b = torch.logsumexp(selected2,dim=0)

        if self.training:
            return nn.LogSigmoid()(((self._count_margin + a - b ) / self._count_tau))
        else:
            return nn.LogSigmoid()(((self._count_margin + a - b ) / self._count_tau))
        #else:
        #    return nn.LogSigmoid()(-10 + 20 * (self.count(selected1) > self.count(selected2)).float()) #this is probably wrong

    def count_less(self, selected1, selected2):
        return self.count_greater(selected2, selected1)

    def count_equal(self, selected1, selected2):
        #selected1 and selected2 are in log probability space

        if self.logit_semantics:
            selected1 = nn.LogSigmoid()(selected1)
            selected2 = nn.LogSigmoid()(selected2)

        a = torch.exp(selected1).sum(dim=-1)
        b = torch.exp(selected2).sum(dim=-1)
        #a = torch.logsumexp(selected1,dim=0)
        #b = torch.logsumexp(selected2,dim=0)
        return nn.LogSigmoid()(((self._count_equal_margin - (a - b).abs()) / self._count_equal_tau))
        

    

    def query(self, selected, group, attribute_groups,index):
        attribute = attribute_groups[group]
        val, index = torch.max(selected,dim=0)

        mask = self._get_attribute_query_masks(attribute,index)
        #print(mask)
        #selected is a probability distribution over objects ( log space)
        #mask is a list consisting of num_objects probability distributions. These probability distributions are in log space.

        #selected = torch.log(selected)
        
        mask = (mask + selected.unsqueeze(1))
        mask = torch.logsumexp(mask,dim=0)
            
        return mask,val, self.word2idx



    def query_ae(self, selected1, selected2, group, attribute_groups,index):
        #selected1 = torch.exp(selected1)
        #selected2 = torch.exp(selected2)
        attribute = attribute_groups[group]
        mask = self._get_attribute_groups_masks(attribute,index)

        mask = torch.logsumexp((mask + selected1.unsqueeze(-1)),dim=-2)
        mask = torch.logsumexp((mask + selected2),dim=-1)

        return mask

    def _get_concept_groups_masks(self, concept_group, k,index):

        
        if k==1:
            if isinstance(concept_group, six.string_types):
                concept_group = [concept_group]
            mask = None
            for c in concept_group:
                attributes = self.taxnomy[1].all_attributes
                concept = self.taxnomy[1].get_concept(c)
                attribute_index = concept.belong.argmax(-1).item()
                attribute = attributes[attribute_index]

                probs = self._attribute_query_masks[attribute]
                concept_index = self.word2idx[c]
                new_mask = probs[index,:,concept_index]
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
        else:
            mask = None
            for c in concept_group:
                
                new_mask  = self._concept_groups_masks[c][index,:]
                mask = torch.min(mask, new_mask) if mask is not None else new_mask

        
        return mask

    def _get_attribute_groups_masks(self, attribute,index):
        #print(attribute)
        mask = self._attribute_groups_masks[attribute][index,:]

        return mask

    def _get_attribute_query_masks(self, attribute,index):
        
        return self._attribute_query_masks[attribute][index,:]

    def init_queries(self,features):
        attributes = ['size','shape','color','material']

        #features = batch_features
        #features = torch.stack(features,dim=0)

        for attribute in attributes:
            masks,word2idx = self.taxnomy[1].query_attribute(features, attribute)
            self._attribute_query_masks[attribute] = masks

        self.word2idx = word2idx



    def init_concepts(self,features):
        concepts = ['blue','green','purple','red','yellow','brown','gray','cyan','sphere','cube','cylinder','rubber','metal','large','small']
        relations = ['front','behind','left','right']

            



        for r in relations:
            #features = batch_features
            #features = torch.stack(features,dim=0)
            probs = self.taxnomy[2].similarity(features, r,k=2)
            probs = do_apply_self_mask(probs)
            self._concept_groups_masks[r] = probs


    def init_attribute_relate(self,features):
        attributes = ['size','shape','color','material']
        #features = batch_features
        #features = torch.stack(features,dim=0)

        for attribute in attributes:
            probs = self.taxnomy[1].cross_similarity(features, attribute)
            probs = do_apply_self_mask(probs)
            self._attribute_groups_masks[attribute] = probs



class DifferentiableReasoning(nn.Module):
    def __init__(self, used_concepts, input_dims, hidden_dims,args=None, parameter_resolution='deterministic', vse_attribute_agnostic=False):
        super().__init__()

        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution


        self.args = args

        for i, nr_vars in enumerate(['attribute', 'relation']):
            if nr_vars not in self.used_concepts:
                continue
            if nr_vars=='relation':
                bilinear = self.args.bilinear_relation
                coord_semantics = self.args.coord_semantics
            else:
                bilinear = False
                coord_semantics = False

            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic,bilinear_relation=bilinear,coord_semantics=coord_semantics))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]


            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for (v, b) in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)



    def forward(self, batch_features, progs, fd=None):

        #assert len(progs) == len(batch_features)

        #print(fd.image_filename)

        programs = []
        buffers = []
        result = []

        prev_image = None

        ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, batch_features, parameter_resolution=self.parameter_resolution, training=self.training, args=self.args)
        ctx.init_queries(batch_features[1])
        ctx.init_concepts(batch_features[2])
        ctx.init_attribute_relate(batch_features[1])


        for i, prog in enumerate(progs):
            buffer = []


            #print(fd['question_raw'][i])
            #print(prog)

            buffers.append(buffer)
            programs.append(prog)


            for block_id, block in enumerate(prog):
                op = block['op']

                if op == 'scene':
                    if self.args.infer_num_objects:
                        buffer.append(torch.log(batch_features[3][i]))
                    else:
                        buffer.append(torch.zeros(batch_features[1].size(1), dtype=torch.float, device=batch_features[1].device))
                    continue

                inputs = []
                for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                    inp = buffer[inp]
                    if inp_type == 'object':
                        #this is done to transform type object_set to object
                        inp = ctx.unique(inp)
                    inputs.append(inp)

                # TODO(Jiayuan Mao @ 10/06): add support of soft concept attention.

                if op == 'filter':
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values'],i))
                elif op == 'relate':
                    buffer.append(ctx.relate(*inputs, block['relational_concept_idx'], block['relational_concept_values'],i))
                elif op == 'relate_attribute_equal':
                    buffer.append(ctx.relate_ae(*inputs, block['attribute_idx'], block['attribute_values'],i))
                elif op == 'intersect':
                    buffer.append(ctx.intersect(*inputs))
                elif op == 'union':
                    buffer.append(ctx.union(*inputs))
                else:
                    assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                    if op == 'query':
                        buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values'],i))
                    elif op == 'query_attribute_equal':
                        buffer.append(ctx.query_ae(*inputs, block['attribute_idx'], block['attribute_values'],i))
                    elif op == 'exist':
                        #print(prog)
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'belong_to':
                        buffer.append(ctx.belong_to(*inputs))
                    elif op == 'count':
                        buffer.append(ctx.count(*inputs))
                    elif op == 'count_greater':
                        buffer.append(ctx.count_greater(*inputs))
                    elif op == 'count_less':
                        buffer.append(ctx.count_less(*inputs))
                    elif op == 'count_equal':
                        buffer.append(ctx.count_equal(*inputs))
                    else:
                        raise NotImplementedError('Unsupported operation: {}.'.format(op))

                if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                    if block_id != len(prog) - 1:
                        buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()

            result.append((op, buffer[-1]))

            #prev_image = current_image

            #quasi_symbolic_debug.embed(self, i, buffer, result, fd)

        return programs, buffers, result

    def inference_query(self, features, one_hot, query_type,args):
        ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, features, parameter_resolution=self.parameter_resolution, training=self.training, args=self.args)
        output, val, d = ctx.query(one_hot,0,query_type)
        m = int(torch.argmax(output))
        reverse_d = dict([(d[k],k) for k in d.keys()])
        return reverse_d[m]

