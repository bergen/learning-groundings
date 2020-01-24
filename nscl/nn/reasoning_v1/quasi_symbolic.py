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
from . import concept_embedding, concept_embedding_ls
from . import quasi_symbolic_debug

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
    def __init__(self, attribute_taxnomy, relation_taxnomy, features, presupposition_semantics, parameter_resolution, training=True):
        super().__init__()

        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

        # None, attributes, relations
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy]
        self._concept_groups_masks = [None, None, None]

        self._attribute_groups_masks = None
        self._attribute_query_masks = None
        self._attribute_query_ls_masks = None
        self._attribute_query_ls_mc_masks = None

        self.presupposition_semantics = presupposition_semantics   

        self.train(training)

    def filter(self, selected, group, concept_groups):
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = torch.min(selected.unsqueeze(0), mask)
        #mask is a list of num_objects log-probabilities. each term is the log-probability that the object has the concept


        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]


    def relate(self, selected, group, concept_groups):
        selected = torch.log(selected)
        #selected is a log probability distribution over objects. Exactly one object is assumed to satisfy the conditions that produced selected 
        #concept groups is a list of relational concepts (e.g. ['front','left','right'])
        #group is an int, which indicates which concept from concept_groups is being used

        mask = self._get_concept_groups_masks(concept_groups, 2)
        #mask is a num_concept_groups x num_objects x num_objects tensor. it contains, for each object pair, the log probability the object pair satisfies a certain relational concept. 
        #the first dimension indexes the specific concept being used (e.g. 'front')

        mask = (mask + selected.unsqueeze(-1).unsqueeze(0))
        mask = torch.logsumexp(mask,dim=-2) #need to verify that this logsumexp is over correct dimension

        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def relate_ae(self, selected, group, attribute_groups):
        selected = torch.log(selected)
        #selected is a log probability distribution over objects. Exactly one object is assumed to satisfy the conditions that produced selected 
        #attribute_groups is a list of attributes (e.g. ['color', 'shape'])
        #group is an int, which indicates which attribute from attribute_groups is being used


        mask = self._get_attribute_groups_masks(attribute_groups)
        #mask is a num_attribute_groups x num_objects x num_objects tensor. it contains, for each object pair, the log probability the object pair has the same attribute value (for example, the same color). 
        #the first dimension indexes the specific attribute being used (e.g. 'color')

        mask = (mask + selected.unsqueeze(-1).unsqueeze(0))
        mask = torch.logsumexp(mask,dim=-2) #need to verify that this logsumexp is over correct dimension

        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def unique(self, selected):
        #this is applied to transform type object_set to objects
        #needs to be performed before a query
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            return jacf.general_softmax(selected, impl='standard', training=self.training)
        # trigger the greedy_max
        return jacf.general_softmax(selected, impl='gumbel_hard', training=self.training)

    def intersect(self, selected1, selected2):
        return torch.min(selected1, selected2)

    def union(self, selected1, selected2):
        return torch.max(selected1, selected2)

    def exist(self, selected):
        #print(selected)
        return selected.max(dim=-1)[0]


    def count(self, selected):
        if self.training:
            return torch.exp(selected).sum(dim=-1)
        else:
            #if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
            #    return (selected > math.log(0.5)).float().sum()
            return torch.exp(selected).sum(dim=-1).round()

    _count_margin = 0.25
    _count_tau = 0.25

    def count_greater(self, selected1, selected2):
        #selected1 and selected2 are tensors of length num_objects. Each is a tensor of log probabilities (not a distribution). Each number is the log probability that an object satisfies a given property
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            a = torch.exp(selected1).sum(dim=-1)
            b = torch.exp(selected2).sum(dim=-1)

            return nn.LogSigmoid()(((a - b - 1 + 2 * self._count_margin) / self._count_tau))
        else:
            return nn.LogSigmoid()(-10 + 20 * (self.count(selected1) > self.count(selected2)).float()) #this is probably wrong

    def count_less(self, selected1, selected2):
        return self.count_greater(selected2, selected1)

    def count_equal(self, selected1, selected2):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            a = torch.exp(selected1).sum(dim=-1)
            b = torch.exp(selected2).sum(dim=-1)
            return nn.LogSigmoid()(((2 * self._count_margin - (a - b).abs()) / (2 * self._count_margin) / self._count_tau))
        else:
            return nn.LogSigmoid()(-10 + 20 * (self.count(selected1) == self.count(selected2)).float()) #this is probably wrong

    def query(self, selected, group, attribute_groups):
        val, index = torch.max(selected,0)

        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        #print(mask)
        #selected is a probability distribution over objects (not log space)
        #mask is a list consisting of num_objects probability distributions. These probability distributions are in log space.

        selected = torch.log(selected)
        
        if self.presupposition_semantics:
            mask = (mask[0][index]).unsqueeze(0)
        else:
            mask = (mask + selected.unsqueeze(-1).unsqueeze(0))
            mask = torch.logsumexp(mask,dim=-2)
        #
        #print(mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx

        return mask[group],val, word2idx



    def query_ae(self, selected1, selected2, group, attribute_groups):
        selected1 = torch.exp(selected1)
        selected2 = torch.exp(selected2)
        mask = self._get_attribute_groups_masks(attribute_groups)

        mask = (mask + selected1.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        mask = (mask + selected2.unsqueeze(0)).sum(dim=-1)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def _get_concept_groups_masks(self, concept_groups, k):
        if self._concept_groups_masks[k] is None:
            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[k].similarity(self.features[k], c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_attribute_groups_masks(self, attribute_groups):
        if self._attribute_groups_masks is None:
            masks = list()
            for attribute in attribute_groups:
                mask = self.taxnomy[1].cross_similarity(self.features[1], attribute)
                if _apply_self_mask['relate_ae']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._attribute_groups_masks = torch.stack(masks, dim=0)
        return self._attribute_groups_masks

    def _get_attribute_query_masks(self, attribute_groups):
        if self._attribute_query_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                # sanity check.
                if word2idx is not None:
                    for k in word2idx:
                        assert word2idx[k] == this_word2idx[k]
                word2idx = this_word2idx

            self._attribute_query_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_masks

class DifferentiableReasoning(nn.Module):
    def __init__(self, used_concepts, input_dims, hidden_dims,args=None, parameter_resolution='deterministic', vse_attribute_agnostic=False):
        super().__init__()

        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution

        try:
            self.presupposition_semantics = args.presupposition_semantics
        except Exception as e:
            self.presupposition_semantics = False

        for i, nr_vars in enumerate(['attribute', 'relation']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]

            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for (v, b) in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)

        for i, nr_vars in enumerate(['attribute_ls', 'relation_ls']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars.replace('_ls', ''), concept_embedding_ls.ConceptEmbeddingLS(
                self.input_dims[1 + i], self.hidden_dims[1 + i], self.hidden_dims[1 + i]
            ))
            tax = getattr(self, 'embedding_' + nr_vars.replace('_ls', ''))
            rec = self.used_concepts[nr_vars]

            if rec['attributes'] is not None:
                tax.init_attributes(rec['attributes'], self.used_concepts['embeddings'])
            if rec['concepts'] is not None:
                tax.init_concepts(rec['concepts'], self.used_concepts['embeddings'])

    def forward(self, batch_features, progs, fd=None):
        assert len(progs) == len(batch_features)

        programs = []
        buffers = []
        result = []
        for i, (features, prog) in enumerate(zip(batch_features, progs)):
            buffer = []

            #print(fd['question_raw'][i])
            #print(prog)

            buffers.append(buffer)
            programs.append(prog)



            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, features, self.presupposition_semantics, parameter_resolution=self.parameter_resolution, training=self.training)

            for block_id, block in enumerate(prog):
                op = block['op']

                if op == 'scene':
                    buffer.append(torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device))
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
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'filter_scene':
                    inputs = [torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device)]
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'filter_most':
                    buffer.append(ctx.filter_most(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                elif op == 'relate':
                    buffer.append(ctx.relate(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                elif op == 'relate_attribute_equal':
                    buffer.append(ctx.relate_ae(*inputs, block['attribute_idx'], block['attribute_values']))
                elif op == 'intersect':
                    buffer.append(ctx.intersect(*inputs))
                elif op == 'union':
                    buffer.append(ctx.union(*inputs))
                else:
                    assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                    if op == 'query':
                        buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'query_ls':
                        buffer.append(ctx.query_ls(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'query_ls_mc':
                        buffer.append(ctx.query_ls_mc(*inputs, block['attribute_idx'], block['attribute_values'], block['multiple_choices']))
                    elif op == 'query_is':
                        buffer.append(ctx.query_is(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'query_attribute_equal':
                        buffer.append(ctx.query_ae(*inputs, block['attribute_idx'], block['attribute_values']))
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

            quasi_symbolic_debug.embed(self, i, buffer, result, fd)

        return programs, buffers, result

    def inference_query(self, features, one_hot, query_type):
        ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, features, parameter_resolution=self.parameter_resolution, training=self.training)
        output, d = ctx.query(one_hot,0,query_type)
        m = int(torch.argmax(output))
        reverse_d = dict([(d[k],k) for k in d.keys()])
        return reverse_d[m]


