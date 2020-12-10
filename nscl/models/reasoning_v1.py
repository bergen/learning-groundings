#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : reasoning_v1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/06/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch.nn as nn
import jactorch.nn as jacnn

from jacinle.logging import get_logger
from nscl.configs.common import make_base_configs
from nscl.datasets.definition import gdef

logger = get_logger(__file__)

__all__ = ['make_reasoning_v1_configs', 'ReasoningV1Model']


def make_reasoning_v1_configs():
    configs = make_base_configs()

    # data configs
    configs.data.image_size = 256
    configs.data.add_full_image_bbox = False

    # model configs for scene graph
    configs.model.sg_dims = [None, 256, 256]

    # model ocnfigs for visual-semantic embeddings
    configs.model.vse_known_belong = True
    configs.model.vse_large_scale = False
    configs.model.vse_ls_load_concept_embeddings = False
    configs.model.vse_hidden_dims = [None, 64, 64]

    # model configs for parser
    configs.model.word_embedding_dim = 300
    configs.model.positional_embedding_dim = 50
    configs.model.word_embedding_dropout = 0.5
    configs.model.gru_dropout = 0.5
    configs.model.gru_hidden_dim = 256

    # supervision configs
    configs.train.discount = 0.9
    configs.train.scene_add_supervision = False
    configs.train.qa_add_supervision = False
    configs.train.parserv1_reward_shape = 'loss'

    return configs


class ReasoningV1Model(nn.Module):
    def __init__(self, args, vocab, configs):
        super().__init__()
        self.args = args
        self.vocab = vocab

        self.resnet_type = args.resnet_type

        import jactorch.models.vision.resnet as jac_resnet
        from nscl.models.resnet import resnet34
        from nscl.models.cmc_resnet import load_pretrained_cmc
        from nscl.models.simclr_resnet import load_pretrained_simclr
        resnet_dict = {'resnet34':jac_resnet.resnet34, 'resnet101':jac_resnet.resnet101}
        

        if self.resnet_type=='cmc_resnet':
            self.resnet = load_pretrained_cmc()
            if args.restrict_finetuning:
                self.resnet.encoder.module.l_to_ab.layer3.requires_grad = False
                self.resnet.encoder.module.ab_to_l.layer3.requires_grad = False
        elif self.resnet_type=='simclr_resnet':
            self.resnet = load_pretrained_simclr()
            if args.restrict_finetuning:
                self.resnet.layer2.requires_grad = False
                self.resnet.layer1.requires_grad = False
                self.resnet.conv1.requires_grad = False
                i=0
                for module in self.resnet.layer3:
                    if i<4:
                        module.requires_grad = False
                    i+=1
        elif self.resnet_type=='resnet34_pytorch':
            self.resnet = resnet34(pretrained=args.pretrained_resnet,restrict_fine_tuning=args.restrict_finetuning)
        else:
            resnet_model = resnet_dict[self.resnet_type]
            self.resnet = resnet_model(pretrained=True, incl_gap=False, num_classes=None)
            self.resnet.layer4 = jacnn.Identity()
            if args.restrict_finetuning:
                self.resnet.layer2.requires_grad = False
                self.resnet.layer1.requires_grad = False
                self.resnet.conv1.requires_grad = False

        import nscl.nn.scene_graph.scene_graph as sng
        import nscl.nn.scene_graph.monet as monet
        # number of channels = 256; downsample rate = 16.
        attention_dispatch = {
                            'monet':monet.MONet,
                            'scene-graph-object-supervised': sng.SceneGraphObjectSupervision,
                            'monet-lite': sng.MonetLiteSceneGraph,
                            'transformer-cnn': sng.TransformerCNN,
                            'transformer-cnn-object-inference': sng.TransformerCNNObjectInference,
                            'transformer-cnn-object-inference-ablate-scope': sng.TransformerCNNObjectInferenceAblateScope,
                            'transformer-cnn-object-inference-ablate-initialization': sng.TransformerCNNObjectInferenceAblateInitialization,
                            'transformer-cnn-object-inference-sequential': sng.TransformerCNNObjectInferenceSequential,
                            'transformer-cnn-object-inference-recurrent': sng.TransformerCNNObjectInferenceRecurrent}

        try:
            if args.attention_type=='monet':
                self.scene_graph = attention_dispatch[args.attention_type](128, 128, 3, configs.model.sg_dims, args=args)
            else:
                if self.resnet_type in ['resnet34','resnet34_pytorch']:
                    feature_dim=256
                    img_input_dim=(16,24)
                elif self.resnet_type=='resnet101':
                    feature_dim=1024
                    img_input_dim=(16,24)
                elif self.resnet_type=='cmc_resnet':
                    feature_dim=1024
                    img_input_dim=(8,12)
                elif self.resnet_type=='simclr_resnet':
                    feature_dim=256
                    img_input_dim=(16,24)

                self.scene_graph = attention_dispatch[args.attention_type](feature_dim, configs.model.sg_dims, 16, args=args,img_input_dim=img_input_dim)
        except Exception as e:
            print(e)
            self.scene_graph = attention_dispatch[args.attention_type](256, configs.model.sg_dims, 16)

        import nscl.nn.reasoning_v1.quasi_symbolic as qs
        self.reasoning = qs.DifferentiableReasoning(
            self._make_vse_concepts(configs.model.vse_large_scale, configs.model.vse_known_belong),
            self.scene_graph.output_dims, configs.model.vse_hidden_dims,
            self.args
        )

        import nscl.nn.reasoning_v1.losses as vqa_losses
        self.scene_loss = vqa_losses.SceneParsingLoss(gdef.all_concepts, add_supervision=configs.train.scene_add_supervision)

        try:
            self.qa_loss = vqa_losses.QALoss(add_supervision=configs.train.qa_add_supervision, args=args)
        except:
            self.qa_loss = vqa_losses.QALoss(add_supervision=configs.train.qa_add_supervision)

        try:
            if args.adversarial_loss:
                self.adversarial_loss = vqa_losses.AdversarialLoss()
        except Exception as e:
            pass

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, large_scale, known_belong):
        if large_scale:
            return {
                'attribute_ls': {'attributes': list(gdef.ls_attributes), 'concepts': list(gdef.ls_concepts)},
                'relation_ls': {'attributes': None, 'concepts': list(gdef.ls_relational_concepts)},
                'embeddings': gdef.get_ls_concept_embeddings()
            }
        return {
            'attribute': {
                'attributes': list(gdef.attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(gdef.relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.relational_concepts.items() for v in vs
                ]
            }
        }
