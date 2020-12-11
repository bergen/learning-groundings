#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/04/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import jactorch

__all__ = ['SigmoidCrossEntropy', 'MultilabelSigmoidCrossEntropy']


class SigmoidCrossEntropy(nn.Module):
    def __init__(self, one_hot=False):
        super().__init__()
        self.one_hot = one_hot
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        if not self.one_hot:
            target = jactorch.one_hot_nd(target, input.size(-1))
        return self.bce(input, target).sum(dim=-1).mean()


class MultilabelSigmoidCrossEntropy(nn.Module):
    def __init__(self, one_hot=False):
        super().__init__()
        self.one_hot = one_hot
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, labels):
        if type(labels) in (tuple, list):
            labels = torch.tensor(labels, dtype=torch.int64, device=input.device)

        assert input.dim() == 1
        if not self.one_hot:
            with torch.no_grad():
                mask = torch.zeros_like(input)
                if labels.size(0) > 0:
                    ones = torch.ones_like(labels, dtype=torch.float32)
                    mask.scatter_(0, labels, ones)
            labels = mask

        return self.bce(input, labels).sum(dim=-1).mean()


class MultitaskLossBase(nn.Module):
    def __init__(self):
        super().__init__()

        self._sigmoid_xent_loss = SigmoidCrossEntropy()
        self._multilabel_sigmoid_xent_loss = MultilabelSigmoidCrossEntropy()

    def _mse_loss(self, pred, label):
        return (pred - label).abs()

    def _bce_loss(self, pred, label):
        #pred is log probability of label 1
        if pred.item()==0: #this only occurs when label==1
            return -pred

        complement_prob = torch.log(1-torch.exp(pred))
        if complement_prob.item()==float("-inf"):
            return -label*pred
        else:
            return -(label*pred + (1-label)*complement_prob)

    def _xent_loss(self, pred, label):
        return -pred[label].mean()
