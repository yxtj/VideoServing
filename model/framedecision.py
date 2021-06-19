# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F


# predict: 1 how many frame should be jumped, 2 which processor to use


class DecisionModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_outs=(5,2)):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_outs = dim_outs
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LeakyReLU(),
            )
        self.heads = nn.ModuleList([ 
            nn.Sequential(nn.Linear(dim_hidden, do), nn.Softmax(1))
            for do in dim_outs
            ])
    
    def forward(self, feat):
        h = self.backbone(feat)
        #f = self.head_fj(h)
        #m = self.head_pm(h)
        #return (f, m)
        ys = [ head(h) for head in self.heads ]
        return ys


class DecisionLoss():
    def __init__(self):
        #self.loss_f = nn.MSELoss()
        #self.loss_f = nn.CrossEntropyLoss()
        #self.loss_m = nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(self, predicted, target):
        #pf, pm = predicted
        #yf, ym = target[:,0]-1, target[:,1]
        #lf = self.loss_f(pf, yf)
        #lm = self.loss_m(pm, ym)
        #return lf+lm
        return sum(self.loss_fn(p,t) for p,t in zip(predicted, target.T))

