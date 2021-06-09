# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F


# predict: 1 how many frame should be jumped, 2 which processor to use


class DecisionModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, frame_jump_max=30):
        super().__init__()
        self.dim_in = dim_in
        self.frame_jump_max = frame_jump_max
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LeakyReLU(),
            )
        self.head_fj = nn.Sequential(
            nn.Linear(dim_hidden, frame_jump_max),
            nn.LeakyReLU(),
            nn.Linear(frame_jump_max, 1),
            )
        self.head_pm = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid(),
            )
    
    def forward(self, feat):
        h = self.backbone(feat)
        f = self.head_fj(h)
        f = torch.clamp(f, 0, self.frame_jump_max-1) + 1
        m = self.head_pm(h)
        return (f, m)


class DicisionLoss():
    def __init__(self):
        self.loss_f = nn.MSELoss()
        self.loss_m = nn.CrossEntropyLoss()
    
    def __call__(self, predicted, target):
        pf, pm = predicted
        yf, ym = target
        lf = self.loss_f(pf, yf)
        lm = self.loss_m(pm, ym)
        return lf+lm

