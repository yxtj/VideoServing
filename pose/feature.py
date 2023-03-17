# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
from scipy.spatial import distance

filepath='E:/Data/video/pose/'

RSL = [240, 360, 480, 720, 840]
NAME_LIST = ['profile-%d.npz'%r for r in RSL]

# %% feature functions


def make_raw_feature(poses, css, ema_f=0.8):
    idx = css != 0
    poses = poses[idx]
    n = len(poses)
    delta = poses[1:,:,[0,1]] - poses[:-1,:,[0,1]]
    # feature speed
    speed = np.sqrt((delta**2).sum(2))

    xmax, ymax = poses[:,:,0].max(1), poses[:,:,1].max(1)
    # xmin, ymin = poses[:,:,0].min(1), poses[:,:,1].min(1)
    xmin, ymin = np.zeros(n), np.zeros(n)
    for i in range(n):
        idx = poses[:,:,2] != 0
        xmin[i] = poses[i,idx,0].min()
        ymin[i] = poses[i,idx,1].min()
    # feature size
    size = (xmax - xmin) * (ymax-ymin)
    # feature aspect ratio
    aratio = (xmax - xmin) / (ymax-ymin)

    # feature distance
    dist = np.zeros(n, 4) # min, mean, max, std
    for i in range(n):
        idx = poses[i,:,2] != 0
        p = poses[i,idx,:2]
        D = distance.cdist(p, p)[np.triu_idices(len(idx), 1)] # upper triangle
        dist[i] = (D.min(), D.mean(), D.max(), D.std())

    return speed, size[1:], aratio[1:], dist[1:,0], dist[1:,1], dist[1:,2], dist[1:,3]


# %% run

folders = [f for f in os.listdir(filepath) if f.startswith('output_') and os.path.isdir(filepath+f)]
