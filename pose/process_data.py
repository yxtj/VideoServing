# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
from .util import computeOKS_1to1


filepath='E:/Data/video/pose/'

RSL = [240, 360, 480, 720, 840]
FN_LIST = ['320x240_25_cmu_estimation_result.tsv',
           '480x352_25_cmu_estimation_result.tsv',
           '640x480_25_cmu_estimation_result.tsv',
           '960x720_25_cmu_estimation_result.tsv',
           '1120x832_25_cmu_estimation_result.tsv',]
NAME_LIST = ['profile-%d.npz'%r for r in RSL]

# %% parse functions

def _parse_pose_one_(s):
    idx = s.rfind(']')
    p = np.fromstring(s[1:idx], sep=',', dtype=np.float32).reshape(17,3)
    cs = float(s[idx+2:])
    return p, cs

def process_pose(poses, nums):
    n = len(nums)
    res = np.zeros((n, 17, 3), dtype=np.float32)
    cs = np.zeros(n, dtype=np.float32)
    for i, (p, n) in enumerate(zip(poses, nums)):
        # if n == 0:
        #     res[i] = None
        if n == 1:
            res[i], cs[i] = _parse_pose_one_(poses[i])
        elif n > 1:
            # use the one with largest confidence score
            l = poses[i].split(';')
            tmp_p = []
            tmp_c = []
            for s in l:
                p,c = _parse_pose_one_(s)
                tmp_p.append(p)
                tmp_c.append(c)
            idx = np.argmax(tmp_c)
            res[i] = tmp_p[idx]
            cs[i] = tmp_c[idx]
    return res, cs

# %% run

folders = [f for f in os.listdir(filepath) if f.startswith('output_') and os.path.isdir(filepath+f)]


for fld in folders:
    print('processing:', fld)
    # use the highest resolution as the ground truth
    d_ref = pd.read_csv(filepath+fld+'/'+FN_LIST[-1], sep='\t')
    num_ref = d_ref.numberOfHumans.values
    poses_ref, cs_ref = process_pose(d_ref.Estimation_result, num_ref)
    oks = np.ones_like(cs_ref)
    times_ref = d_ref.Time_SPF.values
    np.savez(filepath+fld+'/'+NAME_LIST[-1], oks=oks, poses=poses_ref, confscore=cs_ref, times=times_ref)

    for idx, f in enumerate(FN_LIST[:-1]):
        d = pd.read_csv(filepath+fld+'/'+f, sep='\t')
        num = d.numberOfHumans.values
        poses, cs = process_pose(d.Estimation_result, d.numberOfHumans)
        oks = np.zeros(len(poses))
        for i in range(len(poses)):
            if num_ref[i] > 0:
                if num[i] > 0:
                    oks[i] = computeOKS_1to1(poses_ref[i], poses[i])
            else:
                if num[i] == 0:
                    oks[i] = 1
        times = d.Time_SPF.values
        np.savez(filepath+fld+'/'+NAME_LIST[idx], oks=oks, poses=poses, confscore=cs, times=times)
