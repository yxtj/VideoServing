# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

filepath='E:/Data/video/pose/'

RSL = [240, 360, 480, 720, 840]
FPS = [1, 2, 3, 5, 10, 15, 30]
NAME_LIST = ['profile-%d.npz'%r for r in RSL]

# %% functions

def load_profile(fld):
    scores = []
    times = []
    for f in NAME_LIST:
        d = np.load(fld+'/'+f, allow_pickle=True)
        s = d['oks']
        t = d['times']
        scores.append(s)
        times.append(t)
    nmin = min(len(s) for s in scores)
    nmax = max(len(s) for s in scores)
    if nmin != nmax:
        for i in range(len(scores)):
            scores[i] = scores[i][:nmin]
            times[i] = times[i][:nmin]
    scores = np.array(scores)
    times = np.array(times)
    return scores, times


# approximate the score for skipped frames
def get_performance(pf_score, pf_time, rsl_idx, fps_idx):
    fps = FPS[fps_idx]
    fr = 30 // fps
    n = len(pf_score[0])
    n_sec = n // 30
    last = n_sec*30
    # this is approximated score
    # The acctual score should be OKS of estimated poses and ground-truth poses
    factor = np.exp(-0.003*(30/fps-1))
    res_score = pf_score[rsl_idx, :last:fr].reshape((-1, fps)).mean(1)*factor
    res_time = pf_time[rsl_idx, :last:fr].reshape((-1, fps)).sum(1)
    return res_score, res_time


def get_all_performances(pf_score, pf_time):
    nrsl, nfps = len(RSL), len(FPS)
    scores = [[None for _ in range(nfps)] for _ in range(nrsl)]
    times = [[None for _ in range(nfps)] for _ in range(nrsl)]
    for i in range(len(RSL)):
        for j in range(len(FPS)):
            s, t = get_performance(pf_score, pf_time, i, j)
            scores[i][j] = s
            times[i][j] = t
    return np.array(scores), np.array(times)

def adapt_one(pf_score, pf_time, acc_threshold):
    scores, times = get_all_performances(pf_score, pf_time)
    n = len(pf_score[0])
    n_sec = n//30
    res_conf = np.zeros((n_sec, 2), dtype=int)
    res_score = np.zeros(n_sec)
    res_time = np.zeros(n_sec)
    for i in range(n_sec):
        flag = scores[:,:,i] >= acc_threshold
        idx = times[flag, i].argmin()
        xl, yl = np.nonzero(flag)
        x, y = xl[idx], yl[idx]
        res_conf[i] = x, y
        res_score[i] = scores[x,y,i]
        res_time[i] = times[x,y,i]
    return res_conf, res_score, res_time


# %% run

folders = [f for f in os.listdir(filepath) if f.startswith('output_') and os.path.isdir(filepath+f)]

acc_threshold = 0.95

for fld in folders:
    print('processing',fld)
    pf_score, pf_time = load_profile(filepath+fld)
    n = len(pf_score[0])
    n_sec = n//30
    confs, scores, times = adapt_one(pf_score, pf_time, acc_threshold)
    result = np.zeros((n_sec, 4))
    result[:,0] = np.array(RSL)[confs[:,0]]
    result[:,1] = np.array(FPS)[confs[:,1]]
    result[:,2] = scores
    result[:,3] = times
    np.savetxt('data/pose/'+fld+'-conf.csv', result, fmt='%d,%d,%f,%f',
               delimiter=',', header='resolution,fps,score,time')


