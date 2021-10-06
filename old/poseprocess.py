# -*- coding: utf-8 -*-

import numpy as np
from app import posebase
#from ..app import posebase


RS_LIST=[240, 360, 480, 720, 840]


def load_raw_and_filter(fn):
    data = np.load(fn, allow_pickle=True)
    times = data['times']
    poses = data['poses']
    scores = data['scores']
    for i,s in enumerate(scores):
        if len(s) > 1:
            p = s.argmax()
            poses[i] = poses[i][p]
        elif len(s) == 1:
            poses[i] = poses[i].reshape(17,3)
        else:
            poses[i] = np.zeros((17,3))
    poses = np.array([*poses])
    return poses, times

def load_video_profile(folder):
    poses = []
    times = []
    for rs in RS_LIST:
        p, t = load_raw_and_filter(folder+'/raw-%s.npz'%rs)
        poses.append(p)
        times.append(t)
    return poses, times


def estimate_pose(pose_last, pose_now, speed, alpha, n):
    s = pose_now[:,:2] - pose_last[:,:2]
    idx = np.nonzero(np.logical_and(pose_last[:,2]!=0, pose_now[:,2]!=0))
    if len(idx) == 17:
        speed = (1-alpha)*speed + alpha*s
    else:
        speed[idx] = (1-alpha)*speed[idx] + alpha*s[idx]
    poses = np.zeros((n, 17, 3))
    poses[:,:,2] = 2
    for i in range(n):
        poses[i,:,:2] = pose_now[:,:2] + speed*i
    return speed, poses


def process(pose_list, time_list, gpose_list, fps, length, alpha=0.8, max_fps=30):
    speed = np.zeros((17, 2))
    last = np.zeros((17, 3))
    fr = int(np.ceil(max_fps/fps))
    idxes = np.arange(0, max_fps, fr)
    alpha = alpha/fr
    times = np.zeros(length)
    scores = np.zeros(length)
    for i in range(length):
        off = i*max_fps
        t = time_list[off + idxes].sum()
        times[i] = t
        poses = np.zeros((max_fps, 17, 3))
        #poses[idxes] = pose_list[off + idxes]
        for j in range(fps):
            now = pose_list[off + idxes[j]]
            speed, p = estimate_pose(last, now, speed, alpha, fr)
            last = now
            poses[j:j+fr] = p
        s = posebase.computeOKS_NtoN(gpose_list[off:off+max_fps], poses)
        #s=0
        scores[i] = s.mean()
    return scores, times


def process_video(profile, fps_list, rsl_list, length, alpha=0.8, max_fps=30):
    nrsl = len(rsl_list)
    nfps = len(fps_list)
    pose_lists, time_lists = profile
    assert len(pose_lists) == nrsl + 1
    gt_pose_list = pose_lists[-1]
    scores = np.zeros((nrsl, nfps, length))
    times = np.zeros((nrsl, nfps, length))
    for i,rsl in enumerate(rsl_list):
        for j,fps in enumerate(fps_list):
            s, t = process(pose_lists[i], time_lists[i], gt_pose_list, fps, length)
            scores[i,j] = s
            times[i,j] = t
    return scores, times


# %% main

def __main__():
    vn_list=['001_dance.mp4', '002_dance.mp4', '003_dance.mp4', '004_dance.mp4',
             '005_dance.mp4', '006_yoga.mp4', '007_yoga.mp4', '008_cardio.mp4',
             '009_cardio.mp4', '010_cardio.mp4', '011_dance.mp4', '012_dance.mp4',
             '013_dance.mp4', '014_dance.mp4', '015_dance.mp4', '016_dance.mp4',
             '017_dance.mp4', '018_dance.mp4', '019_dance.mp4', '020_dance.mp4',
             '021_dance.mp4', '022_dance.mp4', '023_dance.mp4', '024_dance.mp4',
             '025_dance.mp4', '026_dance.mp4', '027_dance.mp4', '028_dance.mp4',
             '029_dance.mp4', '030_dance.mp4', '031_dance.mp4', '032_dance.mp4',
             '033_dance.mp4', '034_dance.mp4', '035_dance.mp4']
    vfps_list=[23.97, 29.97, 30,    24.97, 24.97, 29.97, 29.97, 29.97,
               29.97, 24.97, 29.97, 29.97, 29.97, 29.97, 29.97, 23.97,
               29.97, 29.97, 29.97, 23.97, 23.97, 30   , 30   , 30,
               30   , 30   , 30   , 30   , 30   , 30   , 30   , 30,
               29.97, 29.97, 29.97]
    vlen_list=[319, 217, 458, 549, 536, 451, 452, 454, 463, 498,
               451, 452, 460, 455, 460, 566, 451, 454, 463, 587,
               582, 475, 468, 474, 469, 468, 470, 473,  28, 471,
               472, 474, 464, 459, 473]
    
    # index of videos with 30 fps and at least 400 seconds
    vidx_list = [2,  5,  6,  7,  8, 10, 11, 12, 13, 14,
                 16, 17, 18, 21, 22, 23, 24, 25, 26, 27,
                 29, 30, 31, 32, 33, 34]
    # 10 selected videos
    vidx_list = [6, 7, 8, 11, 12, 13, 14, 18, 21, 24]
    
    nlength = 400
    
    profile = load_video_profile('E:/Data/video/pose/output_007')
    
    

