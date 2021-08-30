# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:11:06 2021

@author: yanxi
"""

import torch
import torch.nn as nn
import numpy as np
import time

import carcounter

# %% online process with configuration prediction

from util import box_center

class OnlineCarCounterPrediction():
    def __init__(self, cc:carcounter.CarCounter, cmodel:nn.Module,
                 num_prev, decay, rs_list, fr_list, feat_mean, feat_std,
                 conf0=None, pboxes_list=None, times_list=None):
        self.cc = cc
        self.cmodel = cmodel
        self.dim_feat_one = 6 # speed-avg, s-median, s-std, count, rs-idx, fr-idx
        self.dim_feat = cmodel.dim_in
        self.dim_conf = cmodel.dim_outs
        self.num_prev = num_prev
        self.decay = decay
        self.rs_list = rs_list
        self.fr_list = fr_list
        assert len(rs_list) == self.dim_conf[0]
        assert len(fr_list) == self.dim_conf[1]
        self.feat_mean = torch.tensor(feat_mean)
        self.feat_std = torch.tensor(feat_std)
        if conf0 is not None:
            self.last_conf = conf0
        else:
            self.last_conf = [d//2 for d in self.dim_conf]
        self.feature = torch.zeros(self.dim_feat)
        self.sidx = 0 # second idx
        #self.fidx = 0 # frame idx
        self.fps = int(cc.video.fps)
        self.buffer = {}
        assert pboxes_list is not None or len(pboxes_list) == len(rs_list)
        self.pboxes_list = pboxes_list
        assert times_list is not None or len(times_list) == len(rs_list)
        self.times_list = times_list
    
    def reset(self):
        self.cc.reset()
        self.last_conf = [d//2 for d in self.dim_conf]
        self.feature = torch.zeros(self.dim_feat)
        self.sidx = 0
        self.buffer = {}
    
    def cc_update_speed(self, idx, rs_idx):
        if self.pboxes_list is None:
            t0 = time.time()
            frame = self.cc.video.get_frame(idx)
            boxes = self.cc.recognize_cars(frame)
            t0 = time.time() - t0
        else:
            boxes = self.pboxes_list[rs_idx][idx]
            t0 = self.times_list[rs_idx][idx]
        centers = box_center(boxes)
        #boxes = self.filter_cars(boxes, centers)
        t = time.time()
        if len(centers) > 0:
            flag = self.cc.range.in_track(centers)
            centers_in_range = centers[flag]
        else:
            centers_in_range = []
        objects = self.cc.tracker.update(centers_in_range)
        speeds = []
        for oid, c in objects.items():
            if oid in self.buffer:
                old = self.buffer[oid]
                s = c - old
                speeds.append(s)
                self.buffer[oid] = c
            else:
                self.buffer[oid] = c
        c = self.cc.count(idx, objects)
        t = time.time() - t + t0
        return c, t, speeds
    
    def process_one_second(self, rs_idx, fr_idx):
        cnt = 0
        t = 0.0
        speeds = []
        rs = self.rs_list[rs_idx]
        fr = self.fr_list[fr_idx]
        self.cc.change_rs(rs)
        self.cc.change_fr(fr)
        fidx = self.sidx * self.fps
        end_fidx = (self.sidx+1) * self.fps
        while fidx < end_fidx:
            c, tf, s = self.cc_update_speed(fidx, rs_idx)
            cnt += c
            t += tf
            speeds.extend(s)
            fidx += fr
        
        t0 = time.time()
        if len(speeds) == 0:
            sa = sm = ss = 0.0
        else:
            sa = np.mean(speeds)
            sm = np.median(speeds)
            ss = np.std(speeds)
        f = torch.tensor([sa, sm, ss, cnt, rs_idx, fr_idx]).float()
        # normalize
        f = (f - self.feat_mean)/self.feat_std
        # merge into the existing feature
        n = self.dim_feat_one * self.num_prev
        self.feature[0:n] = self.feature[self.dim_feat_one:]*self.decay
        self.feature[n:] = f
        t = time.time() - t0 + t
        return cnt, t
    
    def next_second(self):
        if self.sidx < self.num_prev:
            rs_idx = self.last_conf[0]
            fr_idx = self.last_conf[1]
            c,t = self.process_one_second(rs_idx, fr_idx)
        else:
            t0 = time.time()
            with torch.no_grad():
                rs, fr = self.cmodel(self.feature)
                rs_idx = rs.argmax()
                fr_idx = fr.argmax()
            t0 = time.time() - t0
            c,t = self.process_one_second(rs_idx, fr_idx)
            t += t0
        self.last_conf = (rs_idx, fr_idx)
        self.sidx += 1
        return c, t, (rs_idx, fr_idx)
    
    def have_next(self):
        return self.sidx < self.cc.video.length_second(True)
