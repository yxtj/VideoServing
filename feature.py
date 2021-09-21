# -*- coding: utf-8 -*-

import numpy as np
import util

class FeatureExtractor():
    def __init__(self, dim_conf=2, num_prev=1, clear_threshold=10):
        self.dim_size = 1+5 # avg size of global bbox, min/median/max/mean/std of object size
        self.dim_distance = 5 # min/median/max/mean/std of cross-object distance
        self.dim_speed = 4 # mean/median/max/std of speed
        self.dim_sratio = 5 # min/median/max/mean/std of size change rate
        self.dim_aratio = 4 # mean/median/absmax/std of aspect ratio change rate
        self.dim_count = 4 # mean/min/max/std of object number
        self.dim_conf = dim_conf
        self.dim_unit = self.dim_size + self.dim_distance + \
            self.dim_speed + self.sratio + self.dim_aratio + \
            self.dim_count + self.dim_conf
        self.dim_feat = (num_prev+1)*self.dim_unit
        self.num_prev = num_prev
        self.threshold = clear_threshold
        # running time data
        self.sidx = 0
        self.feature = np.zeros(self.dim_feat)
        self.buffer = {} # last location for speed
        self.bf_size = [] # each object's bounding box size
        self.bf_gsize = [] # global bounging box size (bbox for all objects)
        self.bf_distance = []
        self.bf_speed = [] # speed of each object
        self.bf_aratio = [] # aspect ratio of each object
        self.bf_count = [] # number of objects
    
    def reset(self):
        self.feature = np.zeros(self.dim_feat)
        self.buffer = {}
        self.bf_size = []
        self.bf_gsize = []
        self.bf_distance = []
        self.bf_speed = []
        self.bf_aratio = []
        self.bf_count = []
    
    def update(self, objects, etime, boxes):
        speeds = []
        for oid, c in objects.items():
            if oid in self.buffer:
                old, _ = self.buffer[oid]
                s = (c - old)/etime
                speeds.append(s)
                self.buffer[oid] = (c, self.sidx)
            else:
                self.buffer[oid] = (c, self.sidx)
            a = c
        self.bf_speed.extend(speeds)
        if len(boxes) != 0:
            # global bounding box size
            gbz = util.box_size(util.box_super(boxes))
            self.bf_gsize.append(gbz)
            # object size
            ozs = util.box_size(boxes)
            self.zatemp.extend(ozs)
            # distance
            d = util.box_distance(boxes)
            u = np.triu_indices(len(boxes))
            self.bf_distance(d[u])
        self.bf_count.append(len(boxes))
    
    def move(self, conf):
        self.sidx += 1
        # clear buffer
        bound = self.sidx - self.threshold
        self.buffer = { k:(c,p) for k,(c,p) in self.buffer.items() if p > bound }
        # move existing slots
        if self.num_prev > 0:
            self.feature[:-self.dim_unit] = self.feature[self.dim_unit:]
        # put current buffer into feature
        if len(self.stemp) == 0:
            ss_s = (0.0, 0.0, 0.0)
        else:
            ss_s = FeatureExtractor.__ams_of_list__(self.stemp)
        ss_bz = np.mean(self.zbtemp) if len(self.zbtemp) > 0 else 0.0
        ss_az = FeatureExtractor.__ams_of_list__(self.zatemp)
        ss_r = FeatureExtractor.__ams_of_list__(self.rtemp)
        ss_c = [np.mean(self.ctemp), np.std(self.ctemp)]
        f = np.array([*ss_s, ss_bz, *ss_az, *ss_r, *ss_c, *conf])
        #f = (f - self.feat_mean)/self.feat_std
        self.feature[-self.dim_unit:] = f
        
    def get(self):
        return self.feature
    
    @staticmethod
    def extract(data, method):
        if method == 'min':
            return np.min(data)
        elif method == 'max':
            return np.max(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'median':
            return np.median(data)
        elif method == 'std':
            return np.std(data)
        elif method == 'absmax':
            return np.max(np.abs(data))
        elif method == 'absmin':
            return np.min(np.abs(data))
    
    @staticmethod
    def extract_feature(data, method_list):
        if len(data) == 0:
            return tuple( 0.0 for i in range(len(method_list)) )
        return tuple( FeatureExtractor.extract(data, m) for m in method_list )
        
    
    def __ams_of_list__(temp):
        if len(temp) == 0:
            a,m,s = 0.0, 0.0, 0.0
        else:
            a = np.mean(temp)
            m = np.median(temp)
            s = np.std(temp)
        return a,m,s
    
    
    