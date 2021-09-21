# -*- coding: utf-8 -*-

import numpy as np
import util


def summarize(data, method):
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
    else:
        raise ValueError('Unsupported mehtod: '+str(method))

def summarize_list(data, method_list):
    if len(data) == 0:
        return tuple( 0.0 for i in range(len(method_list)) )
    return tuple( summarize(data, m) for m in method_list )


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
        self.buffer = {} # last position (bbox)
        self.bf_size = [] # each object's bounding box size
        self.bf_gsize = [] # global bounging box size (bbox for all objects)
        self.bf_distance = []
        self.bf_speed = [] # speed of each object
        self.bf_cr_size= [] # size cr of each object
        self.bf_cr_aratio = [] # aspect ratio cr of each object
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
    
    def update(self, objects, etime):
        '''
        Params:
            objects: dictionary of {id:box} for objects.
            etime: elapsed time since last update.
        '''
        # tracking-based features (speed, size-cr, aspect-cr)
        for oid, box in objects.items():
            sz = util.box_size(box)
            cn = util.box_center(box)
            ar = util.box_aratio(box)
            # existing object, add tracking-based features (speed, size-cr, aspect-cr)
            if oid in self.buffer:
                obox, (osz, ocn, oar), _ = self.buffer[oid]
                self.bf_speed.append((cn-ocn)/etime)
                self.bf_cr_size.append((sz-osz)/etime)
                self.bf_cr_aratio.append((ar-oar)/etime)
            # static individual features (size)
            self.buffer[oid] = (box, (sz, cn, ar), self.sidx)
            self.bf_size.append(sz)
        # static global features (gsize, distance)
        boxes = [b for k,b in objects.items()]
        gbz = util.box_size(util.box_super(boxes))
        self.bf_gsize.append(gbz)
        d = util.box_distance(boxes)
        u = np.triu_indices(len(boxes), 1)
        self.bf_distance.extend(d[u])
        self.bf_count.append(len(boxes))
    
    def move(self, conf):
        self.sidx += 1
        # clear buffer
        bound = self.sidx - self.threshold
        self.buffer = { k:(b,s,p) for k,(b,s,p) in self.buffer.items() if p > bound }
        # move existing slots
        if self.num_prev > 0:
            self.feature[:-self.dim_unit] = self.feature[self.dim_unit:]
        # put current buffer into feature
        f_gsize = summarize_list(self.bf_gsize, ['mean'])
        f_size = summarize_list(self.bf_size, ['min','median','max','mean','std'])
        f_dist = summarize_list(self.bf_distance, ['min','median','max','mean','std'])
        f_speed = summarize_list(self.bf_speed, ['min','median','max','mean','std'])
        f_crsize= summarize_list(self.bf_cr_size, ['min','median','max','mean','std'])
        f_craratio = summarize_list(self.bf_cr_aratio, ['min','median','max','mean','std'])
        f_count = summarize_list(self.bf_count, ['min','median','max','mean','std'])
        f = [*f_gsize, *f_size, *f_dist, *f_speed, *f_crsize, *f_craratio, *f_count, *conf]
        self.feature[-self.dim_unit:] = f
        
    def get(self):
        return self.feature
    
