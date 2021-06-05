# -*- coding: utf-8 -*-

import time

import numpy as np
import torch

from app.rangechecker import RangeChecker

from videoholder import VideoHolder
from track import centroidtracker
from util import box_center

__all__ = ['CarRecord',  'FeatureExtractor', 'CarCounter']

class CarRecord():
    def __init__(self, oid, pos, fidx):
        self.oid = oid # object ID
        self.pos = pos # last position
        self.dir = 0.0 # direction
        self.off = 0.0 # offset to checking line
        self.over = False # whether passed the checking line
        self.last = fidx # last appear frame index
        
    def __repr__(self):
        return '{id-%d, (%f, %f) dir: %f off: %f over: %d}' \
            % (self.oid, *self.pos, self.dir, self.off, self.over)
    
    def update(self, fidx, pos, rchecker: RangeChecker):
        dir = rchecker.direction(self.pos, pos)
        self.dir = 0.5*(self.dir + dir)
        self.off = rchecker.offset(pos)
        self.pos = pos
        self.last = fidx
        

class FeatureExtractor():
    def __init__(self, dim_conf=2, num_prev=2, decay=1):
        self.dim_conf = dim_conf
        self.dim_speed = 3 # average/median/std of speed
        self.dim_unit = self.dim_speed + 1 + dim_conf # speed, count, confs
        self.dim_feat = num_prev*self.dim_unit + self.dim_speed
        self.num_prev = num_prev
        self.decay = decay
        # running time data
        self.feature = torch.zeros(self.dim_feat)
        self.buffer = {} # last location for speed
        self.temp = [] # speed of 
    
    def reset(self):
        self.feature = torch.zeros(self.dim_feat)
        self.buffer = {}
        self.temp = []
    
    def update(self, objects):
        speeds = []
        for oid, c in objects.items():
            if oid in self.buffer:
                old = self.buffer[oid]
                s = c - old
                speeds.append(s)
                self.buffer[oid] = c
            else:
                self.buffer[oid] = c
        self.temp.extend(speeds)
    
    def move(self, cnt, conf):
        n = self.dim_unit * self.num_prev
        if self.num_prev > 0:
            self.feature[:n] = self.feature[self.dim_speed:]
            if self.decay != 1:
                # decay the speed
                for i in range(self.num_prev):
                    a = i*self.dim_unit
                    b = a+self.dim_speed
                    self.feature[a:b] *= self.decay
        self.feature[n - self.dim_conf : n] = conf
        if len(self.temp) == 0:
            sa = sm = ss = 0.0
        else:
            sa = np.mean(self.temp)
            sm = np.median(self.temp)
            ss = np.std(self.temp)
        f = torch.tensor([sa, sm, ss, cnt, *conf]).float()
        #f = (f - self.feat_mean)/self.feat_std
        self.feature[n:] = f
        
    def get(self):
        return self.feature
                

class CarCounter():
    
    def __init__(self, video:VideoHolder, rng:RangeChecker,
                 dmodel, rs0, fr0, disappear_time:float=0.5,
                 cmodel=None, feat_gen=None,
                 rs_list=None, fr_list=None,
                 pboxes_list=None, times_list=None
                 ):
        self.video = video
        self.range = rng
        self.dmodel = dmodel
        self.rs = rs0
        self.fr = fr0
        self.cmodel = cmodel
        self.feat_gen = feat_gen
        self.dsap_time = disappear_time
        self.dsap_frame = max(1, int(disappear_time*video.fps))
        n = max(0, int(disappear_time*video.fps/self.fr))
        self.tracker = centroidtracker.CentroidTracker(n)
        # pre-computed result
        self.rs_list = rs_list
        self.fr_list = fr_list
        assert pboxes_list is None or len(pboxes_list) == len(rs_list)
        self.pboxes_list = pboxes_list
        assert times_list is None or len(times_list) == len(rs_list)
        self.times_list = times_list
        # running time data
        self.obj_info = {} # objectID -> CarRecord(dir, over)
        self.sidx = 0 # second id
        
    
    def change_fr(self, fr):
        self.fr = fr
        n = max(1, int(self.dsap_time*self.video.fps/fr))
        self.tracker.maxDisappeared = n
    
    def change_rs(self, rs):
        self.rs = rs
        
    def reset(self):
        self.tracker.reset()
        if self.feat_gen is not None:
            self.feat_gen.reset()
        self.obj_info = {}
        self.sidx = 0
        
    def recognize_cars(self, frame):
        if self.rs is not None:
            lbls, scores, boxes = self.dmodel.process(frame, self.rs)
        else:
            lbls, scores, boxes = self.dmodel.process(frame)
        return boxes
        
    def count(self, fidx, objects):
        c = 0
        # count those passed the checking line
        for oid, center in objects.items():
            if oid in self.obj_info:
                oinfo = self.obj_info[oid]
            else:
                oinfo = CarRecord(oid, center, fidx)
                self.obj_info[oid] = oinfo
            oinfo.update(fidx, center, self.range)
            # count those move over the checking line
            if oinfo.over == False and \
                    ((oinfo.dir > 0 and oinfo.off > 0) or
                     (oinfo.dir < 0 and oinfo.off < 0)):
                oinfo.over = True
                c += 1
        return c
    
    def clear_buffer(self, fidx):
        to_remove = []
        for oid, oinfo in self.obj_info.items():
            if fidx - oinfo.last > self.dsap_frame:
                to_remove.append(oid)
        for oid in to_remove:
            del self.obj_info[oid]
        
    def update(self, fidx):
        frame = self.video.get_frame(fidx)
        if self.pboxes_list is None:
            boxes = self.recognize_cars(frame)
        else:
            # use pre-computed result
            rs_idx = self.rs_list.index(self.rs)
            boxes = self.pboxes_list[rs_idx][fidx]
        centers = box_center(boxes)
        # filter cars far from the checking line
        if len(centers) > 0:
            flag = self.range.in_track(centers)
            centers_in_range = centers[flag]
        else:
            centers_in_range = []
        # count cars
        objects = self.tracker.update(centers_in_range)
        if self.feat_gen is not None:
            self.feat_gen.update(objects)
        c = self.count(fidx, objects)
        return c
    
    def process_one_second(self, rs, fr):
        cnt = 0
        t = 0.0
        self.change_rs(rs)
        self.change_fr(fr)
        fidx = int(self.sidx * self.video.fps)
        end_fidx = int((self.sidx+1) * self.video.fps)
        while fidx < end_fidx:
            cnt += self.update(fidx)
            fidx += fr
        if self.feat_gen is not None:
            self.feat_gen.move(cnt, (rs, fr))
        t = time.time() - t
        return cnt, t

    def process(self):
        n_second = self.video.length_second(True)
        times = np.zeros(n_second, float)
        counts = np.zeros(n_second, int)
        confs = np.zeros((n_second, 2), int)
        for i in range(n_second):
            cnt, t = self.process_one_second(self.rs, self.fr)
            if self.cmodel is not None:
                tt = time.time()
                feature = self.feat_gen.get()
                rs, fr = self.cmodel(feature)
                self.rs = rs
                self.fr = fr
                t += time.time() - tt
            self.times[i] = t
            self.counts[i] = cnt
            self.confs[i] = (self.rs, self.fr)
        return times, counts, confs
    
    
    ##########
    
    def raw_profile(self, idx_start=0, idx_end=None, show_progress=None):
        assert idx_start < self.video.num_frame
        if idx_end is None:
            idx_end = self.video.num_frame
        assert idx_start <= idx_end <= self.video.num_frame
        print(idx_start, idx_end)
        idx = idx_start
        res_times = np.zeros(self.video.num_frame)
        res_boxes = []
        while idx < idx_end:
            t = time.time()
            
            f = self.video.get_frame(idx)
            boxes = self.recognize_cars(f)
            centers = box_center(boxes)
            boxes = self.filter_cars(boxes, centers)

            t = time.time() - t
            res_times[idx] = t
            res_boxes.append(boxes)
            idx += 1
            
            if show_progress is not None and idx % show_progress == 0:
                speed = 1.0/res_times[idx-show_progress:idx].mean()
                eta = (idx_end - idx) / speed
                print('iter %d: total-time(s): %f, speed(fps): %f, eta: %d:%d' %
                      (idx, res_times[:idx].sum(), speed, eta//60, eta%60))
        return res_times, res_boxes
    
    def count_with_raw_boxes(self, boxes, fr=None):
        fps = int(np.ceil(self.video.fps))
        if fr is None:
            fr = self.conf.fr
        else:
            self.change_fr(fr)
        n_second = len(boxes) // fps
        #n_frame = int(n_second * fps) // fr
        
        self.tracker.reset()
        counts = np.zeros(n_second, int)
        times = np.zeros(n_second)
        last_second = 0
        t = time.time()
        c = 0
        for idx in range(0, int(n_second*fps), fr):
            second = idx // fps
            if second != last_second:
                tt = time.time()
                counts[last_second] = c
                times[last_second] = tt - t
                t = tt
                c = 0
                last_second = second
                
            bs = boxes[idx]
            if len(bs) == 0:
                continue
            cs = box_center(bs)
            flag = self.range.in_track(cs)
            objects = self.tracker.update(cs[flag])
            c += self.count(idx, objects)
        if idx // fps == last_second:
            counts[last_second] = c
            times[last_second] = time.time() - t
        return times, counts
    
    @staticmethod
    def group_to_segments(data, segment_legnth):
        n = len(data)
        n_segment = n // segment_legnth
        n = n_segment * segment_legnth
        res = data[:n].reshape((n_segment, segment_legnth)).sum(1)
        return res
    
    @staticmethod
    def compute_accuray(counts, gtruth, segment=1):
        n = len(gtruth)
        n_segment = n // segment
        n = n_segment * segment
        counts = counts[:n].reshape((n_segment, segment)).sum(1)
        gtruth = gtruth[:n].reshape((n_segment, segment)).sum(1)
        up = np.array([counts, gtruth]).max(0)
        down = np.array([counts, gtruth]).min(0)
        accuracy = np.zeros(n_segment)
        for i in range(n_segment):
            if up[i] == 0:
                accuracy[i] = 1.0
            else:
                accuracy[i] = down[i] / up[i]
        return accuracy
    
    def generate_conf_result(self, ptimes, ctimes, counts, gtruth, segment=1):
        # ptimes: frame level
        # ctimes, counts, gtruth: second level
        # segment: number of seconds in each segment
        fps = int(np.ceil(self.video.fps))
        pattern = np.arange(0, fps, self.conf.fr)
        n_second = len(ptimes) // fps
        #n_segment = n_second // segment
        #n = n_segment * segment * fps
        
        accuracy = self.compute_accuray(counts, gtruth, segment)
        t = ptimes[:n_second*fps].reshape((n_second, fps))
        t = t[:,pattern].sum(1)
        times = ctimes + t
        times = self.group_to_segments(times, segment)
        #times = times[:n_segment*segment].reshape((n_segment, segment)).sum(1)
        return times, accuracy
