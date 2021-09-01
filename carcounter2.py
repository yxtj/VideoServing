# -*- coding: utf-8 -*-

import time

import numpy as np

from app.rangechecker import RangeChecker
from framepreprocess import  FramePreprocessor
from model.framedecision import DecisionModel

from videoholder import VideoHolder
from track import centroidtracker
import util

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
    def __init__(self, dim_conf=2, num_prev=1, clear_threshold=10):
        self.dim_speed = 3 # average/median/std of speed
        self.dim_size = 1+3 # avg size of global bbox, ams of object size
        self.dim_aratio = 3 # ams of aspect ratio
        self.dim_count = 2 # average/std of object number
        self.dim_conf = dim_conf
        self.dim_unit = self.dim_speed + self.dim_size + self.dim_aratio + self.dim_count + dim_conf
        self.dim_feat = (num_prev+1)*self.dim_unit
        self.num_prev = num_prev
        self.threshold = clear_threshold
        # running time data
        self.sidx = 0
        self.feature = np.zeros(self.dim_feat)
        self.buffer = {} # last location for speed
        self.stemp = [] # speed of each object
        self.zbtemp = [] # global bounging box size (bbox for all objects)
        self.zatemp = [] # active area size of each objects
        self.rtemp = [] # aspect ratio of each object
        self.ctemp = [] # number of objects
    
    def reset(self):
        self.feature = np.zeros(self.dim_feat)
        self.buffer = {}
        self.stemp = []
        self.zbtemp = []
        self.zatemp = []
        self.rtemp = []
        self.ctemp = []
    
    def update(self, objects, elapse, boxes):
        speeds = []
        for oid, c in objects.items():
            if oid in self.buffer:
                old, _ = self.buffer[oid]
                s = (c - old)/elapse
                speeds.append(s)
                self.buffer[oid] = (c, self.sidx)
            else:
                self.buffer[oid] = (c, self.sidx)
        self.stemp.extend(speeds)
        if len(boxes) != 0:
            bz = util.box_size(util.box_super(boxes))
            self.zbtemp.append(bz)
            azs = util.box_size(boxes)
            self.zatemp.extend(azs)
            r = util.box_aratio(boxes)
            self.rtemp.extend(r)
        self.ctemp.append(len(boxes))
    
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
    
    def __ams_of_list__(temp):
        if len(temp) == 0:
            a,m,s = 0.0, 0.0, 0.0
        else:
            a = np.mean(temp)
            m = np.median(temp)
            s = np.std(temp)
        return a,m,s
    
    

class CarCounter():
    
    def __init__(self, video:VideoHolder, rng:RangeChecker,
                 dmodel, rs0, fr0, # detect
                 disappear_time:float=0.8, # track
                 fpp:FramePreprocessor=None, fmodel:DecisionModel=None, # frame decision
                 feat_gen:FeatureExtractor=None, cmodel=None, # configure prediction
                 rs_list=None, fr_list=None,
                 pboxes_list=None, times_list=None,
                 bsize_list=None, asize_list=None
                 ):
        self.video = video
        self.range = rng
        self.dmodel = dmodel
        self.rs = rs0
        self.rscale = rs0 / max(video.width, video.height) if rs0 else 1
        self.fr = fr0
        self.feat_gen = feat_gen
        self.cmodel = cmodel
        self.dsap_time = disappear_time
        self.dsap_frame = max(1, int(disappear_time*video.fps))
        n = max(0, int(disappear_time*video.fps/self.fr))
        self.tracker = centroidtracker.CentroidTracker(n)
        # frame preprocess and decision
        self.fpp = fpp
        self.fmodel = fmodel
        if fmodel is not None:
            assert fpp is not None
        # pre-computed result
        self.rs_list = rs_list
        self.fr_list = fr_list
        self.pboxes_list = pboxes_list
        self.times_list = times_list
        if rs_list is not None:
            assert pboxes_list is None or len(pboxes_list) == len(rs_list)
            assert times_list is None or len(times_list) == len(rs_list)
        else:
            assert pboxes_list is None or pboxes_list.ndim == 1
            assert times_list is None or times_list.ndim == 1
            assert pboxes_list is None and times_list is None or pboxes_list.shape == times_list.shape
        self.bsize_list = bsize_list
        self.asize_list = asize_list
        # running time data
        self.obj_info = {} # objectID -> CarRecord(dir, over)
        self.sidx = 0 # second id
        
    
    def change_fr(self, fr):
        self.fr = fr
        n = max(1, int(self.dsap_time*self.video.fps/fr))
        self.tracker.maxDisappeared = n
    
    def change_rs(self, rs):
        self.rs = rs
        if rs is not None:
            self.rscale = rs / max(self.video.width, self.video.height)
        else:
            self.rscale = 1
        
    def reset(self):
        self.tracker.reset()
        if self.fpp is not None:
            self.fpp.reset()
        if self.feat_gen is not None:
            self.feat_gen.reset()
        self.change_rs(self.rs)
        self.change_fr(self.fr)
        self.obj_info = {}
        self.sidx = 0
        
    def get_track_state(self):
        return self.tracker.get_state(), self.obj_info.copy()
    
    def set_track_state(self, state):
        self.tracker.set_state(state[0])
        self.obj_info = state[1]
        
    def recognize_cars(self, frame, rs=None):
        if rs is not None:
            lbls, scores, boxes = self.dmodel.process(frame, rs)
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
   
    def __get_boxes__(self, fidx, rs, m_diff=False):
        if self.pboxes_list is None:
            # process online
            t1 = time.time()
            frame = self.video.get_frame(fidx)
            if m_diff and self.fpp is not None:
                rect, f, mask = self.fpp.apply(frame)
                w, h = rect[2]-rect[0], rect[3]-rect[1]
                if f.size != 0:
                    sz = max(w, h)
                    sz = int(sz*self.rscale) if self.rs else sz
                    boxes = self.recognize_cars(f, sz)
                else:
                    boxes = np.zeros((0,4))
            else:
                boxes = self.recognize_cars(frame, rs)
            t1 = time.time() - t1
        elif self.rs_list is not None:
            # use pre-computed result (resolution-frame)
            rs_idx = self.rs_list.index(rs)
            boxes = self.pboxes_list[rs_idx][fidx]
            t1 = self.times_list[rs_idx][fidx]
        else:
            # use pre-compuated result (frame)
            boxes = self.pboxes_list[fidx]
            t1 = self.times_list[fidx]
        return t1, boxes
   
    def update(self, fidx, rs, m_diff=False):
        # part 1: get object boxes from image
        t1, boxes = self.__get_boxes__(fidx, rs, m_diff)
        # part 2: get counting from boxes (filtering, tracking, checking)
        t2 = time.time()
        # filter cars that are far from the checking line
        if len(boxes) > 0:
            centers = util.box_center(boxes)
            flag = self.range.in_track(centers)
            centers_in_range = centers[flag]
        else:
            centers_in_range = []
        # count cars
        objects = self.tracker.update(centers_in_range)
        c = self.count(fidx, objects)
        # part 3: generate features
        if self.feat_gen is not None:
            self.feat_gen.update(objects, self.fr/self.video.fps, boxes)
        t2 = time.time() - t2
        return c, t1 + t2
    
    def process_one_second(self, rs, fr, m_diff=False):
        cnt = 0
        t1 = time.time()
        self.change_rs(rs)
        self.change_fr(fr)
        fidx = int(self.sidx * self.video.fps)
        end_fidx = int((self.sidx+1) * self.video.fps)
        t1 = time.time() - t1
        t2 = 0.0
        while fidx < end_fidx:
            c, t = self.update(fidx, rs, m_diff)
            cnt += c
            t2 += t
            fidx += fr
        return cnt, t1 + t2
    
    def process_period(self, fidx_start, fidx_end, rs, fr, m_diff=False):
        cnt = 0
        self.change_rs(rs)
        self.change_fr(fr)
        tt = 0.0
        for fidx in range(fidx_start, fidx_end, fr):
            c, t = self.update(fidx, rs, m_diff)
            cnt += c
            tt += t
        return cnt, tt

    def process(self, start_second=0, n_second=None):
        n = self.video.length_second(True)
        if n_second is None:
            n_second = n - start_second
        else:
            n_second = min(n_second, n-start_second)
        times = np.zeros(n_second, float)
        counts = np.zeros(n_second, int)
        confs = np.zeros((n_second, 2), int)
        for i in range(start_second, start_second+n_second):
            self.sidx = i
            cnt, t = self.process_one_second(self.rs, self.fr, False)
            if self.feat_gen is not None:
                # update feature
                tt = time.time()
                self.feat_gen.move((self.rs, self.fr))
                t += time.time() - tt
            if self.cmodel is not None:
                # predict next configuration
                tt = time.time()
                feature = self.feat_gen.get()
                #rs, fr = self.cmodel(feature)
                rs, fr, mi = self.cmodel(feature)
                self.rs = rs
                self.fr = fr
                self.mi = mi
                t += time.time() - tt
            times[i] = t
            counts[i] = cnt
            confs[i] = (self.rs, self.fr)
        return times, counts, confs
    
    def process_with_conf(self, conf_list):
        n_second = self.video.length_second(True)
        n_second = min(n_second, len(conf_list))
        times = np.zeros(n_second, float)
        counts = np.zeros(n_second, int)
        for i in range(n_second):
            self.sidx = i
            fr, mi = conf_list[i]
            cnt, t = self.process_one_second(self.rs, fr, mi==1)
            times[i] = t
            counts[i] = cnt
        return times, counts
    
    ##########
    
    def precompute_whole_frame(self, idx_start=0, idx_end=None,
                               filtering:bool=False, show_progress:int=None):
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
            boxes = self.recognize_cars(f, self.rs)
            if filtering and len(boxes) > 0:
                centers = util.box_center(boxes)
                flag = self.range.in_track(centers)
                boxes = boxes[flag]

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
    
    def precompute_frame_difference(self, fr, idx_start=0, idx_end=None,
                                    filtering:bool=False, show_progress:int=None):
        assert isinstance(fr, int) and fr > 0
        assert self.fpp is not None
        assert idx_start < self.video.num_frame
        if idx_end is None:
            idx_end = self.video.num_frame
        assert idx_start <= idx_end <= self.video.num_frame
        n = (idx_end - idx_start + fr - 1) // fr
        print(idx_start, idx_end, n)
        res_times = np.zeros(n)
        res_boxes = [None for _ in range(n)]
        res_rect = np.zeros((n,4), dtype=int)
        res_mask_size = np.zeros(n)
        i = 0
        for idx in range(idx_start, idx_end, fr):
            t = time.time()
            
            frame = self.video.get_frame(idx)
            rect, f, mask = self.fpp.apply(frame)
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            if f.size != 0:
                sz = max(w, h)
                sz = int(sz*self.rscale) if self.rs else sz
                boxes = self.recognize_cars(f, sz)
            else:
                boxes = np.zeros((0,4))
            if filtering and len(boxes) > 0:
                centers = util.box_center(boxes)
                flag = self.range.in_track(centers)
                boxes = boxes[flag]

            t = time.time() - t
            res_times[i] = t
            res_boxes[i] = boxes
            res_rect[i] = rect
            res_mask_size[i] = mask.mean()/255
            i += 1
            if show_progress is not None and i % show_progress == 0:
                speed = 1.0/res_times[i-show_progress:i].mean()
                eta = (n - i) / speed
                print('idx %d: total-time(s): %f, speed(fps): %f, eta: %d:%d' %
                      (idx, res_times[:i].sum(), speed, eta//60, eta%60))
        return res_times, res_boxes, res_rect, res_mask_size
    
    def count_with_raw_boxes(self, boxes, fr=None):
        fps = int(np.ceil(self.video.fps))
        if fr is None:
            fr = self.fr
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
            cs = util.box_center(bs)
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
        pattern = np.arange(0, fps, self.fr)
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

# %% precomputed data io
    
def save_precompute_data_diff(file, rng_param, model_param, fpp_param,
                              width, fr, times, boxes, rects, mask_size):
    np.savez(file, rng_param=np.array(rng_param,object), 
             model_param=np.array(model_param, object),
             fpp_param=np.array(fpp_param, object),
             width=width, fr=fr, 
             times=times, boxes=np.array(boxes, object),
             rects=rects, mask_size=mask_size
             )
    
def load_precompute_data_diff(file):
    with np.load(file, allow_pickle=True) as data:
        rng_param = data['rng_param'].tolist()
        model_param = data['model_param'].tolist()
        fpp_param = data['fpp_param'].tolist()
        width = data['width'].item()
        fr = data['fr'].item()
        times = data['times']
        boxes = data['boxes'].tolist()
        rects = data['rects']
        mask_size = data['mask_size']
        return rng_param, model_param, fpp_param, width, fr, times, boxes, rects, mask_size

# %% test

def __test_conf__():
    v3=VideoHolder('E:/Data/video/s3.mp4')
    rng3=RangeChecker('h', 0.5, 0.2, 0.1)
    
    v4=VideoHolder('E:/Data/video/s4.mp4')
    rng4=RangeChecker('h', 0.5, 0.2, 0.1)
    
    v5=VideoHolder('E:/Data/video/s5.mp4')
    rng5=RangeChecker('v', 0.75, 0.2, 0.1)
    
    v7=VideoHolder('E:/Data/video/s7.mp4')
    rng7=RangeChecker('h', 0.45, 0.2, 0.1)
    
    fps_list = [25,30,20,30]
    fr_list=[1,2,5,15,30]
    
    import framepreprocess
    import detect.yolowrapper
    fpp=framepreprocess.FramePreprocessor()
    dmodel=detect.yolowrapper.YOLO_torch('yolov5s',0.5,(2,3,5,6,7))
    cc=CarCounter(v4,rng4,dmodel,None,2,0.8,fpp)
    
    for fr in fr_list:
        print(fr)
        cc.reset()
        tl,bl,rl,ml=cc.precompute_frame_difference(fr,show_progress=200)
        save_precompute_data_diff(
            'data/s4/s4-diff-raw-%d'% fr,['h', 0.5, 0.2, 0.1],
            ['yolov5s',0.5,(2,3,5,6,7)],
            [True,True,'max',True,True,100,2,10,5,5,0.002,0.2,3],
            None,fr,tl,bl,rl,ml)
        
