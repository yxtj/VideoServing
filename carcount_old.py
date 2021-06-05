# -*- coding: utf-8 -*-

#import cv2
import torch
import numpy as np
import time

from videoholder import VideoHolder
import centroidtracker

from util import box_center

# %% configuration

class Configuation():
    def __init__(self, fr:int, rs:int, model):
        # frame jump rate
        # width of resolution
        assert fr>0
        self.fr = fr
        self.rs = rs
        self.model = model
        # inner
        self.idx = 0
        
    def next_index(self):
        n = self.idx
        self.idx += self.fr
        return n

    def __repr__(self):
        return 'fr: %d, rs: %d, model: %s' % (self.fr, self.rs, self.model)

#%% detect range

class RangeChecker():
    def __init__(self, line_dir='h', line_pos=0.5,
                 detect_rng=0.2, track_rng=0.1):
        assert line_dir in ['h','v']
        assert 0 < track_rng <= detect_rng
        self.dir = line_dir
        self.pos = line_pos
        self.drng = detect_rng
        self.trng = track_rng
        # helpers
        if self.dir == 'h':
            # horizontal line: check the vertical coordinate
            self.idx = 1
        else:
            # vertical line: check the horizontal coordinate
            self.idx = 0
    
    def __repr__(self):
        return '{%c-%g detect-rng: %g, track-rng: %g}' \
            % (self.dir, self.pos, self.drng, self.trng)
    
    def in_detect(self, points):
        if points.ndim == 1:
            return np.abs(points[self.idx] - self.pos) <= self.drng
        else:
            return np.abs(points[:,self.idx] - self.pos) <= self.drng
    
    def in_track(self, points):
        if points.ndim == 1:
            return np.abs(points[self.idx] - self.pos) <= self.trng
        else:
            return np.abs(points[:,self.idx] - self.pos) <= self.trng
        
    def direction(self, old_point, new_point):
        return new_point[self.idx] - old_point[self.idx]
    
    def offset(self, points):
        if points.ndim == 1:
            return points[self.idx] - self.pos
        else:
            return points[:,self.idx] - self.pos

# %% object counter
    
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
        

class CarCounter():
    
    def __init__(self, video: VideoHolder, rng: RangeChecker,
                 conf: Configuation, disappear_time: float=0.5):
        self.video = video
        self.range = rng
        self.conf = conf
        self.dsap_time = disappear_time
        self.dsap_frame = max(1, int(disappear_time*video.fps))
        n = max(0, int(disappear_time*video.fps/conf.fr))
        self.tracker = centroidtracker.CentroidTracker(n)
        self.obj_info = {} # objectID -> CarRecord(dir, over)
    
    def change_fr(self, fr):
        self.conf.fr = fr
        n = max(1, int(self.dsap_time*self.video.fps/fr))
        self.tracker.maxDisappeared = n
    
    def change_rs(self, rs):
        self.conf.rs = rs
        
    def reset(self):
        self.tracker.reset()
        self.change_fr(self.conf.fr)
        self.obj_info = {}
        
    def recognize_cars(self, frame):
        if self.conf.rs is not None:
            lbls, scores, boxes = self.conf.model.process(frame, self.conf.rs)
        else:
            lbls, scores, boxes = self.conf.model.process(frame)
        return boxes
    
    def filter_cars(self, boxes, centers):
        res = []
        for b, c in zip(boxes, centers):
            if self.range.in_detect(c):
                res.append(b)
        return np.array(res)
        
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
        # remove old ones
        to_remove = []
        for oid, oinfo in self.obj_info.items():
            if fidx - oinfo.last > self.dsap_frame:
                to_remove.append(oid)
        for oid in to_remove:
            del self.obj_info[oid]
        return c
    
    def update(self, idx):
        frame = self.video.get_frame(idx)
        boxes = self.recognize_cars(frame)
        centers = box_center(boxes)
        #boxes = self.filter_cars(boxes, centers)

        if len(centers) > 0:
            flag = self.range.in_track(centers)
            centers_in_range = centers[flag]
        else:
            centers_in_range = []
        objects = self.tracker.update(centers_in_range)
        c = self.count(idx, objects)
        return c
    
    def process(self):
        idx = 0
        fps = int(np.ceil(self.video.fps))
        times = np.zeros(self.video.length_second(), float)
        counts = np.zeros(self.video.length_second(), int)
        p = 0
        second = 0
        t = time.time()
        c = 0
        while idx < self.video.num_frame:
            c += self.update(idx)
            
            idx += self.conf.fr
            if idx // fps != second:
                if second % 10 == 0:
                    print(second, idx, t, c)
                second = idx // fps
                t = time.time() - t
                times[p] = t
                counts[p] = c
                p += 1
                t = time.time()
                c = 0
        return times, counts
    
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
    
    def group_to_segments(self, data, segment_legnth):
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

# %% profile io
    
def save_raw_data(file, rng_param, model_param, width, times, boxes):
    np.savez(file, rng_param=np.array(rng_param,object), 
             model_param=np.array(model_param, object),
             width=width, times=times, boxes=np.array(boxes, object))
    
def load_raw_data(file):
    with np.load(file, allow_pickle=True) as data:
        rng_param = data['rng_param'].tolist()
        model_param = data['model_param'].tolist()
        width = data['width'].item()
        times = data['times']
        boxes = data['boxes'].tolist()
        return rng_param, model_param, width, times, boxes

#%% test

def __test_FasterRCNN__():
    import torchvision
    import operation
    
    class MC_FRCNN:
        def __init__(self, model, min_score, target_labels=None):
            model.eval()
            self.model = model
            self.min_score = min_score
            self.target_labels = target_labels
            
        def filter(self, labels, scores, boxes):
            if self.target_labels is None:
                idx = scores > self.min_score
            else:
                idx = [s>self.min_score and l in self.target_labels 
                       for l,s in zip(labels, scores)]
            return labels[idx], scores[idx], boxes[idx]
        
        def process(self, frame, width):
            with torch.no_grad():
                pred = self.model(frame.unsqueeze(0))
            lbls = pred[0]['labels'].cpu().numpy()
            scores = pred[0]['scores'].cpu().numpy()
            boxes = pred[0]['boxes'].cpu().numpy()
            lbls, scores, boxes = self.filter(lbls, scores, boxes)
            return lbls, scores, boxes
        
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(True)
    model = MC_FRCNN(model, 0.7, (3,4,6,8))
    
    v1 = VideoHolder('E:/Data/video/s3.mp4',operation.OptTransOCV2Torch())
    rng = RangeChecker('h', 0.5, 0.1)
    conf = Configuation(5, None, model)
    
    cc = CarCounter(v1, rng, conf)
    times, counts = cc.process()
    np.savez('E:/Data/video/s3-profile.npz', times=times, counts=counts)

def __test_yolo__():
    import yolowrapper
    model = yolowrapper.YOLO_torch('yolov5s', 0.5, (2,3,5,7))
    
    v1 = VideoHolder('E:/Data/video/s3.mp4')
    rng = RangeChecker('h', 0.5, 0.1)
    conf = Configuation(5, None, model)
    
    cc = CarCounter(v1, rng, conf)
    ptimes, pboxes = cc.raw_profile(show_progress=100)
    
    np.savez('data/s3-raw-480', rng_param=np.array(('h',0.5,0.1),object), 
             model_param=np.array(('yolov5s',0.5,(2,3,4,7)), object),
             width=480, times=ptimes, boxes=pboxes)

def __test_conf__():
    v3=VideoHolder('E:/Data/video/s3.mp4')
    rng3=RangeChecker('h', 0.5, 0.1)
    
    v4=VideoHolder('E:/Data/video/s4.mp4')
    rng4=RangeChecker('h', 0.5, 0.1)
    
    v5=VideoHolder('E:/Data/video/s5.mp4')
    rng5=RangeChecker('v', 0.75, 0.2, 0.1)
    
    v7=VideoHolder('E:/Data/video/s7.mp4')
    rng7=RangeChecker('h', 0.45, 0.2, 0.1)
    
    