# -*- coding: utf-8 -*-

from collections import namedtuple
from dataclasses import dataclass # mutable

import cv2
import numpy as np

from ..track import CentroidTracker

from .. import util

DetectionResult = namedtuple('DetectionResult', ['box', 'lbl'])
TrackResult = namedtuple('TrackResult', ['id', 'center', 'box', 'lbl', 'speed'])

SPEED_SMOOTH_ALPHA=0.9

@dataclass
class ObjectInfo:
    center: np.ndarray
    fid: int
    speed: np.ndarray = np.zeros(2)
    
    def update(self, fid, center):
        s = (center-self.center) / (fid - self.fid)
        self.center = center
        self.fid = fid
        self.speed = SPEED_SMOOTH_ALPHA*s + (1-SPEED_SMOOTH_ALPHA)*self.speed


class LiveJob:
    def __init__(self, jid, sid, adapt_fun, detect_fun, analyze_fun,
                 **kwargs):
        self.jid = jid
        self.sid = sid
        self.adapt_fun = adapt_fun
        self.detect_fun = detect_fun
        self.analyze_fun = analyze_fun
        

class LiveExecutor:
    def __init__(self, ljob:LiveJob, task_scheduler):
        self.ljob = ljob
        
        self.tracker = CentroidTracker(30)
        self.objects = {} # id:ObjectInfo
        
        self.task_scheduler = task_scheduler
        
        self.feat_gen = None
        self.n_to_skip = 0
        self.fid = 0

    def restart(self):
        pass
    
    def tracking(self, fid, detections):
        if len(detections) == 0:
            return []
        boxes = np.array([d.box for d in detections])
        centers = util.box_center(boxes)
        tracks = self.tracker.update(centers)
        # tracks: map of (oid, center)
        res = []
        for oid, center in tracks:
            if oid not in self.objects:
                self.objects[oid] = ObjectInfo(center, fid)
            else:
                old = self.objects[oid]
                old.update(fid, center)
            res.append(TrackResult(oid, center, None, None, self.objects[oid].speed))
        return res
    
    def resize_frame(self, frame, rs):
        w,h = frame.shape[:2]
        f = rs/min(w,h)
        f = cv2.resize(frame, None, None, f, f)
        return f
    
    def detect(self, fid, frame):
        self.detect_fun('live', self.jid, self.sid, self.fid, frame)
    
    def update(self, frame):
        if self.n_to_skip > 0:
            self.n_to_skip -= 1
            return None
        feat = self.feat_gen.get()
        rs, fr = self.adapt_fun(feat)
        self.n_to_skip = fr - 1
        f = self.resize_frame(frame, rs)
        detections = self.detect_fun('live', self.jid, self.sid, self.fid, f)
        tracks = self.tracking(self.fid, detections)
        self.feat_gen.update(detections, tracks)
        res = self.analyze_fun(tracks)
        return res 
    

# %% 

class FindCarWithCount(LiveJob):
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        if 'cls' in kwargs:
            self.lbls = kwargs['labels']
        else:
            self.lbls = None
        
    def update(self, objects):
        n = 0
        for obj in objects:
            if self.lbl is None or obj.lbl in self.lbls:
                n += 1
        return n >= self.n
    
        