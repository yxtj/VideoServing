# -*- coding: utf-8 -*-

import numpy as np
from common import DetectionResult

class StreamProcessing():
    
    def __init__(self, sid, knobs, f_post, cinterval=30, clength=1, rbound=0.3,
                 track_mthd='sort', cconf=None, rconf=None, **kwargs):
        self.sid = sid
        #self.knobs = knobs
        self.knob_fps = knobs['fps']
        self.knob_rsl = knobs['rsl']
        self.cinterval = cinterval # certify - sample interval
        self.clength = clength # certify - sample length
        self.rbound = rbound # refine - threshold
        self.set_tracker(track_mthd, **kwargs)
        # buffers
        self.bfFrame = []
        self.bfTask = []
        self.bfDetection = []
        self.bfResult = []
        # input stream
        self.stream = None
        self.sfps = None
        self.srsl = None
        # intermediate
        self.fid = 0
        self.fid_
    
    def set_tracker(self, track_mthd, **kwargs):
        assert track_mthd in ['center', 'sort']
        if track_mthd == 'center':
            from track.centroidtracker import CentroidTracker
            self.tracker = CentroidTracker(kwargs['tck_age'])
        elif track_mthd == 'sort':
            from track.sorttracker import SortTracker
            self.tracker = SortTracker(kwargs['tck_age'], kwargs['tck_age'], 
                                       iou_threshold=kwargs['tck_min_iou'])
                
    def task_frame(self, frame):
        pass
    
    def take_detection(self, detections):
        pass
    
    def take_certify_result(self, lresult, cresult, time):
        v = abs(lresult - cresult) / max(lresult, cresult)
        if v >= self.rbound:
            start_time = time - self.cinterval//2
            end_time = time + self.cinterval//2
            return self.__refine__(start_time, end_time)
        return None
    
    # intermediate
    
    def __adapt__(self, frame):
        tasks = []
        return tasks
    
    def __certify__(self, ):
        tasks = []
        return tasks
    
    def __refine__(self, start_time, end_time):
        tasks = []
        return tasks
    