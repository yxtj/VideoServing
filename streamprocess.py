# -*- coding: utf-8 -*-

#import numpy as np
from common import Configuration
from videoholder import VideoHolder
from track import Tracker


class CertifyConf:
    def __init__(self, interval:float, length:float, conf:Configuration=None):
        # sample interval
        self.interval = interval
        # sample length
        self.length = length
        # configuration used to certify
        self.conf = conf

class RefineConf:
    def __init__(self, bound:float, length:float, conf:Configuration=None):
        # threshold for difference
        self.bound = bound
        # length to refine
        self.length = length
        # configuration used to refine
        self.conf = conf


# %% stream processor

class StreamProcessor():
    
    def __init__(self, streamid, video:VideoHolder, 
                 knobs:list[Configuration], tracker:Tracker,
                 ctf_conf:CertifyConf(30, 1, None), 
                 rfn_conf:RefineConf(0.3, 30, None)):
        self.streamid = streamid
        self.knobs = knobs
        self.knob_fps = [c.fps for c in self.knobs]
        self.knob_rsl = [c.rsl for c in self.knobs]
        self.tracker = tracker
        self.ctf_conf = ctf_conf
        self.rfn_conf = rfn_conf
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
    