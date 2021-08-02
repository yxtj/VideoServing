# -*- coding: utf-8 -*-

import numpy as np
import time
from collections import namedtuple
from threading import Lock

Configure = namedtuple('Configure', ['rs', 'fr', 'roi', 'model'])
FrameInfo = namedtuple('FrameInfo', ['tid', 'jid', 'sid', 'fid', 'time'])

# next: add queues for different models

class FrameHolder():
    
    def __init__(self, rsl_list, bs, mat_pt, policy='come', **kwargs):
        self.rsl_list = rsl_list
        self.rsl_index = { rs:i for i,rs in enumerate(rsl_list) }
        
        self.locks = { rs:Lock() for rs in rsl_list }
        self.queues = { rs:[] for rs in rsl_list }
        self.infos = { rs:[] for rs in rsl_list }
        
        self.batchsize = bs
        self.process_time = mat_pt
        if mat_pt is not None:
            assert mat_pt.ndim == 2 and mat_pt.shape[0] == len(rsl_list)
            self.max_pbs = mat_pt.shape[1]
            assert self.max_pbs >= bs
        
        self.set_ready_method(policy, **kwargs)
        self.tid_lock = Lock()
        self.tid = 0
        
    
    def clear(self):
        self.queues = { rs:[] for rs in self.rsl_list }
        self.infos = { rs:[] for rs in self.rsl_list }
        self.tid = 0

    def put(self, frame:np.ndarray, jobID:int, stmID:int, frmID:int,
            conf:Configure, t:float=None):
        rs = conf.rs
        assert rs in self.rsl_list
        t = t if t is not None else time.time()
        with self.tid_lock:
            tid = self.tid
            self.tid += 1
        with self.locks[rs]:
            self.queues[rs].append(frame)
            self.infos[rs].append(FrameInfo(tid, jobID, stmID, frmID, t))
        return len(self.queues[rs])
    
    def empty(self, rs=None):
        if rs is None:
            return np.all([len(v) == 0 for k,v in self.queue.items()])
        else:
            return len(self.queue[rs]) == 0
    
    def estimate_processing_time(self, rs, n):
        ind = self.rsl_index[rs]
        return self.process_time[ind][min(n, self.max_pbs)-1]*n
    
    # info query functions
    
    def query_queue_length(self, rs=None):
        if rs is None:
            return { k:len(q) for k,q in self.queues.items() }
        else:
            return self.queues[rs]

    def query_queue_length_as_list(self):
        l = np.zeros(len(self.rsl_list), int)
        for i, rs in enumerate(self.rsl_list):
            l[i] = len(self.queues[rs])
        return l
    
    def query_waiting_time(self, rs=None, now=None):
        if now is None:
            now = time.time()
        if rs is None:
            return { k:0 if len(q)==0 else q[0].time - now for k,q in self.infos.items() }
        else:
            iq = self.infos[rs]
            return 0 if len(iq) == 0 else iq[0].time - now
    
    def query_waiting_time_as_list(self, now=None):
        if now is None:
            now = time.time()
        l = np.zeros(len(self.rsl_list), int)
        for i, rs in enumerate(self.rsl_list):
            q = self.info[rs]
            l[i] = 0 if len(q)==0 else q[0].time - now
        return l
    
    # find ready queue        
    
    def set_ready_method(self, m, **kwargs):
        assert m in ['come', 'small', 'finish', 'delay', 'prioirty']
        self.policy = m
        if m == 'come':
            self.ready = lambda now: self.ready_come_first()
        elif m == 'small':
            self.ready = lambda now: self.ready_small_first()
        elif m == 'finish':
            self.ready = self.ready_finish_first
        elif m == 'delay':
            self.ready = self.ready_max_delay_first
        elif m == 'priority':
            self.ready = self.ready_priority_first
        
    def ready_come_first(self):
        t = [info[0].time if len(info)>=self.batchsize else np.inf 
             for rs,info in self.infos.items()]
        ind = np.argmin(t)
        if np.isinf(t[ind]):
            return None
        return self.rsl_list[ind]
    
    def ready_small_first(self):
        for rs in self.rsl_list:
            q = self.queues[rs]
            if len(q) >= self.batchsize:
                return rs
        else:
            return None
   
    def ready_finish_first(self, now=None):
        if now is None:
            now = time.time()
        etds = np.zeros(len(self.rsl_list)) + np.inf
        for i, rs in enumerate(self.rsl_list):
            info = self.infos[rs]
            if len(info) >= self.batchsize:
                etds[i] = now - info[0].time
                etds[i] += self.estimate_processing_time(rs, min(self.batchsize, len(info)))
        ind = etds.argmin()
        if np.isinf(etds[ind]):
            return None
        return self.rsl_list[ind]

    def ready_max_delay_first(self, now=None):
        if now is None:
            now = time.time()
        etds = np.zeros(len(self.rsl_list)) - np.inf
        for i, rs in enumerate(self.rsl_list):
            info = self.infos[rs]
            # may not be a full batch
            if len(info) >= 1:
                etds[i] = now - info[0].time
                etds[i] += self.estimate_processing_time(rs, min(self.batchsize, len(info)))
        ind = etds.argmax()
        if np.isinf(etds[ind]):
            return None
        return self.rsl_list[ind]
    
    def ready_priority_first(self, now=None):
        if now is None:
            now = time.time()
        pass

    # data get functions
    
    def get_batch(self, rs):
        res_f = None
        res_i = None
        if len(self.queues[rs]) > 0:
            with self.locks[rs]:
                res_f = self.queues[rs][:self.batchsize]
                res_i = self.infos[rs][:self.batchsize]
                del self.queues[rs][:self.batchsize]
                del self.infos[rs][:self.batchsize]
        return res_f, res_i
    
    def get_queue(self, rs):
        res_f = None
        res_i = None
        if len(self.queues[rs]) > 0:
            with self.locks[rs]:
                res_f = self.queues[rs]
                res_i = self.infos[rs]
                self.queues[rs] = []
                self.infos[rs] = []
        return res_f, res_i
    
    # -- inner functions --
    
    def __fast_queue_length__(self):
        res = np.zeros(len(self.rsl_list), int)
        for i, (rs, q) in enumerate((self.queues.items())):
            res[i] = len(q)
        return res
    
    def __estimated_delay_one__(self, rs, now):
        q = self.queues[rs]
        n = len(q)
        if n > 0:
            ind_rs = self.rsl_index[rs]
            f = self.infos[rs]
            etd = now - f[0].time
            etd += self.process_time[ind_rs][min(n, self.batchsize)-1]*n
        else:
            etd = 0.0
        return etd
    
    def __estimated_delays__(self):
        t = time.time()
        etds = np.zeros(len(self.rsl_list))
        for i, rs in enumerate(self.rsl_list):
            q = self.queues[rs]
            n = len(q)
            if n > 0:
                f = self.infos[rs]
                etd = t - f[0].time
                etd += self.process_time[i][min(n, self.batchsize)-1]*n
                etds[i] = etd
        return etds

# %% test

def __test__():
    rsl_list=[240,360,480,720]
    bs=5
    mat_pt = np.exp(-np.array([1,1.5,2,2.5,3]))+np.array([1,2,3,4]).reshape(4,1)
    
    mat_pt=np.array([[52,37,28,25,22,27,25,24],
        [98,70,61,60,59,60,60,59],
        [154,131,110,115,110,105,101,96],
        [358,271,226,210,211,212,208,204]])*0.001
    
    fh=FrameHolder(rsl_list,bs,mat_pt,'finish')
    for i in range(5):
        rs=480
        fh.put(np.zeros(rs,rs),0,0,i,Configure(rs,2,False,'yolov5m'))
        rs=240
        fh.put(np.zeros(rs,rs),0,0,i,Configure(rs,2,False,'yolov5m'))
    
    print(fh.__queue_length__())
    print(fh.__estimated_delays__())
    
    r = fh.get()
    assert r is not None
    print(len(r[0]), r[1], r[0][0].shape)
    
    r = fh.get()
    assert r is not None
    print(len(r[0]), r[1], r[0][0].shape)
