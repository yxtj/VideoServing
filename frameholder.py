# -*- coding: utf-8 -*-

import numpy as np
import time
from collections import namedtuple
from threading import Lock

Configure = namedtuple('Configure', ['rs', 'fr', 'roi', 'model'])
FrameInfo = namedtuple('FrameInfo', ['tid', 'jid', 'sid', 'fid', 'time'])

class FrameQueue():
    def __init__(self):
        self.lock = Lock()
        self.queue = []
        self.info = []
    
    def size(self):
        return len(self.queue)
    
    def empty(self):
        return len(self.queue) == 0
    
    def clear(self):
        with self.lock:
            self.queue = []
            self.info = []
    
    def put(self, frame:np.ndarray, info:FrameInfo):
        with self.lock:
            self.queue.append(frame)
            self.info.append(info)
        return len(self.queue)
        
    def get(self, n):
        res_f = None
        res_i = None
        if len(self.queue) > 0:
            with self.lock:
                res_f = self.queue[:n]
                res_i = self.info[:n]
                del self.queue[:n]
                del self.info[:n]
        return res_f, res_i
    
    def get_all(self):
        res_f = None
        res_i = None
        if len(self.queue) > 0:
            with self.lock:
                res_f = self.queue
                res_i = self.info
                self.queue = []
                self.info = []
        return res_f, res_i
    
    def wait_time(self, now):
        if len(self.info) == 0:
            return 0.0
        else:
            return now - self.info[0].time

# next: add queues for different models

class FrameHolder():
    
    def __init__(self, rsl_list, bs:int, levels:int, mat_pt:np.ndarray,
                 policy='come', **kwargs):
        self.rsl_list = rsl_list
        self.rsl_index = { rs:i for i,rs in enumerate(rsl_list) }
        self.nrsl = len(rsl_list)
        assert 0 < bs
        self.batchsize = bs
        assert 0 <= levels
        self.levels = levels
        
        self.queues = [[ FrameQueue() for _ in range(self.nrsl) ] for _ in range(levels)]
        
        self.process_time = mat_pt
        if mat_pt is not None:
            assert mat_pt.ndim == 2 and mat_pt.shape[0] == len(rsl_list)
            self.max_pbs = mat_pt.shape[1]
            assert self.max_pbs >= bs
        
        # assert m in ['come', 'small', 'finish', 'delay', 'priority', 'awt']
        self.set_ready_method(policy, **kwargs)
        self.tid_lock = Lock()
        self.tid = 0 # internal task id
        
    
    def clear(self):
        for qs in self.queues:
            for q in qs:
                q.clear()
        self.tid = 0

    def put(self, frame:np.ndarray, level:int,
            jobID:int, stmID:int, frmID:int, rs:int, t:float=None):
        '''
        level determines the priority. 0 is the highest, 1 is lower
        '''
        assert rs in self.rsl_list
        rs_ind = self.rsl_index[rs]
        assert 0 <= level < self.levels
        t = t if t is not None else time.time()
        with self.tid_lock:
            tid = self.tid
            self.tid += 1
        info = FrameInfo(tid, jobID, stmID, frmID, t)
        q = self.queues[level][rs_ind]
        return q.put(frame, info)
    
    def empty(self, level=None, rs=None):
        if level is None:
            np.all([ self.empty(rs,i) for i in range(self.levels) ])
        else:
            assert 0 <= level < self.levels
            qs = self.queues[level]
            if rs is None:
                return np.all([q.empty() for q in qs])
            else:
                rs_ind = self.rsl_index[rs]
                return len(qs[rs_ind]) == 0
    
    def estimate_processing_time(self, rs, n):
        ind = self.rsl_index[rs]
        return self.process_time[ind][min(n, self.max_pbs)-1]*n
    
    # info query functions
    
    def query_queue_length(self, level=None, rs=None):
        if level is None:
            res = [self.query_queue_length(i, rs) for i in range(self.levels)]
            return res
        else:
            assert 0 <= level < self.levels
            qs = self.queues[level]
            if rs is None:
                return [ q.size() for q in qs ]
            else:
                rs_ind = self.rsl_index[rs]
                return qs[rs_ind].size()
            

    def query_queue_length_as_mat(self):
        l = np.zeros((self.levels, self.nrsl), int)
        for i in range(self.levels):
            for j in range(self.nrsl):
                l[i][j] = self.queues[i][j].size()
        return l
    
    def query_waiting_time(self, level=None, rs=None, now=None):
        if now is None:
            now = time.time()
        if level is None:
            res = [ self.query_waiting_time(i, rs) for i in range(self.levels) ]
            return res
        else:
            qs = self.queues[level]
            if rs is None:
                return [ q.wait_time(now) for q in qs ]
            else:
                rs_ind = self.rsl_index[rs]
                return qs[rs_ind].wait_time(now)
    
    def query_waiting_time_as_mat(self, now=None):
        if now is None:
            now = time.time()
        l = np.zeros((self.levels, self.nrsl))
        for i in range(self.levels):
            for j in range(self.nrsl):
                l[i][j] = self.queues[i][j].wait_time(now)
        return l
    
    # find ready queue        
    
    def set_ready_method(self, m, **kwargs):
        assert m in ['come', 'small', 'finish', 'delay', 'priority', 'awt']
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
            self.ready_param = {'alpha':float(kwargs['param_alpha'])}
            # priority = alpha*wait_time + process_time
            self.ready = self.ready_priority_first
        elif m == 'awt':
            self.ready = self.ready_awt
        
    def ready_come_first(self):
        bf = []
        for lvl in range(self.levels):
            t = [q.info[0].time if q.size()>=self.batchsize else np.inf
                 for q in self.queues[lvl]]
            ind = np.argmin(t)
            if not np.isinf(t[ind]):
                return lvl, self.rsl_list[ind]
            # no queue contains a full batch
            t = [q.info[0].time if q.size()>0 else np.inf for q in self.queues[lvl]]
            ind = np.argmin(t)
            if not np.isinf(t[ind]):
                bf.append((lvl, self.rsl_list[ind]))
        if len(bf) != 0:
            return bf[0]
        return None, None
    
    def ready_small_first(self):
        bf = []
        for lvl in range(self.levels):
            for i in range(self.nrsl):
                q = self.queues[lvl][i]
                if q.size() >= self.batchsize:
                    return lvl, self.rsl_list[i]
                # no queue contains a full batch
                elif q.size() > 0:
                    bf.append((lvl, self.rsl_list[i]))
        if len(bf) != 0:
            return bf[0]
        return None, None
   
    def ready_finish_first(self, now=None):
        if now is None:
            now = time.time()
        bf = []
        for lvl in range(self.levels):
            etds = self.__estimated_delays__(lvl, now, np.inf, self.batchsize)
            ind = etds.argmin()
            if not np.isinf(etds[ind]):
                return lvl, self.rsl_list[ind]
            # no queue contains a full batch
            etds = self.__estimated_delays__(lvl, now, np.inf, 1)
            ind = etds.argmin()
            if not np.isinf(etds[ind]):
                bf.append((lvl, self.rsl_list[ind]))
        if len(bf) != 0:
            return bf[0]
        return None, None

    def ready_max_delay_first(self, now=None):
        if now is None:
            now = time.time()
        bf = []
        for lvl in range(self.levels):
            etds = self.__estimated_delays__(lvl, now, -np.inf, self.batchsize)
            ind = etds.argmax()
            if not np.isinf(etds[ind]): 
                return lvl, self.rsl_list[ind]
            # no queue contains a full batch
            etds = self.__estimated_delays__(lvl, now, -np.inf, 1)
            ind = etds.argmax()
            if not np.isinf(etds[ind]): 
                bf.append((lvl, self.rsl_list[ind]))
        if len(bf) != 0:
            return bf[0]
        return None, None
    
    def ready_priority_first(self, now=None):
        if now is None:
            now = time.time()
        bf = []
        for lvl in range(self.levels):
            priority = self.__estimated_delays__(lvl, now, -np.inf, self.batchsize, self.ready_param['alpha'])
            ind = priority.argmax()
            if not np.isinf(priority[ind]):
                return lvl, self.rsl_list[ind]
            # no queue contains a full batch
            priority = self.__estimated_delays__(lvl, now, -np.inf, 1, self.ready_param['alpha'])
            ind = priority.argmax()
            if not np.isinf(priority[ind]):
                bf.append((lvl, self.rsl_list[ind]))
        if len(bf) != 0:
            return bf[0]
        return None, None
    
    def ready_awt(self, now=None):
        if now is None:
            now = time.time()
        bf = []
        for lvl in range(self.levels):
            priority = self.__waiting_time_queue__(lvl, now, -np.inf, self.batchsize)
            ind = priority.argmax()
            if not np.isinf(priority[ind]):
                return lvl, self.rsl_list[ind]
            # no queue contains a full batch
            priority = self.__waiting_time_queue__(lvl, now, -np.inf, 1)
            ind = priority.argmax()
            if not np.isinf(priority[ind]):
                bf.append((lvl, self.rsl_list[ind]))
        if len(bf) != 0:
            return bf[0]
        return None, None

    # data get functions
    
    def get_batch(self, level, rs):
        ind = self.rsl_index[rs]
        return self.queues[level][ind].get(self.batchsize)
    
    def get_queue(self, level, rs):
        ind = self.rsl_index[rs]
        return self.queues[level][ind].get_all()
    
    # -- inner functions --
    
    def __waiting_time_queue__(self, level, now, pad=np.nan, ql_min=1):
        wt = np.zeros(self.nrsl) + pad
        qs = self.queues[level]
        for i, q in enumerate(qs):
            n = q.size()
            if n >= ql_min:
                wt[i] = q.wait_time(now)
        return wt
    
    def __estimated_delay_one__(self, level, rs, now):
        ind = self.rsl_index(rs)
        q = self.queues[level][ind]
        n = q.size()
        if n > 0:
            etd = q.wait_time(now)
            etd += self.process_time[ind][min(n, self.batchsize)-1]*n
        else:
            etd = 0.0
        return etd
    
    def __estimated_delays__(self, level, now, pad=np.nan, ql_min=-1, alpha=1.0):
        etds = np.zeros(self.nrsl) + pad
        qs = self.queues[level]
        for i, q in enumerate(qs):
            n = q.size()
            if n >= ql_min:
                etd = q.wait_time(now)*alpha
                etd += self.process_time[i][min(n, self.batchsize-1)]*n
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
    
    fh=FrameHolder(rsl_list,bs,2,mat_pt,'finish')
    for i in range(5):
        rs = 480
        fh.put(np.zeros((rs,rs)),0,0,0,i,rs)
        rs = 240
        fh.put(np.zeros((rs,rs)),0,0,1,i,rs)
    for i in range(5):
        rs = 360
        fh.put(np.zeros((rs,rs)),1,0,0,i,rs)
        rs = 720
        fh.put(np.zeros((rs,rs)),1,0,1,i,rs)

    print('queue length')
    print(fh.query_queue_length())
    print(fh.query_queue_length_as_mat())
    print(fh.query_queue_length(level=0))
    print(fh.query_queue_length(rs=240))
    
    print('wait time')
    print(fh.query_waiting_time())
    print(fh.query_waiting_time_as_mat())
    print(fh.query_waiting_time(1, 360))
    
    print('go')
    while rdy := fh.ready(time.time()):
        rdy_lvl, rdy_rs = rdy
        if rdy_lvl is None:
            break
        ql = fh.query_queue_length(rdy_lvl, rdy_rs)
        wt = fh.query_waiting_time(rdy_lvl, rdy_rs)
        d_frm, d_info = fh.get_batch(rdy_lvl, rdy_rs)
        print(rdy_lvl, rdy_rs, rsl_list.index(rdy_rs), len(d_frm))
        print(ql, wt)
    