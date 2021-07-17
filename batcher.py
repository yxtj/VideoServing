# -*- coding: utf-8 -*-

import numpy as np
import time
from collections import namedtuple
from threading import Lock

Configure = namedtuple('Configure', ['rs', 'fr', 'roi', 'model'])
FrameInfo = namedtuple('FrameInfo', ['jid', 'sid', 'fid', 'time'])

# next step: add queues for different models

class FrameHolder():
    
    def __init__(self, rsl_list, bs, max_dl, mat_pt):
        self.rsl_list = rsl_list
        self.rsl_index = { rs:i for i,rs in enumerate(rsl_list) }
        
        self.locks = { rs:Lock() for rs in rsl_list }
        self.queues = { rs:[] for rs in rsl_list }
        self.infos = { rs:[] for rs in rsl_list }
        
        self.batchsize = bs
        self.max_delay = max_dl
        self.process_time = mat_pt
        assert mat_pt is None or \
            (mat_pt.shape[0] == len(rsl_list) and mat_pt.shape[1] >= bs)
    
    def clear(self):
        self.queues = { rs:[] for rs in self.rsl_list }
        self.infos = { rs:[] for rs in self.rsl_list }

    def put(self, frame:np.ndarray, jobID:int, stmID:int, frmID:int,
            conf:Configure, t:float=None):
        rs = conf.rs
        assert rs in self.rsl_list
        t = t if t is not None else time.time()
        with self.locks[rs]:
            self.queues[rs].append(frame)
            self.infos[rs].append(FrameInfo(jobID, stmID, frmID, t))
        return len(self.queues[rs])
    
    def get_batch(self, rs):
        res = None
        if len(self.queues[rs]) >= self.batchsize:
            with self.locks[rs]:
                res = self.queues[rs][:self.batchsize]
                del self.queues[rs][:self.batchsize]
                del self.infos[rs][:self.batchsize]
        return res, rs        
    
    def get_queue(self, rs):
        res = None
        if len(self.queues[rs]) > 0:
            with self.locks[rs]:
                res = self.queues[rs]
                self.queues[rs] = []
                self.infos[rs] = []
        return res, rs
    
    def get_by_queue_length(self, ql_limit=None):
        if ql_limit is None:
            ql_limit = self.batchsize
        res = None
        for rs in self.rsl_list:
            q = self.queues[rs]
            if len(q) >= ql_limit:
                with self.locks[rs]:
                    res = self.queues[rs]
                    self.queues[rs] = []
                    self.infos[rs] = []
                break
        return res, rs
    
    def get_by_queuing_delay(self, delay_limit=2.0, now=None):
        res = None
        if delay_limit is None:
            delay_limit = self.max_delay
        if now is None:
            now = time.time()
        for rs in self.rsl_list:
            qi = self.infos[rs]
            if len(qi) > 0 and now - qi[0].time > delay_limit:
                with self.locks[rs]:
                    res = self.queues[rs]
                    self.queues[rs] = []
                    self.infos[rs] = []
                break
        return res, rs
    
    def get(self):
        # by maximum estimated delay
        res = None
        etds = self.__estimated_delays__()
        ind = etds.argmax()
        rs = self.rsl_list[ind]
        if len(self.queues[rs]) > 0:
            with self.locks[rs]:
                res = self.queues[rs]
                self.queues[rs] = []
                self.infos[rs] = []
        return res, rs
    
    # -- inner functions --
    
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
    
    def __queue_length__(self):
        l = np.zeros(len(self.rsl_list), int)
        for i, rs in enumerate(self.rsl_list):
            l[i] = len(self.queues[rs])
        return l
    
# %% test

def __test__():
    rsl_list=[240,360,480,720]
    bs=5
    dl=1.0
    mat_pt = np.exp(-np.array([1,1.5,2,2.5,3]))+np.array([1,2,3,4]).reshape(4,1)
    
    fh=FrameHolder(rsl_list,bs,dl,mat_pt)
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
    
def __test2__():
    rsl_list=[240,360,480,720]
    bs=8
    dl=2.0
    mat_pt=np.array([[52,37,28,25,22,27,25,24],
        [98,70,61,60,59,60,60,59],
        [154,131,110,115,110,105,101,96],
        [358,271,226,210,211,212,208,204]])*0.001
    
    fh=FrameHolder(rsl_list,bs,dl,mat_pt)
    
    import profiling
    vn_list = ['s3', 's4', 's5', 's7']
    fps_list = [25,30,20,30]
    segment = 1
    
    pts=[]
    pas=[]
    pss=[]
    for i,vn in enumerate(vn_list):
        _,_,sg_list,cts,cas=profiling.load_configurations('data/%s/conf-%s.npz' % (vn,vn))
        sg_idx=sg_list.tolist().index(segment)
        pt,pa,ps=profiling.get_profile_bound_acc(cts[sg_idx],cas[sg_idx],0.9)
        pts.append(pt)
        pas.append(pa)
        pss.append(ps)
    
    nlength = 400
    workload = np.zeros((len(vn_list), nlength, 2), int)
    for i, fps in enumerate(fps_list):
        if fps == 20:
            fr_list = profiling.FR_FOR_20
        elif fps == 25:
            fr_list = profiling.FR_FOR_25
        else:
            fr_list = profiling.FR_FOR_30
        for j, (ind_rs, ind_fr) in enumerate(pss[i]):
            #rs = rsl_list[ind_rs]
            nf = int(fps/fr_list[ind_fr])
            k = 0
            x = j + len(pss[i])*k
            while x < nlength:
                workload[i, x] = (ind_rs, nf)
                k += 1
                x = j + len(pss[i])*k
    
    # batching comparision
    
    bs=5
    
    # no batching
    loads = np.zeros(nlength)
    for i in range(len(vn_list)):
        for j in range(nlength):
            ind_rs = workload[i,j,0]
            rs = rsl_list[ind_rs]
            nf = workload[i,j,1]
            loads[j] += mat_pt[ind_rs,0]*nf
    
    import profiling
    loads = profiling.simulate_workloads(pts, nlength)
    
    delays,_=profiling.get_delay_usage_with_bound(loads, 1.0, 1)
    
    # queue length only
    loads = np.zeros(nlength)
    fh=FrameHolder(rsl_list, bs, dl, mat_pt)
    fids = [0 for _ in range(len(vn_list))]
    for i in range(nlength):
        tasks = []
        for j in range(len(vn_list)):
            ind_rs = workload[j,i,0]
            rs = rsl_list[ind_rs]
            nf = workload[j,i,1]
            if nf == 0:
                continue
            fr = int(fps_list[j]/nf)
            for k in range(nf):
                t = i + (nf/30)*k
                tasks.append((t, j, fr, ind_rs))
        tasks.sort(key=lambda v:v[0])
        for t, sid, fr, ind_rs in tasks:
            rs = rsl_list[ind_rs]
            n = fh.put(np.zeros((int(rs/9*16),rs)), 0, sid, fids[sid], Configure(rs, fr, False, 'yolov5m'))
            fids[sid] += 1
            if n >= bs:
                d, _ = fh.get_queue(rs)
                loads[i] += mat_pt[ind_rs, min(n, bs)-1]*n
    for ind_rs, rs in enumerate(rsl_list):
        d, _ = fh.get_queue(rs)
        if d is not None:
            n = len(d)
            loads[-1] += mat_pt[ind_rs, min(n, bs)-1]*n
    
    delays,_=profiling.get_delay_usage_with_bound(loads, 1.0, 1)
    
    # queue length + delay
    loads = np.zeros(nlength)
    fh=FrameHolder(rsl_list, bs, dl, mat_pt)
    fids = [0 for _ in range(len(vn_list))]
    for i in range(nlength):
        tasks = []
        for j in range(len(vn_list)):
            ind_rs = workload[j,i,0]
            rs = rsl_list[ind_rs]
            nf = workload[j,i,1]
            if nf == 0:
                continue
            fr = int(fps_list[j]/nf)
            for k in range(nf):
                t = i + (nf/30)*k
                tasks.append((t, j, fr, ind_rs))
        tasks.sort(key=lambda v:v[0])
        for t, sid, fr, ind_rs in tasks:
            rs = rsl_list[ind_rs]
            n = fh.put(np.zeros((int(rs/9*16),rs)), 0, i, fids[sid], Configure(rs, fr, False, 'yolov5m'))
            fids[sid] += 1
            tmp = [*fh.infos[rs][-1]]
            tmp[-1] = t
            fh.infos[rs][-1] = FrameInfo(*tmp)
            if n >= bs or t - fh.infos[rs][0].time >= dl:
                d, _ = fh.get_queue(rs)
                loads[i] += mat_pt[ind_rs, min(n, bs)-1]*n
    for ind_rs, rs in enumerate(rsl_list):
        d, _ = fh.get_queue(rs)
        if d is not None:
            n = len(d)
            loads[-1] += mat_pt[ind_rs, min(n, bs)-1]*n

                    
    # estimated delay
    loads = np.zeros(nlength)
    fh=FrameHolder(rsl_list, bs, dl, mat_pt)
    fids = [0 for _ in range(len(vn_list))]
    for i in range(nlength):
        tasks = []
        for j in range(len(vn_list)):
            ind_rs = workload[j,i,0]
            rs = rsl_list[ind_rs]
            nf = workload[j,i,1]
            if nf == 0:
                continue
            fr = int(fps_list[j]/nf)
            for k in range(nf):
                t = i + (nf/30)*k
                tasks.append((t, j, fr, ind_rs))
        tasks.sort(key=lambda v:v[0])
        for t, sid, fr, ind_rs in tasks:
            rs = rsl_list[ind_rs]
            n = fh.put(np.zeros((int(rs/9*16),rs)), 0, i, fids[sid], Configure(rs, fr, False, 'yolov5m'))
            fids[sid] += 1
            tmp = [*fh.infos[rs][-1]]
            tmp[-1] = t
            fh.infos[rs][-1] = FrameInfo(*tmp)
        etds = fh.__estimated_delays__()
        qls = fh.__queue_length__()
        indexes = sorted(np.arange(len(rsl_list)), key=lambda i:etds[i])
        for ind in indexes:
            if qls[ind] >= bs:
                d, _ = fh.get_queue(rs)
                loads[i] += mat_pt[ind_rs, min(n, bs)-1]*n
    for ind_rs, rs in enumerate(rsl_list):
        d, _ = fh.get_queue(rs)
        if d is not None:
            n = len(d)
            loads[-1] += mat_pt[ind_rs, min(n, bs)-1]*n

    # plot
    import matplotlib.pyplot as plt
    import util
    
    plt.figure()
    plt.plot(util.moving_average(loads, 10))
    plt.xlabel('time (s)')
    plt.ylabel('cmp-time (t)')
    plt.tight_layout()
    