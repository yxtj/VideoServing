# -*- coding: utf-8 -*-

import numpy as np
import time
from collections import namedtuple
from threading import Lock

Configure = namedtuple('Configure', ['rs', 'fr', 'roi', 'model'])
FrameInfo = namedtuple('FrameInfo', ['jid', 'sid', 'fid', 'time'])

# next: add queues for different models

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
        if mat_pt is not None:
            assert mat_pt.ndim == 2 and mat_pt.shape[0] == len(rsl_list)
            self.max_pbs = mat_pt.shape[1]
            assert self.max_pbs >= bs
        
        self.set_ready_method('small')
    
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
    
    def ready_early_first(self):
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

    def set_ready_method(self, m):
        assert m in ['early', 'small', 'finish']
        if m == 'early':
            self.ready = lambda now: self.ready_early_first()
        elif m == 'small':
            self.ready = lambda now: self.ready_small_first()
        else:
            self.ready = self.ready_finish_first
        
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
    
    
# %% test utils

from dataclasses import dataclass

@dataclass
class Task:
    time: float
    frame: np.ndarray
    jid: int
    sid: int
    fid: int
    rs: int
    fr: int
    dqueue: float = 0
    dprocess: float = 0
    
    def delay(self):
        return self.dqueue + self.dprocess
    

def compute_distribution_from_real(workload, nrsl, fps_list):
    if workload.ndim == 3:
        w = workload.reshape((-1,2))
    else:
        w = workload
    #nlength = workload.shape[0]
    nfps = len(fps_list)
    bins = [(fps_list[i]+fps_list[i+1])/2 for i in range(nfps-1)]
    distr_rsl = np.zeros(nrsl)
    cdistr_rsl_fps = np.zeros((nrsl, nfps))
    for i in range(nrsl):
        nf = w[w[:,0]==i, 1]
        distr_rsl[i] = nf.sum()
        inds = np.digitize(nf, bins)
        #cdistr_rsl_fps[1,inds]+=nf # this is wrong due to over-write
        for j in range(len(nf)):
            cdistr_rsl_fps[i,inds[j]] += nf[j]
        cdistr_rsl_fps[i,:] /= nf.sum()
    distr_rsl = distr_rsl/distr_rsl.sum()
    return distr_rsl, cdistr_rsl_fps

def simulate_workloads(nsource, nlength, distr_rsl, cdistr_rsl_fps, fps_list):
    assert distr_rsl.ndim == 1
    assert cdistr_rsl_fps.ndim == 2
    nrsl, nfps = cdistr_rsl_fps.shape
    assert nrsl == len(distr_rsl)
    assert nfps == len(fps_list)
    res = np.zeros((nsource, nlength, 2), int)
    smy = np.zeros((nlength, nrsl), int)
    for i in range(nsource):
        inds = np.random.choice(range(nrsl), nlength, p=distr_rsl)
        res[i,:,0] = inds
        for j in range(nrsl):
            p = inds==j
            cnts = np.random.choice(fps_list, sum(p), p=cdistr_rsl_fps[j])
            res[i,p,1] = cnts
            smy[p, j] += cnts
    return res, smy
   
def workload_to_tasks(workload, rsl_list, max_fps=30):
    # result: (time, frame, jid, sid, fid, rs, fr)
    assert workload.ndim == 3
    assert workload.shape[2] == 2
    nsource, nlength = workload.shape[:2]
    res = []
    fids = [0 for _ in range(nsource)]
    for i in range(nlength):
        tasks = []
        for j in range(nsource):
            ind_rs = workload[j,i,0]
            rs = rsl_list[ind_rs]
            nf = workload[j,i,1]
            if nf == 0:
                continue
            fr = int(max_fps/nf)
            for k in range(nf):
                t = i + 1.0/30*k*fr
                tasks.append((t, j, fr, ind_rs))
        tasks.sort(key=lambda v:v[0])
        for t, sid, fr, ind_rs in tasks:
            rs = rsl_list[ind_rs]
            #res.append((t, np.zeros((int(rs/9*16),rs)), 0, sid, fids[sid], rs, fr))
            res.append(Task(t, None, 0, sid, fids[sid], rs, fr))
            fids[sid] += 1
    return res

def optimal_process(tasks, rsl_list, batchsize, mat_pt):
    nrsl = len(rsl_list)
    assert mat_pt.ndim == 2
    assert mat_pt.shape[0] == nrsl
    assert mat_pt.shape[1] >= batchsize
    t0 = tasks[0].time
    nlength = int(np.ceil(tasks[-1].time - t0))
    rsl_index = { rs:i for i,rs in enumerate(rsl_list) }
    costs = mat_pt[:,batchsize-1]*batchsize
    
    queues = [[] for i in range(nrsl)] # the receiving time of each task
    loads = np.zeros((nrsl, nlength))
    delays = []
    for tsk in tasks:
        rs = tsk.rs
        t = tsk.time
        ind_rs = rsl_index[rs]
        q = queues[ind_rs]
        q.append(t)
        if len(q) >= batchsize:
            ind_t = int(t-t0)
            c = costs[ind_rs]
            loads[ind_rs,ind_t] += c
            for tt in q:
                delays.append((t-tt, c))
            q.clear()
    rest = np.zeros(nrsl)
    for ind_rs, q in enumerate(queues):
        c = costs[ind_rs]/batchsize*len(q)
        rest[ind_rs] += c
    delays = np.array(delays)
    return loads, rest, delays

def simulate_process(tasks, fh, nlength, speed_factor):
    ptime = 0.0
    loads = np.zeros(nlength)
    loads_detail = []
    delays = []
    for i in range(len(tasks)):
        tsk = tasks[i] # next task
        #(t, frame, jid, sid, fid, rs, fr) = tsk
        while ptime <= tsk.time:
            rdy_rs = fh.ready(ptime) # ready is a delegate function for scheduling
            if rdy_rs:
                batch,info = fh.get_batch(rdy_rs)
                load = fh.estimate_processing_time(rdy_rs, len(batch))
                loads_detail.append((ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_list()))
                loads[int(ptime)] += load
                eta = load/speed_factor
                for ifo in info:
                    delays.append((ptime - ifo.time, eta))
                ptime = tsk.time + eta
            else:
                break
        fh.put(tsk.frame, tsk.jid, tsk.sid, tsk.fid, Configure(tsk.rs, tsk.fr, False, 'yolov5m'), tsk.time)
    rest_load = 0.0
    for rs in fh.rsl_list:
        #while batch_info := fh.get_batch(rs):
        batch,_ = fh.get_batch(rs)
        while batch:
            load = fh.estimate_processing_time(rs, len(batch))
            rest_load += load
            batch,_ = fh.get_batch(rs)
    delays = np.array(delays)
    print('ft=%.3f rest=%.3f rest_t=%.3f avg_load=%.3f avg_delay=[%.4f %.4f]' %
          (ptime, rest_load, rest_load/speed_factor, loads.mean(), *delays.mean(0)))
    return loads, rest_load, delays, loads_detail


# %% test
import matplotlib.pyplot as plt
import util

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
            n = fh.put(np.zeros((int(rs/9*16),rs)), 0, i, fids[sid], Configure(rs, fr, False, 'yolov5m'), t)
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
            n = fh.put(np.zeros((int(rs/9*16),rs)), 0, i, fids[sid], Configure(rs, fr, False, 'yolov5m'), t)
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

def __test3__():
    rsl_list=[240,360,480,720]
    bs=8
    dl=2.0
    mat_pt=np.array([[52,37,28,25,22,27,25,24],
        [98,70,61,60,59,60,60,59],
        [154,131,110,115,110,105,101,96],
        [358,271,226,210,211,212,208,204]])*0.001
    
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

    # simulate workload
    fps_list = [1, 2, 5, 10, 15, 30]
    nsource = 10
    distr_rsl, cdistr_rsl_fps = compute_distribution_from_real(workload, len(rsl_list), fps_list)
    w,smy = simulate_workloads(nsource, nlength, distr_rsl, cdistr_rsl_fps, fps_list)
    assert w.shape == (nsource, nlength, 2)
    tasks = workload_to_tasks(w, rsl_list, 30)
    
    opt_loads,opt_rest,opt_delays = optimal_process(tasks, rsl_list, bs, mat_pt)
    print(opt_loads.mean(1), opt_rest, opt_delays.mean(0))
    
    plt.figure()
    plt.plot(util.moving_average(opt_loads,10).T)
    plt.ylim((-1, None))
    plt.plot(util.moving_average(opt_loads.sum(0),10).T,'--')
    plt.legend(rsl_list+['sum'])
    plt.ylabel('opt-workload (s)')
    plt.xlabel('time (s)')
    plt.tight_layout()
    
    speed_factor = 10.0
    capacity = 1 * speed_factor
    
    fh=FrameHolder(rsl_list, bs, dl, mat_pt)
    fh.set_ready_method('early')
    
    loads, rest_load, delays, loads_detail = simulate_process(tasks, fh, nlength, speed_factor)
    
    methods = ['early', 'small', 'finish']
    loads8 = np.zeros((4, nlength))
    rloads8 = np.zeros(4)
    delays8 = np.zeros(4)
    for i, m in enumerate(methods):
        fh.set_ready_method(m)
        #loads_detail: (ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_list())
        loads, rest_load, delays, loads_detail = simulate_process(tasks, fh, nlength, speed_factor)
        loads8[i]=loads
        rloads8[i] = rest_load
        delays8[i] = delays.sum(1).mean()
    plt.plot(util.moving_average(loads8[:3], 10).T)
    plt.plot(util.moving_average(opt_loads.sum(0), 10).T, '--')
    plt.ylim((-1,None))
    plt.legend(methods+['opt'])
    
    # analyze queue length
    queuelenght = np.zeros((len(rsl_list), nlength))
    num_process = np.zeros(nlength)
    for t, rs, bs, load, ql in loads_detail:
        ind_t = int(t)
        queuelenght[:,ind_t] += ql
        num_process[ind_t] += 1
    queuelenght /= num_process
    print((queuelenght<bs).mean(1))
    
    plt.hist(queuelenght.T)
    
    # approximate delay and usage
    delay,usage=profiling.get_delay_usage_with_bound(loads, capacity, 1)
    