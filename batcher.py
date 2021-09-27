# -*- coding: utf-8 -*-

import numpy as np
import time
from collections import namedtuple
from dataclasses import dataclass

from frameholder import FrameHolder
#Configure = namedtuple('Configure', ['rs', 'fr', 'roi', 'model'])
#FrameInfo = namedtuple('FrameInfo', ['tid', 'jid', 'sid', 'fid', 'time'])
#from frameholder import Configure
#from frameholder import FrameInfo
from reorderbuffer import ReorderBuffer


class LoadManager:
    def __init__(self, nlength, capacity):
        self.nlength = nlength
        self.loads = np.zeros(nlength)
        self.capacity = capacity
        self.rest = 0.0
    
    def add(self, idx, load):
        self.loads[idx] += load
        if self.loads[idx] <= self.capacity:
            return False
        r = self.loads[idx] - self.capacity
        idx += 1
        while idx < self.nlength and r > 0.0:
            self.loads[idx] += r
            r = max(0.0, self.loads[idx] - self.capacity)
            idx += 1
        return True

    def smooth(self):
        r = 0.0
        for idx, l in enumerate(self.loads):
            self.loads[idx] += r
            r = max(0.0, self.loads[idx] - self.capacity)
        self.rest = r
    
    @staticmethod
    def SmoothLoad(loads, capacity):
        res = np.zeros(len(loads))
        r = 0.0
        for idx, l in enumerate(loads):
            res[idx] = l + r
            if res[idx] > capacity:
                res[idx] = capacity
                r = res[idx] - capacity
            else:
                r = 0.0
        return res, r


Task = namedtuple('Task', ['time', 'frame', 'jid', 'sid', 'fid', 'rs', 'fr'])


def compute_distribution_over_frame(workload, nrsl, fps_list):
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

def compute_distribution_over_slot(workload, nrsl, fps_list):
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
        ns = w[:,0] == i
        distr_rsl[i] = ns.sum()
        nf = w[ns, 1]
        inds = np.digitize(nf, bins)
        for j in range(len(nf)):
            cdistr_rsl_fps[i,inds[j]] += 1
        cdistr_rsl_fps[i,:] /= ns.sum()
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
    tasks = []
    nfbefore = np.zeros((nsource, nlength), int)
    fids = [0 for _ in range(nsource)]
    for i in range(nlength):
        tsks = []
        for j in range(nsource):
            ind_rs = workload[j,i,0]
            rs = rsl_list[ind_rs]
            nf = workload[j,i,1]
            nfbefore[j,i] = nf
            if nf == 0:
                continue
            fr = int(max_fps/nf)
            for k in range(nf):
                t = i + 1.0/nf*k
                tsks.append((t, j, fr, ind_rs))
        tsks.sort(key=lambda v:v[0])
        for t, sid, fr, ind_rs in tsks:
            rs = rsl_list[ind_rs]
            #res.append((t, np.zeros((int(rs/9*16),rs)), 0, sid, fids[sid], rs, fr))
            tasks.append(Task(t, None, 0, sid, fids[sid], rs, fr))
            fids[sid] += 1
    nfbefore = np.cumsum(nfbefore, 1)
    return tasks, nfbefore

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
    for i, tsk in enumerate(tasks):
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
                delays.append((i, t-tt, c))
            q.clear()
    rest = np.zeros(nrsl)
    for ind_rs, q in enumerate(queues):
        c = costs[ind_rs]/batchsize*len(q)
        rest[ind_rs] += c
    delays = np.array(delays)
    return loads, rest, delays

def simulate_process(tasks, ntask_each, fh, nlength, speed_factor):
    nsource = len(ntask_each)
    ptime = 0.0
    rbs = [ ReorderBuffer() for _ in range(nsource) ]
    
    loads = np.zeros(nlength)
    details = []
    # delays: waiting delay, processing delay, commiting delay
    delays = [np.zeros((ntask_each[i], 3)) for i in range(nsource)]
    for tsk in tasks:
        # tsk is the next task
        #(t, frame, jid, sid, fid, rs, fr) = tsk
        now = tsk.time
        while ptime <= now:
            rdy_lvl, rdy_rs = fh.ready(ptime) # ready is a delegate function for scheduling
            if rdy_lvl is None:
                break
            batch,info = fh.get_batch(rdy_lvl, rdy_rs)
            load = fh.estimate_processing_time(rdy_rs, len(batch))
            details.append((ptime, rdy_lvl, rdy_rs, len(batch), load, fh.query_queue_length_as_mat()))
            loads[int(ptime)] += load
            eta = load/speed_factor
            #print(ptime, rdy_lvl, rdy_rs, load)
            for ifo in info:
                rbs[ifo.sid].put(ifo.fid, ptime + eta)
                # waiting delay and processing delay
                delays[ifo.sid][ifo.fid][:2] = (ptime - ifo.time, eta)
                #print(ifo.tid, ifo.sid, ifo.fid, ifo.time)
            ptime = ptime + eta
            for sid,rb in enumerate(rbs):
                fids, ts = rb.get()
                for fid, t in zip(fids, ts):
                    # commit delay
                    delays[sid][fid][2] = ptime - t
        fh.put(tsk.frame, 0, tsk.jid, tsk.sid, tsk.fid, tsk.rs, tsk.time)
        ptime = max(ptime, now)
    rest_load = 0.0
    for rs in fh.rsl_list:
        #while batch_info := fh.get_batch(rs):
        batch,_ = fh.get_batch(0, rs)
        while batch:
            load = fh.estimate_processing_time(rs, len(batch))
            rest_load += load
            batch,_ = fh.get_batch(0, rs)
    avg_delay = np.array([ d.mean(0) for d in delays ])
    print('ft=%.3f rest=%.3f rest_t=%.3f avg_load=%.3f avg_delay=[%.4f, %.4f %.4f]' %
          (ptime, rest_load, rest_load/speed_factor, loads.mean(), *avg_delay.mean(0)))
    return loads, rest_load, delays, details

# %% helper functions for live+certify+reine experiment

# when stream <sid> of type <cond_tp> finishes all tasks before <cond_tm> (inclusive)
@dataclass
class CRSegment:
    type:str
    sid:int
    t_start:int
    t_end:int
    rs:int
    fps:int
    cond_tp:str
    cond_tm:int # inclusive
    cond_nf:int=0
    
    def make_tasks(self, jid, fid0, now, max_fps=30):
        fr = max_fps//self.fps
        tasks = []
        fid = fid0
        for i in range(self.t_start, self.t_end):
            for j in range(self.fps):
                #t = i + 1.0/self.fps*j
                tasks.append(Task(now, None, jid, self.sid, fid, self.rs, fr))
                fid += 1
        return tasks


def generate_cr_segment(nsource, nlength, period, slength, interleave,
                        rrate, nfbefore, ctf_conf=(480,10), rfn_conf=(480,10)):
    assert slength < period < nlength
    assert 0 <= rrate <= 1
    assert nfbefore.shape == (nsource, nlength)
    csegs = []
    rsegs = []
    for i in range(nsource):
        if interleave:
            p = i + period//2
        else:
            p = period//2
        cstart = np.arange(p-slength//2, nlength, period)
        dorfn = np.random.random(len(cstart))<rrate
        for j, (ct, dr) in enumerate(zip(cstart, dorfn)):
            csegs.append(CRSegment('c', i, ct, ct+slength, ctf_conf[0], ctf_conf[1], 'l', ct, nfbefore[i, ct+slength-1]))
            if dr:
                s = ct - period//2
                f = ct + period//2
                rsegs.append(CRSegment('r', i, s, f, rfn_conf[0], rfn_conf[1], 'c', ct+slength-1, (j+1)*ctf_conf[1]))
    csegs.sort(key=lambda s:s.cond_tm)
    rsegs.sort(key=lambda s:s.cond_tm)
    return csegs, rsegs

def simulate_process_with_cr(tasks, ntask_each, csegs, rsegs, fh,
                             nsource, nlength, speed_factor):
    assert len(ntask_each) == nsource
    # reuse jid to identify job type during simulation
    JID_CTF = -1
    JID_RFN = -2
    rbs_l = [ ReorderBuffer() for _ in range(nsource) ]
    rbs_c = [ ReorderBuffer() for _ in range(nsource) ]
    
    pointer_c = 0
    pointer_r = 0
    cmt_nf_c = np.zeros(nsource, int)
    cmt_nf_r= np.zeros(nsource, int)
    
    ptime = 0.0
    loads = np.zeros(nlength)
    details = []
    delays = [np.zeros((ntask_each[i], 3)) for i in range(nsource)]
    
    commit_buffer = []
    
    for tsk in tasks:
        now = tsk.time
        # pick tasks to process (before <now>)
        flag_new_commit = False
        while ptime < now:
            rdy_lvl, rdy_rs = fh.ready(ptime)
            if rdy_lvl is None:
                break
            batch,info = fh.get_batch(rdy_lvl, rdy_rs)
            load = fh.estimate_processing_time(rdy_rs, len(batch))
            details.append((ptime, rdy_lvl, rdy_rs, len(batch), load, fh.query_queue_length_as_mat()))
            #print('%.3f'%ptime, rdy_lvl, rdy_rs, len(batch))
            loads[int(ptime)] += load
            eta = load/speed_factor
            if rdy_lvl == 0: # live
                for b,ifo in zip(batch,info):
                    delays[ifo.sid][ifo.fid][:2] = (ptime - ifo.time, eta)
                    commit_buffer.append((ptime+eta, b, ifo.jid, ifo.sid, ifo.fid))
            else: # ceritfy & refine
                for b,ifo in zip(batch,info):
                    commit_buffer.append((ptime+eta, b, ifo.jid, ifo.sid, ifo.fid))
            flag_new_commit = True
            ptime += eta
        # commit finished tasks (before <now>)
        if flag_new_commit:
            commit_buffer.sort(key=lambda t:t[0])
        for i, (ct,b,jid,sid,fid) in enumerate(commit_buffer):
            if ct > now:
                del commit_buffer[:i]
                break
            if jid >= 0: # live job
                rbs_l[sid].put(fid, b)
            elif jid == JID_CTF:
                rbs_c[sid].put(fid, b)
        # check and insert c-r jobs
        for s in csegs[pointer_c:]:
            if s.cond_tm > now:
                break
            if rbs_l[s.sid].move_and_check(s.cond_nf):
                pointer_c += 1
                ctasks = s.make_tasks(JID_CTF, cmt_nf_c[s.sid], now)
                #print('c-job:',s.sid,s.t_start,s.t_end)
                for ctsk in ctasks:
                    fh.put(ctsk.frame, 1, JID_CTF, ctsk.sid, ctsk.fid, ctsk.rs, ctsk.time)
                cmt_nf_c[s.sid] += len(ctasks)
        for s in rsegs[pointer_r:]:
            if s.cond_tm > now:
                break
            if rbs_c[s.sid].move_and_check(s.cond_nf):
                pointer_r += 1
                ctasks = s.make_tasks(JID_RFN, cmt_nf_r[s.sid], now)
                #print('r-job:',s.sid,s.t_start,s.t_end)
                for ctsk in ctasks:
                    fh.put(ctsk.frame, 1, JID_RFN, ctsk.sid, ctsk.fid, ctsk.rs, ctsk.time)
                cmt_nf_r[s.sid] += len(ctasks)
        # do live job
        fh.put(tsk.frame, 0, tsk.jid, tsk.sid, tsk.fid, tsk.rs, tsk.time)
        ptime = max(ptime, now)
        
    rest_load = np.zeros(2)
    for lvl in range(2):
        for rs in fh.rsl_list:
            #while batch_info := fh.get_batch(rs):
            batch,_ = fh.get_batch(lvl, rs)
            while batch:
                load = fh.estimate_processing_time(rs, len(batch))
                rest_load[lvl] += load
                batch,_ = fh.get_batch(0, rs)
    
    #delays = np.array(delays)
    avg_delay = np.array([ d.mean(0) for d in delays ])
    print('ft=%.3f rest=(%.3f,%.3f) rest_t=%.3f avg_load=%.3f avg_delay=[%.4f %.4f]' %
          (ptime, *rest_load, rest_load.sum()/speed_factor, loads.mean(), *avg_delay.mean(0)))
    return loads, rest_load, delays, details


# %% analyze result

def analyze_delay_overtime(delays, workload, rsl_list):
    assert workload.ndim == 3 and workload.shape[2] == 2
    nsource, nlength = workload.shape[:2]
    nrsl = len(rsl_list)
    
    delay_ot = np.zeros((nrsl, nlength))
    ntask_ot = np.zeros((nrsl, nlength), int)
    for sid, dl in enumerate(delays):
        p = 0
        for tid in range(nlength):
            rs_ind, nf = workload[sid,tid]
            delay_ot[rs_ind, tid] += dl[p:p+nf].sum()
            ntask_ot[rs_ind, tid] += nf
            p += nf
    # prevent warning of divided by zero
    temp = ntask_ot.sum(0)
    temp[temp==0] = 1
    dot_avg = delay_ot.sum(0) / temp
    ntask_ot[ntask_ot==0] = 1
    dot_each = delay_ot/ntask_ot
    return dot_each, dot_avg


def analyze_queue_length(details, rsl_list, nlength):
    queuelength = np.zeros((len(rsl_list), nlength))
    num_process = np.zeros(nlength)
    for t, lvl, rs, bs, load, ql in details:
        ind_t = int(t)
        queuelength[:,ind_t] += ql[0] # live queue
        #queuelength[:,ind_t] += ql[1] # certify queue
        num_process[ind_t] += 1
    queuelength /= num_process # average queue length of each time slot
    return queuelength


# %% plotting script

def show_queue_length(details, rsl_list, nlength, ql_max=None, nbin=8, log=False):
    queuelength = analyze_queue_length(details, rsl_list, nlength)
    if ql_max is None:
        ql_max = queuelength.max()
    
    plt.figure()
    plt.hist(queuelength.T, nbin, (0, ql_max))
    plt.legend(rsl_list)
    plt.xlabel('queue length')
    plt.ylabel('occurence')
    if log:
        plt.yscale('log')
    plt.tight_layout()


def show_delay_distribution(delays, nbin=100, cdf=True, bar=False, newfig=True):
    d = np.concatenate([ d.sum(1) for d in delays ])
    hh,xh=np.histogram(d, nbin)
    x = (xh[1:]+xh[:-1])/2
    h = hh / hh.sum()
    
    if newfig:
        plt.figure()
    if cdf:
        if bar:
            plt.bar(xh, plt.plot(xh, np.pad(np.cumsum(h), (1,0))), x[1]-x[0])
        else:
            plt.plot(xh, np.pad(np.cumsum(h), (1,0)))
    else:
        if bar:
            plt.bar(x, h, x[1]-x[0]) # x[1]-x[0] is the width
        else:
            plt.plot(x, h)
    plt.xlabel('delay (s)')
    if cdf:
        plt.ylabel('CDF')
    else:
        plt.ylabel('PDF')
    plt.tight_layout()

def show_delay_distribution_cmp(delays_list, legends=None, select_idx=None,
                                nbin=100, newfig=True):
    dly_max = max([d.sum(1).max() for dg in delays_list for d in dg])
    if select_idx is None:
        select_idx = list(range(len(delays_list)))
    if newfig:
        plt.figure()
    for i in select_idx:
        d = np.concatenate([d.sum(1) for d in delays_list[i]])
        hh,xh=np.histogram(d, nbin, (0, dly_max))
        h = hh / hh.sum()
        plt.plot(xh, np.pad(np.cumsum(h), (1,0)))
    if legends:
        plt.legend([legends[i] for i in select_idx])
    plt.xlabel('delay (s)')
    plt.ylabel('CDF')
    plt.tight_layout()

def show_delay_overtime(delays, workload, rsl_list, each_rsl=False, newfig=True):
    dot_each, dot_avg = analyze_delay_overtime(delays, workload, rsl_list)
    if newfig:
        plt.figure()
    #plt.plot(ad.sum(0))
    if each_rsl:
        plt.plot(util.moving_average(dot_each, 10).T)
        plt.legend(rsl_list)
    else:
        plt.plot(util.moving_average(dot_avg, 10))
    plt.ylim((0,None))
    plt.xlabel('time')
    plt.ylabel('delay (s)')
    plt.tight_layout()


def show_delay_overtime_cmp(delays_list, workload, rsl_list,
                            legends=None, select_idx=None, newfig=True):
    if newfig:
        plt.figure()
    if select_idx is None:
        select_idx = list(range(len(delays_list)))
    for i in select_idx:
        _, dot_avg = analyze_delay_overtime(delays_list[i], workload, rsl_list)
        plt.plot(util.moving_average(dot_avg, 10))
    if legends:
        plt.legend([legends[i] for i in select_idx])
    plt.ylim((0,None))
    plt.xlabel('time')
    plt.ylabel('delay (s)')
    plt.tight_layout()
        
# %% test
import matplotlib.pyplot as plt
import util

def __test__():
    rsl_list=[240,360,480,720]
    bs=8
    mat_pt=np.array([[52,37,28,25,22,27,25,24],
        [98,70,61,60,59,60,60,59],
        [154,131,110,115,110,105,101,96],
        [358,271,226,210,211,212,208,204]])*0.001
    
    import profiling
    vn_list = ['s3', 's4', 's5', 's7']
    vfps_list = [25,30,20,30]
    segment = 1
    
    pts=[]
    pas=[]
    pss=[]
    for i,vn in enumerate(vn_list):
        #_,_,sg_list,cts,cas,_=profiling.load_configurations('data/%s/conf-%s.npz' % (vn,vn), 1)
        _,_,sg_list,cts,cas,ccs=profiling.load_configurations('data/%s/conf.npz' % (vn), 2)
        sg_idx=sg_list.tolist().index(segment)
        pt,pa,ps=profiling.get_profile_bound_acc(cts[sg_idx],cas[sg_idx],0.9)
        pts.append(pt)
        pas.append(pa)
        pss.append(ps)
    
    nlength = 400
    workload = np.zeros((len(vn_list), nlength, 2), int)
    for i, fps in enumerate(vfps_list):
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
    distr_rsl, cdistr_rsl_fps = compute_distribution_over_slot(workload, len(rsl_list), fps_list)
    w,smy = simulate_workloads(nsource, nlength, distr_rsl, cdistr_rsl_fps, fps_list)
    assert w.shape == (nsource, nlength, 2)
    
    np.savez('data/simulated-workload-10',w=w,smy=smy)
    data=np.load('data/simulated-workload-10.npz',allow_pickle=True)
    w=data['w']; smy=data['smy']
    data.close()
    
    tasks, nfbefore = workload_to_tasks(w, rsl_list, 30)
    
    opt_loads,opt_rest,opt_delays = optimal_process(tasks, rsl_list, bs, mat_pt)
    print(opt_loads.mean(1), opt_rest, opt_delays[:,1:].mean(0))
    
    plt.figure()
    plt.plot(util.moving_average(opt_loads,10).T)
    plt.plot(util.moving_average(opt_loads.sum(0),10).T,'--')
    plt.ylim((-1, None))
    plt.legend(rsl_list+['sum'])
    plt.ylabel('opt-workload (s)')
    plt.xlabel('time (s)')
    plt.tight_layout()
    
    speed_factor = 8
    speed_factor = 11.5
    capacity = 1 * speed_factor
    
    fh=FrameHolder(rsl_list, bs, 1, mat_pt, 'come')
    
    loads, rest_load, delays, details = simulate_process(tasks, nfbefore[:,-1], fh, nlength, speed_factor)
    
    methods = ['come', 'small', 'finish', 'delay', 'priority', 'awt']
    legends = ['FCFS', 'SJFS', 'MLFS', 'LFFS', 'Priority', 'LWFS']
    bss = [1,2,4,8]
    
    loads8 = np.zeros((len(methods), nlength))
    rloads8 = np.zeros(len(methods))
    details8 = [None for _ in range(len(methods))]
    delays8 = [None for _ in range(len(methods))]
    #for i, bs in enumerate(bss):
    #    fh=FrameHolder(rsl_list, bs, 1, mat_pt, 'finish')
    for i, m in enumerate(methods):
        fh=FrameHolder(rsl_list, bs, 1, mat_pt, m, param_alpha=2.0)
        #details: (ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_list())
        loads, rest_load, delays, details = simulate_process(tasks, nfbefore[:,-1], fh, nlength, speed_factor)
        loads8[i]=loads
        rloads8[i] = rest_load
        details8[i] = details
        delays8[i] = delays
    # show loads
    plt.plot(util.moving_average(loads8[:3], 10).T)
    plt.plot(util.moving_average(opt_loads.sum(0), 10).T, '--')
    plt.ylim((-1,None))
    plt.legend(methods+['opt'])
    
    plt.figure()
    plt.plot(util.moving_average(loads8,10).T/capacity*100)
    plt.ylim((0, None))
    plt.xlabel('time (s)')
    plt.ylabel('device occupation (%)')
    plt.legend(['bs=%d'%bs for bs in bss], ncol=2)
    plt.tight_layout()
    
    # analyze queue length
    queuelength = analyze_queue_length(details, rsl_list, nlength)
    print((queuelength<bs).mean(1))
    
    ql_max = queuelength.max()
        
    plt.figure()
    for ql in queuelength:
        hh,xh=np.histogram(ql, 30, (0, ql_max))
        x = (xh[1:]+xh[:-1])/2
        h = hh / hh.sum()
        plt.plot(x, h)
    plt.legend(rsl_list)
    plt.xlabel('queue length')
    plt.ylabel('PDF')
    plt.tight_layout()
    
    show_queue_length(details, rsl_list, nlength, 8, True)
    
    # analyze delay
    
    # delay - total delay distribution
    ## pdf
    show_delay_distribution(delays, 100, False)
    ## cdf
    show_delay_distribution(delays, 100, True)
    ## selected some
    method_idx = [0,1,2,4]
    #method_idx = [0,2]
    show_delay_distribution_cmp(delays8, legends, method_idx)
    
    
    # delay - overtime
    show_delay_overtime(delays, w, rsl_list)
    
    show_delay_overtime_cmp(delays8, w, rsl_list, legends, method_idx)
    
    
# %% test with multiple kinds of jobs (live, certify, refine)

#loads, rest_load, delays, details = simulate_process_with_cr(tasks, nfbefore[:,-1], csegs, rsegs, fh, nsource, nlength, speed_factor)

def __test_lcr__():
    # live + certify + refine
    rsl_list=[240,360,480,720]
    bs=8
    mat_pt=np.array([[52,37,28,25,22,27,25,24],
        [98,70,61,60,59,60,60,59],
        [154,131,110,115,110,105,101,96],
        [358,271,226,210,211,212,208,204]])*0.001
    
    nlength = 400
    nsource = 10
    data=np.load('data/simulated-workload-10.npz',allow_pickle=True)
    w=data['w']; smy=data['smy']
    data.close()
    
    tasks, nfbefore = workload_to_tasks(w, rsl_list, 30)
    csegs, rsegs = generate_cr_segment(nsource, nlength, 30, 1, True, 0.05, nfbefore, (480,10), (480,10))
    
    speed_factor = 11.5
    capacity = 1 * speed_factor
    
    fh=FrameHolder(rsl_list, bs, 2, mat_pt, 'finish')
    loads_l, rload_l, delays_l, detail_l = simulate_process(tasks, nfbefore[:,-1], fh, nlength, speed_factor)
    fh.clear()
    loads_t, rload_t, delays_t, detail_t = simulate_process_with_cr(tasks, nfbefore[:,-1], csegs, rsegs, fh, nsource, nlength, speed_factor)
    
    loads_l, r = LoadManager.SmoothLoad(loads_l, capacity)
    rload_l += r
    loads_t, r = LoadManager.SmoothLoad(loads_t, capacity)
    rload_t += r
    
    #detail: (ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_mat())
    
    # workload (usage)
    plt.figure()
    plt.plot(util.moving_average(loads_l, 10)/capacity*100)
    plt.plot(util.moving_average(loads_t, 10)/capacity*100)
    plt.ylim((0,100))
    plt.legend(['Live-only','L+C+R'])
    plt.xlabel('time (s)')
    plt.ylabel('resource usage (%)')
    plt.tight_layout()
    
    # delay
    # delay - distribution    
    show_delay_distribution(delays_t)
    
    show_delay_distribution_cmp([delays_l, delays_t], ['Live-only','L+C+R'])
    
    # delay - overtime
    show_delay_overtime(delays_l, w, rsl_list)
    show_delay_overtime(delays_l, w, rsl_list, True)
    
    show_delay_overtime_cmp([delays_l, delays_t], w, rsl_list, ['Live-only','L+C+R'])
    plt.ylim((0,0.5))