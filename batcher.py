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


Task = namedtuple('Task', ['time', 'frame', 'jid', 'sid', 'fid', 'rs', 'fr'])


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

def simulate_process(tasks, fh, nlength, speed_factor):
    ptime = 0.0
    loads = np.zeros(nlength)
    loads_detail = []
    delays = []
    for i, tsk in enumerate(tasks):
        # tsk is the next task
        #(t, frame, jid, sid, fid, rs, fr) = tsk
        while ptime <= tsk.time:
            rdy_lvl, rdy_rs = fh.ready(ptime) # ready is a delegate function for scheduling
            if rdy_rs:
                batch,info = fh.get_batch(rdy_lvl, rdy_rs)
                load = fh.estimate_processing_time(rdy_rs, len(batch))
                loads_detail.append((ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_mat()))
                loads[int(ptime)] += load
                eta = load/speed_factor
                #print(ptime, rdy_lvl, rdy_rs, load)
                for ifo in info:
                    delays.append((ifo.tid, ptime - ifo.time, eta))
                    #print(ifo.tid, ifo.sid, ifo.fid, ifo.time)
                ptime = tsk.time + eta
            else:
                break
        fh.put(tsk.frame, 0, tsk.jid, tsk.sid, tsk.fid, tsk.rs, tsk.time)
        ptime = max(ptime, tsk.time)
    rest_load = 0.0
    for rs in fh.rsl_list:
        #while batch_info := fh.get_batch(rs):
        batch,_ = fh.get_batch(0, rs)
        while batch:
            load = fh.estimate_processing_time(rs, len(batch))
            rest_load += load
            batch,_ = fh.get_batch(0, rs)
    delays = np.array(delays)
    print('ft=%.3f rest=%.3f rest_t=%.3f avg_load=%.3f avg_delay=[%.4f %.4f]' %
          (ptime, rest_load, rest_load/speed_factor, loads.mean(), *delays[:,1:].mean(0)))
    return loads, rest_load, delays, loads_detail


# %% plot for test

def show_delay_distribution(delays, nbin=100, cdf=True, bar=False, newfig=True):
    d=delays[:,1:].sum(1)
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
    dly_max = max([d[:,1:].sum(1).max() for d in delays_list])
    if select_idx is None:
        select_idx = list(range(len(delays_list)))
    if newfig:
        plt.figure()
    for i in select_idx:
        d = delays_list[i][:,1:].sum(1)
        hh,xh=np.histogram(d, nbin, (0, dly_max))
        h = hh / hh.sum()
        plt.plot(xh, np.pad(np.cumsum(h), (1,0)))
    if legends:
        plt.legend([legends[i] for i in select_idx])
    plt.xlabel('delay (s)')
    plt.ylabel('CDF')
    plt.tight_layout()

def show_delay_overtime(delays, tasks, rsl_list, nlength, newfig=True):
    delay_ot = np.zeros((len(rsl_list), nlength))
    ntask_ot = np.zeros((len(rsl_list), nlength), int)
    for tid, dq, dp in delays:
        tid = int(np.round(tid))
        rs = tasks[tid].rs
        rid = rsl_list.index(rs)
        pid = int(tasks[tid].time)
        delay_ot[rid, pid] += dq+dp
        ntask_ot[rid, pid] += 1
    ntask_ot[ntask_ot==0] = 1 # prevent warning of divided by zero
    ad = delay_ot/ntask_ot
    if newfig:
        plt.figure()
    #plt.plot(ad.sum(0))
    plt.plot(util.moving_average(ad.sum(0),10))
    plt.xlabel('time')
    plt.ylabel('delay (s)')
    plt.tight_layout()


def show_delay_overtime_cmp(delays_list, tasks, rsl_list, nlength,
                            legends=None, select_idx=None, newfig=True):
    if newfig:
        plt.figure()
    if select_idx is None:
        select_idx = list(range(len(delays_list)))
    for i in select_idx:
        delay_ot = np.zeros((len(rsl_list), nlength))
        ntask_ot = np.zeros((len(rsl_list), nlength), int)
        for tid, dq, dp in delays_list[i]:
            tid = int(np.round(tid))
            rs = tasks[tid].rs
            rid = rsl_list.index(rs)
            pid = int(tasks[tid].time)
            delay_ot[rid, pid] += dq+dp
            ntask_ot[rid, pid] += 1
        ntask_ot[ntask_ot==0] = 1 # prevent warning of divided by zero
        ad = delay_ot/ntask_ot
        plt.plot(util.moving_average(ad.sum(0),10))
    plt.legend([legends[i] for i in select_idx])
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
    fps_list = [25,30,20,30]
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
    
    np.savez('data/simulated-workload-10',w=w,smy=smy)
    data=np.load('data/simulated-workload-10.npz',allow_pickle=True)
    w=data['w']; smy=data['smy']
    data.close()
    
    tasks, _ = workload_to_tasks(w, rsl_list, 30)
    
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
    
    speed_factor = 11.5
    capacity = 1 * speed_factor
    
    fh=FrameHolder(rsl_list, bs, 1, mat_pt, 'come')
    
    loads, rest_load, delays, loads_detail = simulate_process(tasks, fh, nlength, speed_factor)
    
    methods = ['come', 'small', 'finish', 'delay']
    legends = ['FCFS', 'QFS', 'FFFS', 'LFFS']
    
    loads8 = np.zeros((len(methods), nlength))
    rloads8 = np.zeros(len(methods))
    details8 = [None for _ in range(len(methods))]
    delays8 = [None for _ in range(len(methods))]
    #for bs in [1,2,4,8]:
    #    fh=FrameHolder(rsl_list, bs, mat_pt, 'finish')
    for i, m in enumerate(methods):
        fh.clear()
        fh.set_ready_method(m)
        #loads_detail: (ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_list())
        loads, rest_load, delays, loads_detail = simulate_process(tasks, fh, nlength, speed_factor)
        loads8[i]=loads
        rloads8[i] = rest_load
        details8[i] = loads_detail
        delays8[i] = delays
    # show loads
    plt.plot(util.moving_average(loads8[:3], 10).T)
    plt.plot(util.moving_average(opt_loads.sum(0), 10).T, '--')
    plt.ylim((-1,None))
    plt.legend(methods+['opt'])
    
    # analyze queue length
    queuelength = np.zeros((len(rsl_list), nlength))
    num_process = np.zeros(nlength)
    for t, rs, bs, load, ql in loads_detail:
        ind_t = int(t)
        queuelength[:,ind_t] += ql
        num_process[ind_t] += 1
    queuelength /= num_process # average queue length of each time slot
    print((queuelength<bs).mean(1))
    
    ql_max = queuelength.max()
    
    plt.figure()
    plt.hist(queuelength.T, 8)
    plt.legend(rsl_list)
    plt.xlabel('queue length')
    plt.ylabel('number')
    plt.tight_layout()
    
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
    
    
    # analyze delay
    
    # delay - total delay distribution
    ## pdf
    show_delay_distribution(delays, 100, False)
    ## cdf
    show_delay_distribution(delays, 100, True)
    ## selected some
    method_idx = [0,1,2,3]
    #method_idx = [0,2]
    show_delay_distribution_cmp(delays8, legends, method_idx)
    
    
    # delay - overtime
    show_delay_overtime(delays, tasks, rsl_list, nlength)
    
    show_delay_overtime_cmp(delays8, tasks, rsl_list, nlength, method_idx)
    
    
# %% test with multiple kinds of jobs (live, certify, refine)

def generate_certify_tasks(nsource, nlength, period, slength, ctf_conf=(480,10),
                           interleave=True):
    assert 1 <= ctf_conf[1] <= 30
    rs = ctf_conf[0]
    fr = int(30/ctf_conf[1])
    tasks = []
    segs = [[] for _ in range(nsource)]
    fids = [0 for _ in range(nsource)]
    for i in range(nsource):
        if interleave:
            p = i + period//2
        else:
            p = period//2
        while p + slength - 1 < nlength:
            segs[i].append(p-period//2)
            for j in range(slength):
                for k in range(ctf_conf[1]):
                    t = p + j + 1.0/ctf_conf[1]*k
                    tasks.append(Task(t, None, 1, i, fids[i], rs, fr))
                    fids[i] += 1
            p += period
    tasks.sort(key=lambda t:t.time)
    return segs, tasks
    
def generate_refine_tasks(csegs, nlength, period, rrate, rfn_conf=(480,10)):
    assert 1 <= rfn_conf[1] <= 30
    rs = rfn_conf[0]
    fr = int(30/rfn_conf[1])
    nsource = len(csegs)
    tasks = []
    rsegs = [[] for _ in range(nsource)]
    fids = [0 for _ in range(nsource)]
    for i, segs in enumerate(csegs):
        rnds = np.random.random(len(segs))
        for s, r in zip(segs, rnds):
            if r >= rrate:
                continue
            rsegs[i].append(s)
            for p in range(s, s+period):
                for k in range(rfn_conf[1]):
                    t = p + 1.0/rfn_conf[1]*k
                    tasks.append(Task(t, None, 2, i, fids[i], rs, fr))
                    fids[i] += 1
    tasks.sort(key=lambda t:t.time)
    return rsegs, tasks

def merge_tasks(ltasks, ctasks, rtasks):
    tasks = [*ltasks, *ctasks, *rtasks]
    tasks.sort(key=lambda t:t.time)
    return tasks

def filter_delays(tasks, delays, jid):
    idx = delays[:,0].astype(int)
    flags = np.zeros(len(delays), bool)
    for i, tid in enumerate(idx):
        if tasks[tid].jid == jid:
            flags[i] = True
    return delays[flags,:]

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

def simulate_proecss_with_cr(tasks, nfbefore, csegs, rsegs, fh,
                             nsource, nlength, speed_factor):
    assert nfbefore.shape == (nsource, nlength)
    # reuse jid to identify job type during simulation
    JID_CTF = -1
    JID_RFN = -2
    rbs_l = [ ReorderBuffer() for _ in range(nsource) ]
    rbs_c = [ ReorderBuffer() for _ in range(nsource) ]
    used_csegs = np.zeros(len(csegs), bool)
    used_rsegs = np.zeros(len(rsegs), bool)
    cmt_nf_c = np.zeros(nsource, int)
    cmt_nf_r= np.zeros(nsource, int)
    
    ptime = 0.0
    loads = np.zeros(nlength)
    details = []
    delays = [np.zeros((nfbefore[i,-1], 2)) for i in range(nsource)]
    
    commit_buffer = []
    
    now = 0.0
    for tsk in tasks:
        now = tsk.time
        flag_new_commit = False
        while ptime < now:
            rdy_lvl, rdy_rs = fh.ready(ptime)
            if rdy_lvl:
                batch,info = fh.get_batch(rdy_lvl, rdy_rs)
                load = fh.estimate_processing_time(rdy_rs, len(batch))
                details.append((ptime, rdy_lvl, rdy_rs, len(batch), load, fh.query_queue_length_as_mat()))
                loads[int(ptime)] += load
                eta = load/speed_factor
                if rdy_lvl == 0:
                    for b,ifo in zip(batch,info):
                        delays[ifo.sid][ifo.fid] = (ptime - ifo.time, eta)
                        commit_buffer.append((ptime+eta, b, ifo.jid, ifo.sid, ifo.fid))
                else:
                    for b,ifo in zip(batch,info):
                        commit_buffer.append((ptime+eta, b, ifo.jid, ifo.sid, ifo.fid))
                flag_new_commit = True
                ptime += eta
        if flag_new_commit:
            commit_buffer.sort(key=lambda t:t[0])
        for ct,b,jid,sid,fid in commit_buffer:
            if ct > now:
                break
            if jid >= 0:
                rbs_l[sid].put(b, fid)
            elif jid == JID_CTF:
                rbs_c[sid].put(b, fid)
        # check and insert c-r jobs
        for i, (s, f) in enumerate(zip(csegs, used_csegs)):
            if f:
                continue
            if s.cond_tm > now:
                break
            if rbs_l[s.sid].get_check(s.cond_nf):
                used_csegs[i] = True
                ctasks = s.make_tasks(JID_CTF, cmt_nf_c[s.sid])
                for ctsk in ctasks:
                    fh.put(ctsk.frame, 1, JID_CTF, ctsk.sid, ctsk.fid, ctsk.rs, ctsk.time)
                cmt_nf_c[s.sid] += len(ctasks)
        for i, (s, f) in enumerate(zip(rsegs, used_rsegs)):
            if f:
                continue
            if s.cond_tm > now:
                break
            if rbs_l[s.sid].get_check(s.cond_nf):
                used_csegs[i] = True
                ctasks = s.make_tasks(JID_RFN, cmt_nf_r[s.sid])
                for ctsk in ctasks:
                    fh.put(ctsk.frame, 1, JID_RFN, ctsk.sid, ctsk.fid, ctsk.rs, ctsk.time)
                cmt_nf_r[s.sid] += len(ctasks)
        # do live job
        fh.put(tsk.frame, 0, tsk.jid, tsk.sid, tsk.fid, tsk.rs, tsk.time)
        
    rest_load = 0.0
    for rs in fh.rsl_list:
        #while batch_info := fh.get_batch(rs):
        batch,_ = fh.get_batch(0, rs)
        while batch:
            load = fh.estimate_processing_time(rs, len(batch))
            rest_load += load
            batch,_ = fh.get_batch(0, rs)
    delays = np.array(delays)
    print('ft=%.3f rest=%.3f rest_t=%.3f avg_load=%.3f avg_delay=[%.4f %.4f]' %
          (ptime, rest_load, rest_load/speed_factor, loads.mean(), *delays[:,1:].mean(0)))
    return loads, rest_load, delays, details


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
    
    ltasks, nfbefore = workload_to_tasks(w, rsl_list, 30)
    csegs,ctasks = generate_certify_tasks(nsource, nlength, 30, 1, (480,10), True)
    rsegs,rtasks = generate_refine_tasks(csegs, nlength, 30, 0.05, (480,10))
    tasks = merge_tasks(ltasks, ctasks, rtasks)
    
    speed_factor = 11.5
    capacity = 1 * speed_factor
    
    fh=FrameHolder(rsl_list, bs, 2, mat_pt, 'finish')
    loads_l, rload_l, delays_l, detail_l = simulate_process(ltasks, fh, nlength, speed_factor)
    fh.clear()
    loads_t, rload_t, delays_t, detail_t = simulate_process(tasks, fh, nlength, speed_factor)
    
    #delays: (task_id, queue_delay, process_delay)
    #detail: (ptime, rdy_rs, len(batch), load, fh.query_queue_length_as_list())
    
    delays_tl=filter_delays(tasks,delays_t,0)
    
    # workload (usage)
    plt.figure()
    plt.plot(util.moving_average(loads_l, 10)/capacity*100)
    plt.plot(util.moving_average(loads_t, 10)/capacity*100)
    plt.ylim((0,100))
    plt.legend(['live-only','L+C+R'])
    plt.xlabel('time (s)')
    plt.ylabel('resource usage (%)')
    plt.tight_layout()
    
    # delay
    # delay - distribution    
    show_delay_distribution(delays_t)
    
    show_delay_distribution_cmp([delays_l, delays_tl], ['live-only','L+C+R'])
    
    # delay - overtime
    show_delay_overtime(delays_l, tasks, rsl_list, 400)
    
    show_delay_overtime_cmp([delays_l, delays_tl], tasks, rsl_list, 400, ['live-only','L+C+R'])
    