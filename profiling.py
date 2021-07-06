# -*- coding: utf-8 -*-

#import torch
#import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

import videoholder
#import operation

import carcounter


# %% configures

FR_FOR_20=[1,2,5,10,20,40] # fps: 20, 10, 4, 1, 0.5
FR_FOR_25=[1,2,5,12,25,50] # fps: 25, 12, 5, 1, 0.5
FR_FOR_30=[1,2,5,15,30,60] # fps: 30, 15, 6, 1, 0.5

RS=[240,360,480,720]


def generate_conf(cc, video, pbox_files, ground_truth,
                  segment_length, fr_list, rs_list=RS):
    assert len(pbox_files) == len(rs_list)
    if isinstance(ground_truth, str):
        ground_truth = np.loadtxt(ground_truth, int, delimiter=',')
    assert isinstance(ground_truth, np.ndarray)
    
    cc.video=video
    fps = int(np.ceil(video.fps))
    n_second = video.num_frame // fps
    n_segment = n_second // segment_length
    #n = n_segment * segment_length * fps
    #ground_truth = ground_truth[:n].reshape((n_segment, segment_length)).sum(1)
    
    if fr_list is None:
        fr_list = FR_FOR_25 if fps == 25 else FR_FOR_30
    
    shape = (len(rs_list), len(fr_list), n_segment)
    
    #conf_param = np.zeros(shape[:2], int)
    conf_times = np.zeros(shape)
    conf_accuracy = np.zeros(shape)
    
    for i in range(len(rs_list)):
        rs = rs_list[i]
        cc.change_rs(rs)
        pbfile = pbox_files[i]
                            
        data = carcounter.load_precompute_data(pbfile)
        ptimes = data[3]
        pboxes = data[4]
        cc.conf.rs = rs
        for j in range(len(fr_list)):
            fr = fr_list[j]
            cc.change_fr(fr)
            cc.reset()
            
            ctimes, counts = cc.count_with_raw_boxes(pboxes)
            times, accuacy = cc.generate_conf_result(
                ptimes, ctimes, counts, ground_truth, segment_length)
            conf_times[i,j] = times
            conf_accuracy[i,j] = accuacy
    return conf_times, conf_accuracy


def generate_configurations(cc, video, pbox_files, ground_truth,
                            sg_list, fr_list, rs_list):
    cts = []
    cas = []
    for sg in sg_list:
        ct, ca = generate_conf(cc, video, pbox_files, ground_truth,
                               sg, fr_list, rs_list)
        cts.append(ct)
        cas.append(ca)
    return cts, cas
    

def save_configurations(file, fr_list, rs_list, sg_list, 
                        ctime_list, caccuracy_list):
    n = len(sg_list)
    t = {sg_list[i]: (ctime_list[i], caccuracy_list[i]) for i in range(n)}
    np.savez(file, fr_list=fr_list, rs_list=rs_list, 
             sg_list=np.array(sg_list, int), confs=t)
    
def load_configurations(file):
    if not file.endswith('.npz'):
        file += '.npz'
    with np.load(file, allow_pickle=True) as data:
        fr_list = data['fr_list']
        rs_list = data['rs_list']
        sg_list = data['sg_list']
        conf = data['confs'].item()
        cts = []
        cas = []
        for sg in sg_list:
            ct, ca = conf[sg]
            cts.append(ct)
            cas.append(ca)
        return fr_list, rs_list, sg_list, cts, cas
                    
# %% profile

def get_profile_by_selection(conf_time, conf_acc, pfl_sel):
    assert conf_time.shape == conf_acc.shape
    nrs, nfr, ns = conf_time.shape
    assert pfl_sel.shape == (ns, 2)
    assert pfl_sel[:,0].max() <= nrs - 1
    assert pfl_sel[:,1].max() <= nfr - 1
    
    pfl_time = np.zeros(ns)
    pfl_acc = np.zeros(ns)
    
    for i in range(ns):
        rs_idx, fr_idx = pfl_sel[i]
        pfl_time[i] = conf_time[rs_idx,fr_idx,i]
        pfl_acc[i] = conf_acc[rs_idx,fr_idx,i]
    return pfl_time, pfl_acc, pfl_sel


def get_profile_bound_acc(conf_time, conf_acc, acc_bound):
    assert conf_time.shape == conf_acc.shape
    nrs, nfr, ns = conf_time.shape
    shape = (nrs, nfr)
    
    pfl_time = np.zeros(ns)
    pfl_acc = np.zeros(ns)
    pfl_sel = np.zeros((ns, 2), int)
    
    for i in range(ns):
        mask = conf_acc[:,:,i] > acc_bound
        if not np.any(mask):
            # find the most accurate one(s)
            v = conf_acc[:,:,i].max()
            mask = conf_acc[:,:,i] >= v
        # find the fastest one that satisifies the accuracy bound
        xs, ys = mask.nonzero()
        t = [conf_time[x,y,i] for x, y in zip(xs, ys)]
        ind = np.argmin(t)
        x, y = xs[ind], ys[ind]
        pfl_time[i] = conf_time[x,y,i]
        pfl_acc[i] = conf_acc[x,y,i]
        pfl_sel[i] = (x, y)
    return pfl_time, pfl_acc, pfl_sel

    
def show_selection(pf_time, pf_acc, pf_sel, 
                   fr_list=None, rs_list=None, fps=None, show_sel=True):
    plt.figure()
    SP_ID_2 = [211, 212]
    SP_ID_4 = [411, 412, 413, 414]
    if show_sel:
        sp_id = SP_ID_4
    else:
        sp_id = SP_ID_2
    plt.subplot(sp_id[0])
    plt.plot(pf_time)
    plt.ylabel('comp-time')
    
    plt.subplot(sp_id[1])
    plt.plot(pf_acc)
    plt.ylabel('accuracy')
    
    if not show_sel:
        return
    if fr_list is None or rs_list is None or fps is None:
        return
    rs_list = np.array(rs_list)
    plt.subplot(sp_id[2])
    plt.plot(rs_list[pf_sel[:,0]])
    plt.ylabel('resolution')
    
    if not isinstance(fr_list, np.ndarray):
        fr_list = np.array(fr_list)
    plt.subplot(sp_id[3])
    plt.plot(fps/fr_list[pf_sel[:,1]])
    plt.ylabel('fps')
    
    plt.tight_layout()
    

# %% scheduling

def simulate_workloads(cts, length):
    # cts is a list of 1-D array
    res = np.zeros(length)
    for ct in cts:
        l = len(ct)
        if l >= length:
            res += ct[:length]
        else:
            start = 0
            end = l
            off = 0
            while start < length:
                res[start:end] += ct[start-off:end-off]
                start = end
                end = min(end + l, length)
                off += l
    return res

def sum_pad_data_list(data_list, length):
    # data_list is a list of n-D array (n>1)
    shape = data_list[0].shape
    dtype = data_list[0].dtype
    res = np.zeros((length, *shape[1:]), dtype=dtype)
    for data in data_list:
        l = len(data)
        if l >= length:
            res += data[:length]
        else:
            start = 0
            end = l
            off = 0
            while start < length:
                res[start:end] += data[start-off:end-off]
                start = end
                end = min(end + l, length)
                off += l
    return res

def pad_data_list(data_list, length):
    # data_list is a list of 1-D array
    n = len(data_list)
    dtype = data_list[0].dtype
    res = np.zeros((n, length), dtype=dtype)
    for i, data in enumerate(data_list):
        r = (length + len(data) - 1) // len(data)
        d = np.tile(data, r)[:length]
        res[i,:] = d
    return res


def moving_average(array, window, padmethod='mean'):
    assert padmethod in ['edge', 'linear_ramp', 'mean', 'median']
    d = np.pad(array, (window-1, 0), padmethod)
    return np.convolve(d, np.ones(window), 'valid') / window
    

def get_delay_usage_with_bound(loads, bound, unit):
    '''
    Compute the delay of each workloads and the resource usage of each time 
    slot.
    The workload of a time slot is processed after this slot finishes.

    Parameters
    ----------
    loads : np.ndarray
        1D vector of the workloads generated in each time slot.
    bound : float
        the capacity of resource for each time slot.
    unit : float
        time length of each time slot.

    Returns
    -------
    delay : np.ndarray
        the delay of getting the results of workload.
    usage : np.ndarray
        the amount of resource usage in each time slot.
        the number of used time slot may be more than those with workloads.
    '''
    n = len(loads)
    speed = bound/unit # connect workload and delay
    delay = np.zeros(n)
    usage = [0.0]
    
    p_task = 0 # point to current unfinsihed task
    p_slot = 0 # point to current time slot
    work = 0.0
    # loop for each time slot until all tasks are done
    while p_task < n:
        # move to next time slot
        p_slot += 1
        rest = bound # rest available resource
        passed = 0.0 # passed time in this slot
        if work == 0.0 and p_task < n:
            work = loads[p_task]
        while p_task < min(n, p_slot) and rest > 0.0:
            # if a work can be done
            if work <= rest:
                t = work/speed
                delay[p_task] = unit*(p_slot - p_task - 1) + passed + t
                rest -= work
                passed += t
                # move to next task
                p_task += 1
                work = loads[p_task] if p_task < min(p_slot, n) else 0.0
            else:
                work -= rest
                rest = 0.0
        usage.append(bound-rest)
    usage = np.array(usage)
    return delay, usage

def show_delay_usage(loads, delay, usage, bound,
                     xlbl='segment', ylbl='time (s)', title=None,
                     legend=True, new_fig=True):
    if new_fig:
        plt.figure()
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
    if title:
        plt.title(title)
    plt.plot(loads)
    plt.plot(delay)
    plt.plot(usage)
    plt.plot(np.arange(len(usage)), np.zeros_like(usage) + bound,'--')
    if isinstance(legend, bool) and legend == True:
        plt.legend(['workload', 'delay', 'usage', 'bound'])
    elif isinstance(legend, (list, np.ndarray)):
        plt.legend(legend)
    plt.tight_layout()
    

# %% test

def __test_show_profile__():
    import detect.yolowrapper as yolowrapper
    
    v1=videoholder.VideoHolder('E:/Data/video/s3.mp4')
    rng=carcounter.CheckRange('h',0.5,0.1,0.08)
    model=yolowrapper.YOLO_torch('yolov5s',0.5,(2,3,5,7))
    
    fps = int(v1.fps)
    # fps = 25
    
    cc=carcounter.CarCounter(v1,rng,model,480,1)
    box_files = ['data/s3-raw-%d.npz'%r for r in RS]
    segment_length = 5
    
    # get and show profile (accuracy requirement: 0.9)
    ct,ca=generate_conf(cc, v1, box_files, 'data/ground-truth-s3.txt',
                        segment_length, FR_FOR_25, RS)
    pt,pa,ps=get_profile_bound_acc(ct,ca,0.9)
    show_selection(pt,pa,ps,FR_FOR_25,RS,25,True)
    
    # running performance in resource-bounded environment
    bound = pt.mean()*1.2
    delay,usage=get_delay_usage_with_bound(pt,bound,segment_length)
    show_delay_usage(pt,delay,usage,bound)
    
    plt.figure()
    for i,q in enumerate([60,70,80]):
        bound=np.percentile(pt, q)
        delay,usage=get_delay_usage_with_bound(pt,bound,segment_length)
        plt.subplot2grid((3,1),(i,0))
        show_delay_usage(pt,delay,usage,bound,None,
                         'time (s)','percentile-%d'%q,None,False)
    
    # multiple videos
    vn_list = ['s3', 's4', 's5', 's7']
    segment = 1
    
    # cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lines = ["-","--","-.",":"]
    
    pts=[]
    pas=[]
    pss=[]
    for i,vn in enumerate(vn_list):
        _,_,sg_list,cts,cas=load_configurations('data/%s/conf-%s.npz' % (vn,vn))
        sg_idx=sg_list.tolist().index(segment)
        pt,pa,ps=get_profile_bound_acc(cts[sg_idx],cas[sg_idx],0.9)
        pts.append(pt)
        pas.append(pa)
        pss.append(ps)
        
    load = simulate_workloads(pts, 400)
    
    plt.figure()
    for i,b in enumerate([1.1, 1.2, 1.3]):
        bound = load.mean()*b
        delay,usage=get_delay_usage_with_bound(load,bound,1)
        plt.subplot2grid((3,1),(i,0))
        show_delay_usage(load,delay,usage,bound,None,'times(s)',
                         'mean*%g'%b, False, False)
        
    # multiple videos - seperated
    
    b=1.5
    plt.figure()
    plt.title('bound: mean*%g'%b)
    for i in range(4):
        bound=pts[i].mean()*b
        delay,usage=get_delay_usage_with_bound(pts[i],bound,1)
        plt.subplot2grid((2,2),(i//2,i%2))
        show_delay_usage(pts[i],delay,usage,bound,None,'times(s)' if i%2==0 else '',
                         None, False, False)
    
    # profile accuracy-computing time
    
    i = 2
    pt, pa = pts[i], pas[i]
    #vn = vn_list[i]
    #_,_,sg_list,cts,cas=load_configurations('data/%s/conf-%s.npz' % (vn,vn))
    #pt, pa = cts[0][3,0,:], cas[0][3,0,:]
    
    plt.figure()
    plt.subplot2grid((2,1),(0,0))
    plt.plot(moving_average(pt,10))
    #plt.xlabel('time (s)')
    plt.ylabel('cmp-res (s)')
    #plt.ylim((-1,11))
    plt.subplot2grid((2,1),(1,0))
    plt.plot(moving_average(pa,10))
    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    plt.ylim((-0.1,1.1))
    plt.tight_layout()
    
    ## serving multiple videos (separatedly vs jointly)
    
    ptt = pad_data_list(pts,400)
    paa = pad_data_list(pas,400)
    b=1.5
    
    ds, us = [], []
    for i in range(len(pts)):
        pt=ptt[i,:]
        bound = pt.mean()*b
        #t = moving_average(pt,5)
        d,u = get_delay_usage_with_bound(pt,bound,1)
        ds.append(d)
        us.append(u)
    
    dsmm = np.array([(d.mean(),d.max()) for d in ds])
    pt = ptt.sum(0)
    dsum, usum = get_delay_usage_with_bound(pt, pt.mean()*b, 1)
    
    # workload
    
    plt.figure()
    for i in range(4):
        if i % 2 == 0:
            plt.subplot2grid((2,1),(i//2,0))
            #lbls=[]
        plt.plot(moving_average(ptt[i],10), color=colors[i])
        #lbls.append('video-%d'%i)
        #if i % 2 == 1:
        #    plt.legend(lbls)
        plt.ylabel('cmp-res (s)')
    plt.xlabel('time (s)')
    plt.tight_layout()
    
    plt.figure()
    for i in range(4):
        plt.plot(moving_average(ptt[i],10), color=colors[i])
    plt.ylabel('cmp-resource (s)')
    plt.xlabel('time (s)')
    plt.legend(['v%d'%i for i in range(4)], ncol=2)
    #plt.ylim((-0.2,5.1))
    plt.tight_layout()
    
    plt.figure()
    plt.plot(moving_average(pt,10))
    plt.xlabel('time (s)')
    plt.ylabel('total cmp-resource (s)')
    #plt.ylim((-0.2,5.1))
    plt.tight_layout()

    q = 0.95
    t = [np.quantile(ptt[i], q)/ptt[i].mean() for i in range(4)]
    print('%g%% of slots finish within X times of the average:' %q, t)
    t2=[ptt[i,ptt[i]/ptt[i].mean()>t[i]].mean()/ptt[i].mean() for i in range(4)]
    print('the slowest %g%% slots requires X times of the average'%q, t2)

    np.quantile(pt, q)
    t = np.quantile(pt, q)/pt.mean()
    print('%g%% of slots finish no more than %.2f times of the average '%(q*100, t))
    
    # individual delay
    width=0.8
    
    plt.figure()
    x = np.arange(4)
    plt.bar(x-width/4, dsmm[:,0], width=width/2)
    plt.bar(x+width/4, dsmm[:,1], width=width/2)
    #plt.bar(np.arange(5)+width/4, np.concatenate([dsmm[:,0],[dsum.mean()]]), width=width/2)
    #plt.bar(np.arange(5)+width/4, np.concatenate([dsmm[:,1],[dsum.max()]]), width=width/2)
    plt.ylabel('delay (s)')
    plt.legend(['average','maximum'])
    plt.xticks(x, ['video-%d'%i for i in x], rotation=-30)
    plt.tight_layout()
    
    plt.figure()
    for i in range(4):
        bound=ptt[i].mean()*b
        delay,usage=get_delay_usage_with_bound(ptt[i],bound,1)
        plt.plot(delay)
    plt.legend(['video-%d'%i for i in range(4)], ncol=1)
    plt.xlabel('time (s)')
    plt.ylabel('delay (s)')
    plt.tight_layout()
    
    # delay comparision
    plt.figure()
    x = np.arange(2)
    plt.bar(x-width/4, [dsmm[:,0].mean(), dsum.mean()], width=width/2)
    plt.bar(x+width/4, [dsmm[:,1].max(), dsum.max()], width=width/2)
    plt.ylabel('delay (s)')
    plt.legend(['mean-delay','max-delay'])
    plt.xticks(x, ['Separated', 'Joint'], rotation=0)
    plt.tight_layout()

    # mean delay
    plt.figure()
    x = np.arange(2)
    plt.bar(x, [dsmm[:,0].mean(), dsum.mean()], width=0.6)
    plt.xlim((-0.6,1.6))
    plt.ylabel('delay (s)')
    plt.xticks(x, ['individually\nserve', 'jointly\nserve'], rotation=0)
    plt.tight_layout()
    
    # max delay
    plt.figure()
    x = np.arange(2)
    plt.bar(x, [dsmm[:,1].max(), dsum.max()], width=0.6)
    plt.xlim((-0.6,1.6))
    plt.ylabel('delay (s)')
    plt.xticks(x, ['individually\nserve', 'jointly\nserve'], rotation=0)
    plt.tight_layout()
    
    
    ## verification task
    import util.sample_data
    import util.sample_index
    
    i=2
    vn=vn_list[i]
    _,_,sg_list,cts,cas=load_configurations('data/%s/conf-%s.npz' % (vn,vn))
    pag = cas[0][3,0]
    #pag = cas[0].reshape(24,-1).max(0)
    pa = pas[i]
    
    off = 0
    off = 2
    
    # pick 1 second every 30 second
    pad = np.zeros_like(pa)+np.nan
    vidx = util.sample_index(len(pag), 30, 1)+off
    vdata = util.sample_data(moving_average(pag, 10)[off:], 30, 1)
    pad[vidx] = vdata
    #pad[util.sample_index(len(pag),30,1)]=util.sample_data(moving_average(pag,10),30,1)
    #pad[util.sample_index(len(pag)+off,30,1)]=util.sample_data(moving_average(pag,10)[2:],30,1)
    
    plt.figure()
    #t=np.array([moving_average(pa, 10), moving_average(pag, 10)])
    plt.plot(pad, '-X', markersize=8, color=colors[0])
    plt.plot(moving_average(pa, 10), '--', color=colors[1])
    plt.plot(moving_average(pag, 10), '-', color=colors[0])
    plt.ylim((-0.1, 1.1))
    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    #plt.legend(['verify-all', 'verify-smp', 'live'])
    #plt.legend(['verify', '_hidden', 'live'])
    plt.legend(['certify', 'live', 'actual'])
    plt.tight_layout()
    
    
        