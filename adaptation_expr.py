# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import util

# %% basic functions

plt.rcParams["figure.figsize"]=(4,3)
plt.rcParams["font.size"]=13
plt.rcParams["font.family"]='monospace'

#import profiling.load_configurations as load_configurations
#import profiling.get_profile_bound_acc as get_profile_bound_acc

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


# %% adaptation - profile-based

def adapt_profile(ptime, pacc, acc_bounds, pfl_period, pfl_length):
    '''
    Returns
    -------
    res_t : time list
    res_a : accuracy list
    res_s : configure list
    pf_t : profiling time list
    '''
    assert ptime.shape == pacc.shape
    assert ptime.ndim >= 2
    nknob = ptime.ndim - 1
    shape = ptime.shape
    nseg = shape[-1]
    assert isinstance(acc_bounds, (float, list, np.ndarray))
    if isinstance(acc_bounds, float):
        acc_bounds = [acc_bounds]
    acc_bounds = sorted(acc_bounds, reverse=True)
    ptime = ptime.reshape((-1, nseg))
    pacc = pacc.reshape((-1, nseg))
    idxes = util.sample_index(nseg, pfl_period, pfl_length, 'head')
    sel_idx = []
    for m in idxes:
        pt = ptime[:,m].mean(1)
        pa = pacc[:,m].mean(1)
        for ab in acc_bounds:
            mask = pa >= ab
            if np.any(mask):
                break
        else:
            # not break, all conditions are checked
            mask = pa >= pa.max()
        t = pt[mask.nonzero()]
        ind = np.argmin(t)
        sel_idx.append(ind)
    pfl_sel = np.array(np.unravel_index(sel_idx, shape[:-1])).T
    res_t = np.zeros(nseg)
    res_a = np.zeros(nseg)
    res_s = np.zeros((nseg, nknob), int)
    pf_t = np.zeros(nseg)
    for i in range(len(idxes)):
        if i != len(idxes) - 1:
            rng = range(idxes[i,0], idxes[i+1,0])
        else:
            rng = range(idxes[i,0], nseg)
        pf_t[rng[0]] = ptime[:,rng[0]].sum()
        s = sel_idx[i]
        res_t[rng] = ptime[s,rng]
        res_a[rng] = pacc[s, rng]
        res_s[rng] = pfl_sel[i]
    return res_t, res_a, res_s, pf_t

# %% adaptation - profile-free


# %% test

def __test__():
    WLF = 4.4 # factor from time to GFLOPS
    WLF = 220 # factor from time to GFLOPS
    
    vn_list = ['s3', 's4', 's5', 's7']
    segment = 1
    
    # cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lines = ["-","--","-.",":"]
    
    ctss=[]
    cass=[]
    pts=[]
    pas=[]
    pss=[]
    for i,vn in enumerate(vn_list):
        _,_,sg_list,cts,cas=load_configurations('data/%s/conf-%s.npz' % (vn,vn))
        sg_idx=sg_list.tolist().index(segment)
        pt,pa,ps=get_profile_bound_acc(cts[sg_idx],cas[sg_idx],0.9)
        ctss.append(cts)
        cass.append(cas)
        pts.append(pt)
        pas.append(pa)
        pss.append(ps)

    vn='s5'
    _,_,sg_list,cts,cas=load_configurations('data/%s/conf-%s.npz' % (vn,vn))
    
    # no adaptation
    ptg = cts[0][3,0]
    pag = cas[0][3,0]
    # profile-based adaptation (static)
    ao_t,ao_a,ao_s,ao_pt=adapt_profile(cts[0],cas[0],[0.9,0.8,0.7,0.5],len(ptg),30)
    # profile-based adaptation (dynamic)
    ap_t,ap_a,ap_s,ap_pt=adapt_profile(cts[0],cas[0],[0.9,0.8,0.7,0.5],30,1)
    # profile-free adaptation
    pt,pa,ps=get_profile_bound_acc(cts[0],cas[0],0.9)

    # workload
    plt.figure()
    plt.plot(ptg*WLF) # none
    plt.plot(ao_t*WLF) # profile-once
    plt.plot((ap_t+ap_pt)*WLF) # profile-period
    plt.plot(util.moving_average(pt,10)*WLF) # predict
    plt.xlabel('time (s)')
    plt.ylabel('resource (GFLOP)')
    #plt.ylabel('resource (s)')
    plt.legend(['no-adapt', 'prf-once','prf-period','prf-free'],ncol=2,fontsize=12,columnspacing=0.5)
    plt.yscale('log')
    plt.tight_layout()
    
    # zoomin comparison for workload
    plt.figure()
    plt.plot(util.moving_average(ao_t,5)*WLF) # profile-once
    plt.plot((util.moving_average(ap_t,5)+ap_pt)*WLF) # profile-period
    plt.plot(util.moving_average(pt,5)*WLF) # predict
    plt.xlabel('time (s)')
    plt.ylabel('resource (GFLOP)')
    #plt.ylabel('resource (s)')
    plt.legend(['prf-once','prf-period','prf-free'])
    #plt.ylim((0,10))
    plt.yscale('log')
    plt.ylim(0.5,5000)
    plt.tight_layout()

    # accuracy
    plt.figure()
    plt.plot(util.moving_average(pag,10)) # none
    plt.plot(util.moving_average(ao_a,10)) # profile-once
    plt.plot(util.moving_average(ap_a,10)) # profile-period
    plt.plot(util.moving_average(pa,10)) # predict
    plt.ylim((-0.1,1.1)) # plt.ylim((-0.05,1.05))
    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    plt.legend(['no-adapt', 'prf-once','prf-period','prf-free'],ncol=2,fontsize=12,columnspacing=0.5)
    plt.tight_layout()
    
    al = [pag.mean(), ao_a.mean(), ap_a.mean(), pa.mean()]
    #al.reverse()
    x = np.arange(4)
    plt.figure()
    plt.bar(x, al, width=0.7)
    plt.xticks(x, ['no-adapt', 'prf-once','prf-period','prf-free'], rotation=-30)
    plt.ylim((0,1))
    plt.ylabel('accuracy')
    plt.tight_layout()
    
    
    # workload and accuracy
    plt.figure()
    plt.subplot2grid((2,1),(0,0))
    plt.plot(pt*WLF)
    plt.ylim((0,None))
    plt.ylabel('rsc (GFLOP)')
    plt.subplot2grid((2,1),(1,0))
    plt.plot(util.moving_average(pa,10))
    plt.ylim((-0.05,1.05))
    plt.ylabel('accuracy')
    plt.xlabel('time (s)')
    plt.tight_layout()

