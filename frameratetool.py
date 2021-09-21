# -*- coding: utf-8 -*-

import numpy as np
import time

# %% make features

import carcount

import profiling
from centroidtracker import CentroidTracker
import groundtruth

def extract_speed(pboxes, tracker, fps:int, fr:int, segment:int, abs=True):
    # pboxes is a list of numpy.ndarray for bounding boxes
    n_frm = len(pboxes)
    n_sec = n_frm // fps
    n_seg = n_sec // segment
    #n = n_seg * segment
    speed_avg = np.zeros(n_seg)
    speed_med = np.zeros(n_seg)
    speed_std = np.zeros(n_seg)
    tracker.reset()
    buffer = {}
    for i in range(n_seg):
        idx_base = i*segment
        speeds = []
        for j in range(0, fps*segment, fr):
            centers = pboxes[idx_base + j]
            objs = tracker.update(centers)
            for oid, c in objs.items():
                if oid in buffer:
                    old = buffer[oid]
                    s = c - old
                    speeds.append(s)
                    buffer[oid] = c
                else:
                    buffer[oid] = c
        if len(speeds) != 0:
            if abs is True:
                speeds = np.abs(speeds)
            speed_avg[i] = np.mean(speeds)
            speed_med[i] = np.median(speeds)
            speed_std[i] = np.std(speeds)
    return speed_avg, speed_med, speed_std

def prepare_data(vn_list, fps_list, rs_list, acc_bound=0.9,
                 ft_reso=480, ft_fr=2):
    vn_list = ['s3','s4','s5','s7']
    fps_list = [25,30,20,30]
    
    # input (feature)
    fsas = [] # feature speed average
    fsms = [] # feature speed median
    fsss = [] # feature speed standard derivation
    fcs = [] # feature count
    for fn, fps in zip(vn_list, fps_list):
        pboxes=carcount.load_raw_data('data/%s-raw-%d.npz' % (fn, ft_reso))[4]
        tracker = CentroidTracker(fps/2)
        fa,fm,fs=extract_speed(pboxes,tracker,fps,ft_fr,1,True)
        fsas.append(fa)
        fsms.append(fm)
        fsss.append(fs)
        
        gt=groundtruth.load_ground_truth('data/ground-truth-%s.txt' % fn)
        fcs.append(gt)
    
    fsas = np.concatenate(fsas)
    fsms = np.concatenate(fsms)
    fsss = np.concatenate(fsss)
    fcs = np.concatenate(fcs)
    
    feat = np.array([fsas, fsms, fsss, fcs]).T
    
    n_slot = []
    # output (configuration)
    pss = [] # profile selections
    conf_res = []
    for fn, fps in zip(vn_list, fps_list):
        if fps == 20:
            fr_list = profiling.FR_FOR_20
        elif fps == 25:
            fr_list = profiling.FR_FOR_25
        else:
            fr_list = profiling.FR_FOR_30
        _,_,sg_list,cts,cas=profiling.load_configurations('data/conf-%s.npz' % fn)
        sg_idx = sg_list.tolist().index(1)
        _,_,ps=profiling.get_profile_bound_acc(cts[sg_idx],cas[sg_idx],acc_bound)
        n_slot.append(ps.shape[0])
        pss.append(ps)
        conf = np.array([(rs_list[i], fr_list[j]) for i,j in ps])
        conf_res.append(conf)
        
    n_slot = np.array(n_slot)
    conf_res = np.concatenate(conf_res)
    return n_slot, feat, conf_res

def turn_to_trainable(feats, confs, num_prev):
    n1,m1 = feats.shape
    n2,m2 = confs.shape
    assert n1 == n2
    n = n1
    m = m1 + m2
    no = n - num_prev
    f = np.zeros((no, (m1+m2)*num_prev + m1))
    for i in range(num_prev):
        idx1 = i*m
        idx2 = i*m + m1
        idx3 = (i+1)*m
        f[:,idx1:idx2] = feats[i:no+i,:]
        f[:,idx2:idx3] = confs[i:no+i,:]
    return f, confs[num_prev:]

def augment_feat(x, deg=1, offset=True, log=False):
    res = x.copy()
    if deg > 1:
        for d in range(2,deg+1):
            t = x**d
            res = np.hstack([res, t])
    if log:
        res = np.hstack([res, np.log(x+1)])
    if offset:
        res = np.hstack([res, np.ones((len(res),1))])
    return res

def sample_balance(x, y, method='log', ratios={}):
    assert method in ['log', 'even', 'given']
    n = len(y)
    assert len(x) == n
    # group data
    data = {}
    for i in range(n):
        a, b = x[i], y[i]
        if y.ndim == 2:
            b = tuple(b.tolist())
        if b not in data:
            data[b] = [a]
        else:
            data[b].append(a)
    neach = { k:len(v) for k,v in data.items() }
    ngroup = len(neach)
    # generate ratios
    if method == 'log':
        a = np.array([v for v in neach.values()])
        t = np.log(a/a.sum()+np.e/2)
        ratios = { k:r for k,r in zip(neach.keys(), t) }
    elif method == 'even':
        ratios = { k: 1.0/v for k,v in neach.items() }
    else:
        assert sorted(ratios.keys()) != sorted(neach.keys())
    # assign number to each group
    s = sum(ratios.values())
    nassign = { k:int(np.round(v/s*n)) for k,v in ratios.items() }
    s = sum(nassign.values())
    if s != n:
        keys = sorted(nassign.keys(), key=lambda k:nassign[k])
        if s<n:
            for i in range(n-s):
                nassign[keys[i]] += 1
        else:
            for i in range(s-n):
                nassign[keys[ngroup - 1 - i]] -= 1
    # sample
    res_x = []
    res_y = []
    for k in data.keys():
        dx = np.array(data[k])
        nr = nassign[k]
        ind = np.random.randint(0, len(dx), nr)
        res_x.append(dx[ind])
        res_y.extend([k]*nr)
    res_x = np.concatenate(res_x)
    res_y = np.array(res_y)
    return res_x, res_y

# %% fit and meassure

import scipy.optimize

def fit_via_data(x, y, offset=False):
    assert x.ndim == 2
    if offset is True:
        l = len(x)
        x = np.hstack([x, np.ones((l,1))])
    w, residual, rank, s = np.linalg.lstsq(x, y, None)
    return w

def fit_via_func(func, x, y):
    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    return popt


def postprocess_fr(fr):
    if isinstance(fr, (np.ndarray,list)):
        res = np.array(fr, dtype=int)
        return np.clip(res, 1, None)
    else:
        res = int(fr)
        return max(1, fr)

def discretize_fr(fr, fr_list):
    pass

def measure_fr(output, predicted, thresholds=[10,5,3,1,0]):
    d = output - predicted
    n_close = []
    for th in thresholds:
        t = sum(np.logical_and(-th<=d, d<=th))
        n_close.append(t)
    residual = sum(d**2)
    std = np.sqrt(residual/len(d))
    return residual, std, n_close

# %% test

import matplotlib.pyplot as plt

def figure_setting():
    plt.rcParams["figure.figsize"]=(4,3)
    plt.rcParams["font.size"]=14

def evaluate(x, yf, w, thresholds):
    o = np.dot(x, w)
    o = postprocess_fr(o)
    d = o - yf
    r, s, dis = measure_fr(o, yf, thresholds)
    return (d, r, s, dis)

def show_difference(d, nbins=50, both=False,
                    xlbl='difference (frame)', ylbl='probability'):
    plt.figure()
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.hist(d, nbins, density=True, color=color[0])
    # cannot use the return value of plt.hist for h,x when density=True
    h,x = np.histogram(d, nbins)
    plt.tick_params(axis='y', labelcolor=color[0])
    if xlbl:
        plt.xlabel(xlbl)
    if ylbl:
        plt.ylabel(ylbl)
    if both:
        ax=plt.gca()
        ax2=ax.twinx()
        y = np.concatenate([[0],np.cumsum(h)])
        ax2.plot(x, y/y[-1], '--', color=color[1])
        ax2.set_ylim(0, None)
        #ax2.plot(x, y, '--', color=color[1])
        #ax2.set_ylabel('CDF')
        ax2.tick_params(axis='y', labelcolor=color[1])
    plt.grid(True, linestyle='--')
    plt.tight_layout()

def __test__():
    vn_list = ['s3','s4','s5','s7']
    fps_list = [25,30,20,30]
    n_video = len(vn_list)
    rs_list = profiling.RS
    
    thresholds=[10,5,3,1,0]
    
    n_slot, feats, confs = prepare_data(vn_list, fps_list, rs_list, 0.9)
    x,y = turn_to_trainable(feats,confs,2)
    yf = y[:,1]
    
    # degree 1 without log
    x1 = augment_feat(x, 1, True, False)
    w1 = fit_via_data(x1, yf)
    d1, r1, s1, dis1 = evaluate(x1, yf, w1, thresholds)
    show_difference(d1)
    print(r1, s1, dis1)
    
    # degree 2 without log
    x2 = augment_feat(x, 2, True, False)
    w2 = fit_via_data(x2, yf)
    d2, r2, s2, dis2 = evaluate(x2, yf, w2, thresholds)
    show_difference(d2)
    print(r2, s2, dis2)
    
    # degree 1 with log
    x3 = augment_feat(x, 1, True, True)
    w3 = fit_via_data(x3, yf)
    d3, r3, s3, dis3 = evaluate(x3, yf, w3, thresholds)
    show_difference(d3)
    print(r3, s3, dis3)
    
    # degree 2 with log
    x4 = augment_feat(x, 2, True, True)
    w4 = fit_via_data(x4, yf)
    d4, r4, s4, dis4 = evaluate(x4, yf, w4, thresholds)
    show_difference(d4)
    print(r4, s4, dis4)
    
    # balanced sample
    xx,yyf = sample_balance(x, yf, 'log')
    x5 = augment_feat(xx, 1, True, True)
    w5 = fit_via_data(x5, yyf)
    d5, r5, s5, dis5 = evaluate(x5, yyf, w5, thresholds)
    show_difference(d5)
    print(r5, s5, dis5)
    