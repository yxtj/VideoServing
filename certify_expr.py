# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import util
#import common

# %% util functions

def compute_accuray(counts, gtruth, segment=1):
    n = len(gtruth)
    n_segment = n // segment
    n = n_segment * segment
    counts = counts[:n].reshape((n_segment, segment)).sum(1)
    gtruth = gtruth[:n].reshape((n_segment, segment)).sum(1)
    up = np.array([counts, gtruth]).max(0)
    down = np.array([counts, gtruth]).min(0)
    accuracy = np.zeros(n_segment)
    for i in range(n_segment):
        if up[i] == 0:
            accuracy[i] = 1.0
        else:
            accuracy[i] = down[i] / up[i]
    return accuracy

def pick_by_conf(ccs, ps):
    nrs, nfr, nseg = ccs.shape
    assert ps.shape == (nseg, 2)
    assert ps[:,0].max() < nrs
    assert ps[:,1].max() < nfr
    res = [ ccs[a,b,i] for i,(a,b) in enumerate(ps) ]
    return np.array(res)

# %% key part of certify and refine

def certify(low_result, high_result, high_time, threshold, speriod, slength):
    assert len(low_result) == len(high_result) == len(high_time)
    assert 0.0 <= threshold <= 1.0
    n = len(low_result)
    idxes = util.sample_index(n, speriod, slength, 'middle')
    rngs = []
    ctime = np.zeros(len(idxes))
    cerror = np.zeros(len(idxes))
    for i, sidx in enumerate(idxes):
        diff = low_result[sidx] - high_result[sidx]
        upper = np.array([low_result[sidx], high_result[sidx]]).max(0)
        e = sum(diff[j]/upper[j] if upper[j]!=0 else 0 for j in range(len(sidx)))/len(sidx)
        cerror[i] = e
        if e > threshold:
            rngs.append((i*speriod, min((i+1)*speriod, n)))
        ctime[i] = high_time[sidx].sum()
    return rngs, ctime, cerror


def refine(low_result, low_time, high_result, high_time, refine_segments):
    assert len(low_result) == len(low_time)
    n = len(low_result)
    assert len(high_result) == len(high_time)
    assert n >= len(high_result)
    res_r = low_result.copy()
    res_t = low_time.copy()
    for f,l in refine_segments:
        res_r[f:l] = high_result[f:l]
        res_t[f:l] = high_time[f:l]
    return res_r, res_t

# %% main script

def __test__():
    rsl_list=[240,360,480,720]
    
    import profiling
    vn_list = ['s3', 's4', 's5', 's7']
    fps_list = [25,30,20,30]
    segment = 1
    
    ctss=[]; cass=[]; ccss=[]
    pts=[]; pas=[]; pss=[]
    for i,vn in enumerate(vn_list):
        _,_,sg_list,cts,cas,ccs=profiling.load_configurations('data/%s/conf.npz' % (vn))
        sg_idx=sg_list.tolist().index(segment)
        ctss.append(cts[sg_idx])
        cass.append(cas[sg_idx])
        ccss.append(ccs[sg_idx])
        pt,pa,ps=profiling.get_profile_bound_acc(cts[sg_idx],cas[sg_idx],0.9)
        pts.append(pt)
        pas.append(pa)
        pss.append(ps)
    
    
    vidx = 2
    vn = vn_list[vidx]
    gt = np.loadtxt('data/%s/ground-truth-%s.txt'%(vn,vn), int, delimiter=',')
    live_count = pick_by_conf(ccss[vidx], pss[vidx])
    live_time = pick_by_conf(ctss[vidx], pss[vidx])
    live_acc = pick_by_conf(cass[vidx], pss[vidx])
    certify_count = ccss[vidx][2,2,:]
    certify_time = ctss[vidx][2,2,:]
    
    rf_segs, ct_t, ct_e = certify(live_count, certify_count, 0.3, 30, 1)
    rf_c, rf_t = refine(live_count, live_time, certify_count, certify_time, gt, rf_segs)
    rf_a = compute_accuray(rf_c, gt)
    ct_t_pad = util.pad_by_sample(ct_t, len(live_time), 30, 1, 0, 'middle', 0.0)
    
    # accuracy by other segment size
    la2 = compute_accuray(live_count, gt, 2)
    ra2 = compute_accuray(rf_c, gt, 2)
    print(la2.mean(), ra2.mean())
    
    # time comparison
    plt.figure()
    plt.plot(live_time)
    plt.plot(ct_t_pad)
    plt.plot(rf_t)
    plt.legend(['live', 'certify', 'refine'])
    plt.tight_layout()
    
    print('live=%.2f certify=%.2f refine=%.2f overhead=%.2f (%.2f%%), '
          'o_c=%.2f%% o_r=%.2f%%' %
          (live_time.sum(), ct_t.sum(), rf_t.sum(), ct_t.sum()+rf_t.sum(),
           (ct_t.sum()+rf_t.sum())/live_time.sum()*100,
           ct_t.sum()/live_time.sum()*100, rf_t.sum()/live_time.sum()*100
           ))
    
    # accuracy comparison
    plt.figure()
    plt.plot(util.moving_average(live_acc, 10))
    plt.plot(util.moving_average(rf_a, 10))
    plt.ylim((-0.05, 1.05))
    plt.legend(['live', 'refine'])
    plt.legend(['w/o refine', 'w/ refine'])
    plt.tight_layout()
    
    print('live=%.4f refine=%.4f gain=%.4f' % (live_acc.mean(), rf_a.mean(), rf_a.mean()-live_acc.mean()))
    
    