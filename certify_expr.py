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
    rf_r = np.zeros_like(low_result)
    rf_t = np.zeros_like(low_time)
    final_r = low_result.copy()
    final_t = low_time.copy()
    for f,l in refine_segments:
        rf_r[f:l] = high_result[f:l]
        rf_t[f:l] = high_time[f:l]
        final_r[f:l] = high_result[f:l]
        final_t[f:l] = high_time[f:l]
    return rf_r, rf_t, final_r, final_t

# %% main script

def __test__():
    WLF = 220 # factor from time to GFLOPS
    rsl_list=[240,360,480,720]
    
    import profiling
    vn_list = ['s3', 's4', 's5', 's7']
    vfps_list = [25,30,20,30]
    segment = 1
    
    ctss=[]; cass=[]; ccss=[]
    pts=[]; pas=[]; pss=[]
    for i,vn in enumerate(vn_list):
        _,_,sg_list,cts,cas,ccs=profiling.load_configurations('data/%s/conf.npz' % (vn), 2)
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
    
    vn = 's5'
    gt = np.loadtxt('data/%s/ground-truth-%s.txt'%(vn,vn), int, delimiter=',')
    _,_,sg_list,cts,cas,ccs=profiling.load_configurations('data/%s/conf.npz' % (vn), 2)
    sg_idx=sg_list.tolist().index(segment)
    cts, cas, ccs = cts[sg_idx], cas[sg_idx], ccs[sg_idx]
    pt,pa,ps=profiling.get_profile_bound_acc(cts,cas,0.9)
    live_count = pick_by_conf(ccs, ps)
    live_time = pick_by_conf(cts, ps)
    live_acc = pick_by_conf(cas, ps)
    certify_count = ccs[2,2,:]
    certify_time = cts[2,2,:]
    
    srate=30
    
    rf_segs, ct_t, ct_e = certify(live_count, certify_count, certify_time, 0.4, srate, 1)
    rf_c, rf_t, final_c, final_t = refine(live_count, live_time, certify_count, certify_time, rf_segs)
    rf_a = compute_accuray(final_c, gt, 1)
    ct_t_pad, _ = util.pad_by_sample(ct_t, len(live_time), srate, 1, 0, 'middle', 0.0)
    total_t = live_time + ct_t_pad + rf_t
    
    print(srate, ct_t.sum(), rf_t.sum(), rf_a.mean())
    
    srates = [10, 20, 30, 45, 60, 120]
    loads=np.zeros((len(srates), len(live_count)))
    accs=np.zeros((len(srates), len(live_count)))
    for i, srate in enumerate(srates):
        rf_segs, ct_t, ct_e = certify(live_count, certify_count, certify_time, 0.2, srate, 1)
        rf_c, rf_t, final_c, final_t = refine(live_count, live_time, certify_count, certify_time, rf_segs)
        rf_a = compute_accuray(final_c, gt, 1)
        ct_t_pad, _ = util.pad_by_sample(ct_t, len(live_time), srate, 1, 0, 'middle', 0.0)
        total_t = live_time + ct_t_pad + rf_t
        loads[i] = total_t
        accs[i] = rf_a
        print(srate, ct_t.sum(), rf_t.sum(), total_t.mean(), rf_a.mean())
    
    plt.figure()
    plt.plot(srates, loads.mean(1)*WLF)
    plt.xlabel('sampling rate (s)')
    plt.ylabel('resource per second (GFLOP)')
    plt.tight_layout()
    
    plt.figure()
    plt.plot(srates, accs.mean(1))
    plt.xlabel('sampling rate (s)')
    plt.ylabel('average accuracy')
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()

    # accuracy by other segment size
    la2 = compute_accuray(live_count, gt, 2)
    ra2 = compute_accuray(final_c, gt, 2)
    print(la2.mean(), ra2.mean())
    
    # workload comparison
    plt.figure()
    plt.plot(util.moving_average(live_time,5)*WLF)
    plt.plot(ct_t_pad*WLF)
    plt.plot(rf_t*WLF)
    plt.plot(total_t*WLF, '--')
    #plt.legend(['prf-free', 'certify', 'refine', 'total'])
    plt.legend(['live', 'certify', 'refine', 'total'])
    plt.xlabel('time (s)')
    plt.ylabel('resource (GFLOP)')
    plt.tight_layout()
    
    print('live=%.2f certify=%.2f refine=%.2f overhead=%.2f (%.2f%%), '
          'o_c=%.2f%% o_r=%.2f%%' %
          (live_time.sum(), ct_t.sum(), rf_t.sum(), ct_t.sum()+rf_t.sum(),
           (ct_t.sum()+rf_t.sum())/live_time.sum()*100,
           ct_t.sum()/live_time.sum()*100, rf_t.sum()/live_time.sum()*100
           ))
    
    # compare workload with profile-based
    import adaptation_expr
    # ap_t and ap_pt are from "adaptation_expr.py"
    ap_t,ap_a,ap_s,ap_pt=adaptation_expr.adapt_profile(ctss[vidx],cass[vidx],[0.9,0.8,0.7,0.5],srate,1)
    
    plt.figure()
    plt.plot((util.moving_average(ap_t,5)+ap_pt)*WLF) # profile
    plt.plot((util.moving_average(live_time,5)+ct_t_pad+rf_t)*WLF)
    plt.xlabel('time (s)')
    plt.ylabel('resource (GFLOP)')
    plt.legend(['prf-based','prf-free+C+R'])
    #plt.yscale('log')
    plt.ylim((None,380))
    plt.tight_layout()
    
    # accuracy comparison
    plt.figure()
    plt.plot(util.moving_average(ap_a, 10))
    plt.plot(util.moving_average(live_acc, 10))
    plt.plot(util.moving_average(rf_a, 10))
    plt.ylim((-0.05, 1.05))
    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    #plt.legend(['live only', 'live+refine'])
    #plt.legend(['w/o refine', 'w/ refine'])
    plt.legend(['prf-based','prf-free','prf-free+C+R'])
    plt.tight_layout()
    
    print('live=%.4f refine=%.4f gain=%.4f' % (live_acc.mean(), rf_a.mean(), rf_a.mean()-live_acc.mean()))
    
    