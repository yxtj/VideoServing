# -*- coding: utf-8 -*-

import numpy as np
import os
import re

from carcounter2 import load_precompute_data_diff
from carcounter import load_precompute_data
from track.centroidtracker import CentroidTracker
from util import box_center
from carcounter2 import FeatureExtractor

# %% prepare training data - load and align raw data

def pick_precomputed_diff_files(folder, prefix, max_jump, sort=True):
    dfiles = list(filter(lambda fn: fn.startswith(prefix) and
                         fn.endswith('.npz'), os.listdir(folder)))
    ptn = prefix+'-?(\d+)\.npz'
    dmap = { int(re.match(ptn, fn)[1]): fn for fn in dfiles }
    keys = filter(lambda k: k<=max_jump, dmap)
    if sort:
        keys = sorted(keys)
    dfiles = [ dmap[k] for k in keys ]
    return dfiles

def organize_diff_data(folder, file_list, nfrm, max_jump):
    nfiles = len(file_list)
    fr_list = np.zeros(nfiles, int)
    dtime_list = np.zeros((nfrm, nfiles))
    dbox_list = np.array([[None for _ in range(nfiles)] for _ in range(nfrm)],
                         dtype=object)
    bsize_list = np.zeros((nfrm, nfiles))
    asize_list = np.zeros((nfrm, nfiles))
    for i, fn in enumerate(file_list):
        data = load_precompute_data_diff(folder+'/'+fn)
        fr = data[4]
        if fr > max_jump:
            continue
        tl = data[5]
        bl = data[6]
        rl = data[7]
        rl = (rl[:,2]-rl[:,0])*(rl[:,3]-rl[:,1])/ 720 / 1280
        ml = data[8]
        fr_list[i] = fr
        rng = range(0, nfrm, fr)
        dtime_list[rng, i] = tl
        dbox_list[rng, i] = bl
        bsize_list[rng, i] = rl
        asize_list[rng, i] = ml
    return fr_list, dtime_list, dbox_list, bsize_list, asize_list

def get_aligned_point(idx, wtime_list, wbox_list,
                      fr_list, dtime_list, dbox_list, max_jump):
    nfrm = len(wtime_list)
    ndfile = len(fr_list)
    res_jump = []
    res_time = np.zeros((ndfile,2))
    res_box = np.array([(None,None) for i in range(ndfile)], dtype=object)
    for p, (fr, dtime, dbox) in enumerate(zip(fr_list, dtime_list, dbox_list)):
        end = min(idx + max_jump, nfrm)
        for i in range(idx+int(np.ceil(fr/2)), end):
            if dbox[i] is not None:
                next_idx = i
                res_jump.append(next_idx - idx)
                res_time[p] = ( wtime_list[next_idx], dtime[next_idx] )
                res_box[p] = ( wbox_list[next_idx], dbox[next_idx] )
                break
    return res_time, res_box

def align_precomputed_data(n, ndfile, fr_list,
                           wtime_list, wbox_list, dtime_list, dbox_list):
    assert dtime_list.shape[1] == ndfile
    res_time = np.zeros((n, ndfile, 2))
    res_box = np.array([[(None,None) for _ in range(ndfile)] for _ in range(n)], dtype=object)
    for i, fr in enumerate(fr_list):
        rng = np.arange(0,n,fr)
        res_time[rng,i,0] = wtime_list[rng]
        res_box[rng,i,0] = wbox_list[rng]
    res_time[:,:,1] = dtime_list[:n]
    res_box[:,:,1] = dbox_list[:n]
    return res_time, res_box
    #res_time = np.zeros((n, ndfile, 2))
    #res_box = np.array([[(None,None) for _ in range(ndfile)] for _ in range(n)], dtype=object)
    #for idx in range(n):
    #    atl, abl = get_aligned_point(idx, wtime_list, wbox_list,
    #                                 fr_list, dtime_list, dbox_list, max_jump)
    #    res_time[idx] = atl
    #    res_box[idx] = abl
    #return fr_list, res_time, res_box

# %% prepare training data - configuration

def __compute_accuracy__(target, predict):
    up = max(target, predict)
    down = min(target, predict)
    return down/up if up != 0 else 1.0
    
# return configuration index (in fr_list), time and accuracy
def select_configuration(cc, acc_bound, time_list, box_list,
                         fr_list, ground_truth):
    assert time_list.shape == box_list.shape
    nfrm, nfr, nmdl = time_list.shape
    assert nmdl == 2
    assert len(fr_list) == nfr
    if isinstance(ground_truth, str):
        ground_truth = np.loadtxt(ground_truth, int, delimiter=',')
    assert isinstance(ground_truth, np.ndarray)
    fps = int(np.ceil(cc.video.fps))
    nsecond = nfrm//fps 
    assert nsecond <= len(ground_truth)
    
    res_conf = np.zeros((nsecond, 2), dtype=int) # frame jump, method(0,1)
    res_time = np.zeros(nsecond)
    res_acc = np.zeros(nsecond)
    
    fidx_mat = np.zeros((nfr, nmdl), int)
    cc.reset()
    state = cc.get_track_state()
    for sidx in range(nsecond):
        #print('sidx',sidx)
        fidx_end = fps * (sidx+1)
        acc_buff = np.zeros((nfr, nmdl))
        time_buff = np.zeros((nfr, nmdl))
        state_buff = [[None for j in range(nmdl)] for i in range(nfr)]
        cc.set_track_state(state)
        for i, fr in enumerate(fr_list):
            cc.change_fr(fr)
            for j in range(nmdl):
                #print(i,fr,j,":",fidx_mat[i,j])
                pboxes = box_list[:,i,j]
                times = time_list[:,i,j]
                cc.set_track_state(state)
                cc.pboxes_list = pboxes
                cc.times_list = times
                cnt, t = cc.process_period(fidx_mat[i,j], fidx_end, None, fr)
                acc_buff[i,j] = __compute_accuracy__(ground_truth[sidx], cnt)
                time_buff[i,j] = t
                state_buff[i][j] = cc.get_track_state()
                fidx_new = fidx_mat[i,j] + fr * ((fidx_end - fidx_mat[i,j] + fr - 1) // fr)
                fidx_mat[i,j] = fidx_new
        # pick the best (fastest one with accuracy >= acc_bound)
        mask = acc_buff >= acc_bound
        if not np.any(mask):
            # find the most accurate one(s)
            v = acc_buff.max()
            mask = acc_buff[:,:] >= v
        xs, ys = mask.nonzero()
        t = [time_buff[x,y] for x, y in zip(xs, ys)]
        ind = np.argmin(t)
        x, y = xs[ind], ys[ind]
        res_conf[sidx] = (x,y)
        res_time[sidx] = time_buff[x,y]
        res_acc[sidx] = acc_buff[x,y]
        state = state_buff[x][y]
    return res_conf, res_time, res_acc
    
# %% prepare training data - feature

def get_feature(cc, fr_list, time_list, box_list, bsize_list, asize_list,
                conf_list, ground_truth):
    assert cc.feat_gen is not None
    assert time_list.shape == box_list.shape
    nfrm, nfr, nmdl = time_list.shape
    assert nmdl == 2
    assert bsize_list.shape == asize_list.shape == (nfrm, nfr)
    fps = int(np.ceil(cc.video.fps))
    nsecond = nfrm//fps 
    assert len(fr_list) == nfr
    assert conf_list.shape == (nsecond, 2)
    
    #feat_gen = FeatureExtractor(2, 2, 1)
    feat_gen = cc.feat_gen
    feat_gen.reset()
    #tracker = CentroidTracker(int(fps*0.5))
    tracker = cc.tracker
    tracker.reset()
    rngchecker = cc.range
    
    feature = np.zeros((nsecond, cc.feat_gen.dim_feat))
    for sidx in range(nsecond):
        fi, mi = conf_list[sidx]
        fr = fr_list[fi]
        dis_n = max(1, int(cc.dsap_time*fps/fr))
        tracker.maxDisappeared = dis_n
        fidx_start = fr * ((sidx*fps + fr - 1)//fr)
        fidx_end = fr * ((sidx*fps + fps + fr - 1)//fr)
        
        for fidx in range(fidx_start, fidx_end, fr):
            boxes = box_list[fidx, fi, mi]
            if len(boxes) > 0:
                centers = box_center(boxes)
                flag = rngchecker.in_track(centers)
                centers_in_range = centers[flag]
            else:
                centers_in_range = []
            objects = tracker.update(centers_in_range)
            if mi == 0:
                bsize, asize = 1, 1
            else:
                bsize, asize = bsize_list[fidx, fi], asize_list[fidx, fi]
            feat_gen.update(objects, bsize, asize)
        feat_gen.move(ground_truth[sidx], (fr, mi))
        feature[sidx] = cc.feat_gen.get()
    return feature

# %% prepare training data

def prepare_training_date(cc, folder,
                          whole_proc_file, diff_proc_prefix, gtruth_file,
                          max_jump, acc_bound):
    assert os.path.isfile(folder+'/'+whole_proc_file)
    dfiles = pick_precomputed_diff_files(folder, diff_proc_prefix, max_jump)
    #print(dfiles)
    ndfile = len(dfiles)
    assert len(dfiles) != 0
    # part 1: load data
    wdata = load_precompute_data(folder+'/'+whole_proc_file)
    wtime_list = wdata[3]
    wbox_list = np.array(wdata[4], dtype=object)
    nfrm = len(wbox_list)
    fr_list, dtime_list, dbox_list, bsize_list, asize_list = \
        organize_diff_data(folder, dfiles, nfrm, max_jump)
    # part 2: align difference data with frame
    n = nfrm-max_jump
    time_list, box_list = align_precomputed_data(
        n, ndfile, fr_list, wtime_list, wbox_list, dtime_list, dbox_list)
    bsize_list = bsize_list[:n]
    asize_list = asize_list[:n]
    # part 3: generate configuration
    ground_truth = np.loadtxt(folder+'/'+gtruth_file, int, delimiter=',')
    t_conf, t_time, t_acc = select_configuration(
        cc, acc_bound, time_list, box_list, fr_list, ground_truth)
    # part 4: generate feature
    feat = get_feature(cc, fr_list, time_list, box_list,
                       bsize_list, asize_list, t_conf, ground_truth)
    return fr_list, t_conf, t_time, t_acc, feat

# %% test

def __test__():
    from app.rangechecker import RangeChecker
    from videoholder import VideoHolder
    import carcounter2
    
    # generate data example
    video_folder = 'E:/Data/video'
    v3=VideoHolder('E:/Data/video/s3.mp4')
    rng3=RangeChecker('h', 0.5, 0.2, 0.1)
    cc=carcounter2.CarCounter(v3,rng3,None,None,2,0.8,None)
    ground_truth = np.loadtxt('data/s3/ground-truth-s3.txt', int, delimiter=',')
    
    fr_list, time_list, box_list, bsize_list, asize_list = align_precomputed_data(
        'data/s3/', 's3-raw-720.npz', 's3-diff-480-', 30)
    
    t_conf, t_time, t_acc = select_configuration(
        cc, 0.9, time_list, box_list, fr_list, ground_truth)
    np.savez('data/s3/conf-diff',fr_list=fr_list,
             conf=t_conf,time=t_time,acc=t_acc)
    
    # train with generated data
    vn_list = ['s3', 's4', 's5', 's7']
    rng_param_list = [('h', 0.5, 0.2, 0.1), ('h', 0.5, 0.2, 0.1),
                      ('v', 0.75, 0.2, 0.1), ('h', 0.45, 0.2, 0.1)]
    len_each = []
    tcl, ttl, tal, tfl = [], [], [], []
    for vn, rngp in zip(vn_list, rng_param_list):
        v = VideoHolder(video_folder+'/'+vn+'.mp4')
        rng = RangeChecker(*rngp)
        cc = carcounter2.CarCounter(v,rng,None,None,2,0.8,None)
        fr_list, t_conf, t_time, t_acc, feat = prepare_training_date(
            cc, 'data/%s'%vn, '%s-raw-720.npz'%vn, '%s-diff-480-'%vn,
            'ground-truth-%s'%vn, 30)
        np.savez('data/s3/conf-diff', fr_list=fr_list,
                 conf=t_conf, time=t_time, accuracy=t_acc, feature=feat)
        len_each.append(len(ttl))
        t_conf[:,0] = fr_list[t_conf[:,0]]
        tcl.append(t_conf)
        ttl.append(t_time)
        tal.append(t_acc)
        tfl.append(feat)
    tcl = np.concatenate(tcl)
    ttl = np.concatenate(ttl)
    tal = np.concatenate(tal)
    tfl = np.concatenate(tfl)

    from model.framedecision import DecisionModel
    
    dsm = DecisionModel(15, 30)
    
    
