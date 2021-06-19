# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import torch

from carcounter2 import load_precompute_data_diff
from carcounter import load_precompute_data
from track.centroidtracker import CentroidTracker
from util import box_center

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
    cc.bsize_list = np.zeros(nfrm)
    cc.asize_list = np.zeros(nfrm)
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

# %% train

def train_epoch(loader, model, optimizer, loss_fn):
    lss=[]
    for i,(bx,by) in enumerate(loader):
        optimizer.zero_grad()
        o=model(bx)
        l=loss_fn(o,by)
        l.backward()
        optimizer.step()
        lss.append(l.item())
    return lss

def evaluate(x, y, model):
    model.eval()
    with torch.no_grad():
        p = model(x)
        ps = [t.argmax(1) for t in p]
    n = len(x)
    flag1 = ps[0] == y[:,0]
    flag2 = ps[1] == y[:,1]
    acc1 = sum(flag1) / n
    acc2 = sum(flag2) / n
    acc = sum(torch.logical_and(flag1,flag2)) / n
    ps = np.array([t.numpy() for t in ps], dtype=int).T
    return acc, (acc1, acc2), ps


# %% test

def __test__():
    from app.rangechecker import RangeChecker
    from videoholder import VideoHolder
    import carcounter2
    
    # generate data example
    video_folder = 'E:/Data/video'
    
    feat_gen = carcounter2.FeatureExtractor(2, 2, 1)
    
    vn_list = ['s3', 's4', 's5', 's7']
    rng_param_list = [('h', 0.5, 0.2, 0.1), ('h', 0.5, 0.2, 0.1),
                      ('v', 0.75, 0.2, 0.1), ('h', 0.45, 0.2, 0.1)]
    fr_list_each = []
    
    len_each = []
    tcrl, tcl, ttl, tal, tfl = [], [], [], [], []
    ### compute
    for vn, rngp in zip(vn_list, rng_param_list):
        v = VideoHolder(video_folder+'/'+vn+'.mp4')
        rng = RangeChecker(*rngp)
        cc = carcounter2.CarCounter(v,rng,None,None,2,0.8,None,feat_gen=feat_gen)
        fr_list, t_conf, t_time, t_acc, feat = prepare_training_date(
            cc, 'data/%s'%vn, '%s-raw-720.npz'%vn, '%s-diff-480-'%vn,
            'ground-truth-%s.txt'%vn, 30, 0.9)
        np.savez('data/%s/conf-diff.npz'%vn, fr_list=fr_list,
                 conf=t_conf, time=t_time, accuracy=t_acc, feature=feat)
        fr_list_each.append(fr_list)
        len_each.append(len(t_conf))
        tcrl.append(t_conf.copy())
        t_conf[:,0] = fr_list[t_conf[:,0]]
        tcl.append(t_conf)
        ttl.append(t_time)
        tal.append(t_acc)
        tfl.append(feat)
    tcrl = np.concatenate(tcrl)
    tcl = np.concatenate(tcl)
    ttl = np.concatenate(ttl)
    tal = np.concatenate(tal)
    tfl = np.concatenate(tfl)

    ### load 
    for vn in vn_list:
        with np.load('data/%s/conf-diff.npz'%vn, allow_pickle=True) as data:
            len_each.append(len(data['conf']))
            fr_list = data['fr_list']
            fr_list_each.append(fr_list)
            t_conf = data['conf']
            tcrl.append(t_conf.copy())
            t_conf[:,0] = fr_list[t_conf[:,0]]
            tcl.append(t_conf)
            ttl.append(data['time'])
            tal.append(data['accuracy'])
            tfl.append(data['feature'])
    tcrl = np.concatenate(tcrl)
    tcl = np.concatenate(tcl)
    ttl = np.concatenate(ttl)
    tal = np.concatenate(tal)
    tfl = np.concatenate(tfl)

    # train model with generated data
    import model.framedecision
    
    dsm = model.framedecision.DecisionModel(20, 30, (5,2))
    loss_fn = model.framedecision.DicisionLoss()
    x = torch.from_numpy(tfl).float()
    #y = torch.from_numpy(tcl).long()
    y = torch.from_numpy(tcrl).long()
    ds = torch.utils.data.TensorDataset(x, y)
    
    epoch = 1000
    lr = 0.001
    bs = 20

    loader = torch.utils.data.DataLoader(ds, bs)
    optimizer = torch.optim.Adam(dsm.parameters(), lr=lr)
    
    dsm.train()
    losses = []
    for i in range(epoch):
        lss = train_epoch(loader, dsm, optimizer, loss_fn)
        loss = sum(lss)
        if i % 100 == 0:
            print("epoch: %d, loss: %f" % (i, loss))
        losses.append(loss)
        
    acc_overall, acc_individual, ps = evaluate(x, y, dsm)
    print("overall accuracy: %f" % acc_overall)
    print("individual accuracy:", acc_individual)
    
    # save model
    torch.save(dsm, 'data/model-fr-pm-1.pth')
    # load model
    dsm = torch.load('data/model/fr-pm-1.pth')
    
    import framepreprocess
    import detect.yolowrapper
    
    dmodel=detect.yolowrapper.YOLO_torch('yolov5s',0.5,(2,3,5,6,7))
    fpp=framepreprocess.FramePreprocessor()
    
    off_each=np.cumsum(np.pad(len_each,(1,0)))
    p=dsm(x)
    ps = np.array([t.argmax(1).numpy() for t in p]).T
    
    eval_res_time = []
    eval_res_count = []
    eval_res_acc = []
    for i, (vn, rngp) in enumerate(zip(vn_list, rng_param_list)):
        print(i,vn)
        v = VideoHolder(video_folder+'/'+vn+'.mp4')
        rng = RangeChecker(*rngp)
        fpp.reset()
        cc = carcounter2.CarCounter(v,rng,dmodel,480,2,0.8,fpp)
        fr_list = fr_list_each[i]
        conf_list = ps[off_each[i]:off_each[i+1]].copy()
        conf_list[:,0] = fr_list[conf_list[:,0]]
        #conf_list = tcl[off_each[i]:off_each[i+1]]
        times,counts=cc.process_with_conf(conf_list)
        gtruth = np.loadtxt('data/%s/ground-truth-%s.txt'%(vn,vn),int,delimiter=',')
        n = len(counts)
        acc = cc.compute_accuray(counts, gtruth[:n], 1)
        eval_res_time.append(times)
        eval_res_count.append(counts)
        eval_res_acc.append(acc)
    
    np.savez('data/eval_res_predicted.npz', len_each=len_each,
             eval_res_time=np.concatenate(eval_res_time),
             eval_res_count=np.concatenate(eval_res_count),
             eval_res_acc=np.concatenate(eval_res_acc))

    
