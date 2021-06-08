# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2
import time

import visualization
import videoholder

from detect.yolowrapper import yolowrapper
import carcount


# %% running

def run_offline_profile(file, chk_dir, chk_pos, mdl_name, segment, sel_list):
    #file = 'E:/Data/Video/s3.mp4'
    #chk_dir = 'h'
    #chk_pos = 0.5
    #mdl_name = 'yolov5s'
    
    v1 = videoholder.VideoHolder(file)
    rng = carcount.CheckRange(chk_dir, chk_pos, 0.2, 0.1)
    model = yolowrapper.YOLO_torch(mdl_name, 0.5, (2,3,5,7))
    conf = carcount.Configuation(1, 480, model)
    
    fps = int(np.ceil(v1.fps))
    
    #FR_FOR_20=[1,2,5,10,20,40] # fps: 20, 10, 4, 1, 0.5
    #FR_FOR_25=[1,2,5,12,25,50] # fps: 25, 12, 5, 1, 0.5
    #FR_FOR_30=[1,2,5,15,30,60] # fps: 30, 15, 6, 1, 0.5
    
    FR_LIST = [1,2,5] + [int(fps/r) for r in [2, 1, 0.5]]
    RS_LIST = [240,360,480,720]
    
    cc = carcount.CarCounter(v1, rng, conf)
    n = int(cc.video.length_second()/segment)
    #assert n == len(sel_list)
    
    idx = 0
    last = 0
    counts = np.zeros(n, int)
    times = np.zeros(n)
    for fr, rs in sel_list:
        cc.change_fr(fr)
        cc.change_rs(rs)
        t = time.time()
        c = 0
        while idx // fps // segment == last:
            c += cc.update(idx)
            idx += fr
        counts[last] = c
        times[last] = time.time() - t
        last = idx // fps // segment
    return counts, times


# %% make features

import profiling
from centroidtracker import CentroidTracker
import feature
import groundtruth

def prepare_raw_training_data(video_name_list, fps_list, segment=1,
                              acc_bound=0.9, ft_reso=480, ft_fr=2):
    # input / features
    fsas = [] # feature speed average
    fsms = [] # feature speed median
    fsss = [] # feature speed standard derivation
    fcs = [] # feature count
    for fn, fps in zip(video_name_list, fps_list):
        pboxes=carcount.load_raw_data('data/%s-raw-%d.npz' % (fn, ft_reso))[4]
        tracker = CentroidTracker(fps/2)
        fa,fm,fs=feature.extract_speed(pboxes,tracker,fps,ft_fr,segment)
        fsas.append(fa)
        fsms.append(fm)
        fsss.append(fs)
        
        gt=groundtruth.load_ground_truth('data/ground-truth-%s.txt' % fn)
        fcs.append(gt)
    
    fsas = np.concatenate(fsas)
    fsms = np.concatenate(fsms)
    fsss = np.concatenate(fsss)
    fcs = np.concatenate(fcs)
    
    features = np.array([fsas, fsms, fsss, fcs]).T
    
    # output / configuration
    ctss = [] # time of each configuration
    cass = [] # accuracy of each configuration
    pss = [] # profile selections
    for fn in video_name_list:
        _,_,sg_list,cts,cas=profiling.load_configurations('data/conf-%s.npz' % fn)
        sg_idx = sg_list.tolist().index(segment)
        _,_,ps=profiling.get_profile_bound_acc(cts[sg_idx],cas[sg_idx],acc_bound)
        ctss.append(cts[sg_idx])
        cass.append(cas[sg_idx])
        pss.append(ps)
    
    ctss = np.concatenate(ctss, 2).transpose((2,0,1))
    cass = np.concatenate(cass, 2).transpose((2,0,1))
    pss = np.concatenate(pss)
    
    return features, ctss, cass, pss


def normalize_features(features):
    m = features.mean(0)
    s = features.std(0)
    f = (features - m)/s
    return f

from typing import Union

def generate_training_data(features:np.ndarray, outputs:Union[list, np.ndarray],
                           num_prev:int, norm=True,
                           decay:Union[float,int,list,np.ndarray]=1.0):
    if not isinstance(outputs, list):
        outputs = [outputs]
    n, m = features.shape
    no = n - num_prev - 1
    # decay factor
    if isinstance(decay, (list, np.ndarray)):
        assert len(decay) == m
        factors = decay
    else:
        factors = [decay**(num_prev-i) for i in range(num_prev+1)]
    if norm:
        features = normalize_features(features)
    # add previous segments to feature
    f = np.zeros((no, m*(1+num_prev)))
    for i in range(num_prev+1):
        idx_s = i*m
        idx_e = (i+1)*m
        f[:,idx_s:idx_e] = features[i:no+i,:] * factors[i]
    # align outputs
    os = []
    for out in outputs:
        os.append(out[num_prev+1:])
    return f, *os
    
# %% prediction common

import torch.nn as nn

def test_selection(output, target):
    n1 = output.shape[0]
    n2 = target.shape[0]
    assert n1 == n2
    cmp = output == target
    correct_col = cmp.sum(0)
    correct_all = (cmp.sum(1)==2).sum()
    return correct_all/n1, correct_col/n1

def performance_of_selection(cass, ctss, selections):
    # now it assumes there are only two knots i.e.: len(prediction) == 2
    n1 = cass.shape[2]
    n2 , m= selections.shape
    off = n1-n2
    times = np.zeros(n2)
    accuracies = np.zeros(n2)
    for i in range(n2):
        s1, s2 = selections[i]
        times[i] = ctss[off+i, s1, s2]
        accuracies[i] = cass[off+i, s1, s2]
    return times, accuracies
    

# %% predition with classfication

class ConfPred_classification(nn.Module):
    def __init__(self, dim_in, dim_outs):
        super().__init__()
        assert isinstance(dim_outs, (list,tuple))
        self.dim_in = dim_in
        self.dim_outs = dim_outs
        self.fcs = nn.ModuleList(
            [nn.Linear(dim_in, dim_o) for dim_o in dim_outs])
        self.act = nn.Sigmoid()
        for p in self.fcs:
            nn.init.xavier_uniform_(p.weight)

    def forward(self, x):
        hs = [fc(x) for fc in self.fcs]
        ys = [self.act(h) for h in hs]
        return ys

class ConfPred_classification_loss:
    def __init__(self):
        self.lf = nn.CrossEntropyLoss()
        
    def __call__(self, output, target):
        dim = len(output)
        l = torch.zeros(1)
        for i in range(dim):
            l += self.lf(output[i], target[:,i])
        return l

def train_epoch_classification(loader,model,optimizer,loss_fn):
    lss=[]
    for i,(bx,by) in enumerate(loader):
        optimizer.zero_grad()
        o=model(bx)
        l=loss_fn(o,by)
        l.backward()
        optimizer.step()
        lss.append(l.item())
    return lss

def test_classification(x, y, model):
    model.eval()
    with torch.no_grad():
        ps = model(x)
        ps = [p.argmax(1) for p in ps]
        os = torch.stack(ps).T
    return test_selection(os, y)

def performance_of_prediction(cass, ctss, predictions):
    # now it assumes there are only two knots i.e.: len(prediction) == 2
    os = torch.stack([p.argmax(1) for p in predictions]).T
    return performance_of_selection(cass, ctss, os)
    

# %% prediction with regression

class ConfPred_regression(nn.Module):
    def __init__(self, dim_in, dim_outs, dim_hidden):
        super().__init__()
        assert isinstance(dim_outs, list)
        self.dim_in = dim_in
        self.dim_outs = dim_outs
        n_o = np.product(dim_outs)
        self.hidden = torch.nn.Linear(dim_in, dim_hidden)
        self.head_time = torch.nn.Linear(dim_hidden, n_o)
        self.head_acc = torch.nn.Linear(dim_hidden, n_o)
        for p in [self.hidden, self.head_time, self.head_acc]:
            nn.init.xavier_uniform_(p.weight)
        
    def forward(self, x):
        n = len(x)
        shape = [n, *self.dim_outs]
        h = torch.nn.functional.relu(self.hidden(x))
        ot = self.head_time(h)
        oa = self.head_acc(h)
        return ot.reshape(shape), oa.reshape(shape)


class ConfPred_regression_loss():
    def __call__(self, output_t, output_a, target_t, target_a):
        f = torch.nn.functional.mse_loss
        return f(output_t, target_t) + f(output_a, target_a)

    
def train_epoch_regression(loader,model,optimizer,loss_fn):
    lss=[]
    for i,(bx,bt,ba) in enumerate(loader):
        optimizer.zero_grad()
        ot, oa = model(bx)
        l = loss_fn(ot, oa, bt, ba)
        l.backward()
        optimizer.step()
        lss.append(l.item())
    return lss

def performance_of_regression(cass, ctss, predictions):
    os = torch.stack([p.argmax(1) for p in predictions]).T
    n1 = cass.shape[2]
    n2 , m= os.shape
    off = n1-n2
    times = np.zeros(n2)
    accuracies = np.zeros(n2)
    for i in range(n2):
        s1, s2 = os[i]
        times[i] = ctss[s1, s2, off+i]
        accuracies[i] = cass[s1, s2, off+i]
    return times, accuracies
    

# %% test

def __test_online__():
    #box_files = ['data/s3-raw-%d.npz'%r for r in RS_LIST]
    #ct,ca=prifiling.generate_conf(cc, v1, box_files, 'data/ground-truth-s3.txt',
    #                              segment, FR_LIST, RS_LIST)
    #pt,pa,ps=prifiling.get_profile_bound_acc(ct,ca,0.9)
    
    #sel_list = np.array([np.random.randint(0, len(FR_LIST), n),
    #                     np.random.randint(0, len(RS_LIST), n)])
    pass

def __test_train__():
    vn_list = ['s3','s4','s5','s7']
    fps_list = [25,30,20,30]
    # feature, time, accuracy, selection
    features, ctss, cass, pss = prepare_raw_training_data(vn_list, fps_list,
                                                          1, 0.9, 480, 2)
    aug_feat = np.hstack([features, pss])
    
    n, m = aug_feat.shape
    _, n_rs, n_fs = ctss.shape
    
    num_prev = 2
    decay = 0.9
    f,p,ct,ca = generate_training_data(aug_feat, [pss, ctss, cass], num_prev, decay)
    n, m = f.shape
    
    epoch = 1000
    lr = 0.05
    bs = 20
    
    # classification method
    cmodel = ConfPred_classification(m, [n_rs, n_fs])
    c_loss_fn = ConfPred_classification_loss()
    #cmodel.load_state_dict(torch.load('data/cmodel-2p'))
    
    x = torch.tensor(f, dtype=torch.float)
    yc = torch.tensor(p, dtype=torch.long)
    ds_c = torch.utils.data.TensorDataset(x, yc)
    loader_c = torch.utils.data.DataLoader(ds_c, bs)
    
    #optimizer_c = torch.optim.SGD(cmodel.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(cmodel.parameters(), lr=lr)
    
    losses = []
    for i in range(epoch):
        lss = train_epoch_classification(loader_c, cmodel, optimizer_c, c_loss_fn)
        loss = sum(lss)
        if i % 100 == 0:
            print("epoch: %d, loss: %f" % (i, loss))
        losses.append(loss)
        
    acc_overall, acc_individual= test_classification(x,yc,cmodel)
    print("overall accuracy: %f" % acc_overall)
    print("individual accuracy:", acc_individual)
    
    ##########################
    # regression method
    rmodel = ConfPred_regression(m, [n_rs, n_fs], 50)
    r_loss_fn = ConfPred_regression_loss()
    
    yrt = torch.tensor(ct, dtype=torch.float)
    yra = torch.tensor(ca, dtype=torch.float)
    ds_r = torch.utils.data.TensorDataset(x, yrt, yra)
    loader_r = torch.utils.data.DataLoader(ds_r, bs)
    
    optimizer_r = torch.optim.Adam(rmodel.parameters(), lr=lr)
    losses = []
    for i in range(epoch):
        lss = train_epoch_regression(loader_r, rmodel, optimizer_r, r_loss_fn)
        loss = sum(lss)
        if i % 100 == 0:
            print("epoch: %d, loss: %f" % (i, loss))
        losses.append(loss)
   
# %% online with prediction

from carcountprediction import OnlineCarCounterPrediction

# video specific
def __one_preprocessed_video(vfld, vname, model, rs_list, fr_list, 
                           cmodel, num_prev, decay, feat_mean, feat_std):
    pboxes_list=[]
    times_list=[]
    for rs in rs_list:
        data=carcount.load_raw_data('data/%s-raw-%d.npz' % (vname, rs))
        rng_param = data[0]
        #model_param = data[1]
        times_list.append(data[3])
        pboxes_list.append(data[4])
    v=videoholder.VideoHolder(vfld+'/'+vname+'.mp4')
    rng = carcount.RangeChecker(*rng_param)
    conf=carcount.Configuation(2,480,model)
    cc=carcount.CarCounter(v,rng,conf,0.5)
    occp=OnlineCarCounterPrediction(cc,cmodel,num_prev,decay,rs_list,fr_list,
                        feat_mean,feat_std,(3,3),pboxes_list, times_list)
    
    cntlist = np.zeros(occp.cc.video.length_second(True))
    timelist = np.zeros(occp.cc.video.length_second(True))
    conflist = np.zeros((occp.cc.video.length_second(True),2), int)
    for i in range(v.length_second(True)):
        cnt,tm,conf=occp.next_second()
        cntlist[i] = cnt
        timelist[i] = tm
        conflist[i] = conf
       
    return cntlist, timelist, conflist

def __evaluate_conf_video(vname, cntlist, conflist, duration_list):
    sg_idx=0
    acc_bound=0.9
    # prediction accuracy
    _,_,sg_list,cts,cas=profiling.load_configurations('data/conf-%s.npz'%vname)
    _,_,ps=profiling.get_profile_bound_acc(cts[sg_idx],cas[sg_idx],acc_bound)
    pao, pas = test_selection(conflist, ps)
    # analytic accuracy
    gt=groundtruth.load_ground_truth('data/ground-truth-%s.txt'%vname)
    aal = np.zeros(len(duration_list))
    for i,d in enumerate(duration_list):
        a = carcount.CarCounter.compute_accuray(cntlist, gt, d)
        aal[i] = a.mean()
    return pao, pas, aal


def __test_online_prediction__():
    num_prev=2
    #feat_mean = aug_feat.mean(0)
    #feat_std = aug_feat.std(0)
    feat_mean = np.array([3.6511e-05,-0.00010775,0.01389,0.55556,0.4793,3.5599])
    feat_std = np.array([0.0012985,0.001723,0.017293,0.8035,0.87641,1.0217])
    
    decay=0.9
    
    cmodel=ConfPred_classification(18,(4,6))
    cmodel.load_state_dict(torch.load('data/cmodel-2p'))
    
    model=yolowrapper.YOLO_torch('yolov5s',0.5,(2,3,5,7))
    
    vn_list = ['s3','s4','s5','s7']
    fps_list = [25,30,20,30]
    
    rs_list=[240,360,480,720]
    fr_list=[1,2,5,15,30,60]
    
    duration_list = [1,5,10]
    
    # example of s4
    cntlist, timelist, conflist = __one_preprocessed_video('E:/Data/video',
        's4', model, rs_list, fr_list, cmodel, num_prev, decay, feat_mean, feat_std)
    pao, pas, aal = __evaluate_conf_video('s4', conflist, duration_list)
    print('prediction accuracy overall: %f' % pao)
    print("individual accuracy separated:", pas)
    print("analytic accuracy:", aal)
    
    # on all videos
    n_slot = np.zeros(len(vn_list), int)
    cnt_list_idv = []
    time_list_idv = []
    conf_list_idv = []
    pred_acc_overall = np.zeros(len(vn_list))
    pred_acc_idv = np.zeros((len(vn_list),2))
    anal_acc = np.zeros((len(vn_list),len(duration_list)))
    for i,(vn,fps) in enumerate(zip(vn_list, fps_list)):
        if fps == 20:
            fr_list = profiling.FR_FOR_20
        elif fps == 25:
            fr_list = profiling.FR_FOR_25
        else:
            fr_list = profiling.FR_FOR_30
        cl,tl,fl = __one_preprocessed_video('E:/Data/video',
            vn, model, rs_list, fr_list, cmodel, num_prev, decay, feat_mean, feat_std)
        pao, pas, aal = __evaluate_conf_video(vn, cl, fl, duration_list)
        cnt_list_idv.append(cl)
        time_list_idv.append(tl)
        conf_list_idv.append(fl)
        
        n = len(cl)
        n_slot[i] = n
        pred_acc_overall[i] = pao
        pred_acc_idv[i] = pas
        anal_acc[i] = aal
        
    cnt_list = np.concatenate(cnt_list_idv)
    time_list = np.concatenate(time_list_idv)
    conf_list = np.concatenate(conf_list_idv, dtype=int)
    
    n_total = n_slot.sum()
    pao_m = (pred_acc_overall*n_slot).sum()/n_total
    pas_m = np.apply_along_axis(lambda x:(x*n_slot).sum()/n_total, 0, pred_acc_idv)
    aa_m = np.apply_along_axis(lambda x:(x*n_slot).sum()/n_total, 0, anal_acc)
    
    print('prediction accuracy overall: %f' % pao_m)
    print("prediction accuracy separated:", pas_m)
    print("analytic accuracy:", aa_m)
    
    np.savez('data/online-result',n_slot=n_slot,
             cnt_list_idv=np.array(cnt_list_idv, 'object'),
             time_list_idv=np.array(time_list_idv, 'object'),
             conf_list_idv=np.array(conf_list_idv, 'object'),
             pred_acc_overall=pred_acc_overall,pred_acc_idv=pred_acc_idv,
             anal_acc=anal_acc)

def load_online_result(filename):
    with np.load(filename, allow_pickle=True) as data:
        n_slot=data['n_slot']
        cnt_list_idv = data['cnt_list_idv'].tolist()
        time_list_idv = data['time_list_idv'].tolist()
        conf_list_idv = data['conf_list_idv'].tolist()
        pred_acc_overall = data['pred_acc_overall']
        pred_acc_idv = data['pred_acc_idv']
        anal_acc = data['anal_acc']
        return n_slot,cnt_list_idv,time_list_idv,conf_list_idv,pred_acc_overall,pred_acc_idv,anal_acc

# %% analyze for batch size

import matplotlib.pyplot as plt

def configuration_distribution(n_slot, conf_list_idv):
    vn_list = ['s3','s4','s5','s7']
    fps_list = [25,30,20,30]
    
    rs_list=[240,360,480,720]
    fr_list=[1,2,5,15,30,60]
    
    n_rs = 4
    n_fr = 6
    n_video = len(n_slot)

    n_total = n_slot.sum()
    n_seg = 400
    
    conf_list_idv_pad = [profiling.pad_data_list([cl], 400) for cl in conf_list_idv]
    
    frame_count = np.zeros((n_seg, n_rs), dtype=int)
    for vidx, fps in zip(range(n_video), fps_list):
        if fps == 20:
            fr_list = profiling.FR_FOR_20
        elif fps == 25:
            fr_list = profiling.FR_FOR_25
        else:
            fr_list = profiling.FR_FOR_30
        for i in range(n_seg):
            rs_idx, fr_idx = conf_list_idv_pad[vidx][i]
            fr = fr_list[fr_idx]
            n_frm = fps // fr
            frame_count[i,rs_idx] += n_frm
    
    plt.figure()
    for i in range(4):
        plt.subplot2grid((4,1),(i,0))
        plt.plot(frame_count[:,i])
        plt.legend([rs_list[i]],loc='upper right')
        plt.ylabel('# frames')
    plt.tight_layout()


