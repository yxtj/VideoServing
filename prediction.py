# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

import util
#from carcounter2 import load_precompute_data_diff
from carcounter import load_precompute_data
#from track.centroidtracker import CentroidTracker

# %% model 1: predict configuration (classification)

class ConfPred_classification(nn.Module):
    def __init__(self, dim_in, dim_outs):
        super().__init__()
        assert isinstance(dim_outs, (tuple, list))
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

# %% model 2: predict accuracy (regression)

class AccPred_regression(nn.Module):
    def __init__(self, dim_in, dim_outs, dim_hidden):
        super().__init__()
        assert isinstance(dim_outs, (tuple, list))
        self.dim_in = dim_in
        self.dim_outs = dim_outs
        n_o = np.product(dim_outs)
        self.hidden = torch.nn.Linear(dim_in, dim_hidden)
        self.head_acc = torch.nn.Linear(dim_hidden, n_o)
        for p in [self.hidden, self.head_time, self.head_acc]:
            nn.init.xavier_uniform_(p.weight)
        
    def forward(self, x):
        n = len(x)
        shape = [n, *self.dim_outs]
        h = torch.nn.functional.relu(self.hidden(x))
        oa = self.head_acc(h)
        return oa.reshape(shape)


class AccPred_regression_loss():
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
    
# %% model 3: predict accuracy range (classification)

class AccPred_rangeclass(nn.Module):
    def __init__(self, dim_in, dim_outs, dim_range, dim_hidden=None):
        super().__init__()
        assert isinstance(dim_outs, (tuple, list))
        self.dim_in = dim_in
        self.dim_outs = dim_outs
        self.dim_range = dim_range
        n_o = np.product(dim_outs)
        self.n_o = n_o
        if dim_hidden is None or dim_hidden == 0:
            self.hidden = nn.Identity()
        else:
            self.hidden = nn.Sequential(
                nn.Linear(dim_in, dim_hidden), nn.ReLU(),
                )
        self.heads = nn.ModuleList(
            [torch.nn.Linear(dim_hidden, dim_range) for i in range(n_o)])
        
    def forward(self, x):
        n = len(x)
        shape = [n, *self.dim_outs, self.dim_range]
        h = self.hidden(x)
        h = torch.cat([self.heads[i](h) for i in range(self.n_o)])
        h = h.view(shape)
        #h = torch.nn.functional.sigmoid(h)
        return h

class AccPred_rangeclass_loss():
    def __init__(self):
        self.lf = nn.CrossEntropyLoss()
        
    def __call__(self, output, target):
        shape = output.shape
        #n = shape[0]
        r = shape[-1]
        output = output.view(-1, r)
        l = torch.zeros(1)
        for o, t in zip(output, target):
            l += self.lf(o, t)
        return l

def acc2range(accuracies, thresholds):
    return np.digitize(accuracies, thresholds, right=True)

def selectConfByAcc(accuracies, threshold_acc):
    shape = accuracies.shape
    n = shape[-1]
    if len(shape) == 2:
        return divmod(accuracies.argmax(), n)
    return np.array([divmod(v.argmax(), n) for v in accuracies])

def selectConfByAccRange(accrng, threshold_idx):
    pass

# %% prepare data

def gen_feature_one(feat_gen, tracker, rngchecker, nsecond, fps,
                    rs_list, rs_idx, fr_list, fr_idx, box_list):
    feature = np.zeros((nsecond, feat_gen.dim_feat))
    fr = fr_list[fr_idx]
    for sidx in range(nsecond):
        for fidx in range(0, fr, fps):
            fidx += sidx*fps
            boxes = box_list[fidx]
            if len(boxes) > 0:
                centers = util.box_center(boxes)
                flag = rngchecker.in_track(centers)
                centers_in_range = centers[flag]
            else:
                centers_in_range = []
            objects = tracker.update(centers_in_range)
            feat_gen.update(objects, fr/fps, boxes)
        feat_gen.move((rs_idx, fr_idx))
        feature[sidx] = feat_gen.get()
    return feature

def gen_feature_time_accuracy(cc, box_file_list, rs_list, fr_list, ground_truth):
    assert cc.feat_gen is not None
    if isinstance(ground_truth, str):
        ground_truth = np.loadtxt(ground_truth, int, delimiter=',')
    fps = int(np.ceil(cc.video.fps))
    nfrm = cc.video.num_frame
    nsecond = nfrm//fps
    nrs = len(rs_list)
    nfr = len(fr_list)
    #assert conf_list.shape == (nsecond, 2)
    
    features = np.zeros((nrs, nfr, nsecond, cc.feat_gen.dim_feat))
    times = np.zeros((nrs, nfr, nsecond))
    accuracies = np.zeros((nrs, nfr, nsecond))
    for ridx, rs in enumerate(rs_list):
        data = load_precompute_data(box_file_list[ridx])
        time_list = data[3]
        box_list = np.array(data[4], dtype=object)
        cc.rs = rs
        cc.times_list = time_list
        cc.pboxes_list = box_list
        for fidx, fr in enumerate(fr_list):
            cc.fr = fr
            cc.reset()
            tms, cnts, _ = cc.process()
            accs = cc.compute_accuray(cnts, ground_truth)
            times[ridx, fidx] = tms
            accuracies[ridx, fidx] = accs
            cc.feat_gen.reset()
            cc.tracker.reset()
            f = gen_feature_one(cc.feat_gen, cc.tracker, cc.range,
                                nsecond, fps, rs_list, ridx, fr_list, fidx,
                                box_list)
            features[ridx, fidx] = f
    return features, times, accuracies

def match_feature_accuracy(features, accuracies, num_prev):
    nrs, nfr, n, ndim = features.shape
    assert accuracies.shape == (nrs, nfr, n)
    m = n - num_prev - 1 - 1 # head prune: num_prev + 1, tail prune: 1
    f = np.zeros((m*nrs*nfr, ndim))
    a = np.zeros((m*nrs*nfr, nrs, nfr))
    off = 0
    for ridx in range(nrs):
        for fidx in range(nfr):
            for i in range(off, n-1):
                f[off] = features[ridx, fidx, i, :]
                a[off] = accuracies[:,:,i+1]
                off += 1
    return f, a


def gen_feature_one_config(feat_gen, tracker, rngchecker, nsecond, fps,
                    rs_list, fr_list, boxes_list, conf_list):
    feature = np.zeros((nsecond, feat_gen.dim_feat))
    for sidx in range(nsecond):
        rs_idx, fr_idx = conf_list[sidx]
        fr = fr_list[fr_idx]
        for fidx in range(0, fr, fps):
            fidx += sidx*fps
            boxes = boxes_list[rs_idx][fidx]
            if len(boxes) > 0:
                centers = util.box_center(boxes)
                flag = rngchecker.in_track(centers)
                centers_in_range = centers[flag]
            else:
                centers_in_range = []
            objects = tracker.update(centers_in_range)
            feat_gen.update(objects, fr/fps, boxes)
        feat_gen.move((rs_idx, fr_idx))
        feature[sidx] = feat_gen.get()
    return feature


def gen_feature_with_random_config(cc, box_file_list, rs_list, fr_list, n=1):
    assert cc.feat_gen is not None
    fps = int(np.ceil(cc.video.fps))
    nfrm = cc.video.num_frame
    nsecond = nfrm//fps
    nrs = len(rs_list)
    nfr = len(fr_list)
    #assert conf_list.shape == (nsecond, 2)
    
    cc.rs_list = rs_list
    cc.fr_list = fr_list
    times_list = []
    pboxes_list = []
    for bf in box_file_list:
        data = load_precompute_data(bf)
        time_list = data[3]
        box_list = np.array(data[4], dtype=object)
        times_list.append(time_list)
        pboxes_list.append(box_list)
    cc.times_list = times_list
    cc.pboxes_list = pboxes_list
    
    conf_list = np.zeros((nsecond*n, 2), dtype=int)
    conf_list[:,0] = np.random.randint(0, len(rs_list), nsecond*n)
    conf_list[:,1] = np.random.randint(0, len(fr_list), nsecond*n)
    
    features = []
    for i in range(n):
        cc.reset()
        f = gen_feature_one_config(cc.feat_gen, cc.tracker, cc.range,
                                   nsecond, fps, rs_list, fr_list,
                                   pboxes_list, conf_list)
        features.append(f)
    features = np.concatenate(features)
    return features


# %% training and testing functions

def train_epoch(loader,model,optimizer,loss_fn):
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
    p = p.view((len(p), -1)).argmax(1)
    # TODO: finish this
    

# %% test

def __test__():
    from app.rangechecker import RangeChecker
    from videoholder import VideoHolder
    import carcounter2
    
    # generate data example
    video_folder = 'E:/Data/video'
    
    vn_list = ['s3', 's4', 's5', 's7']
    rng_param_list = [('h', 0.5, 0.2, 0.1), ('h', 0.5, 0.2, 0.1),
                      ('v', 0.75, 0.2, 0.1), ('h', 0.45, 0.2, 0.1)]
    vfps_list = [25,30,20,30]
    
    rs_list = [240,360,480,720]
    fps_list = [30, 10, 5, 2, 1]
    
    num_prev = 2
    
    len_each = []
    features = []
    times = []
    accuracies = []
    for vidx, vn in enumerate(vn_list):
        fr_list = np.ceil(vfps_list[vidx]/np.array(fps_list)).astype(int)
        print(vn, fr_list)
        v = VideoHolder(video_folder+'/%s.mp4'%vn)
        rng = RangeChecker(*rng_param_list[vidx])
        feat_gen = carcounter2.FeatureExtractor(2, num_prev)
        cc = carcounter2.CarCounter(v,rng,None,240,2,0.8,None,feat_gen=feat_gen)
        box_file_list = ['data/%s/%s-raw-%d.npz'%(vn,vn,rs) for rs in rs_list]
        f, t, a = gen_feature_time_accuracy(cc, box_file_list, rs_list, fr_list, 'data/%s/ground-truth-%s.txt'%(vn,vn))
        len_each.append(a.shape[2])
        features.append(f)
        times.append(f)
        accuracies.append(a)
    features = np.concatenate(features, 2)
    times = np.concatenate(times, 2)
    accuracies = np.concatenate(accuracies, 2)
    
    np.savez('data/feature-2',features=features, times=times, accuracies=accuracies,
             rs_list=rs_list,fps_list=fps_list)
    
    
    data = np.load('data/feature-2.npz', allow_pickle=True)
    features = data['features']
    times = data['times']
    accuracies = data['accuracies']
    del data
    
    # train
    acc_range = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    nele = features.shape[2]
    
    off_each=np.cumsum(np.pad(len_each,(1,0)))
    
    #x = features.transpose((2,0,1,3)).reshape((nele, -1, 42))
    #y = acc2range(accuracies, acc_range).transpose((2,0,1).reshape((-1, 42)))
    x, y = match_feature_accuracy(features, accuracies, num_prev)
    y = acc2range(y, acc_range)
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    ds = torch.utils.data.TensorDataset(x, y)
    
    model = AccPred_rangeclass(42, (4,5), 4, 42)
    loss_fn = AccPred_rangeclass_loss()
    
    epoch = 1000
    lr = 0.001
    bs = 20

    loader = torch.utils.data.DataLoader(ds, bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    losses = []
    for i in range(epoch):
        lss = train_epoch(loader, model, optimizer, loss_fn)
        loss = sum(lss)
        if i % 100 == 0:
            print("epoch: %d, loss: %f" % (i, loss))
        losses.append(loss)
        
    acc_overall, acc_individual, ps = evaluate(x, y, model)
    print("overall accuracy: %f" % acc_overall)
    print("individual accuracy:", acc_individual)

    