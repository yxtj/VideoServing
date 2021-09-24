# -*- coding: utf-8 -*-

from common import Configuration
import numpy as np
import torch
import torch.nn as nn

#Configuration = namedtuple('Configuration', ['fps', 'rsl'])

class Adapter():
    def __init__(self, dim_feat, fps_list, rsl_list, unit_rsl_time, **kwargs):
        self.dim_feat = dim_feat
        assert fps_list == sorted(fps_list)
        assert rsl_list == sorted(rsl_list)
        self.fps_list = np.array(fps_list)
        self.rsl_list = np.array(rsl_list)
        self.nrsl = len(rsl_list)
        self.nfps = len(fps_list)
        # selection
        self.unit_rsl_time = unit_rsl_time
        assert len(unit_rsl_time) == self.nrsl
        self.time_matrix = np.outer(fps_list, unit_rsl_time)
        # network
        self.network = None
    
    def load_model(self, model_file):
        self.network = torch.load(model_file)
    
    def save_model(self, model_file):
        assert self.model is not None
        torch.save(self.network, model_file)
    
    def load_checkpoint(self, checkpoint_file):
        assert self.model is not None
        self.network.load_state_dict(torch.load(checkpoint_file))
    
    def save_checkpoint(self, checkpoint_file):
        assert self.model is not None
        torch.save(self.network.state_dict(), checkpoint_file)
    
    def make_model(self):
        pass
    
    def get(self, feature, acc_bounds):
        with torch.no_grad():
            pred = self.predict(feature).numpy()
        conf_index = self.pick_config(pred, acc_bounds)
        fps = self.trans_fps_index(conf_index[0])
        rsl = self.trans_rsl_index(conf_index[1])
        return Configuration(fps, rsl)
    
    # internal functions
    
    def predict(self, feature):
        o = self.network(feature)
        return o
    
    def pick_config(self, predicted_acc, acc_bounds):
        '''
        Params:
            predicted_acc : matrix of predicted accuracies of all configurations.
            acc_bounds : list of accuracy requirements. Must be DESCENDING order
        Returns the index of fps and resolution
        '''
        for bound in acc_bounds:
            m = predicted_acc < bound # mask out those less than the bound
            if not m.all():
                # pick the fastest one
                ind = np.ma.MaskedArray(self.time_matrix, m).argmin()
                x, y = divmod(ind, self.time_matrix.shape[1])
                return x, y
        # no one satisfies any accuracy bound, return the best one
        ind = predicted_acc.argmax()
        x, y = divmod(ind, self.time_matrix.shape[1])
        return x, y
    
    def trans_fps_index(self, fps_index):
        return self.fps_list[fps_index]
    
    def trans_rsl_index(self, rsl_index):
        return self.rsl_list[rsl_index]
    
    def trans_config_index(self, fps_index, rsl_index):
        return self.fps_list[fps_index], self.rsl_list[rsl_index]

# %% model

class AdaptationModel(nn.Module):
    def __init__(self, dim_in, dim_outs, dim_range, dim_hidden=None):
        super().__init__()
        assert isinstance(dim_outs, (tuple, list))
        self.dim_in = dim_in
        self.dim_outs = dim_outs
        self.dim_range = dim_range
        self.dim_hidden = dim_hidden
        self.n_o = np.product(dim_outs)
        if dim_hidden is None or dim_hidden == 0:
            self.hidden = nn.Identity()
        else:
            self.hidden = nn.Sequential(
                nn.Linear(dim_in, dim_hidden), nn.ReLU(),
                )
        self.heads = nn.ModuleList(
            [torch.nn.Linear(dim_hidden, dim_range) for i in range(self.n_o)])
        
    def forward(self, x):
        n = len(x)
        shape = [n, *self.dim_outs, self.dim_range]
        h = self.hidden(x)
        h = torch.cat([self.heads[i](h) for i in range(self.n_o)])
        h = h.view(shape)
        #h = torch.sigmoid(h)
        return h

class AdaptationLoss():
    def __init__(self):
        self.lf = nn.CrossEntropyLoss()
        
    def __call__(self, output, target):
        shape = output.shape
        n = shape[0]
        nc = np.product(shape[1:-1])
        r = shape[-1]
        assert n == len(target)
        output = output.view(n, -1, r)
        l = torch.zeros(1)
        for i in range(nc):
            l += self.lf(output[:,i,:], target)
        return l

def acc2range(accuracies, thresholds):
    return np.digitize(accuracies, thresholds, right=True)

def selectConfByAcc(accuracies, threshold_acc):
    shape = accuracies.shape
    n = shape[-1]
    if len(shape) == 2:
        return divmod(accuracies.argmax(), n)
    return np.array([divmod(v.argmax(), n) for v in accuracies])
