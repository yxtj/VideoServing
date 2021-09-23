# -*- coding: utf-8 -*-

from common import Configuration
import numpy as np
import torch

#Configuration = namedtuple('Configuration', ['fps', 'rsl'])

class AdaptationModel():
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
        conf = self.trans_config_index(conf_index)
        return conf
    
    # 2nd-level functions
    
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
    
    def trans_config_index(self, fps_index, rsl_index):
        return self.fps_list[fps_index], self.rsl_list[rsl_index]
    

