# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 02:19:49 2021

@author: yanxi
"""

from common import Configuration

#Configuration = namedtuple('Configuration', ['rsl', 'fps', 'roi', 'model'])

class AdaptationModel():
    def __init__(self, rs_list, fr_list, mdl_list,
                 num_prev, decay, **kwargs):
        self.rs_list = rs_list
        self.fr_list = fr_list
        self.mdl_list = mdl_list
        
        self.feat = None
        
    def update(self, speed):
        pass
    
    def move(self, last_rs_idx, last_fr_idx, last_mdl_idx):
        pass
    
    def get(self):
        return Configuration(360, 10, False, 'yolov5m')
    