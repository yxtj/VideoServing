# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:55:10 2021

@author: yanxi
"""

import torch

class GPUState():
    def __init__(self, gpu_id=0):
        self.available = torch.cuda.is_available()
        self.gid = gpu_id
        self.device = torch.cuda.device(gpu_id)
        self.properties = torch.cuda.get_device_properties(gpu_id)
    
    def name(self):
        return self.properties.name
    
    def processor_count(self):
        return self.properties.multi_processor_count
    
    def memory_total(self):
        return self.properties.total_memory
    
    def memory_allocated(self):
        return torch.cuda.memory_allocated(self.gid)
    
    def memory_cached(self):
        return torch.cuda.memory_reserved(self.gid) 
        
    def memory_used(self):
        r = self.memory_cached()
        a = self.memory_allocated()
        return r + a
        
    def memory_avaliable(self):
        t = self.memory_total()
        a = self.memory_allocated()
        return t - a  # free inside reserved
    
    def memory_clear(self):
        torch.cuda.empty_cache()
        