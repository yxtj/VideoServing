# -*- coding: utf-8 -*-

import numpy as np

class FrameProcessorModel():
    
    def __init__(self):
        pass
        
    def reset(self):
        pass
        
    def update(self, fidx, **kwargs):
        '''
        Returns whether to use the frame difference processor.
        '''
        return False

# %% confidence score based

# 3 types of factors. f(0)=1, f(max_value)=0
class __Exp_Type__():
    # y = 2 exp(-wx) - 1
    def __init__(self, max_value):
        self.w = np.log(2) / max_value
    def calc(self, value):
        return 2*np.exp(-self.w * value) - 1

class __Linear_Type__():
    # y = 1 - wx
    def __init__(self, max_value):
        self.w = 1 / max_value
    def calc(self, value):
        return 1 - self.w * value
    
class __Sqrt_Type__():
    # y = sqrt(1 - wx)
    def __init__(self, max_value):
        self.w = 1 / max_value
    def calc(self, value):
        return np.sqrt(1 - self.w*value)

        
class FrameProcessorModelConfidence(FrameProcessorModel):
    
    def __init__(self, max_time:int, 
                 max_box_area_chg:float=0.5, max_act_area_chg:float=0.5,
                 min_conf:float=0.3,
                 ema_decay=0.9):
        self.max_time = max_time
        self.max_box_area_chg = max_box_area_chg
        self.max_act_area_chg = max_act_area_chg
        self.min_conf = min_conf
        self.decay = ema_decay
        # weights
        self.f_time = __Sqrt_Type__(max_time)
        self.f_box_chg = __Sqrt_Type__(max_box_area_chg)
        self.f_act_chg = __Exp_Type__(max_act_area_chg)
        # progress
        self.last_whole_idx = 0
        self.last_box_area = 1
        self.last_act_area = 1

    def reset(self, fidx=0):
        self.last_whole_idx = fidx
        self.last_box_area = 1
        self.last_act_area = 1
        
    def update(self, fidx, box_area, active_area):
        '''
        returns whether to use frame difference
        '''
        ct = self.f_time.calc(fidx - self.last_whole_idx)
        b_chg = np.abs(box_area - self.last_box_area)/self.last_box_area
        cb = self.f_box_chg.calc(b_chg)
        a_chg = np.abs(active_area - self.last_act_area)/self.last_act_area
        ca = self.f_box_chg.calc(a_chg)
        c = (ct + cb + ca)/3
        self.last_box_area = self.decay*box_area + (1-self.decay)*self.last_box_area
        self.last_act_area = self.decay*active_area + (1-self.decay)*self.last_act_area
        if c >= self.min_conf:
            return False,c
        else:
            self.last_whole_idx = fidx
            return True,c
        