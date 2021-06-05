# -*- coding: utf-8 -*-


class Operation():
    def __call__(self, data):
        return data


# %% basic operations for raw input processing

import torch
import numpy as np
import cv2

class OptTransOCV2Torch(Operation):
    def __init__(self, device=None):
        self.device = device
        
    def __call__(self, frames):
        assert isinstance(frames, (list, np.ndarray))
        if isinstance(frames, list) or \
            (isinstance(frames, np.ndarray) and frames.ndim == 4):
            return torch.tensor(frames, device=self.device).permute(0,3,1,2).float()/255
        else:
            return torch.tensor(frames, device=self.device).permute(2,0,1).float()/255


class OptTransTorch2OCV(Operation):
    def __call__(self, data):
        assert isinstance(data, (list, torch.Tensor))
        if isinstance(data, list):
            return [d.mul(255).to(torch.uint8).permute(1,2,0).cpu().numpy() for d in data]
        elif data.dim() == 4:
            return data.mul(255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
        else:
            return data.mul(255).to(torch.uint8).permute(1,2,0).cpu().numpy()

def transOCV2Torch(image):
    o = OptTransOCV2Torch()
    return o(image)

def transTorch2OCV(image):
    o = OptTransTorch2OCV()
    return o(image)



class OptResize(Operation):
    def __init__(self, dsize):
        self.dsize = dsize

    def __call__(self, frames):
        assert isinstance(frames, (list, np.ndarray))
        if isinstance(frames, list):
            return [cv2.resize(f, self.dsize) for f in frames]
        else:
            return cv2.resize(frames, self.dsize)


class OptModelCall(Operation):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def __call__(self, frames):
        with torch.no_grad():
            p = self.model(frames)
        return p

#%% input is prediction results

class OptFilter(Operation):
    def __init__(self, labels, min_score=0.8, fun_pred=None):
        if isinstance(labels, int):
            labels = [labels]
        self.labels = set(labels)
        self.min_score = min_score
        self.fun = fun_pred
    
    def __call__(self, predictions):
        assert isinstance(predictions, list)
        res = []
        for pred in predictions:
            lbls = pred['labels']
            scrs = pred['scores']
            boxs = pred['boxes']
            n = len(pred['labels'])
            idx = []
            for i in range(n):
               if lbls[i].item() in self.labels \
                   and scrs[i].item() >= self.min_score \
                   and (self.fun is None or self.fun(lbls[i], scrs[i], boxs[i])):
                   idx.append(i)
            tmp = {'labels': lbls[idx], 'scores': scrs[idx], 'boxes': boxs[idx]}
            res.append(tmp)
        return res
        

class OptCountFrame(Operation):
    def __call__(self, predictions):
        return len(predictions)


class OptCountObj(Operation):
    def __init__(self, labels, min_score=0.8):
        if isinstance(labels, int):
            labels = [labels]
        self.labels = set(labels)
        self.min_score = min_score
        
    def __call__(self, predictions):
        assert isinstance(predictions, list)
        res = 0
        for pred in predictions:
            lbls = pred['labels']
            scrs = pred['scores']
            n = len(lbls)
            for i in range(n):
                if lbls[i].item() in self.labels and scrs[i].item() >= self.min_score:
                   res += 1
        return res
    
    
class OptArea(Operation):
    def __init__(self, labels, min_score=0.8):
        if isinstance(labels, int):
            labels = [labels]
        self.labels = set(labels)
        self.min_score = min_score
    
    def __call__(self, predictions):
        assert isinstance(predictions, list)
        res = 0.0
        for pred in predictions:
            lbls = pred['labels']
            scrs = pred['scores']
            n = len(pred['labels'])
            for i in range(n):
               if lbls[i] in self.labels and scrs[i] >= self.min_score:
                   box = pred['boxes'][i]
                   res += (box[2]-box[0]) * (box[3]-box[1])
        return res
    
#%% input is processed data
    
class OptSquare(Operation):
    def __call__(self, number):
        return number * number
    

class OptFunc1(Operation):
    def __init__(self, func):
        self.func = func
        
    def __call__(self, data):
        return self.func(data)


class OptFunc2(Operation):
    def __init__(self, func_trans, func_agg, init_v = 0):
        self.func_trans = func_trans
        self.func_agg = func_agg
        self.init_v = init_v
        
    def __call__(self, data):
        res = self.init_v
        for d in data:
            dt = self.func_trans(d)
            res = self.func_agg(res, dt)
        return res
    
#%% tracking

class OptTrackSingle(Operation):
    def __init__(self, height, width, center=(0.5,0.5), padding=False):
        self.center = torch.tensor((center[0]*width, center[1]*height))
        self.padding = padding
        
    def __call__(self, predictions):
        res = []
        last = None
        for pred in predictions:
            s = self.select(pred)
            if s is not None:
                res.append((s, True))
            elif self.padding == True:
                res.append((last, False))
            else:
                res.append((None, False))
            last = s
        return res
    
    def select(self, prediction):
        # a filter operation before is required before
        #lbl = prediction['labels']
        #scr = prediction['scores']
        boxes = prediction['boxes']
        if len(boxes) == 1:
            return boxes[0]
        elif len(boxes) == 0:
            return None
        else:
            with torch.no_grad():
                dx = (boxes[:,0] + boxes[:,2])/2 - self.center[0]
                dy = (boxes[:,1] + boxes[:,3])/2 - self.center[1]
                ds = dx**2 + dy**2
            idx=ds.argmin()
            return boxes[idx]

