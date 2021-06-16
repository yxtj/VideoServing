# -*- coding: utf-8 -*-

# %% torch version

import os, sys
import torch
from PIL import Image

from .detectorbase import DetectorBase

class YOLO_torch(DetectorBase):
    
    def __init__(self, name, conf_threshold=0.5, target_labels=None,
                 device='cpu'):
        # <target_labels>: COCO labels starting from 0
        assert name in ['yolov5s','yolov5m','yolov5l','yolov5x']
        path = os.environ['TORCH_HOME']+'/hub/ultralytics_yolov5_master'
        if not os.path.isdir(path):
            model = torch.hub.load('ultralytics/yolov5', name, pretrained=True)
        else:
            sys.path.append(path)
            # https://github.com/ultralytics/yolov5/blob/master/hubconf.py
            ckpt = torch.load(path+'/weights/'+name+'.pt')
            model = ckpt['model'].float()
            model.eval()
            model.names = ckpt['model'].names
            model = model.autoshape()
        p = next(model.parameters())
        if device != p.device:
            model.to(device)
        self.model = model
        self.names = model.names
        self.conf_threshold = conf_threshold
        self.target_labels = target_labels
    
    def _forward_(self, frame, psize):
        with torch.no_grad():
            results = self.model(frame, psize)
        return results
    
    def _filtering_(self, xyxycl, conf_threshold):
        confs = xyxycl[:,4]
        idx = confs > conf_threshold
        tmp = xyxycl[idx,:]
        
        if self.target_labels is not None:
            lbls = tmp[:,5].astype(int)
            idx2 = [l in self.target_labels for l in lbls]
            tmp = tmp[idx2]
        
        boxes = tmp[:, :4]
        confs = tmp[:, 4]
        lbls = tmp[:, 5].astype(int)
        return lbls, confs, boxes
    
    def process(self, frame, psize=640, conf_threshold=None):
        # <frame> is a np.ndarray of 3D: [H, W, C] where C=3
        # <psize> is an int of the maximum of H and W
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        img = Image.fromarray(frame)
            
        results = self._forward_(img, psize)
        #results.print()
        xyxycl = results.xyxyn[0].cpu().numpy()
        lbls, confs, boxes = self._filtering_(xyxycl, conf_threshold)
        return lbls, confs, boxes


# %% test

def __test__():
    import cv2
    
    v=cv2.VideoCapture('E:/Data/video/s3.mp4')
    _,f1=v.read()
    
    ytch = YOLO_torch('yolov5s', 0.3, (2,3,5,6,7))
    l1,c1,b1 = ytch.process(f1)
    