# -*- coding: utf-8 -*-

# %% opencv version

import cv2
import numpy as np

from .detectorbase import DetectorBase

class YOLO_cv(DetectorBase):
    
    def __init__(self, name, conf_threshold=0.3, target_labels=None,
                 cfg_file=None, weight_file=None, root_dir=None):
        self.name = name
        assert name in ['yolov3','yolov3-tiny','yolov3-ssp',
                        'yolov2','yolov2-tiny','yolov2-voc']
        self.conf_threshold = conf_threshold
        self.target_labels = target_labels
        if cfg_file is not None:
            self.cfg_file = cfg_file
        elif root_dir is not None:
            self.cfg_file = root_dir + '/cfg/' + name + '.cfg'
        else:
            raise ValueError('cfg file is not given')
        if weight_file is not None:
            self.weight_file = weight_file
        elif root_dir is not None:
            self.weight_file = root_dir + '/weights/' + name + '.weights'
        else:
            raise ValueError('weights file is not given')
        print(self.cfg_file, self.weight_file)
        self.model = cv2.dnn.readNetFromDarknet(self.cfg_file, self.weight_file)
        ln = self.model.getLayerNames()
        oln = [ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        self.olayer = oln
        
    def _forward_(self, frame, tsize):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (tsize, tsize),
                                     swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_outputs = self.model.forward(self.olayer)
        return layer_outputs
        
    def _pick_boxes_(self, layer_outputs, conf_threshold, W=None, H=None):
        bboxes = []
        confidences = []
        classIDs = []
        
        # loop over each of the layer outputs
        for output in layer_outputs:
            scores = output[:,5:]
            cids = scores.argmax(1)
            confs = scores.max(1)
            
            idx = confs > conf_threshold
            if sum(idx) == 0:
                continue
            p = output[idx,:4]
            cx = p[:,0]
            cy = p[:,1]
            w = p[:,2]
            h = p[:,3]
            bs = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2]).T
            
            bboxes.append(bs)
            confidences.append(confs[idx])
            classIDs.append(cids[idx])
        
        bboxes = np.vstack(bboxes)
        confidences = np.hstack(confidences)
        classIDs = np.hstack(classIDs).astype('int')
        
        if W is not None and H is not None:
            # cvbox: UL.x, UL.y, w, h
            cvbox = bboxes.copy()
            cvbox[:,2:] -= cvbox[:,:2]
            cvbox *= np.array([W,H,W,H])
            cvbox = cvbox.astype('int')
            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(cvbox, confidences, self.conf_threshold,
                                    self.conf_threshold)
            bboxes = bboxes[idxs,:]
            confidences = confidences[idxs]
            classIDs = classIDs[idxs]
            
        return classIDs, confidences, bboxes
        
    def process(self, frame, size=416, conf_threshold=None):
        # <frame> is a np.ndarry of 3D: [H, W, C] where C=3
        # <size> is a scalar of the processing resolution
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
            
        layer_outputs = self._forward_(frame, size)
        
        classIDs, confidences, bboxes = self._pick_boxes_(
            layer_outputs, conf_threshold)
        
        if self.target_labels is not None:
            idxs = [x in self.target_labels for x in classIDs]
            if sum(idxs) != len(classIDs):
                bboxes = bboxes[idxs]
                confidences = confidences[idxs]
                classIDs = classIDs[idxs]
        
        return classIDs, confidences, bboxes

# %% test

def __test__():
    import cv2
    
    v=cv2.VideoCapture('E:/Data/video/s3.mp4')
    _,f1=v.read()
    
    ycv = YOLO_cv('yolov3-tiny', 0.3, (2,3,5,6,7), 
                  root_dir='D:/pytorch_data/darknet/')
    l2,c2,b2 = ycv.process(f1)
    