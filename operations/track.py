# -*- coding: utf-8 -*-

from .operation import Operation
import cv2
import torch


OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create, # accurate
	"kcf": cv2.TrackerKCF_create, # balanced
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create # fast
}


class OptTrackSingleOCV(Operation):
    def __init__(self, bbox, model, tracker='kcf'):
        self.init_bbox = bbox
        #initBB = cv2.selectROI("Frame", frames[0], fromCenter=False)
        self.model = model
        self.tracker = OPENCV_OBJECT_TRACKERS[tracker]()
        
        
    def __call__(self, frames):
        assert isinstance(frames, list)
        self.tracker.init(frames[0], self.init_bbox)
        res = []
        for f in frames[0:]:
            success, box = self.tracker.update(f)
            res.append(box)
        return res
    

class OptTrackMultiOCV(Operation):
    def __init__(self, labels, model, min_score=0.8, tracker='kcf'):
        if isinstance(labels, int):
            labels = [labels]
        self.labels = set
        self.model = model
        self.min_score = min_score(labels)
        self.tracker = tracker
    
    def __call__(self, frames):
        assert isinstance(frames, list)
        pred = self.model([frames[0]])
        n = len(pred[0]['labels'])
        idx = []
        trackers = cv2.MultiTracker_create()
        res = []
        for i in range(n):
            if pred[0]['labels'][i].item() not in self.labels:
                continue
            if pred[0]['scores'][i].item() < self.min_score:
                continue
            idx.append(i)
            tck = OPENCV_OBJECT_TRACKERS[self.tracker]()
            trackers.add(tck, frames[0], pred[0]['boxes'][i])
        for f in frames[0:]:
            success, boxes = trackers.update(f)
            res.append(boxes)
        return res


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
            

#%% speed

class OptSpeed(Operation):
    REF_SIZE = {
        1: (0.3, 0.5, 1.7), 2: (0.8, 1.4), 3: (2.0, 4.5),
        4: (0.8, 1.5), 5: (100, 100), 6: (2.2, 10.0),
        7: (2.2, 100), 8: (2.5, 5.0),
    }
    def __init__(self, bbox, model, tracker='kcf'):
        self.init_bbox = bbox
        self.model = model
        
    def __call__(self, trackings):
        pass