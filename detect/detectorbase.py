# -*- coding: utf-8 -*-

class DetectorBase():
    
    def process(self, frame, psize=None, conf_threshold=None):
        # <frame> is a np.ndarray of 3D: [H, W, C] where C=3
        # <psize> is a int of height.
        # return: labels, scores, boxes (labels and scores are None)
        return [], [], []
    