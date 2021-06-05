# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

class VideoHolder():
    def __init__(self, path, transform=None):
        assert os.path.isfile(path)
        self.path = path
        self.transform = transform
        self.cap = cv2.VideoCapture(path)
        # some constants
        self.num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.num_frame / self.fps
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap=None
        
    def __del__(self):
        self.close()

    def second2frame(self, second):
        return int(self.fps*second)

    def length_second(self, complete=False):
        if complete:
            return int(self.num_frame // self.fps)
        else:
            return int(np.ceil(self.duration))
        
    def get_frame(self, idx, raw=False):
        assert 0 <= idx < self.num_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = self.cap.read()
        if success:
            b = not raw and self.transform is not None
            return self.transform(frame) if b else frame
        else:
            return None
        
    def get_frames(self, idx_s, idx_e, raw=False):
        assert 0 <= idx_s < idx_e < self.num_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx_s)
        b = not raw and self.transform is not None
        res = [ None for i in range(idx_e - idx_s) ]
        for i in range(idx_e - idx_s):
            success, frame = self.cap.read()
            res[i] = self.transform(frame) if b else frame
        return res
    
    def get_frame_next(self, raw=False):
        success, frame = self.cap.read()
        if success:
            b = not raw and self.transform is not None
            return self.transform(frame) if b else frame
        else:
            return None
    