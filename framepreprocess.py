# -*- coding: utf-8 -*-

# reference
#https://github.com/mmheydari97/obj-Tracking/blob/master/frame_diff.py
#https://github.com/mmheydari97/obj-Tracking/blob/master/backgnd_subtract.py


import numpy as np
import cv2

__all__ = ['FramePreprocessor']

class FramePreprocessor():
    def __init__(self, modeBG=True, modeDiff=True, mergeMethod='max',
                 bg_history=100, diff_num=2,
                 removeBG = True,
                 flt_kernel_size=5, flt_min_size=0.002, flt_max_size=0.2,
                 flt_aratio_rng=(1/3, 3)):
        assert modeBG or modeDiff
        assert isinstance(flt_aratio_rng, int) and flt_aratio_rng > 0
        assert 0.0 < flt_min_size <= flt_max_size < 1.0
        assert flt_aratio_rng > 0
        assert mergeMethod in ['max', 'min', 'and', 'or']
        self.modeBG = modeBG
        self.modeDiff = modeDiff
        if mergeMethod == 'max':
            self.mode_agg_fun = cv2.max
        elif mergeMethod == 'min':
            self.mode_agg_fun = cv2.min
        elif mergeMethod == 'and':
            self.mode_agg_fun = cv2.bitwise_and
        elif mergeMethod == 'or':
            self.mode_agg_fun = cv2.bitwise_or
        if modeBG:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.bg_lr = 1.0/bg_history
        if modeDiff:
            self.diff_num = diff_num
            self.frame_history = [None for _ in range(diff_num)]
            self.frame_init = False
        self.removeBG = removeBG
        # filtering
        self.flt_kernel = np.ones((flt_kernel_size, flt_kernel_size), np.uint8)
        self.flt_min_size = flt_min_size
        self.flt_max_size = flt_max_size
        if isinstance(flt_aratio_rng, (int, float)):
            if flt_aratio_rng > 1:
                flt_aratio_rng = 1.0 / flt_aratio_rng
            flt_aratio_rng = (flt_aratio_rng, 1.0 / flt_aratio_rng)
        self.flt_aratio_rng = flt_aratio_rng
    
    def option_remove_background(self, on_off: bool):
        self.removeBG = on_off
    
    def get_mask_bg(self, frame):
        mask = self.bg_subtractor.apply(frame, self.bg_lr)
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        return mask
    
    def update_frame_history(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.frame_init == True:
            self.frame_history[:self.diff_num-1] = self.frame_history[1:self.diff_num]
            self.frame_history[self.diff_num-1] = gray
        else:
            self.frame_init = True
            self.frame_history = [gray for _ in range(self.diff_num)]
        
    def get_mask_diff(self, frame):
        mask = FramePreprocessor.frame_diff(self.frame_history, self.mode_agg_fun)
        return mask
    
    def update(self, frame):
        # return an active region from the input frame using given methods
        if self.modeBG:
            mask1 = self.get_mask_bg(frame)
        if self.modeDiff:
            self.update_frame_history(frame)
            mask2 = self.get_mask_diff(frame)
        # merge mask
        if self.modeBG and self.modeDiff:
            mask = self.mode_agg_fun(mask1, mask2)
        elif self.modeBG:
            mask = mask1
        else:
            mask = mask2
        # process
        _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        mask = self.remove_mask_noise(mask)
        x1, y1, x2, y2, mask = self.get_active_region(mask)
        if self.removeBG:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            f = mask & frame
        else:
            f = frame
        return (x1, y1, x2, y2), f[y1:y2, x1:x2]
    
    def remove_mask_noise(self, mask):
        #_, pmask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        #pmask = cv2.dilate(pmask, None, iterations=1)
        pm1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.flt_kernel)
        pm2 = cv2.morphologyEx(pm1, cv2.MORPH_CLOSE, self.flt_kernel)
        pm3 = cv2.dilate(pm2, self.flt_kernel, 3)
        pm4 = cv2.morphologyEx(pm3, cv2.MORPH_CLOSE, self.flt_kernel)
        return pm4
    
    def get_active_region(self, mask):
        H, W = mask.shape
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_mask = np.zeros_like(mask)
        x1, y1, x2, y2 = W, H, 0, 0
        for contour in contours:
            #a = cv2.contourArea(contour) / W / H
            (x, y, w, h) = cv2.boundingRect(contour)
            a = w*h / W / H
            if a < self.flt_min_size or a > self.flt_max_size:
                continue
            if w/h < self.flt_aratio_rng[0] or w/h > self.flt_aratio_rng[1]:
                continue
            #print(x, y, w, h)
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x+w)
            y2 = max(y2, y+h)
            new_mask[y:y+h, x:x+w] = 255
            #new_mask = cv2.fillPoly(new_mask, [contour], 255)
        return x1, y1, x2, y2, new_mask

    @staticmethod
    def frame_diff(gray_frame_list, merge_fun):
        diff = cv2.absdiff(gray_frame_list[0], gray_frame_list[1])
        #_, diff = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        for i in range(2,len(gray_frame_list)):
            temp = cv2.absdiff(gray_frame_list[i-1], gray_frame_list[i])
            #_, temp = cv2.threshold(temp, 40, 255, cv2.THRESH_BINARY)
            diff = merge_fun(diff, temp)
        return diff
    
# %% test

def __test__(video_name, skip=1):
    fpp = FramePreprocessor()
    
    cap = cv2.VideoCapture(video_name)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    idx = 0
    success, frame = cap.read()
    while success:
        rect,f = fpp.update(frame)
        w, h = rect[2]-rect[0], rect[3]-rect[1]
        print(rect, w, h, '%.4f'%(w*h/W/H), '%.4f'%(f.mean()/255))
        frame=cv2.rectangle(frame,rect[0:2],rect[2:],(0,255,0),2)
        cv2.imshow('Input', frame)
        cv2.imshow('Cut', f)
        if cv2.waitKey(20) & 0xff == ord('q'):
            break
        idx += skip
        if skip != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
