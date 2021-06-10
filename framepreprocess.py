# -*- coding: utf-8 -*-

# reference
#https://github.com/mmheydari97/obj-Tracking/blob/master/frame_diff.py
#https://github.com/mmheydari97/obj-Tracking/blob/master/backgnd_subtract.py


import numpy as np
import cv2

__all__ = ['FramePreprocessor']

class FramePreprocessor():
    def __init__(self, methodBG=True, methodDiff=True, mergeMethod='max',
                 removeBG=True, regionGrid=True,
                 bg_history=100, diff_num=2,
                 grid_line_num=10, grid_threshold=5,
                 flt_kernel_size=5, flt_min_size=0.002, flt_max_size=0.2,
                 flt_aratio_rng=(1/3, 3)
                 ):
        assert methodBG or methodDiff
        assert isinstance(flt_aratio_rng, tuple) or \
            (isinstance(flt_aratio_rng, (float, int)) and flt_aratio_rng > 0)
        assert 0.0 < flt_min_size <= flt_max_size < 1.0
        assert mergeMethod in ['max', 'min', 'and', 'or']
        self.methodBG = methodBG
        self.methodDiff = methodDiff
        if mergeMethod == 'max':
            self.method_agg_fun = cv2.max
        elif mergeMethod == 'min':
            self.method_agg_fun = cv2.min
        elif mergeMethod == 'and':
            self.method_agg_fun = cv2.bitwise_and
        elif mergeMethod == 'or':
            self.method_agg_fun = cv2.bitwise_or
        if methodBG:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.bg_lr = 1.0/bg_history
        if methodDiff:
            self.diff_num = diff_num
            self.frame_history = [None for _ in range(diff_num)]
            self.frame_init = False
        # active region (grid)
        self.regionGrid = regionGrid
        self.grid_num = grid_line_num
        self.grid_threshold = grid_threshold
        # output
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
        
    def reset(self):
        self.bg_subtractor.clear()
        if self.methodDiff:
            self.frame_history = [None for _ in range(self.diff_num)]
            self.frame_init = False
    
    def option_remove_background(self, on_off: bool):
        self.removeBG = on_off
        
    # main API
    
    def apply(self, frame):
        # return an active region from the input frame using given methods
        mask = self.get_basic_mask(frame)
        # process
        mask = self.remove_mask_noise(mask)
        x1, y1, x2, y2, mask = self.get_active_region(mask)
        if self.removeBG:
            f = self.remove_back_ground(frame, mask)
        else:
            f = frame
        return (x1, y1, x2, y2), f[y1:y2, x1:x2], mask
    
    # 2rd-level API
    
    def get_basic_mask(self, frame):
        if self.methodBG:
            mask1 = self._get_mask_bg_(frame)
        if self.methodDiff:
            self._update_frame_history_(frame)
            mask2 = self._get_mask_diff_(frame)
        # merge mask
        if self.methodBG and self.methodDiff:
            mask = self.method_agg_fun(mask1, mask2)
        elif self.methodBG:
            mask = mask1
        else:
            mask = mask2
        _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        return mask
        
    def remove_mask_noise(self, mask):
        #_, pmask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        #pmask = cv2.dilate(pmask, None, iterations=1)
        pm1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.flt_kernel)
        pm2 = cv2.morphologyEx(pm1, cv2.MORPH_CLOSE, self.flt_kernel)
        pm3 = cv2.dilate(pm2, self.flt_kernel, 5)
        pm4 = cv2.morphologyEx(pm3, cv2.MORPH_CLOSE, self.flt_kernel)
        return pm4
    
    def get_active_region(self, mask):
        H, W = mask.shape
        gh = int(np.ceil(H/self.grid_num))
        gw = int(np.ceil(W/self.grid_num))
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_mask = np.zeros_like(mask)
        x1, y1, x2, y2 = W, H, 0, 0
        heat_map = np.zeros((self.grid_num, self.grid_num), dtype=np.int32)
        for contour in contours:
            #a = cv2.contourArea(contour) / W / H
            (x, y, w, h) = cv2.boundingRect(contour)
            if w/h < self.flt_aratio_rng[0] or w/h > self.flt_aratio_rng[1]:
                continue
            a = w*h / W / H
            if a < self.flt_min_size:
                # for small contours
                heat_map[y//gh, x//gw] += 1
                continue
            elif a > self.flt_max_size:
                # for big contours
                continue
            # for middle contours
            #print(x, y, w, h)
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x+w)
            y2 = max(y2, y+h)
            new_mask[y:y+h, x:x+w] = 255
            #new_mask = cv2.fillPoly(new_mask, [contour], 255)
        for i,j in zip(range(y1//gh, y2//gh), range(x1//gw, x2//gw)):
            if  heat_map[i,j] >= self.grid_threshold:
                new_mask[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = 255
        return x1, y1, x2, y2, new_mask

    def remove_back_ground(self, frame, gray_mask):
        color_mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        f = color_mask & frame
        return f
    
    # mask - background-difference
    
    def _get_mask_bg_(self, frame):
        mask = self.bg_subtractor.apply(frame, self.bg_lr)
        #_, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        return mask
    
    def _update_frame_history_(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.frame_init == True:
            self.frame_history[:self.diff_num-1] = self.frame_history[1:self.diff_num]
            self.frame_history[self.diff_num-1] = gray
        else:
            self.frame_init = True
            self.frame_history = [gray for _ in range(self.diff_num)]
    
    # mask - consecutive-frame-difference
    
    def _get_mask_diff_(self, frame):
        mask = FramePreprocessor.frame_diff(self.frame_history, self.method_agg_fun)
        return mask
    
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

def __test__(video_name, step=1, n_second=None, show=True, showCut=False,
             save_video_name=None):
    import time
    
    fpp = FramePreprocessor()
    
    cap = cv2.VideoCapture(video_name)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ft = 1000/fps*step
    
    n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) if n_second is None else int(fps*n_second)
    
    idx = 0
    area_bounding_box_list = []
    area_active_box_list = []
    time_list = []
    
    curr_progress = 0
    success, frame = cap.read()
    if save_video_name is not None:
        out = cv2.VideoWriter(
            save_video_name, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (W, H))
    else:
        out = None
    while success and idx <= n_frame:
        t0 = time.time()
        rect,f,mask = fpp.apply(frame)
        w, h = rect[2]-rect[0], rect[3]-rect[1]
        ba = w*h/W/H
        aa = mask.mean()/255
        t = time.time() - t0
        area_bounding_box_list.append(ba)
        area_active_box_list.append(aa)
        time_list.append(t)
        if show or out:
            frame = cv2.rectangle(frame,rect[0:2],rect[2:],(0,255,0),2)
            g = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, None)
            redimg = np.zeros_like(frame, frame.dtype)
            redimg[:,:,2] = 255
            redmask = cv2.bitwise_and(redimg, redimg, mask=g)
            frame=cv2.addWeighted(redmask, 1, frame, 1, 0)
            cv2.putText(frame, 'b-area=%.2f%%,a-area=%.2f%%,FPS=%.1f'%(ba,aa,1/t),
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0))
            if out:
                out.write(frame)
            if show:
                print(rect, w, h, '%.4f'%ba, '%.4f'%aa, '%.4f'%t)
                cv2.imshow('Image', frame)
            if showCut and f.size != 0:
                cv2.imshow('Cut', f)
            if show or showCut:
                t = time.time() - t0
                if cv2.waitKey(max(0, int(ft-t))) & 0xff == ord('q'):
                    break
        else:
            c = idx // int(fps)
            if c % 30 == 0 and c != curr_progress:
                curr_progress = c
                print(idx, c, '%.4f'%ba, '%.4f'%aa, '%.4f'%t, '%.2f'%sum(time_list))
        idx += step
        if step != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
    return area_bounding_box_list, area_active_box_list, time_list
