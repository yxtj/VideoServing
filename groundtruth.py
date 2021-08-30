# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

from videoholder import VideoHolder

from app.rangechecker import RangeChecker
from util import box_center

# %% ground truth functions


def __trans_boxes__(boxes, W, H):
    if len(boxes) == 0:
        return np.empty(0, int)
    elif isinstance(boxes, list):
        boxes = np.array(boxes)
    b = boxes.max() <= 1
    if b:
        boxes = boxes * np.array([W, H, W, H])
    return boxes.astype(int)


def show_result(image, rng:RangeChecker, boxes, 
                time=None, info=None, win_name='image'):
    H, W = image.shape[:2]
    ncenters = box_center(boxes)
    boxes = __trans_boxes__(boxes, W, H)
    centers = box_center(boxes).astype(int)
    
    img = image.copy()
    # check line
    if rng.dir == 'h':
        ckp=int(H*rng.pos)
        cv2.line(img, (0,ckp), (W,ckp), (0,255,0), 1)
    else:
        ckp=int(W*rng.pos)
        cv2.line(img, (ckp,0), (ckp,H), (0,255,0), 1)
    # boxes
    for i in range(len(boxes)):
        box = boxes[i]
        center = centers[i]
        if rng.offset(ncenters[i]) < 0:
            color = (0,0,255)
        else:
            color = (255,0,0)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.circle(img, (center[0], center[1]), 2, color, 2)
    # time
    if time is not None:
        cv2.putText(img, 'time: %d'%time, (30, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    # info
    if info is not None and len(info) > 0:
        cv2.putText(img, info, (30, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.imshow(win_name, img)


def help_gen_ground_truth(video:VideoHolder, boxes, rng:RangeChecker, 
                          start_s=0, end_s=None, show_fps=None):
    fps = int(np.ceil(video.fps))
    n_group = video.num_frame // fps
    wait_time = max(1, int(1000/fps)) # make sure it is not 0
    if show_fps is None:
        show_fps = fps
    pattern = np.linspace(0,fps,show_fps,False,dtype=int)
    end_s = n_group if end_s is None else min(end_s, n_group)
    
    i = start_s
    res = np.zeros(end_s - start_s, int)
    while i < end_s:
        if i > start_s:
            info = 'last: %d, current: %d' % (res[i-start_s-1], res[i-start_s])
        else:
            info = ''
        idx = i * fps
        for j in pattern:
            t = time.time()
            f = video.get_frame(idx+j)
            show_result(f, rng, boxes[idx+j], i, info)
            t = int(1000 * (time.time() - t))
            cv2.waitKey(max(1, wait_time - t))
        # wait until any key pressed
        key = cv2.waitKey(0) & 0xff
        if key == ord('p'): # previous second
            i = max(0, i-1)
        elif key == ord('r'): # replay currend second
            pass
        elif key == ord('q'): # quit
            break
        elif ord('1') <= key <= ord('9'): # input count (1-9)
            n = key - ord('0')
            res[i-start_s] = n
            i += 1
        elif key == ord('x'): # input count (>=10)
            buff = []
            key = cv2.waitKey(0) & 0xff
            while key != ord('\n'):
                buff.append(key)
                key = cv2.waitKey(0) & 0xff
            res[i-start_s] = int(''.join(buff))
            i += 1
        elif key == ord('n'): # keep current count
            i += 1
        else: # set 0
            res[i-start_s] = 0
            i += 1
    return res


def save_ground_truth(file, gt):
    np.savetxt(file, gt, '%d', ',')


def load_ground_truth(file):
    gt = np.loadtxt(file, int, delimiter=',')
    return gt
    