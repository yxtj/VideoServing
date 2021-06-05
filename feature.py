# -*- coding: utf-8 -*-

import numpy as np


def box_center(boxes):
    c = (boxes[:,:2]+boxes[:,2:])/2
    return c


def extract_speed(pboxes, tracker, fps:int, fr:int, segment:int, abs=True):
    # pboxes is a list of numpy.ndarray for bounding boxes
    n_frm = len(pboxes)
    n_sec = n_frm // fps
    n_seg = n_sec // segment
    #n = n_seg * segment
    speed_avg = np.zeros(n_seg)
    speed_med = np.zeros(n_seg)
    speed_std = np.zeros(n_seg)
    tracker.reset()
    buffer = {}
    for i in range(n_seg):
        idx_base = i*segment
        speeds = []
        for j in range(0, fps*segment, fr):
            centers = pboxes[idx_base + j]
            objs = tracker.update(centers)
            for oid, c in objs.items():
                if oid in buffer:
                    old = buffer[oid]
                    s = c - old
                    speeds.append(s)
                    buffer[oid] = c
                else:
                    buffer[oid] = c
        if len(speeds) != 0:
            if abs is True:
                speeds = np.abs(speeds)
            speed_avg[i] = np.mean(speeds)
            speed_med[i] = np.median(speeds)
            speed_std[i] = np.std(speeds)
    return speed_avg, speed_med, speed_std

def extract_count(gtruth:np.ndarray, segment:int):
    n_sec = len(gtruth)
    n_seg = n_sec // segment
    n = n_seg * segment
    res = gtruth[:n].reshape((n_seg, segment)).sum(1)
    return res
