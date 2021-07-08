# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 03:09:50 2021

@author: yanxi
"""

import numpy as np

def compIoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of each rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
    
def compCenter(box):
    x1, y1, x2, y2 = box
    return (x1+x2)/2, (y1+y2)/2

def compDistance(box, center):
    c = compCenter(box)
    return (c[0]-center[0])**2 + (c[1]-center[1])**2

def box_center(boxes):
    if boxes.ndim == 1:
        c = (boxes[:2]+boxes[2:])/2
    else:
        c = (boxes[:,:2]+boxes[:,2:])/2
    return c

def sample_index(n, period, slength, pos='tail'):
    assert period >= slength
    assert pos in ['head', 'middle', 'tail']
    nf_p = int(period)
    nf_s = int(slength)
    if pos == 'head':
        ns = n // nf_p
        if n % nf_p >= nf_s:
            ns += 1
        start = 0
    elif pos == 'middle':
        ns = n // nf_p
        if n % nf_p >= (nf_p + nf_s)//2:
            ns += 1
        start = (nf_p - nf_s) // 2
    else: # tail
        ns = n // nf_p
        start = nf_p - nf_s
    res = []
    for i in range(start, n, nf_p):
        t = np.arange(i,i+nf_s)
        if len(t) == nf_s: # in case the last period is not complete
            res.append(t)
    return np.array(res, int)

def sample_data(data, period, slength, pos='tail'):
    assert period >= slength
    assert pos in ['head', 'middle', 'tail']
    nf_p = int(period)
    nf_s = int(slength)
    n = len(data)
    if pos == 'head':
        ns = n // nf_p
        if n % nf_p >= nf_s:
            ns += 1
        start = 0
    elif pos == 'middle':
        ns = n // nf_p
        if n % nf_p >= (nf_p + nf_s)//2:
            ns += 1
        start = (nf_p - nf_s) // 2
    else: # tail
        ns = n // nf_p
        start = nf_p - nf_s
    res = []
    for i in range(start, n, nf_p):
        t = data[i:i+nf_s]
        if len(t) == nf_s: # in case the last period is not complete
            res.append(t)
    return np.array(res)

def pad_with_sample(data, period, slength=1, off=0, pos='middle', line=True):
    vidx=sample_index(len(data[off:]), period, slength, pos)+off
    vdata=sample_data(data[off:], period, slength, pos)
    if line == False:
        pad=np.zeros_like(data)+np.nan
        pad[vidx] = vdata
    else:
        pad=np.zeros_like(data)
        if pos == 'head':
            vidx2=np.pad(vidx.ravel(),(0,1),constant_values=len(data))
            vidx2[0]=0
        elif pos == 'middle':
            vidx2 = vidx.ravel().copy()
            vidx2 = [(vidx2[i]+vidx2[i+1])//2 for i in range(len(vidx2)-1)]
            vidx2=np.pad(vidx2,1,constant_values=(0, len(data)))
        else:
            vidx2=np.pad(vidx.ravel(),(1,0),constant_values=0)
            vidx2[-1]=len(data)
        for i in range(len(vidx2)-1):
            pad[vidx2[i]:vidx2[i+1]]=vdata[i]
    return pad,vidx
