# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 03:09:50 2021

@author: yanxi
"""

import numpy as np

# %% bound box

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

def compBAsize(boxes):
    # global bounding box size and actual sizes
    if boxes.ndim == 1:
        s = np.prod(boxes[2:]-boxes[:2])
        return s,s
    else:
        sa = np.prod(boxes[:,2:]-boxes[:,:2], 1).sum()
        sb = np.prod(boxes[:,2:].max(0)-boxes[:,:2].min(0))
        return sb, sa

def box_center(boxes):
    if boxes.ndim == 1:
        c = (boxes[:2]+boxes[2:])/2
    else:
        c = (boxes[:,:2]+boxes[:,2:])/2
    return c

def box_size(boxes):
    if len(boxes) == 0:
        return []
    if boxes.ndim == 1:
        s = np.prod(boxes[2:]-boxes[:2])
    else:
        s = np.prod(boxes[:,2:]-boxes[:,:2], 1)
    return s

def box_aratio(boxes):
    if len(boxes) == 0:
        return []
    if boxes.ndim == 1:
        w, h = boxes[2:] - boxes[:2]
        r = w / h
    else:
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        r = w / h
    return r

def box_super(boxes):
    if boxes.ndim == 1:
        r = boxes
    else:
        r = np.concatenate([boxes[:,:2].min(0), boxes[:,2:].max(0)])
    return r

# %% data sampling

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

def sample_and_pad(data, period, slength=1, off=0, pos='middle', padvalue='same'):
    assert pos in ['head', 'middle', 'tail']
    assert padvalue == 'same' or isinstance(padvalue, (int, float)) # np.nan is float
    vdata=sample_data(data[off:], period, slength, pos)
    return pad_by_sample(vdata, len(data), period, slength, off,pos, padvalue)

def pad_by_sample(data, n, period, slength, off=0, pos='middle', padvalue='same'):
    assert pos in ['head', 'middle', 'tail']
    assert padvalue == 'same' or isinstance(padvalue, (int, float)) # np.nan is float
    vidx=sample_index(n-off, period, slength, pos)+off
    assert data.ndim == 1 and slength == 1 and data.shape == (len(vidx),) or \
        data.ndim == 2 and data.shape == (len(vidx), slength)
    if padvalue != 'same':
        pad = np.zeros(n) + padvalue
        if data.ndim == 1:
            pad[vidx] = data.reshape((-1, 1))
        else:
            pad[vidx] = data
    else:
        pad=np.zeros(n)
        if pos == 'head':
            vidx2=np.pad(vidx.ravel(),(0,1),constant_values=n)
            vidx2[0]=0
        elif pos == 'middle':
            vidx2 = vidx.ravel().copy()
            vidx2 = [(vidx2[i]+vidx2[i+1])//2 for i in range(len(vidx2)-1)]
            vidx2=np.pad(vidx2,1,constant_values=(0, n))
        else:
            vidx2=np.pad(vidx.ravel(),(1,0),constant_values=0)
            vidx2[-1]=n
        # len(data) == len(vidx2) - 1
        for i in range(len(data)):
            pad[vidx2[i]:vidx2[i+1]]=data[i]
    return pad, vidx

# %%

def moving_average(array, window, padmethod='mean'):
    assert padmethod in ['edge', 'linear_ramp', 'mean', 'median']
    pshape = [(0,0) for _ in range(array.ndim-1)] + [(window-1, 0)]
    d = np.pad(array, pshape, padmethod)
    kernel = np.ones(window)/window
    if array.ndim == 1:
        return np.convolve(d, kernel, 'valid')
    else:
        return np.apply_along_axis(lambda x:np.convolve(x,kernel,'valid'),-1,d)