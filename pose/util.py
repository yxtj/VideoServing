# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 02:33:57 2019

@author: yanxi
"""

import numpy as np


# -------- part 1: key point operation --------

'''
0	nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle
'''

__KPT_NAME__ = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar',
               'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
               'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
               'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']

__KPT_MAP__ = {'nose': 0,
               'leftEye':1, 'leye':1, 'ly':1, 'rightEye':2, 'reye':2, 'ry':2,
               'leftEar':3, 'lear':3, 'le':3, 'rightEar':4, 'rear':4, 're':4,
               'leftShoulder':5, 'lshoulder':5, 'ls':5,
               'rightShoulder':6, 'rshoulder':6, 'rs':6,
               'leftElbow':7, 'lelbow':7, 'lb':7, 'rightElbow':8, 'relbow':8, 'rb':8,
               'leftWrist':9, 'lwrist':9, 'lw':8, 'rightWrist':10, 'rwrist':10, 'rw':10,
               'leftHip':11, 'lhip':11, 'lh':11, 'rightHip':12, 'rhip':12, 'rh':12,
               'leftKnee':13, 'lknee':13, 'lk':13, 'rightKnee':14, 'rknee':14, 'rk':14,
               'leftAnkle':15, 'lankle':15, 'la':15, 'rightAnkle':15, 'rankle':16, 'ra':16}

__NUM_KPT__ = 17

def name2kp(data):
    if isinstance(data, str):
        return __KPT_MAP__[data]
    elif isinstance(data, tuple):
        return tuple([name2kp(v) for v in data])
    elif isinstance(data, list):
        return [name2kp(v) for v in data]
    elif isinstance(data, np.ndarray) and data.ntype.kind in ['U','S','a']:
        s=data.shape
        res=np.apply_along_axis(lambda n:__KPT_MAP__[n], 0, data.ravel())
        return res.reshape(s)
    else:
        assert hasattr(data, '__iter__')
        return [name2kp(v) for v in data]


def kp2name(data):
    if isinstance(data, int):
        return __KPT_NAME__[data]
    elif isinstance(data, tuple):
        return tuple(kp2name(v) for v in data)
    elif isinstance(data, list):
        return [kp2name(v) for v in data]
    elif isinstance(data, np.ndarray):
        s=data.shape
        res=np.apply_along_axis(lambda n:__KPT_NAME__[n], 0, data.ravel())
        return res.reshape(s)
    else:
        assert hasattr(data, '__iter__')
        return [kp2name(v) for v in data]

# -------- part 2: pose matrix operation --------

def fillUnseen(kpm, method='same'):
    assert kpm.ndim == 4
    assert kpm.shape[-2:] == (17,3)
    assert method in ['same', 'linear']
    n, m = kpm.shape[:-2]
    res = kpm.copy()
    if method == 'same':
        for i in range(17):
            nz = np.nonzero(kpm[:,:,i,2] - 2)
            for x,y in zip(nz[0],nz[1]):
                p=y-1
                while p >= 0 and res[x,p,i,2] == 0:
                    p-=1
                if p >= 0:
                    res[x,y,i,:2] = res[x,p,i,:2]
    else:
        for i in range(17):
            nz = np.nonzero(kpm[:,:,i,2] - 2)
            for x,y in zip(nz[0],nz[1]):
                p=y-1
                while p >= 0 and res[x,p,i,2] == 0:
                    p-=1
                q=p-1
                while q >= 0 and res[x,q,i,2] == 0:
                    q-=1
                if q >= 0:
                    s = res[x,p,i,:2] - res[x,q,i,:2]
                    res[x,y,i,:2] = res[x,p,i,:2] + s / (p-q) * (y-p)
    return res

# -------- part 3: OKS --------


__KPT_OKS_SIGMAS__ = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
__KPT_OKS_VARIANCE__ = (__KPT_OKS_SIGMAS__ * 2)**2

def computeOKS_1to1(gts, dts, sigmas = None):
    '''
    Object Keypoint Similarity
    http://cocodataset.org/#keypoints-eval
    OKS = Σi[exp(-di2/2s2κi2)δ(vi>0)] / Σi[δ(vi>0)]
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    input:
        <gts>: ground truth key points. Format: 2d array with shape (17,3) for the (x,y) coordinates of the 17 keypoints.
        <dts>: destination key points. Format: the same as <gts>.
    output:
        a scalar of oks
    '''
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dts, np.ndarray) and dts.shape == (17,3)
    if sigmas is None:
        sigmas = __KPT_OKS_SIGMAS__
        vars = __KPT_OKS_VARIANCE__
    else:
        assert sigmas.shape == (17,)
        vars = (sigmas * 2)**2

    xg = gts[:,0]
    yg = gts[:,1]
    vg = gts[:,2]
    k1 = np.count_nonzero(vg > 0)

    xmin = xg.min(); xmax = xg.max(); xdif = xmax - xmin;
    ymin = yg.min(); ymax = yg.max(); ydif = ymax - ymin;
    area = (xmax - xmin)*(ymax - ymin)

    xd = dts[:,0]
    yd = dts[:,1]
    #vd = np.zeros_like(dg) + 2
    #k2 = np.count_nonzero(vd > 0)

    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        #bb = gt['bbox']
        x0 = xmin - xdif; x1 = xmax + xdif;
        y0 = ymin - ydif; y1 = ymax + ydif;
        z = np.zeros((17))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
    e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2
    if k1 > 0:
        e=e[vg > 0]
    return np.sum(np.exp(-e)) / e.shape[0]


def computeOKS_NtoN(gtsMat, dtsMat, sigmas = None):
    assert gtsMat.shape == dtsMat.shape
    assert gtsMat.ndim >= 3
    assert gtsMat.shape[-2:] == (17, 3)
    s = gtsMat.shape[:-2]
    n = s[0] if len(s) == 1 else np.multiply(*s)
    res = np.zeros(n)
    g = gtsMat.reshape(-1, 17, 3)
    d = dtsMat.reshape(-1, 17, 3)
    for i in range(n):
        res[i] = computeOKS_1to1(g[i], d[i], sigmas)
    return res.reshape(s)


def computeOKS_1toN(gts, dtsMat, sigmas = None):
    '''
    return a matrix of OKS for each dts in <dtsMat> using gts as the reference
    '''
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dtsMat, np.ndarray) and dtsMat.shape[-2:] == (17,3)
    assert dtsMat.ndim > 2
    sigmas = np.array(__KPT_OKS_SIGMAS__ if sigmas is None else sigmas)
    assert sigmas.shape == (17,)
    vars = (sigmas * 2)**2

    matShape = dtsMat.shape[:-2]
    dtsList = dtsMat.reshape(-1,17,3)

    xg = gts[:,0]
    yg = gts[:,1]
    vg = gts[:,2]
    k1 = np.count_nonzero(vg > 0)

    xmin = xg.min(); xmax = xg.max(); xdif = xmax - xmin;
    ymin = yg.min(); ymax = yg.max(); ydif = ymax - ymin;
    area = (xmax - xmin)*(ymax - ymin)

    n = dtsList.shape[0] if isinstance(dtsList, np.ndarray) else len(dtsList)
    res=np.zeros(n)
    # normal case
    if k1>0:
        for i in range(n):
            dts = dtsList[i]
            xd = dts[:,0]
            yd = dts[:,1]
            dx = xd - xg
            dy = yd - yg
            e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2
            e = e[vg > 0]
            res[i] = np.sum(np.exp(-e)) / e.shape[0]
    else:
        x0 = xmin - xdif; x1 = xmax + xdif;
        y0 = ymin - ydif; y1 = ymax + ydif;
        z = np.zeros((17))
        for i in range(n):
            dts = dtsList[i]
            xd = dts[:,0]
            yd = dts[:,1]
            dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
            dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2
            res[i] = np.sum(np.exp(-e)) / 17
    return res.reshape(matShape)


def computeOKS_NtoMN(gtsMat, dtsMat, sigmas = None):
    '''
    gts is 3d.
    dts is 4d or higher.
    '''
    assert gtsMat.ndim == 3 and gtsMat.shape[-2:] == (17,3)
    assert dtsMat.ndim >= 4 and dtsMat.shape[-2:] == (17,3)
    assert dtsMat.shape[-3] == gtsMat.shape[0]
    sigmas = np.array(__KPT_OKS_SIGMAS__ if sigmas is None else sigmas)
    assert sigmas.shape == (17,)
    n = gtsMat.shape[0]
    matShape = dtsMat.shape[:-2]
    if len(matShape) > 2:
        dtsMat = dtsMat.reshape((-1, n, 17, 3))
    m = dtsMat.shape[0]
    oks = np.zeros((m,n))
    for i in range(n):
        oks[:,i] = computeOKS_1toN(gtsMat[i], dtsMat[:,i], sigmas)
    if len(matShape) > 2:
        oks.resize(matShape)
    return oks



# -------- part 4: analysis of OKS --------


def computeDiff_1to1(gts, dts):
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dts, np.ndarray) and dts.shape == (17,3)

    res = np.zeros([17, 2])
    res[:,0] = np.sqrt(((gts[:,0:2] - dts[:,0:2])**2).sum(1))
    res[:,1] = gts[:,2] + dts[:,2]

    return res


def computeDiff_1toN(gts, dtsMat):
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dtsMat, np.ndarray) and dtsMat.ndim > 2 and dtsMat.shape[-2:] == (17,3)
    s = dtsMat.shape[:-2]
    dtsList = dtsMat.reshape(-1,17,3)
    n = dtsList.shape[0]
    res = np.zeros([n, 17, 2])
    for i in range(n):
        res[i] = computeDiff_1to1(gts, dtsList[i])
    res = res.reshape([*s, 17, 2])
    return res


def computeDiff_NtoN(gtsMat, dtsMat):
    assert isinstance(gtsMat, np.ndarray) and gtsMat.shape[-2:] == (17,3)
    assert isinstance(dtsMat, np.ndarray) and gtsMat.shape == dtsMat.shape
    s = gtsMat.shape[:-2]
    n = s[0] if len(s) == 1 else np.multiply(*s)
    g = gtsMat.reshape(-1, 17, 3)
    d = dtsMat.reshape(-1, 17, 3)
    res = np.zeros(n, 17, 2)
    for i in range(n):
        res[i] = computeDiff_1to1(g[i], d[i])
    res = res.reshape([*s, 17, 2])
    return res
