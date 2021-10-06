# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re

FN_LIST=['320x240_25_cmu_estimation_result',
         '480x352_25_cmu_estimation_result',
         '640x480_25_cmu_estimation_result',
         '960x720_25_cmu_estimation_result',
         '1120x832_25_cmu_estimation_result']

RS_LIST=[240, 360, 480, 720, 840]

def parse_pose(line):
    '''
    return poses (3d)
    D1: number of persons
    D2: 17 key points
    D3: 3 for x, y, v
    v: 0->fail, 1->estimate, 2->seen
    '''
    if len(line) == 0:
        return [], []
    poses = []
    scores = []
    for m in re.findall('\[(.+?)\],(\d\.\d+?)', line):
        s = float(m[1])
        if s >= 1.0:
            line = m[0].split(', ')
            r = np.array(line, dtype=float).reshape(17,3)
            poses.append(r)
            scores.append(s)
    return np.array(poses), np.array(scores)


def read_data(fn):
    df = pd.read_csv(fn, sep='\t', usecols=['Estimation_result', 'Time_SPF'],
                     converters={'Estimation_result':str})
    r = df['Estimation_result']
    times = df['Time_SPF'].to_numpy()
    poses = []
    scores = []
    for line in r:
        p, s = parse_pose(line)
        poses.append(p)
        scores.append(s)
    return times, poses, scores


def trans_folder(folder):
    print(folder)
    poses = []
    times = []
    scores = []
    for fn,h in zip(FN_LIST, RS_LIST):
        print(fn)
        t, p, s = read_data(folder+'/'+fn+'.tsv')
        poses.append(p)
        times.append(t)
        scores.append(s)
        np.savez(folder+'/raw-%d' % h, height=h, times=t,
                 poses=np.array(p,dtype=object),
                 scores=np.array(s,dtype=object))
    #poses = np.array(poses, dtype=object)
    #times = np.array(times)
    #np.savez(folder+'/raw-all', heights=RS_LIST, times=times, poses=poses)
    
    
