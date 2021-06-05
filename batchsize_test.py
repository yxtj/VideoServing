# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:20:50 2021

@author: yanxi
"""

import torch
import time


def batchsize_test_cpu(dataset, model, n, bs):
    if(next(model.parameters()).is_cuda):
      model.cpu()
    model.eval()
    t=time.time()
    groups=torch.randint(0,len(dataset),[n//bs, bs])
    for idx in groups:
        imgs = [dataset[i][0] for i in idx]
        with torch.no_grad():
            p=model(imgs)
    np = (n//bs) * bs
    t=time.time()-t
    print(np/t)
    return t,np,np/t

def batchsize_test_cuda(dataset, model, n, bs):
    if(not next(model.parameters()).is_cuda):
      model.cuda()
    model.eval()
    tt=0.0
    groups=torch.randint(0,len(dataset),[n//bs, bs])
    for idx in groups:
        imgs = [dataset[i][0].to('cuda') for i in idx]
        t=time.time()
        with torch.no_grad():
            p=model(imgs)
        tt+=time.time()-t
    np = (n//bs) * bs
    print(np/tt)
    return tt,np,np/tt