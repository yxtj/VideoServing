# -*- coding: utf-8 -*-

import torch
import torchvision


class ObjectSummarizer:
    def __init__(self, size=128):
        self.backbone = torchvision.models.alexnet(True).features
        self.backbone.eval()
        self.trans = torchvision.transforms.Resize((size,size))
    
    def get_feature(self, image, box):
        assert image.ndim == 3
        box = box.int().cpu().numpy()
        img = image[:,box[1]:box[3],box[0]:box[2]].unsqueeze(0)
        img = self.trans(img)
        with torch.no_grad():
            feat = self.backbone(img)
        return feat
    
    def get_features(self, image, boxes):
        assert image.ndim == 3
        assert isinstance(boxes, (list, torch.Tensor))
        x = []
        for box in boxes:
            box = box.int().cpu().numpy()
            img = image[:,box[1]:box[3],box[0]:box[2]]
            img = self.trans(img)
            x.append(img)
        x = torch.stack(x)
        with torch.no_grad():
            feat = self.backbone(x)
        return feat
        
    def fast_xcorr(self, z, x):
        # fast cross correlation
        # z: target feature (4D, with first dimension is 1)
        # x: input feature(s) (4D with the same 2-4 dimension as z)
        with torch.no_grad():
            nz = z.size(0)
            nx, c, h, w = x.size()
            x = x.view(-1, nz * c, h, w)
            out = torch.nn.functional.conv2d(x, z, groups=nz)
            #out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out.squeeze()
    
    def xcorr(self, z, x):
        assert z.ndim == 4 and z.size(0) == 1
        assert x.ndim == 4 and x.shape[1:] == z.shape[1:]
        _, c, h, w = z.shape
        f = c*h*w
        mxz = self.fast_xcorr(z, x) / f
        mz = z.mean()
        mx = x.mean((1,2,3))
        return mxz - mz*mx
        
    def xcorr_mat(self, a, b):
        na = a.size(0)
        #nb = b.size(0)
        ma = a.mean((1,2,3))
        mb = b.mean((1,2,3))
        res = -ma.view(-1, 1) * mb
        for i in range(na):
            mab = self.fast_xcorr(a[i].unsqueeze(0), b)
            res[i] += mab
        return res
        
