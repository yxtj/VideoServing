# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageDraw
import numpy as np

import cv2
import torch

def addbox_matplot(ax,box,text=None,linewidth=1,color='r'):
    x1,y1,x2,y2=box
    bbox=patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=linewidth,
                           edgecolor=color,facecolor='none')
    #ax = plt.gca()
    ax.add_patch(bbox)
    if text is not None:
        plt.text(x1,y1,text,color=color,verticalalignment='top')


def __trans_boxes__(boxes, W, H):
    if len(boxes) == 0:
        return np.empty(0, int)
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    elif isinstance(boxes, list):
        boxes = np.array(boxes)
    b = boxes.max() <= 1
    if b:
        boxes = boxes * np.array([W, H, W, H])
    return boxes.astype(int)


def show_box_pil(img, boxes, scores=None, ground_trouth=[]):
    if isinstance(img, torch.Tensor):
        image = Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
    elif isinstance(img, np.ndarray):
        image = Image.fromarray(img)
    else:
        image = img
    W, H = image.size
    draw = ImageDraw.Draw(image)
    boxes = __trans_boxes__(boxes, W, H)
    # draw groundtruth
    for i in range(len(ground_trouth)):
        draw.rectangle([(ground_trouth[i][0], ground_trouth[i][1]),
                       (ground_trouth[i][2], ground_trouth[i][3])], 
                      outline ="green", width=3)
    for i in range(len(boxes)):
        box = boxes[i]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                       outline ="red", width=3)
        if scores:
            score = np.round(scores[i], decimals=4)
            draw.text((box[0], box[1]), text=str(score))
    image.show()

def show_box_cv(img, boxes, scores=None, ground_trouth=[], win_name='image'):
    if isinstance(img, torch.Tensor):
        image = img.mul(255).to(torch.uint8).permute(1,2,0).cpu().numpy()
    else:
        image = img.copy()
    H, W = image.shape[:2]
    boxes = __trans_boxes__(boxes, W, H)
    for i in range(len(ground_trouth)):
        # color space BGR
        cv2.rectangle(image, (ground_trouth[i][0], ground_trouth[i][1]),
                      (ground_trouth[i][2], ground_trouth[i][3]), 
                      (0,255,0), 2)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        box = boxes[i]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
        if scores:
            score = np.round(scores[i], decimals=4)
            cv2.putText(image, str(score), (int(box[0]+2), int(box[1]-2)),
                        font, 0.8, (0,0,255), 1)
    cv2.imshow(win_name, image)
    