# -*- coding: utf-8 -*-

# idea from: https://www.zhihu.com/question/307282137/answer/1518052588

import threading
import torch
import cv2

class ImagePreLoader(threading.Thread):
    def __init__(self, img_list, device, q):
        super(self).__init__()
        self.img_list = img_list
        self.device = device
        self.q = q
        self.stream = torch.cuda.Stream()

    def run(self):
        with torch.cuda.stream(self.stream):
            for img_path in self.img_list:
                img = cv2.imread(img_path)
                if img is None:
                    print('NoneType: %s' % img_path)
                    continue
                # some other processing here
                img = torch.Tensor(img).to(device=self.device)
                self.q.put((img, img_path))
                
# %% test

def __test__():
    import queue
    q = queue.SimpleQueue()
    #import multiprocessing
    #q = multiprocessing.SimpleQueue()
    img_list=['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    
    # load
    load_thread = ImagePreLoader(img_list, 'cuda:0', q)
    load_thread.start()
    
    # consume
    n = 0
    while n < len(img_list):
        img = q.get()
        print(n, img.shape)
    print('done')