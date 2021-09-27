# -*- coding: utf-8 -*-

class ReorderBuffer:
    
    def __init__(self, init_v=0):
        self.buffer_data = {}
        self.buffer_idx = set()
        self.pfirst = init_v # the next ID to pick
        self.plast = init_v # one over the last ordered ID
    
    def put(self, idx, data):
        self.buffer_data[idx] = data
        self.buffer_idx.add(idx)
        if idx == self.plast:
            self.__move_last__()
        
    def get_one(self):
        if self.pfirst < self.plast:
            idx = self.pfirst
            data = self.buffer_data.pop(idx)
            self.pfirst += 1
            return idx, data
        return None, None
    
    def get(self, n:int=0):
        if n == 0:
            n = self.plast - self.pfirst
        res_i = []
        res_d = []
        while self.pfirst < self.plast and n > 0:
            idx = self.pfirst
            data = self.buffer_data.pop(idx)
            self.pfirst += 1
            res_i.append(idx)
            res_d.append(data)
            n -= 1
        return res_i, res_d
    
    def move_and_check(self, target_idx, n:int=0):
        if n == 0:
            n = self.plast - self.pfirst
        while self.pfirst < self.plast and n > 0:
            idx = self.pfirst
            self.buffer_data.pop(idx)
            self.pfirst += 1
            n -= 1
            if idx == target_idx:
                return True
        return False
    
    # size related 
    
    def size(self):
        return len(self.buffer_data)
    
    def size_inorder(self):
        return self.plast - self.pfirst
    
    def size_unorder(self):
        return len(self.buffer_data) - self.size_inorder()
        
    # helpers
    
    def __move_last__(self):
        while self.plast in self.buffer_idx:
            self.buffer_idx.remove(self.plast)
            self.plast += 1

# %% test

def __test1__():
    idxes = [7, 4, 9, 0, 5, 2, 3, 1, 6, 8]
    rb=ReorderBuffer()
    for idx in idxes:
        print('put:',idx)
        rb.put(idx*10, idx)
        print('  size:',rb.size(), rb.size_inorder(), rb.size_unorder())
        i,d = rb.get()
        print('  get:',i,d)
        print('  idx:',rb.buffer_idx)
    
    print('finish:', rb.size(), rb.buffer_idx, rb.buffer_data)

def __test2__():
    idxes = [7, 4, 9, 0, 5, 2, 3, 1, 6, 8]
    rb=ReorderBuffer()
    for idx in idxes:
        print('put:',idx)
        rb.put(idx*10, idx)
        print('  size:',rb.size(), rb.size_inorder(), rb.size_unorder())
        i,d = rb.get(2)
        print('  get:',i,d)
        print('  idx:',rb.buffer_idx)
    while True:
        print('pick')
        i,d = rb.get(2)
        if len(i) == 0:
            break
        print('  size:',rb.size(), rb.size_inorder(), rb.size_unorder())
        print('  get:',i,d)
        print('  idx:',rb.buffer_idx)

def __test3__():
    import numpy as np
    
    idxes = np.arange(10)
    np.random.shuffle(idxes)
    print('order:',idxes)
    
    rb=ReorderBuffer()
    for idx in idxes:
        print('put:',idx)
        rb.put(idx*10, idx)
        print('  size:',rb.size(), rb.size_inorder(), rb.size_unorder())
        i,d = rb.get()
        print('  get:',i,d)
        print('  idx:',rb.buffer_idx)
    
    print('finish:', rb.size(), rb.buffer_idx, rb.buffer_data)
