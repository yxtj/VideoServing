# -*- coding: utf-8 -*-

import numpy as np

from reorderbuffer import ReorderBuffer

class Engine():
    
    def __init__(self, f_detect, nsource):
        self.f_detect = f_detect
        self.task_buffer = []
        self.rb = ReorderBuffer()
    
    def receive_tasks(self, tasks):
        pass
    
    
    def process(self):
        pass
    
    