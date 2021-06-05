# -*- coding: utf-8 -*-


class Task():
    def __init__(self, task_id, operations=[]):
        self.task_id = task_id
        self.operations = operations
        self.num_opt = len(operations)
        
    def __repr__(self):
        return 'Task(id=%d, #op=%d, addr: %x)' % \
            (self.task_id, self.num_opt, id(self))
        
    def go_whole(self, data):
        d = data
        for op in self.operations:
            d = op(d)
        return d
    
    def go_phase(self, pid, data):
        assert 0 <= pid < self.num_opt
        return self.operations[pid](data)
        