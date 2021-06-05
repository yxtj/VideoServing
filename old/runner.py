# -*- coding: utf-8 -*-

from videoholder import VideoHolder
from query import Query
from task import Task
from operation import Operation
from typing import List

class Runner():
    def __init__(self, videos: List[VideoHolder]):
        self.videos = videos
        self.queries = {}
        self.qid_counter = 0
        self.tasks = {}
        self.id2opt = {}
        self.opt2id = {}
        self.oid_counter = 0
        
    def add_task(self, tid: int, operations: List[Operation]):
        self.tasks[tid] = Task(tid, operations)
        for opt in operations:
            if opt not in self.opt2id:
                oid = self.oid_counter
                self.oid_counter += 1
                self.id2opt[oid] = opt
                self.opt2id[opt] = oid
    
    def remove_task(self, tid: int):
        assert tid in self.tasks
        del self.tasks[tid]
    
    def clear_task(self):
        self.tasks = {}
        self.id2opt = {}
        self.opt2id = {}
    
    def add_query(self, query: Query):
        assert 0 <= query.channel < len(self.videos)
        assert query.time_end is None or query.time_end < self.videos[query.channel].length
        qid = self.qid_counter
        self.qid_counter += 1
        ff, lf = self._get_frame_range_(query)
        self.queries[qid] = {'query': query, 'result': None,
                             'frm_first': ff, 'frm_last': lf}
        return qid
    
    def remove_query(self, qid: int):
        assert qid in self.queries
        del self.queries[qid]
    
    def check_query(self, qid: int):
        if qid not in self.queries:
            return None
        elif self.queries[qid]['result'] is None:
            return False
        else:
            return True

    def clear_query(self):
        self.queries = []
        
    def get_result(self, qid: int):
        if qid in self.queries and self.queries[qid]['result'] is not None:
            res = self.queries[qid]['result']
            del self.queries[qid]
            return res
        else:
            return None
    
    def _get_frame_range_(self, query):
        v = self.videos[query.channel]
        ff = v.second2frame(query.time_start)
        lf = v.second2frame(query.time_end)
        return ff, lf
            
    def process(self):
        frames = {}
        # step 1: get frames
        for qid, qr in self.queries.items():
            q = qr['query']
            r = qr['result']
            if r is not None:
                continue
            v = self.videos[q.channel]
            ff = qr['frm_first']
            lf = qr['frm_last']
            fms = v.get_frames(ff, lf)
            for i,f in zip(range(ff, lf), fms):
                frames[i] = f
        # step 2: process frames
        res1 = {}
        for qid, qr in self.queries.items():
            q = qr['query']
            r = qr['result']
            if r is not None:
                continue
            ff = qr['frm_first']
            lf = qr['frm_last']
            tid = q.target
            opt = self.tasks[tid].operations[0]
            oid = self.opt2id[opt]
            for i in range(ff, lf):
                f = frames[i]
                res1[(i,oid)] = opt([f])
        # step 3: other processing
        for qid, qr in self.queries.items():
            q = qr['query']
            r = qr['result']
            if r is not None:
                continue
            ff = qr['frm_first']
            lf = qr['frm_last']
            tid = q.target
            opt = self.tasks[tid].operations[0]
            oid = self.opt2id[opt]
            data = [ res1[(i,oid)] for i in range(ff, lf) ]
            for o in self.tasks[tid].operations[1:]:
                data = o(data)
            r = data
        
