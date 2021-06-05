# -*- coding: utf-8 -*-
import re

class Query():
    def __init__(self, task_id, period=None, channel=0, other=None):
        self.tid = task_id
        if period is None:
            self.time_start = 0
            self.time_end = None
        else:
            l = period.split('-')
            assert 1 <= len(l) <= 2
            if len(l) == 2:
                self.time_start_hms = Query.ParseTime(l[0])
                self.time_end_hms = Query.ParseTime(l[1])
            else:
                self.time_start_hms = Query.ParseTime(l[0])
                self.time_end_hms = self.time_start
            self.time_start = Query.HMS2second(self.time_start_hms)
            self.time_end = Query.HMS2second(self.time_end_hms)
        self.channel = channel
        self.other = other
        # parse arguments
    
    def __repr__(self):
        return 'Query(task=%d, channel=%d, period=%d-%d)' % \
            (self.tid, self.channel, self.time_start, self.time_end)
    
    @staticmethod
    def generate(string):
        return None
    
    @staticmethod
    def ParseTime(period):
        if period is None:
            return None
        pat = re.compile('''(?:(?:(\d{1,2}):)?(\d{1,2}):)?(\d{1,2})''')
        m = pat.match(period)
        if m is None:
            return None
        return tuple(int(i) if i is not None else 0 for i in m.groups())
        
    @staticmethod    
    def HMS2second(hms):
        return hms[0]*3600+hms[1]*60+hms[2]
    