# -*- coding: utf-8 -*-

import numpy as np

class RangeChecker():
    def __init__(self, line_dir='h', line_pos=0.5,
                 detect_rng=0.2, track_rng=0.1):
        assert line_dir in ['h','v']
        assert 0 < track_rng <= detect_rng
        self.dir = line_dir
        self.pos = line_pos
        self.drng = detect_rng
        self.trng = track_rng
        # helpers
        if self.dir == 'h':
            # horizontal line: check the vertical coordinate
            self.idx = 1
        else:
            # vertical line: check the horizontal coordinate
            self.idx = 0
    
    def __repr__(self):
        return '{%c-%g detect-rng: %g, track-rng: %g}' \
            % (self.dir, self.pos, self.drng, self.trng)
    
    def in_detect(self, points):
        if points.ndim == 1:
            return np.abs(points[self.idx] - self.pos) <= self.drng
        else:
            return np.abs(points[:,self.idx] - self.pos) <= self.drng
    
    def in_track(self, points):
        if points.ndim == 1:
            return np.abs(points[self.idx] - self.pos) <= self.trng
        else:
            return np.abs(points[:,self.idx] - self.pos) <= self.trng
        
    def direction(self, old_point, new_point):
        return new_point[self.idx] - old_point[self.idx]
    
    def offset(self, points):
        if points.ndim == 1:
            return points[self.idx] - self.pos
        else:
            return points[:,self.idx] - self.pos
