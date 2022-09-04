# -*- coding: utf-8 -*-
import numpy as np

# %% basic class

class Tracker():
    def __int__(self):
        self.oid = 0
    
    def reset(self):
        self.oid = 0
    
    def update(self, boxes:np.ndarray, frames:int=None):
        '''
        Track bounding-boxes.

        Parameters
        ----------
        boxes : np.ndarray
            bounding boxes seen in current frame with shape: (n, 4)
        frames : int, optional
            number of frames since last update.

        Returns
        -------
        dict
            key is an integer object id. value is a bounding box.

        '''
        return {}

# %% generate tracker

def set_tracker(track_mthd, **kwargs):
    assert track_mthd in ['center', 'sort']
    tracker = None
    if track_mthd == 'center':
        from track.centroidtracker import CentroidTracker
        tracker = CentroidTracker(kwargs['age'])
    elif track_mthd == 'sort':
        from track.sorttracker import SortTracker
        tracker = SortTracker(kwargs['age'], kwargs['age'], 
                              iou_threshold=kwargs['min_iou'])
    return tracker