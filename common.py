# -*- coding: utf-8 -*-

#from collections import namedtuple
import typing.NamedTuple
import numpy as np

#Configuration = namedtuple('Configuration', ['fps', 'rsl', 'roi', 'model'])

class Configuration(typing.NamedTuple):
    fps: int
    rsl: int
    #roi: bool
    #model: str

#Task = namedtuple('Task', ['time', 'frame', 'jid', 'sid', 'fid', 'rs', 'fr'])
class Task(typing.NamedTuple):
    frame: np.ndarry
    sid: int
    fid: int
    tag: int
    time: float

#DetectionResult = namedtuple('DetectionResult', ['box', 'lbl'])
class DetectionResult(typing.NamedTuple):
    box: np.ndarray
    lbl: int
    time: float


#TrackResult = namedtuple('TrackResult', ['id', 'center', 'box', 'lbl', 'speed'])
class TrackResult(typing.NamedTuple):
    id: int
    box: np.ndarray
    lbl: int
    speed: np.ndarray
    time: float

