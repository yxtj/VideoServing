# -*- coding: utf-8 -*-

#from collections import namedtuple
import typing.NamedTuple
import numpy as np

#Configuration = namedtuple('Configuration', ['rsl', 'fps', 'roi', 'model'])

class Configuration(typing.NamedTuple):
    rsl: int
    fps: int
    roi: bool
    model: str


#DetectionResult = namedtuple('DetectionResult', ['box', 'lbl'])
class DetectionResult(typing.NamedTuple):
    box: np.ndarray
    lbl: int


#TrackResult = namedtuple('TrackResult', ['id', 'center', 'box', 'lbl', 'speed'])
class TrackResult(typing.NamedTuple):
    id: int
    center: np.ndarray
    box: np.ndarray
    lbl: int
    speed: np.ndarray

