# -*- coding: utf-8 -*-

import numpy as np
import cv2

__all__ = ["get_background", "Detector_FrameDifference"]

# another online background extraction method:
# https://docs.opencv.org/4.5.1/d1/dc5/tutorial_background_subtraction.html

from .detectorbase import DetectorBase

def get_background(cap=None, file=None, size=50):
    if cap is None and file is None:
        raise ValueError('One of cap or file must be valid')
    if cap is None and isinstance(file, str):
        cap = cv2.VideoCapture(file)
    # we will randomly select 50 frames for the calculating the median
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=size)
    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    if cap is None and isinstance(file, str):
        cap.release()
    return median_frame

# %% detector

class Detector_FrameDifference(DetectorBase):
    def __init__(self, cap, size_threshold=0.001, bg_size=50):
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        median_frame = get_background(cap, size=50)
        self.background = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
        self.threshold = size_threshold*self.width*self.height
        self.factor = np.array([self.width, self.height, self.width, self.height])
    
    def process(self, frame, psize=None):
        # <frame> is a np.ndarray of 3D: [H, W, C] where C=3
        # <psize> is int of height. Not used here
        # return: labels, scores, boxes (labels and scores are None)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray, self.background)
        _, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        contours, hierarchy = cv2.findContours(
            dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            if cv2.contourArea(contour) < self.threshold:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            result.append((x, y, x+w, y+h))
        return None, None, np.array(result)/self.factor

# %% standalone detect and visualization function

def detect_base_difference(inFile, outFile=None,
                           consecutive_frame=4, show=True):
    cap = cv2.VideoCapture(inFile)
    # define codec and create VideoWriter object
    if outFile is not None:
    # get the video frame height and width
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(
            outFile,
            cv2.VideoWriter_fourcc(*'mp4v'), 10, 
            (frame_width, frame_height)
        )
    else:
        out = None
    
    # get the background model
    background = get_background(cap)
    # convert the background model to grayscale format
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_count += 1
            orig_frame = frame.copy()
            # IMPORTANT STEP: convert the frame to grayscale first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_count % consecutive_frame == 0 or frame_count == 1:
                frame_diff_list = []
            # find the difference between current frame and base frame
            frame_diff = cv2.absdiff(gray, background)
            # thresholding to convert the frame to binary
            ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
            # dilate the frame a bit to get some more white area...
            # ... makes the detection of contours a bit easier
            dilate_frame = cv2.dilate(thres, None, iterations=2)
            # append the final result into the `frame_diff_list`
            frame_diff_list.append(dilate_frame)
            # if we have reached `consecutive_frame` number of frames
            if len(frame_diff_list) == consecutive_frame:
                # add all the frames in the `frame_diff_list`
                sum_frames = sum(frame_diff_list)
                # find the contours around the white segmented areas
                contours, hierarchy = cv2.findContours(
                    sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # draw the contours, not strictly necessary
                for i, cnt in enumerate(contours):
                    cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
                for contour in contours:
                    # continue through the loop if contour area is less than 500...
                    # ... helps in removing noise detection
                    if cv2.contourArea(contour) < 500:
                        continue
                    # get the xmin, ymin, width, and height coordinates from the contours
                    (x, y, w, h) = cv2.boundingRect(contour)
                    # draw the bounding boxes
                    if out is not None:
                        cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                if out is not None:
                    out.write(orig_frame)
                if show:
                    cv2.imshow('Detected Objects', orig_frame)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# %% test


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to the input video',
                        required=True)
    parser.add_argument('-c', '--consecutive-frames', default=4, type=int,
                        dest='consecutive_frames', help='path to the input video')
    args = vars(parser.parse_args())
    
    save_name = f"outputs/{args['input'].split('/')[-1]}"
    detect_base_difference(args['input'], save_name, args['consecutive-frames'])

