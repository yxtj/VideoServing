# modified from:
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
from scipy.spatial import distance as dist
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
    
    def reset(self):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        
    def get_state(self):
        return self.nextObjectID, self.objects.copy(), self.disappeared.copy()
    
    def set_state(self, state):
        self.nextObjectID = state[0]
        self.objects = state[1]
        self.disappeared = state[2]
        
    def register(self, centroid, box):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = (centroid, box)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def bbox2centroid(self, rect):
        return (rect[:2]+rect[2:]) / 2
    
    def bboxes2centroids(self, rects):
        return np.apply_along_axis(lambda r:(r[:2]+r[2:])/2, 1, rects)

    def get_result(self):
        return { id:b for id,(c,b) in self.objects.items() }

    def update(self, boxes):
        if len(boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.get_result()
        
        # initialize an array of input centroids for the current frame
        #inputCentroids = self.bboxes2centroids(boxes)
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        inputCentroids = (boxes[:,:2] + boxes[:,2:]) / 2

        if len(self.objects) == 0:
            for c,b in zip(inputCentroids, boxes):
                self.register(c, b)
            return self.get_result()
        
        # grab the set of object IDs and corresponding centroids
        objectIDs = list(self.objects.keys())
        objectCentroids, objectBoxes = list(zip(*self.objects.values()))
        objectCentroids = np.array(objectCentroids)
        objectBoxes = np.array(objectBoxes)
        D = dist.cdist(np.array(objectCentroids), inputCentroids)
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value is at the *front* of the index
        # list
        rows = D.min(axis=1).argsort()
        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = D.argmin(axis=1)[rows]
        
        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        usedRows = set()
        usedCols = set()
        # loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it
            if row in usedRows or col in usedCols:
                continue
            # otherwise, grab the object ID for the current row,
            # set its new centroid, and reset the disappeared counter
            objectID = objectIDs[row]
            self.objects[objectID] = (inputCentroids[col], boxes[col])
            self.disappeared[objectID] = 0
            # indicate that we have examined each of the row and
            # column indexes, respectively
            usedRows.add(row)
            usedCols.add(col)
            
        # compute both the row and column index we have NOT yet examined
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        
        # unmatched existing objects
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            # check to see if the number of consecutive
            # frames the object has been marked "disappeared"
            # for warrants deregistering the object
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)
        
        # unmatched new objects
        for col in unusedCols:
            self.register(inputCentroids[col], boxes[col])
        
        return self.get_result()
        
        