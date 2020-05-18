import collections
import math

existRule = []
enterRule = []
exitRule = []
lostRule = []

CLASS_ID_PERSON = 0

class Tracker:

    def __init__(self, numberOfFrames):
        self.currentFrame = 0
        self.trackID = 0
        self.maxFramesBeforeLost = 5
        self.totalNumberOfPeople = 0
        self.ObjData = collections.namedtuple('ObjData', ['trackID', 
                                                'xmin', 'xmax', 'ymin', 'ymax', 
                                                'enter', 'lost'])

        self.frameHistory = [None] * numberOfFrames
        self.uniqueIDs = set()


    def increateFrameCounter(self):
        self.currentFrame += 1
        return

    def getNumberOfPeopleInFrame(self):
        if self.frameHistory[self.currentFrame] != None:
            people = [ True for obj in self.frameHistory[self.currentFrame] if obj.class_id == CLASS_ID_PERSON ]
            numOfNewObjects = len(people)
        else:
            numOfNewObjects = 0    
        return numOfNewObjects

    def updateFrame(self, objects):

        objInFrame = []
        for obj in objects:
            if(obj['class_id']==CLASS_ID_PERSON):
                objInFrame.append(self.ObjData(
                        trackID=self.trackID, 
                        xmin=obj['xmin'], ymin=obj['ymin'], 
                        xmax = obj['xmax'], ymax = obj['ymax'],
                        enter = None, lost = None))
                self.trackID += 1
            
        self.frameHistory[self.currentFrame] = objInFrame

    def _calculateCentroid(self, xmin, ymin, xmax, ymax):
        return ( (xmin+xmax)/2.0, (ymin+ymax)/2.0 )
    
    def _calculateBoxSize(self, xmin, ymin, xmax, ymax):
        return (xmax - xmin) * (ymax - ymin)

    def detectEnter(self):
        return

    def _cosineSimilarity(self, A, B):
        return ( A.xmin * B.xmin + A.xmax * B.xmax + A.ymin * B.ymin + A.ymax * B.ymax) /           \
            math.sqrt(A.xmin * A.xmin + A.xmax * A.xmax + A.ymin * A.ymin + A.ymax * A.ymax) /      \
            math.sqrt(B.xmin * B.xmin + B.xmax * B.xmax + B.ymin * B.ymin + B.ymax * B.ymax)

    def findSimilarity(self):
        currentObjects = self.frameHistory[self.currentFrame]
        previousObjects = self.frameHistory[self.currentFrame - 1]
        currentObjectIndexes = [i for i in range(len(currentObjects))]
        previousObjectIndexes = [i for i in range(len(previousObjects))]
        similarityMap = [ [None for y in range(len(previousObjects))] for x in range(len(currentObjects))]
        
        for x in currentObjectIndexes:
            for y in previousObjectIndexes:
                similarityMap[x][y] = self._cosineSimilarity(currentObjects[x], previousObjects[y])

        pairs = []
        Pair = collections.namedtuple("Pair", ["current", "previous"])
        while( len(currentObjectIndexes) > 0 and len(previousObjectIndexes)>0):
            maxX = currentObjectIndexes[0]
            maxY = previousObjectIndexes[0]
            for x in currentObjectIndexes:
                for y in previousObjectIndexes:
                    if similarityMap[x][y] > similarityMap[maxX][maxY]:
                        maxX = x
                        maxY = y
            pairs.append(Pair(current = maxX, previous = maxY))
            currentObjectIndexes.remove(maxX)
            previousObjectIndexes.remove(maxY)

        return pairs, currentObjectIndexes, previousObjectIndexes

    def updateTrackID(self, pairs):
        for pair in pairs:
            trackID = self.frameHistory[self.currentFrame-1][pair.previous].trackID
            self.frameHistory[self.currentFrame][pair.current] = self.frameHistory[self.currentFrame][pair.current]._replace(trackID = trackID)

    def updateUniqueIDs(self):
        for obj in self.frameHistory[self.currentFrame]:
            self.uniqueIDs.add(obj.trackID)
        
    def countTotalIDs(self):
        return len(self.uniqueIDs)
        
        




    # def detectEnter(self, obj):
    #     return True

    # def detectExit(self, obj):
    #     return False

    # def detectLost(self, obj):
    #     return False

    # def whenNumIncreased(self, objects):
    #     return

    # def whenNumDecreased(self, objects):
    #     return

    
if __name__ == '__main__':
    tracker = Tracker(4)
    objectA1 = {'class_id':0, 'xmin':1, 'ymin':2, 'xmax':4, 'ymax':5, 'confidence':0.7}
    objectB1 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':3, 'ymax':4, 'confidence':0.7}
    objectA2 = {'class_id':0,'xmin':1, 'ymin':2, 'xmax':4, 'ymax':5, 'confidence':0.7}
    objectB2 = {'class_id':0,'xmin':3, 'ymin':4, 'xmax':4, 'ymax':5, 'confidence':0.7}
    objectC2 = {'class_id':0,'xmin':4, 'ymin':4, 'xmax':8, 'ymax':8, 'confidence':0.7}
    objectA3 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':5, 'ymax':6, 'confidence':0.7}
    objectB3 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':3, 'ymax':4, 'confidence':0.7}

    frame1 = [objectA1, objectB1]
    frame2 = [objectA2, objectB2, objectC2]
    frame3 = [objectA3, objectB3]
    tracker.updateFrame(frame1)
    tracker.updateUniqueIDs()
    print(tracker.countTotalIDs())
    tracker.increateFrameCounter()
    tracker.updateFrame(frame2)
    pairs, new, lost = tracker.findSimilarity()
    tracker.updateTrackID(pairs)
    tracker.updateUniqueIDs()
    print(tracker.countTotalIDs())
    tracker.increateFrameCounter()
    tracker.updateFrame(frame3)
    pairs, new, lost = tracker.findSimilarity()
    tracker.updateTrackID(pairs)
    tracker.updateUniqueIDs()
    print(tracker.countTotalIDs())
    print(tracker.frameHistory)