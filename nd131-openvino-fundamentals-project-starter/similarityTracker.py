import collections
import math

SHOW_CONSOLE = False
CLASS_ID_PERSON = 0

class Tracker:

    def __init__(self, numberOfFrames):
        self.currentFrame = 0
        self.trackID = 0
        self.lostTolerance = 10
        self.similaritytolerance = 0.99
        self.totalNumberOfPeople = 0

        self.frameHistory = [None] * numberOfFrames
        self.uniqueIDs = set()

    def setTolerance(self, tolerance):
        self.lostTolerance = tolerance

    def increateFrameCounter(self):
        self.currentFrame += 1

    def countPeopleInFrame(self):
        if self.frameHistory[self.currentFrame] != None:
            return len(self.frameHistory[self.currentFrame])
        else:
            return 0
        
    def updateFrame(self, objects):
        objInFrame = []
        for obj in objects:
            if(obj.get('class_id')==CLASS_ID_PERSON):
                objInFrame.append({'trackID':self.trackID, 'xmin':obj['xmin'], 'ymin':obj['ymin'], 'xmax':obj['xmax'], 'ymax': obj['ymax']})
                self.trackID += 1

        self.frameHistory[self.currentFrame] = objInFrame

    def _cosineSimilarity(self, A, B):
        Axmin = A.get('xmin')
        Axmax = A.get('xmax')
        Aymin = A.get('ymin')
        Aymax = A.get('ymax')
        Bxmin = B.get('xmin')
        Bxmax = B.get('xmax')
        Bymin = B.get('ymin')
        Bymax = B.get('ymax')
        return ( Axmin * Bxmin + Axmax * Bxmax + Aymin * Bymin + Aymax * Bymax) /           \
            math.sqrt(Axmin * Axmin + Axmax * Axmax + Aymin * Aymin + Aymax * Aymax) /      \
            math.sqrt(Bxmin * Bxmin + Bxmax * Bxmax + Bymin * Bymin + Bymax * Bymax)

    def findSimilarity(self, currentFrame, frameToBeCompared):
        currentObjects = self.frameHistory[currentFrame]
        previousObjects = self.frameHistory[frameToBeCompared]
        currentObjectIndexes = [i for i in range(len(currentObjects))]
        previousObjectIndexes = [i for i in range(len(previousObjects))]
        similarityMap = [ [None for y in range(len(previousObjects))] for x in range(len(currentObjects))]
        for x in currentObjectIndexes:
            for y in previousObjectIndexes:
                similarityMap[x][y] = self._cosineSimilarity(currentObjects[x], previousObjects[y])

        pairs = []
        Pair = collections.namedtuple("Pair", ["current", "previous", "similarity"])
        while( len(currentObjectIndexes) > 0 and len(previousObjectIndexes)>0):
            maxX = currentObjectIndexes[0]
            maxY = previousObjectIndexes[0]
            for x in currentObjectIndexes:
                for y in previousObjectIndexes:
                    if similarityMap[x][y] > similarityMap[maxX][maxY]:
                        maxX = x
                        maxY = y
            pairs.append(Pair(current = maxX, previous = maxY, similarity = similarityMap[maxX][maxY]))
            currentObjectIndexes.remove(maxX)
            previousObjectIndexes.remove(maxY)

        return pairs, currentObjectIndexes, previousObjectIndexes

    def findSimilarityWithTolerance(self):
        if self.frameHistory[self.currentFrame] == None or len(self.frameHistory[self.currentFrame]) == 0:
            return False, False, False, False

        def findIndex():
            for i in range(1, self.lostTolerance):
                if self.currentFrame - i < 0:
                    break
                if self.frameHistory[self.currentFrame - i] != None and len(self.frameHistory[self.currentFrame - i]) > 0:
                    return self.currentFrame - i
            return False
                
        foundIndex = findIndex()
        if SHOW_CONSOLE:
            print("Curr: ", self.currentFrame, "Found: ", foundIndex)

        if foundIndex is not False:
            pairs, new, lost = self.findSimilarity(self.currentFrame, foundIndex)
            return pairs, new, lost, foundIndex
        else:
            return False, False, False, False        

    def updateTrackID(self, pairs, foundIndex):
        for pair in pairs:
            if pair.similarity > self.similaritytolerance:
                trackID = self.frameHistory[foundIndex][pair.previous].get('trackID')
                self.frameHistory[self.currentFrame][pair.current]['trackID'] = trackID
                self.frameHistory[self.currentFrame][pair.current]['similarity'] = pair.similarity
                if SHOW_CONSOLE:
                    print("Updated: ", self.frameHistory[self.currentFrame][pair.current])

    def updateUniqueIDs(self):
        for obj in self.frameHistory[self.currentFrame]:
            self.uniqueIDs.add(obj.get('trackID'))
        
    def countTotalIDs(self):
        return len(self.uniqueIDs)
        
    def update(self, objects):
        self.updateFrame(objects)
        pairs, _, _, foundIndex = self.findSimilarityWithTolerance()
        if pairs is not False:
            self.updateTrackID(pairs, foundIndex)
            self.updateUniqueIDs()
        if SHOW_CONSOLE:
            print("Current Frame: ", self.frameHistory[self.currentFrame])        

    def run(self, objects):
        self.update(objects)
        counts_in_frame = self.countPeopleInFrame()
        total_counts = self.countTotalIDs()
        self.increateFrameCounter()
        return counts_in_frame, total_counts

if __name__ == '__main__':
    tracker = Tracker(100)
    objectA1 = {'class_id':0,'xmin':1, 'ymin':2, 'xmax':4, 'ymax':5, 'confidence':0.7}
    objectB1 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':3, 'ymax':4, 'confidence':0.7}
    objectA2 = {'class_id':0,'xmin':1, 'ymin':2, 'xmax':4, 'ymax':5, 'confidence':0.7}
    objectB2 = {'class_id':0,'xmin':3, 'ymin':4, 'xmax':4, 'ymax':5, 'confidence':0.7}
    objectC2 = {'class_id':0,'xmin':4, 'ymin':4, 'xmax':8, 'ymax':8, 'confidence':0.7}
    objectA3 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':5, 'ymax':6, 'confidence':0.7}
    objectB3 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':3, 'ymax':4, 'confidence':0.7}
    objectA5 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':5, 'ymax':6, 'confidence':0.7}
    objectB5 = {'class_id':0,'xmin':2, 'ymin':3, 'xmax':3, 'ymax':4, 'confidence':0.7}

    objectTest = {'xmin': 196, 'xmax': 471, 'ymin': 247, 'ymax': 428, 'class_id': 0, 'confidence': 0.7201187}
    # frame1 = [objectA1, objectB1]
    # frame2 = [objectA2, objectB2, objectC2]
    # frame3 = [objectA3, objectB3]
    
    
    frameBlank = []
    frameTestOne = [objectTest]
    frameTestTwo = [objectTest, objectA1]
    frameTestThree =  [objectA1, objectB1, objectC2]
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameTestOne)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameTestTwo)
    tracker.run(frameTestThree)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameBlank)
    tracker.run(frameTestOne)

