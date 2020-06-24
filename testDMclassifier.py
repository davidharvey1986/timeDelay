

from dmModelClassifier import dmClassifier
import numpy as np
from matplotlib import pyplot as plt

def main():

    dmClassifierClass = dmClassifier()
    dmClassifierClass.getTrainingData()
    dmClassifierClass.extractPrincipalComponents()
    dmClassifierClass.generateTestAndTrainingSets()
    dmClassifierClass.getTimeDelayModel()
    score = dmClassifierClass.classifier.score( dmClassifierClass.testSet['features'],dmClassifierClass.testSet['label'])
    print(score)


def getTestData( classifier, haloNumber=0):

    x = classifier.timeDelays
    
    y = classifier.pdfArray[haloNumber,:]

    return {'x':x, 'y':y}
    

if __name__ == '__main__':
    main()
