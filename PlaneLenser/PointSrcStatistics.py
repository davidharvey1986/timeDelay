import PlaneLenser.MockLens as MockLens
import PlaneLenser.ValueBinner as ValueBinner
import numpy as np
import math

def getValueCountSimpleBrightestToSecondRatio(entry):

    if ( len(entry) < 2 ):
        return [], []

# remove central image for testing:
#    if ( len(entry) > 2 ):
#        # remove the central image as a hack
#        nonZeroCount = 0
#        distToOrigin = np.zeros(len(entry))
#        for i in range(0, len(entry)):
#            distToOrigin[i] = math.sqrt( math.pow( entry[i][0], 2) + math.pow(entry[i][1], 2) )
#            if distToOrigin[i] > 1e-12:
#                nonZeroCount += 1
#
#        if nonZeroCount > 2:
#            for i in range(0, len(entry)):
#                iMin = i % 3
#                iMax = (i + 1) % 3
#                iCent = (i + 2) % 3
#
#                d = (entry[iCent][0:2] - entry[iMin][0:2]) / (entry[iMax][0:2] - entry[iMin][0:2])
#                if (d > 0).all() and (d < 1).all():
##                    if (not iCent == 1 ):
##                        print(iCent, d, entry)
#                    entry[iCent][0] = 0
#                    entry[iCent][1] = 0
#                    entry[iCent][2] = 0
#                    break

    # first things first, which images in this entry are actual images?
    onlyHotEntries = []
    for it in entry:
        if not ( it[0] == 0. and it[1] == 0. and it[2] == 0.):
            onlyHotEntries.append(it)

    # Now watch it: critical curves are at mu^-1 == 0,
    # and that's where the caustics are (in the source plane)
    # So, our collected values here, are mu^-1 so they
    # are the inverse of the brightness.
    # -> smallest values is brightest image.
    # -> regular sort, smallest = brightest value first.
    absEntry = np.array([abs(it[2]) for it in onlyHotEntries])
    timeArrival = np.array([abs(it[3]) for it in onlyHotEntries])

    timeArrival = timeArrival[ np.argsort(timeArrival)][::-1]
    absEntry = absEntry[ np.argsort(absEntry)]

    imageCount = len(absEntry)
    if imageCount < 2:
        return [], []

    brightest_INVERSE = absEntry[0]
    second2Brightest_INVERSE = absEntry[1]

#    if brightest_INVERSE != 0 and second2Brightest_INVERSE != 0:
#        result = brightest_INVERSE / second2Brightest_INVERSE
    # remember: resultList has the magnification == *inverse* brightness
    def myRatio(a, b):
        a = abs(a)
        b = abs(b)
        if b < a:
            acp = a
            a = b
            b = acp
        if b == 0 and a == 0:
            return 1
        return a / b

    resultMag = []
    resultTime = []
    
    for i in range(1, imageCount):
        resultMag.append(myRatio(brightest_INVERSE, absEntry[i]))

        resultTime.append(timeArrival[0] - timeArrival[i])
        print("Time diff",timeArrival[0], timeArrival[i])
#    print("Pixel ratios:", result);
    return resultMag, resultTime



def CountSimpleBrightestToSecondRatio(oneSourcePlane, nBins = 100, linLog = "linear"):
    """Takes the output of SingleLensProcessor.LensOneSourcePlane, and counts the ratios of brightest to second-to-brightest magnifications."""
    
    # remember: resultList has the *inverse* magnification
    
    def IncludeThisPos(ix, iy):
        # We are only including pixels inside the largest circle that fits in the source plane.
        sx = oneSourcePlane.source2Lens(ix, 0) / (oneSourcePlane.lensLayout[0] - 1) - 0.5
        sy = oneSourcePlane.source2Lens(iy, 1) / (oneSourcePlane.lensLayout[1] - 1) - 0.5
        return sx*sx + sy*sy <= 0.25

    doubleImageRatios = []
    doubleMus = []
    
    doubleTimeDelay = []
    doubleTimes = []
    
    quadImageRatios = [[], [], []]
    quadMus = []

    quadTimeDelays = [[], [], []]
    quadTimes = []

    positionX = []
    positionY = []
    
    countMultiples = 0
    for x in range(oneSourcePlane.sourceLayout[0]):
        for y in range(oneSourcePlane.sourceLayout[1]):
            # here we select only the value inside a circle, in order not to
            # have a non-uniform weighting in the angular distance to the center
            # of the lens.
            if ( IncludeThisPos(x, y) ) :
                nextRes, nextResTime = \
                  getValueCountSimpleBrightestToSecondRatio( oneSourcePlane.resultList[x][y])
                if len(nextRes) > 0:
                    countMultiples += 1
                if len(nextRes) > 0 and abs(nextRes[0]) > 1e7:
                    print("big one at", x, y, nextRes,":\n",oneSourcePlane.resultList[x][y])
            else:
                nextRes = []
            onlyMus = [jt[2] for jt in oneSourcePlane.resultList[x][y]]
            onlyTimes = [jt[3] for jt in oneSourcePlane.resultList[x][y]]
            onlyX = [jt[0] for jt in oneSourcePlane.resultList[x][y]]
            onlyY = [jt[1] for jt in oneSourcePlane.resultList[x][y]]
            if len(nextRes) < 3 and len(nextRes) > 0:
                doubleImageRatios.append(nextRes[0])
                doubleMus.append(onlyMus)

                doubleTimeDelay.append(nextResTime[0])
                doubleTimes.append(onlyTimes)

                positionX.append(onlyX)
                positionY.append(onlyY)
                
                
            elif len(nextRes) > 2:
                quadMus.append(onlyMus)
                quadTimes.append(onlyTimes)
                for ii in range(3):
                    quadImageRatios[ii].append(nextRes[ii])
                    quadTimeDelays[ii].append(nextResTime[ii])

    print("Number of multiply projected source pixels:", countMultiples)

    return doubleImageRatios, quadImageRatios, doubleMus, quadMus, \
      doubleTimeDelay, quadTimeDelays, doubleTimes, quadTimes, positionX, positionY

def CountSimpleMagnifications(oneSourcePlane, nBins = 100, linLog = "linear"):
    """Takes the output of SingleLensProcessor.LensOneSourcePlane, and counts the ratios of brightest to second-to-brightest magnifications."""
    
    # remember: resultList has the *inverse* magnification
    
    def IncludeThisPos(ix, iy):
        sx = oneSourcePlane.source2Lens(ix, 0) / (oneSourcePlane.lensLayout[0] - 1) - 0.5
        sy = oneSourcePlane.source2Lens(iy, 1) / (oneSourcePlane.lensLayout[1] - 1) - 0.5
        return sx*sx + sy*sy <= 0.25

    allFlat = []
    for x in range(oneSourcePlane.sourceLayout[0]):
        for y in range(oneSourcePlane.sourceLayout[1]):
            for it in oneSourcePlane.resultList[x][y]:
                if not it[2] == 0.:
                    allFlat.append(1. / abs(it[2]))

    return ValueBinner.BinThis(allFlat, linLog = linLog), allFlat

def CountValues(oneSourcePlane, nBins = 100, linLog = "linear"):
    """Takes the output of SingleLensProcessor.LensOneSourcePlane, and counts the ratios of brightest to second-to-brightest magnifications."""
    
    # remember: resultList has the *inverse* magnification
    
    def IncludeThisPos(ix, iy):
        sx = oneSourcePlane.source2Lens(ix, 0) / (oneSourcePlane.lensLayout[0] - 1) - 0.5
        sy = oneSourcePlane.source2Lens(iy, 1) / (oneSourcePlane.lensLayout[1] - 1) - 0.5
        return sx*sx + sy*sy <= 0.25
    
    allFlatMagnOverBrightest = []
    allFlatrtotOverBrightest = []
    allFlatrcusp = []
    allFlatrfold = []
    for x in range(oneSourcePlane.sourceLayout[0]):
        for y in range(oneSourcePlane.sourceLayout[1]):
            # here we select only the value inside a circle, in order not to
            # have a non-uniform weighting in the angular distance to the center
            # of the lens.
            if ( IncludeThisPos(x, y) ) :
                nextRes = getValue( oneSourcePlane.resultList[x][y])
            else:
                nextRes = [None], None, None, None
            if nextRes[0]!=None:
                for el in nextRes[0]:
                    allFlatMagnOverBrightest.append(el)
            allFlatrtotOverBrightest.append(nextRes[1])
            allFlatrcusp.append(nextRes[2])
            allFlatrfold.append(nextRes[3])
    #return allFlat, 0
    #return allFlatMagnOverBrightest, allFlatrtotOverBrightest, allFlatrcusp, allFlatrfold
    if not(all(v is None for v in allFlatMagnOverBrightest)) and  not(all(v is None for v in allFlatrtotOverBrightest)) and not(all(v is None for v in allFlatrcusp)) and not(all(v is None for v in allFlatrfold)) :
        return True, ValueBinner.BinThis(allFlatMagnOverBrightest, linLog = "log"), ValueBinner.BinThis(allFlatrtotOverBrightest, linLog = "log"), ValueBinner.BinThis(allFlatrcusp, linLog = "log"), ValueBinner.BinThis(allFlatrfold, linLog = "linear") # log?????????
    else:
        print('Error message: No values for certainly allFlatrcusp')
        return False, ValueBinner.BinThis(allFlatMagnOverBrightest, linLog = "log"), ValueBinner.BinThis(allFlatrtotOverBrightest, linLog = "log"), None, None

def getValue(entry):
    if ( len(entry) < 2 ):
        return None
    firstTwo = [abs(entry[0][2]), abs(entry[1][2])]
    brightest = max(firstTwo)
    second2Brightest = min(firstTwo)

    for i in range(2, len(entry)):
        next = abs(entry[i][2])
        if next > brightest:
            second2Brightest = brightest
            brightest = next
        elif next > second2Brightest:
            second2Brightest = next

        if brightest == 0:
            brightest = 1.e-15

    result = None
    rtot = None
    rcusp = None
    rfold = None
    imgSep = []
    imgSepIndices = []
    if brightest != 0 and second2Brightest!=0: # check really 2 images min
        result = []
        for i in range(0, len(entry)):
            if entry[i][2]!=0:
                result.append(brightest/np.abs(entry[i][2])) # take abs value ??????? -> inverse axis x!!!!
                for j in range(i+1, len(entry)):
                    if entry[j][2]!=0:
                        imgSepIndices.append([i,j])
                        imgSep.append(np.sqrt((entry[i][0]-entry[j][0])**2+(entry[i][1]-entry[j][1])**2))# real distances? on lens plane???? are they usable???
        rtot = np.abs(brightest/np.sum(entry, axis=0)[2]) # abs????????
        if len(imgSep) == 6:
            lmin = np.argmin(imgSep)
            lmax = np.argmax(imgSep)
            a = np.array(imgSep)
            a[lmin] = a[lmax]
            lsecmin = np.argmin(a)
            b = np.array(imgSep)
            b[lmax] = b[lmin]
            lsecmax = np.argmax(b)
            if 0.5*imgSep[lmax] > imgSep[lmin]: # ?????????
                if 0.5*imgSep[lsecmax] > imgSep[lmin]:
                    if 0.5*imgSep[lmax] > imgSep[lsecmin]:
                        if imgSepIndices[lmin][0] in imgSepIndices[lsecmin]:
                            rcusp = np.abs((1/entry[imgSepIndices[lmin][1]][2]+1/entry[imgSepIndices[lsecmin][0]][2]+1/entry[imgSepIndices[lsecmin][1]][2])/(np.abs(1/entry[imgSepIndices[lmin][1]][2])+np.abs(1/entry[imgSepIndices[lsecmin][0]][2])+np.abs(1/entry[imgSepIndices[lsecmin][1]][2]))) # abs??????
                        if imgSepIndices[lmin][1] in imgSepIndices[lsecmin]:
                            rcusp = np.abs((1/entry[imgSepIndices[lmin][0]][2]+1/entry[imgSepIndices[lsecmin][0]][2]+1/entry[imgSepIndices[lsecmin][1]][2])/(np.abs(1/entry[imgSepIndices[lmin][0]][2])+np.abs(1/entry[imgSepIndices[lsecmin][0]][2])+np.abs(1/entry[imgSepIndices[lsecmin][1]][2])))
                    else:
                        rfold = np.abs((1/entry[imgSepIndices[lmin][0]][2]+1/entry[imgSepIndices[lmin][1]][2])/(np.abs(1/entry[imgSepIndices[lmin][0]][2])+np.abs(1/entry[imgSepIndices[lmin][1]][2]))) # ??????? sould I inverse everything since inverse magnification?????
    # remember: resultList has the *inverse* magnification
    return result, rtot, rcusp, rfold
#imgSepIndices = []
#imgSep = []
#for i in range(0, len(entry)):
#    if entry[i][2]!=0:
#        for j in range(i+1, len(entry)):
#            if entry[j][2]!=0:
#                imgSepIndices.append([i,j])
#                imgSep.append(np.sqrt((entry[i][0]-entry[j][0])**2+(entry[i][1]-entry[j][1])**2))
#if len(imgSep) == 6:
#    lmin = np.argmin(imgSep)
#    lmax = np.argmax(imgSep)
#    a = np.array(imgSep)
#    a[lmin] = a[lmax]
#    lsecmin = np.argmin(a)
#    b = np.array(imgSep)
#    b[lmax] = b[lmin]
#    lsecmax = np.argmax(b)
#    if 0.5*imgSep[lmax] > imgSep[lmin]: # ?????????
#        if 0.5*imgSep[lsecmax] > imgSep[lmin]:
#            if 0.5*imgSep[lmax] > imgSep[lsecmin]:
#                if imgSepIndices[lmin][0] in imgSepIndices[lsecmin]:
#                    rcusp = np.abs((1/entry[imgSepIndices[lmin][1]][2]+1/entry[imgSepIndices[lsecmin][0]][2]+1/entry[imgSepIndices[lsecmin][1]][2])/(np.abs(1/entry[imgSepIndices[lmin][1]][2])+np.abs(1/entry[imgSepIndices[lsecmin][0]][2])+np.abs(1/entry[imgSepIndices[lsecmin][1]][2])))
#                if imgSepIndices[lmin][1] in imgSepIndices[lsecmin]:
#                    rcusp = np.abs((1/entry[imgSepIndices[lmin][0]][2]+1/entry[imgSepIndices[lsecmin][0]][2]+1/entry[imgSepIndices[lsecmin][1]][2])/(np.abs(1/entry[imgSepIndices[lmin][0]][2])+np.abs(1/entry[imgSepIndices[lsecmin][0]][2])+np.abs(1/entry[imgSepIndices[lsecmin][1]][2])))
#            else:
#                rfold = np.abs((1/entry[imgSepIndices[lmin][0]][2]+1/entry[imgSepIndices[lmin][1]][2])/(np.abs(1/entry[imgSepIndices[lmin][0]][2])+np.abs(1/entry[imgSepIndices[lmin][1]][2])))


#if __name__ == "__main__":
#    dimBase = 400
#    lensLayout = [dimBase, dimBase]
#    sourceLayout = [dimBase, dimBase]
#    myMock = ProcessSingleLens.SingleLensProcessor(MockLens.MockLens(lensLayout, dimBase//10, dimBase//4, MockLens.W))
#    
#    stats, np = CountSimpleBrightestToSecondRatio(myMock.LensOneSourcePlane(100))
#
#    for i in stats:
#        print(i[0], i[1], i[2], i[3])
#
