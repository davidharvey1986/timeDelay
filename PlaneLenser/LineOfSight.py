from math import floor
import numpy as np
import PlaneLenser.ReadLineOfSightPDFTable as ReadLineOfSightPDFTable
import PlaneLenser.ValueBinner as ValueBinner

def getMinMaxNonZero(inPDF):
    """
    returns the smallest and largest entry in a PDF with nonzero weight.
    """
    minNonZero = inPDF[0][0]
    maxNonZero = inPDF[-1][0]

    for i in range(len(inPDF)):
        if inPDF[i][3] != 0.:
            minNonZero = inPDF[i][0]
            break

    for i in reversed(range(len(inPDF))):
        if inPDF[i][3] != 0.:
            maxNonZero = inPDF[i][0]
            break
    return minNonZero, maxNonZero

turboGLConvolvedCacheList = {};

def ConvolveLineOfSight(purePDF, lineOfSightPDF, nBins = 100, fixRange = None, linLog = "log", zFloat = None, LoSFileRoot = None):
    """
    Takes two BinThisResultHolders which are (1) a PDF,
    and convolves it (twice!) with (2)
    a line-of-sight-magnification PDF such as what you get for example from turboGL.
    
    The computed integral is:
    
    P(h) = \int dg_2 dg_1 g_2 / g_1 P(g_1) P(g_2) P(f = h g_2 / g_1).
    
    """

    ValueBinner.NormalizePDF(purePDF)
    ValueBinner.NormalizePDF(lineOfSightPDF)


    # Looking up values in pdfs would involve interpolation, which would be bad.
    # so we don't perform the actual integral in its bare form.
    # instead, do a 2d integral for each value in the purePDF list (f),
    # and collect the value h = f g_1 / g_2 with the weight P(h),
    # finally to a weighted binning of the found h's.

    # So, first step, create a list of same length as purePDF,
    # of values h and weights.
    
    # in order to be flexible with binning, we use ValueBinner.BinThis.
    # but in order to save memory, we first bin here with nBins^2 bins.

    internalNBins = nBins * nBins

    gMin, gMax = getMinMaxNonZero(lineOfSightPDF)
    fMin, fMax = getMinMaxNonZero(purePDF)

    valuesAndWeights = np.zeros([internalNBins, 2])
    hMin = fMin * gMin / gMax
    hMax = fMax * gMax / gMin
    hRange = hMax - hMin

    for i in range(internalNBins):
        valuesAndWeights[i][0] = hMin + ((i + 0.5) * hRange) / (internalNBins - 1)

    def putValueAndWeight(hV, hW):
        hVBin = int(floor((internalNBins * (hV - hMin)) / hRange))
        if hVBin < 0:
            hVBin = 0
        elif hVBin > internalNBins - 1:
            hVBin = internalNBins - 1
#        print("hValue:", hV, "to bin", hVBin, ", from range: ", hMin, hMax)
        valuesAndWeights[hVBin][1] += hW
    

#    FOR REFERENCE: this is the explicit double integral for the convolution, once for each value in purePDF
#    Of course, the inner double integral does not depend on purePDF, so we can precompute it.
#    for i in range(len(purePDF)):
#
#        fValue = purePDF[i][0]
#        fWeight = purePDF[i][3]
#
#        # perform the 2D integral
#        for it in lineOfSightPDF:
#            g1Value = it[0]
#            g1Weight = it[3]
#            for jt in lineOfSightPDF:
#                g2Value = jt[0]
#                g2Weight = jt[3]
#
#                hValue = fValue * g1Value / g2Value
#                hWeight = fWeight * g1Weight * g2Weight
#
##                print(hValue, hWeight, fValue, g1Value, g2Value)
#                # Now for something not irrelevant: Ratios is defined as smallest over largest, hence strictly <= 1.
#                # The convolution does allow for values > 1
#                # in which case we must invert the value...
#                if hValue > 1:
#                    hValue = 1. / hValue
#
#                putValueAndWeight(hValue, hWeight)

    doubleIntegral = np.zeros([len(lineOfSightPDF) * len(lineOfSightPDF), 2])
    
    zString = None
    if not zFloat is None and not LoSFileRoot is None:
        zString = str(round(10 * zFloat)) + LoSFileRoot
    
    if not zString is None and zString in turboGLConvolvedCacheList:
    
        doubleIntegral = turboGLConvolvedCacheList[zString]
        print("Obtained cached turboGL result for z", zString);

    else:
        doubleIntegral_index = 0;
        for it in lineOfSightPDF:
            g1Value = it[0]
            g1Weight = it[3]
            for jt in lineOfSightPDF:
                g2Value = jt[0]
                g2Weight = jt[3]
                
                g1g2Value = g1Value / g2Value
                g1g2Weight = g1Weight * g2Weight
                doubleIntegral[doubleIntegral_index][0] = g1g2Weight
                doubleIntegral[doubleIntegral_index][1] = g1g2Value
                doubleIntegral_index += 1
        if not zString is None:
            turboGLConvolvedCacheList[zString] = doubleIntegral
            print("Storing cached turboGL result for z", zString);


    print("purePDF: ", len(purePDF), "doubleIntegral:", len(doubleIntegral))

    for i in range(len(purePDF)):

        fValue = purePDF[i][0]
        fWeight = purePDF[i][3]

        # perform the 2D integral
        for it in doubleIntegral:
            hValue = fValue * it[1]
            hWeight = fWeight * it[0]
            # Now for something not irrelevant: Ratios is defined as smallest over largest, hence strictly <= 1.
            # The convolution does allow for values > 1
            # in which case we must invert the value...
            if hValue > 1:
                hValue = 1. / hValue

            putValueAndWeight(hValue, hWeight)


    returnValue, _, _ = ValueBinner.BinThis(valuesAndWeights, nBins = nBins, fixRange = fixRange, linLog = linLog)

    print("Convolved:", returnValue)
    
    print("Before binning:", valuesAndWeights)

    ValueBinner.NormalizePDF(returnValue)

    # verify that sum is 1:
    sum = 0
    for it in returnValue:
        sum += it[3] * (it[2] - it[1])

    print("Verified normalized:", sum)

    return returnValue
