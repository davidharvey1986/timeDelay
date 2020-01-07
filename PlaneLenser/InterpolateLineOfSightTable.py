import os
import sys
import re
import math
import PlaneLenser.ReadLineOfSightPDFTable as ReadLineOfSightPDFTable
import PlaneLenser.ValueBinner as ValueBinner
import numpy as np
import matplotlib.pyplot as plt

def getPathBase(fBase):
    return os.path.join("../WesselsCodeOriginal/", "LoS_PDFs", fBase)

def getFilesAndRedshifts(fBase):
    """
    Lists all the files that match our pattern, extracts the redshift from the name, and returns the list of filenames with redshifts.
    """

    pattern = re.compile(fBase + "hpdf_z_(.*).dat", re.IGNORECASE)
    pBase = getPathBase(fBase)
    result = []
    for root, dirs, files in os.walk(pBase):
        for filename in files:
            zs = pattern.match(filename);
            if zs:
                redshift = zs.group(1)
                zAsFloat = float(redshift)
                result.append({"z" : zAsFloat, "filename" : os.path.join(pBase, filename)})

    result.sort(key = lambda entry : entry["z"])

    return result


def getValueFromPDF(pd, it, xVal):
    """
    Looks for value xVal on or after index it in pdf pd. Returns interpolated x and next value it.
    """
    if ( xVal < pd[it][0] ): # before start of list
#        print("Start of list", it, xVal, pd[it][0])
        return 0, it

    if ( xVal > pd[-1][0] ): # beyond end of list
        return 0, len(pd) - 1

    if ( xVal == pd[it][0] ): # spot on
        return pd[it][3], it

    if ( xVal > pd[it][0]): # somewhere above current point
        while ( it < len(pd) - 1) and xVal > pd[it + 1][0]:
            it += 1
        # now either xVal is within it and it + 1, or beyond the list.
        if ( it == len(pd) - 1 ):
            if xVal > pd[it][0]:
                return 0, len(pd) - 1
        
        # now it is most certainly bracketed.
        interPolValue = pd[it][3] * (pd[it + 1][0] - xVal) / (pd[it + 1][0] - pd[it][0]) + pd[it + 1][3] * (xVal - pd[it][0]) / (pd[it + 1][0] - pd[it][0])

#        print(pd[it], pd[it + 1], "xVal:", xVal, "result: ", interPolValue)
        return interPolValue, it
    return 0, it

def InterpolatePDFs(pdf1, w1, pdf2, w2):
    """
    Interpolates two pdfs linearly. Simply takes the largest range possible given the ranges of the two, and interpolates each entry. When out-of-range for either of the PDF's, zeros are substituted.
    Returns a fresh new BinThisResultHolder.
    """

    xMin = min(pdf1[0][0], pdf2[0][0])
    xMax = max(pdf1[-1][0], pdf2[-1][0])
    
    xRange = xMax - xMin

    nBin = max(len(pdf1), len(pdf2))

    print(xMin, xMax, nBin)
    
    resultList = np.zeros([nBin, 4])

    # walk three lists simultaneously, three indices
    i1 = 0 # pdf 1
    i2 = 0 # pdf 2
    ir = 0 # result

    halfBinWidth = 0.5 / (nBin - 1) * xRange
    
    for ir in range(nBin):
        x = xMin + ir / (nBin - 1) * xRange
        resultList[ir][0] = x
        resultList[ir][1] = x - halfBinWidth
        resultList[ir][2] = x + halfBinWidth
        value1, i1 = getValueFromPDF(pdf1, i1, x)
        value2, i2 = getValueFromPDF(pdf2, i2, x)
#        print("value1:", value1, "value2:", value2)
        resultList[ir][3] = w1 * value1 + w2 * value2
    
    maxVal = 0
    for i in range(len(resultList)):
        if resultList[i][3] > maxVal:
            maxVal = resultList[i][3]

    if maxVal == 0:
        resultList = []
    else:

        trimmedResultList = []
        for i in range(len(resultList)):
            if resultList[i][3] / maxVal > 0.01:
                trimmedResultList.append(resultList[i])
#            print("resultList:", resultList[i])
        resultList = trimmedResultList
#        for i in range(len(resultList)):
#            print("resultList:", resultList[i])
#    exit(0)
    return ValueBinner.BinThisResultHolder(resultList)


def InterpolateLineOfSightTable(fBase, redshift):
    """
    Looks for all files with the basename LoS_PDFs/fBase/fBasehpdf_z_ , finds two files bracketing the input redshift, and linearly interpolates between them to get your answer.
    """
    redshift = float(redshift)

    # find the entries in the list bracketing this redshift.
    fileList = getFilesAndRedshifts(fBase)

    indexInList = len(fileList) - 1;
    for i in range(len(fileList)):
        if fileList[i]["z"] > redshift:
            indexInList = i - 1
            break;

    result = None

    # the two extremes: the redshift is out of range. Just return the last / first PDF that we got.
    if indexInList == -1:

        result = ReadLineOfSightPDFTable.ReadLineOfSightPDFTable(fileList[0]["filename"])

    elif indexInList == len (fileList) - 1:

        result = ReadLineOfSightPDFTable.ReadLineOfSightPDFTable(fileList[-1]["filename"])

    else:
    
        halfResult1 = ReadLineOfSightPDFTable.ReadLineOfSightPDFTable(fileList[indexInList]["filename"])
        weight1 = (fileList[indexInList + 1]["z"] - redshift) / (fileList[indexInList + 1]["z"] - fileList[indexInList]["z"])

        halfResult2 = ReadLineOfSightPDFTable.ReadLineOfSightPDFTable(fileList[indexInList + 1]["filename"])
        weight2 = (redshift - fileList[indexInList]["z"]) / (fileList[indexInList + 1]["z"] - fileList[indexInList]["z"])

        # yes, this way weight1 == 1 - weight2

        result = InterpolatePDFs(halfResult1, weight1, halfResult2, weight2)


#    inFig = plt.figure()
#    axes = plt.axes()
#    result.show(axes)
#    plt.show()

    return result

if __name__ == "__main__":
    inFig = plt.figure()
    result = InterpolateLineOfSightTable(sys.argv[1], sys.argv[2])
    result.show(inFig)
    plt.show()
