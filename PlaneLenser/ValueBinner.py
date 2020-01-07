import numpy as np
import math
import re
import random
import matplotlib.pyplot as plt

def NormalizePDF(inPDF):
    # normalize
    sum = 0;
    for it in inPDF:
#        print(it)
        if it[3] < 0:
            it[3] = 0
        sum += it[3] * (it[2] - it[1])

    print("Sum before normalizing:", sum)

    if ( sum > 0 ) :
        for it in inPDF:
            it[3] /= sum

    sum = 0;
    for it in inPDF:
        sum += it[3] * (it[2] - it[1])

    print("Sum after normalizing:", sum, ", length of data:", len(inPDF))

    return sum



class NegativeRangeForLogarithmicBinning(Exception):
    pass


class AllDataAreNAN_Binning(Exception):
    pass



def GetMinMaxFromMany(listOfIterables):
    mins = min(min(it) for it in listOfIterables)
    maxs = max(max(it) for it in listOfIterables)
    return [mins, maxs]

class BinThisResultHolder:
    """
    Adds the capability to plot to a binning result.
    """
    def __init__(self, resultList):
        self.data = resultList
        self.defaultLogPlot = False

    def __getitem__(self, args):
        return self.data[args]

    def __str__(self):
        result = "# Plotting distribution:\n"
        for it in self.data:
            result += str(it[0]) + " " + str(it[3]) + "\n"
        result += "\n"
        return result
        
    def setLogarithmicDefault(self):
        self.defaultLogPlot = True

    def makeBarPlotData(self):
        barx = [it[0] for it in self.data]
        barw = [it[2] - it[1] for it in self.data]
        invNorm = sum([it[3] for it in self.data])
        normh = 1 / invNorm if invNorm != 0 else 0
        barh = [it[3] * normh for it in self.data]
        return barx, barw, barh
    
    def show(self, axis, xlog = None, title = None):
    
        if xlog is None:
            xlog = self.defaultLogPlot
    
        NormalizePDF(self.data)

#        print(self)
        barx, barw, barh = self.makeBarPlotData()
    

        axis.bar(barx, barh, width = barw, align = "center", alpha = 0.65 )
        if not title is None:
            axis.set_title(title)
        if xlog:
            axis.set_xscale("log", nonposx='clip')

    def __len__(self):
        return len(self.data)

_emptyBins = np.zeros([1, 4])
_emptyBinResult = BinThisResultHolder(_emptyBins)

def BinThis(iterable, nBins = 100, linLog = "linear", fixRange = None):
    """Returns a 1D accounting of the distribution of the values in your iterable / list. You choose the binning: logarithmic or linear, and whatever number of bins.
        Also supports values with associated weights, if your list is [[value, weight], [value, weight], ...].
        Otherwise weights are 1 if your list is [value, value, value, ...].
    """

    automaticBinRange = fixRange is None
    pattern = re.compile("log.*", re.IGNORECASE)
    logarithmic = None != pattern.match(linLog)

    # just check the first not-None entry
    hasWeights = False
    for it in iterable:
        if it is None:
            continue
        if hasattr(it, "__len__") and hasattr(it, "__getitem__"):
            hasWeights = True
        break

    def getValueFromEntry(entry):
        if hasWeights:
            return entry[0]
        return entry

    def getWeightFromEntry(entry):
        if hasWeights:
            return entry[1]
        return 1

    def getExtremum(minMax):
        # get first valid value
        noValidValues = True
        for next in iterable:
            if ( next != None ):
                isANumber = False
                if not np.isnan(getValueFromEntry(next)):
                    extr = next
                    noValidValues = False
                    break
        if ( noValidValues ):
            raise AllDataAreNAN_Binning()
        for next in iterable:
            # compare other valid values
            if ( next != None and not np.isnan(getValueFromEntry(next)) ):
                extr = minMax([extr, next])
        return extr

    largest = 0
    smallest = 0
    validData = True
    if not automaticBinRange:
        largest = fixRange[1]
        smallest = fixRange[0]
    else:
        try:
            largest = getExtremum(max)
            smallest = getExtremum(min)
        except AllDataAreNAN_Binning:
            validData = False

    # setup the bins, also if we have no data: fixed range might mean someone needs a result from us no matter what.
    binmin = math.log10(max(1e-300, smallest)) if logarithmic else smallest
    binmax = math.log10(max(1e-300, largest)) if logarithmic else largest

    if (binmax == binmin):
        binmax += 1
    bindist = 1. / (binmax - binmin)

    result = np.zeros([nBins, 4])
    totalCount = 0

    if (bindist == 0) :
    
        print("You asked for binning of data with zero range: data in [", binmax, ",", binmax, ">.")

    else:

        # set the bin centers
        for i in range(nBins):
            result[i][0] = binmin + (i + 0.5) / nBins / bindist
            result[i][1] = binmin + (i ) / nBins / bindist
            result[i][2] = binmin + (i + 1) / nBins / bindist
            if ( logarithmic ):
                for j in range(3):
                    result[i][j] = pow(10., result[i][j])

    if validData:
        # how many orders of mangitude do we span?
    #    oom = math.log10(largest / smallest)
    
#        print("logarithmic: ", logarithmic, "raw range:", smallest, largest)

        def BinForValue(value):
            if ( value != None and not np.isnan(value)):
                if ( value < 0 and logarithmic):
                    raise  NegativeRangeForLogarithmicBinning("Value: " + str(value) + ", cannot take real-valued logarithm.")
                mvalue = math.log10(value) if logarithmic else value
                rel = (mvalue - binmin) * bindist
                index = int(math.floor(rel * nBins))
                return 0 if index < 0 else nBins - 1 if index >= nBins else index
            return None

        for next in iterable:
            if getValueFromEntry(next) != None:
                nextWeight = getWeightFromEntry(next)
                totalCount += nextWeight
                bindex = BinForValue(getValueFromEntry(next))
                if bindex != None:
                    result[bindex][3] += nextWeight

        # normalize:
        if totalCount > 0:
            norm = 1. / totalCount
            for binPosAndValue in result:
                binPosAndValue[3] *= norm


    else:
        print("BinThis returns empty result")

    mainResult = BinThisResultHolder(result)
    if logarithmic  :
        mainResult.setLogarithmicDefault()

    return mainResult, totalCount, [largest, smallest]



if __name__ == "__main__":
    nTest = 100000
    testList = np.zeros(nTest)

    for i in range(nTest):
        testList[i] = abs(random.gauss(10, 1)) # values < 0 should almost never occur, but take abs so we can safely test log binning
    
    test1, t = BinThis(testList, nBins = 30, linLog = "linear")
    test2, _ = BinThis(testList, nBins = 30, linLog = "log")

    def showThis(l):
        for i in l:
            print(i[0], i[1], i[2], i[3])
        print()
        print()

    showThis(test1)
    showThis(test2)

