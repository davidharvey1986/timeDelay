import PlaneLenser.ValueBinner as ValueBinner
import numpy as np


def ReadLineOfSightPDFTable(fileName = "/Users/wesselvalkenburg/Desktop/turboGL3.0/results/hpdf_1.dat", offset = 1):
    """
    Reads a table such as what turboGL outputs, parses it,
    and returns a BinThisResultHolder holding the table.
    Argument 'offset' is added to the bare value of the quantity,
    from the file (the x-axis of your pdf plot...). Default
    assumption is offset = 1.
    """
    fileContents = np.loadtxt(fileName)
    nBins = len(fileContents)
    result = np.zeros([nBins, 4])
    for i in range(nBins):
        result[i][0] = offset + fileContents[i][0]
        result[i][3] = fileContents[i][1]

    for i in range(1, nBins - 1):
        result[i][1] = 0.5 * ( result[i][0] + result[i - 1][0] )
        result[i][2] = 0.5 * ( result[i + 1][0] + result[i][0] )

    # the bins at the end are simply assumed to be symmetric */
    result[0][2] = result[1][1]
    result[0][1] = 2 * result[0][0] - result[0][2]

    result[nBins - 1][1] = result[nBins - 2][2]
    result[nBins - 1][2] = 2 * result[nBins - 1][0] - result[nBins - 1][1]

    # normalize
    sum = ValueBinner.NormalizePDF(result)
    print ("Read", fileName, ", normalized:", sum)

    returnValue = ValueBinner.BinThisResultHolder(result)

#    print(returnValue)

    return returnValue

if __name__ == "__main__":
    ReadLineOfSightTable()

