import PlaneLenser.InterpolateLineOfSightTable as InterpolateLineOfSightTable
import PlaneLenser.LineOfSight as LineOfSight
import PlaneLenser.ValueBinner as ValueBinner
import PlaneLenser.Distances as Distances

def BinListOfIterablesAndWeights(iterablesAndWeights, nBins = 100, linLog = "linear", fixRange = None, LoSFileRoot = None):

    nonEmptyList = [it["values"] for it in iterablesAndWeights if len(it["values"]) > 0]
    nonEmptyIterablesList = [it for it in iterablesAndWeights if len(it["values"]) > 0]
    
#    LineOfSight.ConvolveLineOfSight(CDM_doublesBinned, InterpolateLineOfSightTable.InterpolateLineOfSightTable("CDM", defs["zSource"]), nBins = nBins, fixRange = fixRange, linLog = linLog)

    if len(nonEmptyList) == 0:
        print("Got no information to bin.")
        return ValueBinner._emptyBinResult
#    else:
#        print("Got information to bin, continuing here.");

    thisRange = ValueBinner.GetMinMaxFromMany(nonEmptyList)

    if not fixRange is None:
        thisRange = fixRange

    allBins = None

    for it in nonEmptyIterablesList:

        w = it["weight"]

        lensRedshift = it["zLens"]

        myTmpDistances = Distances.Distances(0, lensRedshift, 0.272, 0, it["H0"])
        lensWeight, _, _, _ = myTmpDistances.Kernel_And_dVdZ(lensRedshift)

        # correct for the error in the weighting
        myTmpDistances = Distances.Distances(lensRedshift, it["z"], 0.272, 0, it["H0"])
        sourceWeight, _, _, wrongSourceWeight = myTmpDistances.Kernel_And_dVdZ(it["z"])


        print("HEY\n\n\n Hard-coded omega_matter = 0.272 here. Also, source plane weight:", w, "lensplane weight (z=", lensRedshift, "):", lensWeight, ", combined:", w * lensWeight / wrongSourceWeight * sourceWeight, "with bug fixing rescale", 1./ wrongSourceWeight * sourceWeight, "\n\n\n")

        # add the lens weight for the case where we integrate over lenses.
        w *= lensWeight / wrongSourceWeight * sourceWeight

        theseBins = ValueBinner.BinThis(it["values"], nBins = nBins, linLog = linLog, fixRange = thisRange )[0]
        if not LoSFileRoot is None:
            LoSTable = InterpolateLineOfSightTable.InterpolateLineOfSightTable(LoSFileRoot, it["z"])
            tmpBins = LineOfSight.ConvolveLineOfSight(theseBins, LoSTable, nBins = nBins, fixRange = fixRange, linLog = linLog, zFloat = it["z"], LoSFileRoot = LoSFileRoot)
            theseBins = tmpBins
        if allBins is None:
            for jt in theseBins:
                jt[3] *= w
            allBins = theseBins
        else:
            if not len(theseBins) == len(allBins):
                raise Exception("Bug: number of bins is not stable in BinListOfIterablesAndWeights")
            for i in range(len(theseBins)):
                allBins[i][3] += w * theseBins[i][3]

    return allBins if not allBins is None else ValueBinner._emptyBinResult

