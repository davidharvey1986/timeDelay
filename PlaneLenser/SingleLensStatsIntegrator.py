import PlaneLenser.ProcessSingleLens as ProcessSingleLens
import PlaneLenser.LensModels as LensModels
import PlaneLenser.Distances as Distances
import PlaneLenser.ValueBinner as ValueBinner
import PlaneLenser.LensOneSourcePlaneResult as LensOneSourcePlaneResult
import PlaneLenser.IntegrateStatsFromMultiplePlanes as IntegrateStatsFromMultiplePlanes
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm

# for joyplots
import joypy


import json

class SingleLensStatsIntegrator:

    def __init__(self, defs):
        self.defs = defs
        
        self.lens = LensModels.LensAndTitleFromDefs(defs)
        #DH ADDED
        self.lens.zLens = defs["zLens"]
        
        self.processor = ProcessSingleLens.SingleLensProcessor(self.lens, filters = (defs["filters"] if "filters" in defs else []))

        self.distances = Distances.Distances(zLens = defs["zLens"], zSource =  defs["zSource"], OmegaM = defs["Omega_matter"], OmegaK = 0, H0=defs["H0"])
        self.processor.distances = self.distances
        print("#", self.distances)

        self.lens.ReportEinsteinRadius(self.distances)
        
        self.nIntegrationBins = defs["integrationSteps"] if "integrationSteps" in defs else 10
        
        self.minimalKernelDelta = defs["minimalKernelDelta"] if "minimalKernelDelta" in defs else 100
        
        self.integrationWeights = self.distances.ComputeIntegrationDVolumeAndKernelForLensAndMinimalKernel(self.processor.GetMinimalDistanceKernelSurfaceToPotential(), self.nIntegrationBins, self.minimalKernelDelta)

        self.finalLensingKernel = self.distances.DistanceKernelSurfaceToPotential()
        if "fixedLensKernel" in defs:
            self.finalLensingKernel = defs["fixedLensKernel"] * processor.GetMinimalDistanceKernelSurfaceToPotential()
        
        # oversample the source plane:
        self.sourceLayout = [int((defs["sourcePlaneSampling"] if "sourcePlaneSampling" in defs else 1) * it) for it in self.lens.surface.shape]

    def NoStrongLensing(self):
        return len(self.integrationWeights) == 0

    def SinglePlaneForKernel(self, lensingKernel, show = False, showBlocking = False, showName = ""):

        if lensingKernel < 0:
            # take the maximum kernel
            lensingKernel = self.finalLensingKernel

        result = self.processor.LensOneSourcePlane(lensingKernel, sourceLayout = self.sourceLayout, bands = self.lens.bands)

        if show:
            result.ShowYourSelf(showName, block = showBlocking, fixRange = [1.e-2, 1])


        print("central source pixel:", result.resultList[result.resultList.shape[0]//2, result.resultList.shape[1]//2])

        return result

    def IntegrateStats(self):

        allRawStats = []

        for it in self.integrationWeights:
            print("Getting stats at z:", it["z"], "kernel:", it["kernel"])
            result = self.processor.LensOneSourcePlane(it["kernel"], sourceLayout = self.sourceLayout, bands = self.lens.bands)
            result.ComputeStatistics(fixRange = [1.e-2, 1])

            allRawStats.append({**{"weight" : it["weight"], "z" : it["z"]}, **result.ForRawData()})

        return allRawStats

    def MergeMultiRawStats(a, b):
        """
        Takes two sets of statistics and tries to merge them, preserving the redshift binning.
        """
#        result = []
#
#        takenFromB = [False for it in b]
#        for obj in a:
#            z = obj["z"]
#            objFromB = None
#            # dumb search in b
#            for i in range(len(b)):
#                equalZ = (z == 0 and b[i]["z"] == 0) or (abs(z - b[i]["z"]) / abs(z + b[i]["z"]) < 1e-7)
#                if equalZ:
#                    takenFromB[i] = True
#                    objFromB = b[i]
#                    break
#            if objFromB is None:
#                raise Exception("Redshift " + str(z) + " found in a, but not in b.")
#            # simply add the weights, no re-normalizing: we do not know
#            # if each entry a and b has the same number of lenses
#            # already added in them.
#            entry = { **{"weight" : obj["weight"] + objFromB["weight"], "z" : z}, ** LensOneSourcePlaneResult.LensOneSourcePlaneResult.MergeTwoForRawDatas(obj, objFromB)}
#            result.append(entry)
        if a is None:
            return b
        if b is None:
            return a
        return a + b
    
        
    
        return result
    

    def DumpAllRawStats(allRawStats, filename):
        with open(filename, "w") as f:
            json.dump(allRawStats, f)
        return

    def ReadAllRawStats(filename):
        result = {}
        with open(filename, "r") as f:
            result = json.load(f)
        return result
    
    def RawStats3D(allRawStats, fixRange = [1.e-2, 1], nBins = 30, showBlocking = False, showName = None):
        def getValues(entry, ii):
            if ii == 0:
                return entry["doubleRatios"], "Doubles"
            return entry["quadRatios"][ii - 1], "Quads " + str(ii)
            
        def _show3d(_block, indexForGet):
            fig = plt.figure()
            fig.canvas.set_window_title(getValues(allRawStats[0], indexForGet)[1] + " - " + showName)
            ax = fig.add_subplot(111, projection='3d')
            
            # make the under-the-hood y-values:
            ys = []
            yLabels = []
            y_cumulative = 0
            for it in allRawStats:
                y_cumulative += it["weight"]
                ys.append(y_cumulative)
                yLabels.append("%7.2f" % it["z"])
            
            ax.set_yticks(ys)
            ax.set_yticklabels(yLabels)
            ax.set_xlabel("Flux ratio")
            ax.set_ylabel("z")
            ax.set_zlabel("Count")
            # Log scale doesn't seem to play well.
#            ax.set_xscale("log", nonposx='clip')

            forJoyPlot = []
            firstBarx = None
            for i in range(len(allRawStats)):
                it = allRawStats[i]
                zPos = ys[i]
                thisBinned = ValueBinner.BinThis(getValues(it, indexForGet)[0], nBins = nBins, linLog = "linear", fixRange = fixRange)[0]
                barx, barw, barh = thisBinned.makeBarPlotData()
                ax.bar(barx, barh, width = barw, align = "center", zs=zPos, zdir='y', alpha=0.8)
                forJoyPlot.append(barh)
                firstBarx = barx
            fig, ax = joypy.joyplot(forJoyPlot, kind="values", x_range = [i for i in range(len(firstBarx))], xlabels = [str(it) for it in firstBarx], ylabels = yLabels, labels = yLabels, colormap=cm.autumn_r, linewidth = 0.5, figsize=(4, 4))
            fig.canvas.set_window_title(getValues(allRawStats[0], indexForGet)[1] + " - " + showName)

#            plt.show(block = _block)

            return
        
        _show3d(False, 0)
        _show3d(False, 1)
        _show3d(False, 2)
        _show3d(showBlocking, 3)

        return

    


    def AnalyseRawStats(allRawStats, nBins = 30, show = False, showBlocking = False, showName = "", fixRange = None, figAndAxarr = None, linLog = "log", LoSFileRoot = None):
        

        #DH Change this to doubleTimeDelay instead of doubleRatio
        doublesBinned = IntegrateStatsFromMultiplePlanes.BinListOfIterablesAndWeights([{"weight" : it["weight"], "H0": it["H0"], "values" : it["doubleTimeDelay"], "z" : it["z"], "zLens" : it["zLens"]} for it in allRawStats], nBins = nBins, linLog = linLog, fixRange = fixRange, LoSFileRoot = LoSFileRoot)

        quadsBinned = [ IntegrateStatsFromMultiplePlanes.BinListOfIterablesAndWeights([{"weight" : it["weight"], "H0": it["H0"], "values" : it["quadTimeDelay"][j], "z" : it["z"], "zLens" : it["zLens"]} for it in allRawStats], nBins = nBins, linLog = linLog, fixRange = fixRange)
            for j in range(3) ]

        if showName == "All combined":
            for it in allRawStats:
                print("weight", it["weight"])

        fig, axarr = None, None
        if show:
            if not figAndAxarr is None:
                fig, axarr = figAndAxarr[0], figAndAxarr[1]
            else:
                fig, axarr = plt.subplots(2, 2)
            fig.canvas.set_window_title(showName)
            fig.tight_layout() # https://stackoverflow.com/a/9827848/2295722 Very much better spacing.
            
            array = [doublesBinned] + quadsBinned
            names = ["Doubles", "Quads 1", "Quads 2", "Quads 3"]
            for x in range(2):
                for y in range(2):
                    z = x * 2 + y
                    data = array[z]

                    data.show(axarr[x, y], xlog = True, title = names[z])

#            plt.show(block = showBlocking)



        return doublesBinned, quadsBinned, [fig, axarr]




