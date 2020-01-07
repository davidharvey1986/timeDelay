import numpy as np
import math
import PlaneLenser.PointSrcStatistics as PointSrcStatistics
import PlaneLenser.ValueBinner as ValueBinner
import PlaneLenser.ContourCounter as ContourCounter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LensOneSourcePlaneResult:
    """
    This class is the object that is returned by SingleLensProcessor.LensOneSourcePlane.
    It holds a bunch of stuff, most importantly
    a single source plane with a full accounting of all the
    lens-plane images of each source plane pixel. Also counts the
    flux ratio statistics. And can display figures.
    Needs layout at construction, all other info is
    set by getters/setters.
    """
    # A member variable without 'self', is a static variable ( i.e., if you change it at runtime, ALL instances
    # of this class will see the changed value. So don't change it.
    entryCount = 5 # why 3??????????????????? -> got 7 max and 3 min for cluster 0 CDM EAGLE 0.25, restriction size 100
    pass
    def __init__(self, lensRef, lensLayout, sourceLayout, bands):
        self.lens = lensRef
        self.bands = bands
        self.sourceLayout = [sourceLayout[0], sourceLayout[1]]
        self.lensLayout = [lensLayout[0], lensLayout[1]]
        # [self._layout[0], self._layout[1]]
        self.lens2SourceConversion = [
            sourceLayout[0] / (self.lensLayout[0] - bands[0][0] - bands[0][1]) ,
            sourceLayout[1] / (self.lensLayout[1] - bands[1][0] - bands[1][1])
        ]

        # python sucks: https://stackoverflow.com/a/12791510/2295722
        #    resultList = [[[] * 1] * sourceLayout[1]] * sourceLayout[0]
        resultLayout = [sourceLayout[0], sourceLayout[1], self.entryCount, 4]
#                print(resultLayout)
#                self.resultList = [[[] for y in range(sourceLayout[1])] for x in range(sourceLayout[0])]
        self.resultList = np.zeros(resultLayout)
        self.muInvGrid = None
        self.timeArrivalGrid = None

        self.ownStatsComputed = 0
        self.imageCount = np.zeros(sourceLayout)

    def Set(self, sx, sy, lxly_etc):
#                v = sx == 199 and sy == 199
        v = False
        target = self.resultList[sx][sy]
        if v:
            print("with:", lxly_etc, "\nbefore:", target)
        # test if this new value has a larger muInv than the currect record.
        needToCompare = True
        for i in range(self.entryCount):
             if target[i][2] == 0:
                  for j in range(4):
                       target[i][j] = lxly_etc[j]
                  needToCompare = False
                  self.imageCount[sx][sy] = i + 1
                  break # the i loop.

        if needToCompare:
            for i in range(self.entryCount):
                 if abs(lxly_etc[2]) > abs(target[i][2]):
                      for j in range(3):
                           target[i][j] = lxly_etc[j]
                      break # the i loop.
        if v:
            print("after:", target, "\n")


#               if ( sx > 200 and sy > 200 and len(self.resultList[sx][sy]) > 1) :
#               if (sx is 22 and sy is 50) or (sx is 23 and sy is 50):
#                if len(self.resultList[sx][sy]) > 2:
#                    print("New size at ", sx, sy, ":", len(self.resultList[sx][sy]), self.resultList[sx][sy])
#                if len(self.resultList[sx][sy]) > 2:
#                    print("New size at ", sx, sy, ":", len(self.resultList[sx][sy]), self.resultList[sx][sy])

    def setMuInvGrid(self, newGrid):
        self.muInvGrid = newGrid
        
    def setTimeArrivalGrid(self, newGrid):
        self.timeArrivalGrid = newGrid

    def setLensPot(self, _lensPot):
        self.lensPot = _lensPot
    
    def setFilteredImage(self, _rho):
        self.filteredRho = _rho

    def lens2Source(self, lx, i):
        return (lx - self.bands[i][0]) * self.lens2SourceConversion[i]

    def lens2SourceScaling(self, lx, i):
        return lx * self.lens2SourceConversion[i]

    def source2Lens(self, sx, i):
        return (sx / self.lens2SourceConversion[i]) + self.bands[i][0]

    def source2LensScaling(self, sx, i):
        return (sx / self.lens2SourceConversion[i])

    def CountRatios(self):
        """
        Called after the lensing computation, by someone else, is done.
        """
        if self.ownStatsComputed > 0:
            return
        self.simpleStats_doubles, self.simpleStats_quads, \
          self.doubleImages, self.quadImages, \
          self.simpleStats_doublesTime, self.simpleStats_quadsTime, \
          self.doubleImagesTime, self.quadImagesTime,\
          self.positionX, self.positionY = \
          PointSrcStatistics.CountSimpleBrightestToSecondRatio(self)

        self.contourInfo = ContourCounter.ContourCounter(self.muInvGrid).Measure([-1e30, 0])
        
        self.ownStatsComputed = 1
        return

    def ComputeStatistics(self, fixRange = [1.e-2, 1]):
        """
        Called after the lensing computation, by someone else, is done.
        """
        if self.ownStatsComputed > 1:
            return
        if self.ownStatsComputed < 1:
            self.CountRatios()

        self.distribution = ValueBinner.BinThis(self.simpleStats_doubles, \
                                    linLog = "log", fixRange = fixRange)[0]
        print(self.simpleStats_doublesTime)
        self.distribution = ValueBinner.BinThis(self.simpleStats_doublesTime, \
                                    linLog = "log")[0]
    
        self.peakBin = 0
        for i in range(len(self.distribution)):
            if self.distribution[i][3] > self.distribution[self.peakBin][3]:
                self.peakBin = i

        self.ownStatsComputed = 2
        return

    def ForRawData(self):
        self.ComputeStatistics()
        return {"doubleRatios" : self.simpleStats_doubles, \
                "quadRatios" : self.simpleStats_quads, \
                "doubles" : self.doubleImages, \
                "quads" : self.quadImages, \
                "contours" : self.contourInfo, \
                "peakBin" : self.distribution[self.peakBin][0], \
                "doubleTimeDelay":self.simpleStats_doublesTime, \
                "quadTimeDelay":self.simpleStats_quadsTime, \
                "doublesTime":self.doubleImagesTime, \
                "quadsTime":self.quadImagesTime,\
                "positionX":self.positionX, \
                "positionY":self.positionY   }
                
                                

            
            #"resultList" : self.resultList # nope, not serializable
            

    def MergeTwoForRawDatas(a, b):
        """
        Assumes that a and b are exactly results of a call to ForRawData, no checking.
        """
        result = {}
        result["doubleRatios"] = a["doubleRatios"] + b["doubleRatios"]
        result["doubles"] = a["doubles"] + b["doubles"]
        result["quads"] = a["quads"] + b["quads"]
        result["quadRatios"] = [[], [], []]
        for i in range(3):
            result["quadRatios"][i] = a["quadRatios"][i] + b["quadRatios"][i]

        return result

    def ShowYourSelf(self, windowTitle, block = True, fixRange = None):

        if self.ownStatsComputed < 2:
            self.ComputeStatistics(fixRange = fixRange)
        ShowOneSourcePlane(self, windowTitle, block = block, fixRange = fixRange)

def ShowOneSourcePlane(self, windowTitle, block = True, fixRange = None):
    """
    Takes either a LensOneSourcePlaneResult, or a dict that just
    looks very much like LensOneSourcePlaneResult.
    """
    print("# About to construct pyplot results for one source plane. May take a moment...")
    cmap = plt.get_cmap('PiYG')
    fig, axarr = plt.subplots(3, 3)
    fig.canvas.set_window_title(self.lens.toString() + " - " + windowTitle)
    fig.tight_layout() # https://stackoverflow.com/a/9827848/2295722 Very much better spacing.


    ld = self.lens.surface
    ld_xs = [i for i in range(len(ld))]
    ld_ys = [i for i in range(len(ld[0]))]

    # plot the input density map
    axarr[0, 0].pcolormesh(ld_xs, ld_ys, ld, cmap = cmap)
    axarr[0, 0].set_title("ρ")

    # plot the determinant of the jacobian of the lens mapping
    axarr[0, 1].pcolormesh(ld_xs, ld_ys, self.muInvGrid, cmap = cmap)
    axarr[0, 1].set_title("μ⁻¹")

    myYellow = [(1, 0.8, 0.2, 1)]
    myBlues = [(0, 0, 1, 1), (0, 0, 0.7, 1), (0, 0, 0.5, 1)]


    # plot critical curves: contours where the determinant of the jacobian of the lens mapping equals 0.
    axarr[1, 1].contour(ld_xs, ld_ys, self.muInvGrid, [0], colors = myYellow)
    axarr[1, 1].set_title("μ⁻¹ == 0")


    # plot contours of the determinant of the jacobian of the lens mapping. equals 0.
    axarr[2, 1].contour(ld_xs, ld_ys, self.imageCount, [2.9, 4.9, 6.9], colors = myBlues)
    axarr[2, 1].set_title("Caustics (μ⁻¹ == 0 in source plane)")

#        # plot the lensing potential
#        axarr[1,2].pcolormesh(ld_xs, ld_ys, self.lensPot, cmap = cmap)
#        axarr[1,2].set_title("ψ")


    # plot critical curves: contours where the determinant of the jacobian of the lens mapping equals 0.
    axarr[1, 2].contour(ld_xs, ld_ys, self.muInvGrid, [0], colors = myYellow)
    # plot contours of the determinant of the jacobian of the lens mapping. equals 0.
    axarr[1, 2].contour(ld_xs, ld_ys, self.imageCount, [2.9, 4.9, 6.9], colors = myBlues)
    axarr[1, 2].set_title("Caustics and critical curves")

    # plot the filtered density map (same as input minus <rho> if there are no filters).
    axarr[0, 2].pcolormesh(ld_xs, ld_ys, self.filteredRho)
    axarr[0, 2].set_title("ρ (filtered)")

    # plot the lensed source plane: a dot for each image of a source plane pixel, drawn in the lens plane.
    forPlot = {"x" : [], "y" : [], "s" : []}
    scatterPlotSizeScale = min( max( 100 / (self.sourceLayout[0]*self.sourceLayout[0]), 0.00001 ), 10)
    for it in self.resultList:
        for kt in it:
            imageCount = 0
            for jt in kt:
                if not jt[2] == 0.:
                    imageCount += 1
                    forPlot["x"].append(jt[0])
                    forPlot["y"].append(jt[1])
                    # jt[2] = mu^-1 which is magnification and inverse brightness. Let size reflect brightness = 1/jt[2].
                    thisSize = min(abs(scatterPlotSizeScale / jt[2]), 50)
                    forPlot["s"].append(thisSize)
#            if imageCount > 1:
#                print("Double images:", kt)

    plot = axarr[2, 0].scatter(forPlot["x"], forPlot["y"], s = forPlot["s"])
    axarr[2, 0].set_title("Lensed source plane")



    # overlay lensed source plane and density map
    axarr[1, 0].pcolormesh(ld_xs, ld_ys, ld, cmap = cmap)
    axarr[1, 0].scatter(forPlot["x"], forPlot["y"], s = forPlot["s"])
    axarr[1, 0].set_title("Lensed source plane + ρ")



    barx = [it[0] for it in self.distribution]
    barw = [it[2] - it[1] for it in self.distribution]
    barh = [it[3] for it in self.distribution]
    barplot = axarr[2, 2].bar(barx, barh, width = barw, align = "center" )
    axarr[2, 2].set_title("Flux ratios")
    axarr[2, 2].set_xscale("log", nonposx='clip')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x in range(len(self.resultList)):
        if not x % 2 == 0:
            continue
        for y in range(len(self.resultList[x])):
            if not y % 2 == 0:
                continue
            countNonZeros = 0
            obj = self.resultList[x][y]
            for it in obj:
                if it[2] != 0:
                    countNonZeros += 1
            if countNonZeros == 3:
                takeThese = [obj[0], obj[1]]
                if abs(obj[2][2]) < abs(obj[0][2]):
                    takeThese[0] = obj[2]
                elif abs(obj[2][2]) < abs(obj[1][2]):
                    takeThese[1] = obj[2]
                xs = [takeThese[0][0], takeThese[1][0]]
                ys = [takeThese[0][1], takeThese[1][1]]
                zs = [takeThese[0][2], takeThese[1][2]]
                zs = [math.log(abs(it)) for it in zs]
#                    for it in obj:
#                        if it[2] == 0:
#                            break
#                        xs.append(it[0])
#                        ys.append(it[1])
#                        zs.append(abs(it[2]))

                ax.plot(xs, ys, zs = zs, alpha = 0.3)

    plt.show(block = block)
    return


