import PlaneLenser.FFT_able as FFT_able
import PlaneLenser.MultiFFT_able as MultiFFT_able
import PlaneLenser.MockLens as MockLens
import numpy as np
import math
import PlaneLenser.CloudInCell as CloudInCell
import json
import PlaneLenser.rgba as rgba
import sys
import PlaneLenser.MockDensity as MockDensity
import PlaneLenser.Filters as Filters
import PlaneLenser.LensOneSourcePlaneResult as LensOneSourcePlaneResult

import cProfile, pstats, io


def CopyUnequalSizeArraysLooper(fromArray, toArray, shapeTo):
    if isinstance(shapeTo, list) and len(shapeTo) > 1:
        for i in range(shapeTo[0]):
            CopyUnequalSizeArraysLooper(fromArray[i], toArray[i], shapeTo[1:])
    else:
        imax = 0
        if isinstance(shapeTo, list):
            imax = shapeTo[0]
        else:
            imax = shapeTo
        for i in range(imax):
            fromArray[i] = toArray[i]



def CopyUnequalSizeArrays(fromArray, toArray):
    """Copy from one numpy array to another, not going out of bounds of either. Elements outside of the range of the fromArray are untouched. """
    shapeFrom = list(fromArray.shape)
    shapeTo = list(toArray.shape)
    if not (len(shapeTo) is len(shapeFrom)):
        raise "Cannot copy arrays of different dimensionality in this routine. Do your own reshape first."
    for i in range(len(shapeTo)):
        shapeTo[i] = min(shapeTo[i], shapeFrom[i])

    CopyUnequalSizeArraysLooper(fromArray, toArray, shapeTo)


def PointInsideTriangle(p, tr, epsilon = 0.):
    """Test if 2d point p lies in a triangle defined by three coordinates. Epsilon: return True for points that are only epsilon away."""
    # see https://stackoverflow.com/a/2049712/2295722
    # solve for s and t in p = p0 + (p1 - p0) * s + (p2 - p0) * t,
    # point is inside if 0 <= s <= 1 and 0 <= t <= 1 AND s + t <= 1.
    # Mathematica says: {{s -> -((-x2 y + x2 y0 + x y2 - x0 y2)/(x2 y1 - x1 y2)),
    #                     t -> -((x1 y - x1 y0 - x y1 + x0 y1)/(x2 y1 - x1 y2))}}
    # where x = p[0], x1 = (p1 - p0)[0], x2 = (p2 - p0][0], etc.
    #
    # s <= 1 if abs(-((-x2 y + x2 y0 + x y2 - x0 y2)) < abs(x2 y1 - x1 y2))
    # s >= 0 if -((-x2 y + x2 y0 + x y2 - x0 y2)/(x2 y1 - x1 y2)) >= 0,
    # but also if -((-x2 y + x2 y0 + x y2 - x0 y2) * (x2 y1 - x1 y2)) >= 0
    # last version saves a potential division by zero.

    x = p[0]
    y = p[1]

    x0 = tr[0][0]
    x1 = tr[1][0] - tr[0][0]
    x2 = tr[2][0] - tr[0][0]

    y0 = tr[0][1]
    y1 = tr[1][1] - tr[0][1]
    y2 = tr[2][1] - tr[0][1]

    onePlusEpsilon = 1 + epsilon

    sInside = (abs((-x2 * y + x2 * y0 + x * y2 - x0 * y2)) < onePlusEpsilon * abs(x2 * y1 - x1 * y2) ) and ( -((-x2 * y + x2 * y0 + x * y2 - x0 * y2) * (x2 * y1 - x1 * y2)) >= - epsilon)

    if not sInside:
        return False

    tInside = (abs((x1 * y - x1 * y0 - x * y1 + x0 * y1)) < onePlusEpsilon * abs(x2 * y1 - x1 * y2)) and ( -((x1 * y - x1 * y0 - x * y1 + x0 * y1) * (x2 * y1 - x1 * y2)) >= - epsilon)

    if not tInside:
        return False

    sAndTSumInside = abs(-(-x2 * y + x2 * y0 + x * y2 - x0 * y2) - (x1 * y - x1 * y0 - x * y1 + x0 * y1)) < onePlusEpsilon * abs(x2 * y1 - x1 * y2)

    if not sAndTSumInside:
        return False

    return True
#    result = sInside and tInside and sAndTSumInside
#    if ( x == 53 and y == 46): # or ( x == 46 and y == 53):
#        print("testing insideness of source", x, y, ": s, t, s+t ==> ", -((-x2 * y + x2 * y0 + x * y2 - x0 * y2)) / (x2 * y1 - x1 * y2), -((x1 * y - x1 * y0 - x * y1 + x0 * y1)) / (x2 * y1 - x1 * y2), -((-x2 * y + x2 * y0 + x * y2 - x0 * y2)) / (x2 * y1 - x1 * y2) + -((x1 * y - x1 * y0 - x * y1 + x0 * y1)) / (x2 * y1 - x1 * y2) - 1.)
#        if (result): print("inside corners:", tr)
#        print("Graphics[{Triangle[{")
#        firstit = True
#        for it in tr:
#            if not firstit:
#                print(", ")
#            firstit = False
#            print ( "{", it[0], ", ", it[1], "}")
#        print("}], Point[{", x, ", ", y, "}]}]")

#    return sInside and tInside and sAndTSumInside

def PointInsideFourPoints(p, corners, epsilon = 0.) :
    """The four corners define two triangles. Test if p is in either triangle. Epsilon: return True for points that are only epsilon away."""
    fcorners = corners.reshape((4, 2))
    return PointInsideTriangle(p, fcorners[:3], epsilon) or PointInsideTriangle(p, fcorners[1:], epsilon)


class DiamondCutter:
    """Class that computes the *exact* iteration ranges for successive
       lines, that carve out a diamond shape in a 2d plane. That is,
       the area between 4 arbitrary points in a plane."""
    def __init__(self, lensCorners2source):
        # Get the ordered corners: the one with smallest x, largest x, smallest y, largest y. That should always cover exactly
        # all four points: in case of overlap in one dimension,
        # points will most likely be different in another dimension.
        # If exactly equal, deal with it...
        
        self.lensCorners2source = lensCorners2source
        flatInput = lensCorners2source.reshape(4, 2)

        i_extrema = [0] * 4;
        
        sortedPoints = [[0 for j in range(2)] for k in range(4)]

        extrema = [[flatInput[0][k] for j in range(2)] for k in range(2)]
        for i in range(4):
            for j in range(2):
                sortedPoints[i][j] = flatInput[i][j]
                if flatInput[i][j] < extrema[j][0]:
                    extrema[j][0] = flatInput[i][j]
                if flatInput[i][j] > extrema[j][1]:
                    extrema[j][1] = flatInput[i][j]
        # numpy transposes a list when sorting..
#        sortedPoints = np.sort(sortedPoints, axis = 0)

        self.emptyXRange = math.ceil(extrema[0][1]) == math.ceil(extrema[0][0])

        self.emptyYRange = math.ceil(extrema[1][1]) == math.ceil(extrema[1][0])
        if not self.emptyYRange and not self.emptyXRange:
            sortedPoints = sorted(sortedPoints, key=lambda pt: pt[0])

            # now, there are two posibilities:
            # the two middle points (1, 2) are each on a
            # side of the diamond shape,
            # or they are on the same side.
            # which is it?
            
            borderLine = [ sortedPoints[3][0] - sortedPoints[0][0], sortedPoints[3][1] - sortedPoints[0][1] ]
            l1 = [ sortedPoints[1][0] - sortedPoints[0][0], sortedPoints[1][1] - sortedPoints[0][1] ]
            l2 = [ sortedPoints[2][0] - sortedPoints[0][0], sortedPoints[2][1] - sortedPoints[0][1] ]

            side1 = l1[0] * borderLine[1] - l1[1] * borderLine[0]
            side2 = l2[0] * borderLine[1] - l2[1] * borderLine[0]

            self.oneTwoOnSameSide = side1 * side2 >= 0
            
    #        if self.oneTwoOnSameSide:
    #            print("Gotcha", lensCorners2source)

    #        if sortedPoints[0][0] <= 199 and sortedPoints[3][0] >= 199:
    #            print("\n\ninput:\n", flatInput, "\nsorted:\n", sortedPoints, "\n")
            self.sortedPoints = sortedPoints

    def xRange(self):
        if self.emptyYRange or self.emptyXRange:
            return range(0, 0)
#        if self.sortedPoints[0][0] <= 199 and self.sortedPoints[3][0] >= 199:
#            print("xRange:", range(math.ceil(self.sortedPoints[0][0]), math.ceil(self.sortedPoints[3][0])))
        return range(math.ceil(self.sortedPoints[0][0]), math.ceil(self.sortedPoints[3][0]))

    def yRange(self, sx):
        if self.oneTwoOnSameSide:
            return self.yRangeSameSide(sx)
        return self.yRangeOpposingSides(sx)

    def yRangeOpposingSides(self, sx):
        # Given this x, what is the range of y?
        # compute where the line [sx, lambda] crosses the
        # two sides of the square for this part of the diamond.
        # which part of the diamond?
        # We are guaranteed to be inside the
        # xRange spanned by sortedPoints[0] and sortedPoints[3].
        # For both points sortedPoints[1] and sortedPoints[2],
        # simply test whether this sx is before or after
        # the point's x. Before? Cross the line sortedPoints[0] - sortedPoints[..].
        # after? Cross the line sortedPoints[3] - sortedPoints[..].
        yMin = None
        yMax = None
        sortedPoints = self.sortedPoints
        for p in range(1, 3):
            thisPoint = sortedPoints[p]
            otherPoint = sortedPoints[0]
            if sx > thisPoint[0]:
                otherPoint = sortedPoints[3]

            # simple: where do we cross?
            dx = otherPoint[0] - thisPoint[0]
            if dx == 0.:
                # this can happen... No line to check against!
                # But well, that would mean we are spot-on
                # walking along exactly this line.
                yMin = min(otherPoint[1], thisPoint[1])
                yMax = max(otherPoint[1], thisPoint[1])
                break
            else:

                yCross = thisPoint[1] + (sx - thisPoint[0]) * (otherPoint[1] - thisPoint[1]) / dx
                if p == 1:
                    yMin = yCross
                    yMax = yCross
                else:
                    yMin = min(yMin, yCross)
                    yMax = max(yMax, yCross)

#        if sortedPoints[0][0] <= 199 and sortedPoints[3][0] >= 199:
#            print("sx:", sx, "yRange:", range(math.ceil(yMin), math.ceil(yMax)), "from", yMin, yMax)

#            if PointInsideFourPoints([sx, math.ceil(yMin) - 1], self.lensCorners2source, 0):
#                print("BUGGER: boundaries are wrong.", [sx, math.ceil(yMin) - 1], " is inside.")
#                exit()
#            if PointInsideFourPoints([sx, math.ceil(yMax)], self.lensCorners2source, 0):
#                print("BUGGER: boundaries are wrong.", [sx, math.ceil(yMax)], " is inside.")
#                exit()

        return range(math.ceil(yMin), math.ceil(yMax))

    def yRangeSameSide(self, sx):
        # Given this x, what is the range of y?
        # compute where the line [sx, lambda] crosses the
        # two sides of the square for this part of the diamond.
        # which part of the diamond?
        # We are guaranteed to be inside the
        # xRange spanned by sortedPoints[0] and sortedPoints[3].
        # sx is in one of three ranges:
        # p0.x < p1.x < p2.x < p3.x.
        # test for intersecting the line of the current range.
        sortedPoints = self.sortedPoints

        firstLine = [sortedPoints[0], sortedPoints[3]];


        secondLine = []
        for p in range(1, 4):
            if sx < sortedPoints[p][0] or p == 3:
                secondLine = [sortedPoints[p - 1], sortedPoints[p]]
                break

        twoLines = [firstLine, secondLine]

        yMin = None
        yMax = None
        for l in range(2):
            thisPoint = twoLines[l][0]
            otherPoint = twoLines[l][1]

            # simple: where do we cross?
            dx = otherPoint[0] - thisPoint[0]
            if abs(dx) < 1e-14:
                # this can happen... No line to check against!
                # But well, that would mean we are spot-on
                # walking along exactly this line.
                yMin = min(otherPoint[1], thisPoint[1])
                yMax = max(otherPoint[1], thisPoint[1])
                break
            else:

                yCross = thisPoint[1] + (sx - thisPoint[0]) * (otherPoint[1] - thisPoint[1]) / dx
                if l == 0:
                    yMin = yCross
                    yMax = yCross
                else:
                    yMin = min(yMin, yCross)
                    yMax = max(yMax, yCross)

#        if sortedPoints[0][0] <= 199 and sortedPoints[3][0] >= 199:
#            print("sx:", sx, "yRange:", range(math.ceil(yMin), math.ceil(yMax)), "from", yMin, yMax)

#            if PointInsideFourPoints([sx, math.ceil(yMin) - 1], self.lensCorners2source, 0):
#                print("BUGGER: boundaries are wrong.", [sx, math.ceil(yMin) - 1], " is inside.")
#                exit()
#            if PointInsideFourPoints([sx, math.ceil(yMax)], self.lensCorners2source, 0):
#                print("BUGGER: boundaries are wrong.", [sx, math.ceil(yMax)], " is inside.")
#                exit()

        return range(math.ceil(yMin), math.ceil(yMax))

class SingleLensProcessor:
    """An object that takes a surface density (2D) as input,
       and computes all the lensing information about it. """

#    enableProfiling = True
    enableProfiling = False

    class NamedEntries:
        psi = 0
        gradPsi_0 = 1
        gradPsi_1 = 2
        d2Psi_d00 = 3
        d2Psi_d01 = 4
        d2Psi_d11 = 5
#        size = 6
        causticScale0 = 6
        causticScale1 = 7
        verificationImage = 8
        size = 9


    def __init__(self, mLensModel, interpolator = CloudInCell.CloudInCell, filters = []):

        self._lensModel = mLensModel;

        self._layout = [ len(mLensModel.surface), len(mLensModel.surface[0]) ]
        # provoke a crash if things aren't in order
        testValue = mLensModel.surface[self._layout[0] - 1, self._layout[1] - 1]
        verifyTestValue = float(testValue)
    
    
        self._unitWaveNumber = 2 * math.pi / self._lensModel.comovingSideLength
        
        self.filter = Filters.PolyFilters(self._layout[0], filters)
        
        self._interpolator = interpolator

        # copy the mLensModel.surface field
        self._densityField = np.array(mLensModel.surface)
        minMaxStats = [np.min(self._densityField), np.max(self._densityField)]
        self._potentialFFT = FFT_able.FFT_ableFromField(self._densityField)

        print("# FFT R2C.")
        self._potentialFFT.R2C();
        self.eightPiGNewtonOverC2 = self._lensModel.EightPiGNewtonOverC2()
        print("# Taking (inverse) derivatives in Fourier space.");
        self._potentialFFT.RodLoopWithWaveNumbers(self.RhokToPhik)
        
        # phi is now in units Mpc/h

        self._allFields = MultiFFT_able.MultiFFT_able_OneToMany(self._potentialFFT, self.NamedEntries.size)

        self._allFields.RodLoopWithWaveNumbers( self.PrepareFields )
    
        print("# FFT C2R.")
        self._allFields.C2R()
        self.recordedMinCausticScale = 1e300

        print("# Computing caustic scale at each vertex.")
        self._allFields.RodLoopWithWaveNumbers( self.PostpareFields )

        print("# Finished preparing this lens. Minimal caustic scale:", self.GetMinimalDistanceKernelSurfaceToPotential(), "Mpc/h")

        minMaxStats2 = [1e30, 0]
        for x in range(self._densityField.shape[0]):
            for y in range(self._densityField.shape[1]):
                next = self._allFields._realView[x, y, self.NamedEntries.verificationImage]
                minMaxStats2 = [min(minMaxStats2[0], next), max(minMaxStats2[1], next)]
        verificationImageRelDiff = abs( 0.5 * (minMaxStats[1] - minMaxStats[0] - minMaxStats2[1] + minMaxStats2[0])/(minMaxStats[1] - minMaxStats[0] + minMaxStats2[1] - minMaxStats2[0]) )
        if self.filter.size() < 1:
            print("# [max(rho) - min(rho)] of image before and after fft, relative difference:", verificationImageRelDiff, "That's GOOD" if verificationImageRelDiff < 1.e-15 else "That's BAD.")


#        print("at edge:", self._allFields._realView[ self._layout[0] // 100,  self._layout[1] // 100, :])

    def GetDensityRange(self):
        minDens = self._densityField[0, 0]
        maxDens = minDens
        for x in np.nditer(self._densityField):
            minDens = x if x < minDens else minDens
            maxDens = x if x > maxDens else maxDens

        return minDens, maxDens

    def GetMinimalDistanceKernelSurfaceToPotential(self):
        return self.recordedMinCausticScale
        
    def RhokToPhik(self, rhok, waveNumber):
        waveNumberMod2 = 0
        zero = True
        for k in waveNumber:
            zero = zero and k == 0
            waveNumberMod2 += k * k
        
        waveNumberMod2 *= (self._unitWaveNumber * self._unitWaveNumber)

        phik = 0
        if not zero:
            phik = - self.eightPiGNewtonOverC2 * rhok / waveNumberMod2
            phik *= self.filter(waveNumber, waveNumberMod2)
        
        # rhok was in units M_sun / (Mpc/h)^2.
        # so phik is now in units M_sun / (Mpc/h)^2 * (Mpc/h)^2 * (Mpc/h) / M_sun = (Mpc/h).

        return phik

    def PrepareFields(self, vectorOnGridVertex, waveNumbers):
        # FFTW_BACKWARD = 1, f(x) = sum_k exp( +i k x) f_k
        # => d/dx f(x) = sum_k exp( +i k x) (+i k) f_k
        
        ikx = (0 + waveNumbers[0] * 1j) * self._unitWaveNumber
        iky = (0 + waveNumbers[1] * 1j) * self._unitWaveNumber

        vectorOnGridVertex[self.NamedEntries.gradPsi_0] *= ikx
        vectorOnGridVertex[self.NamedEntries.gradPsi_1] *= iky
        
        # phi_k in Mpc/h -> gradient is dimensionless

        vectorOnGridVertex[self.NamedEntries.d2Psi_d00] *= (ikx * ikx)
        vectorOnGridVertex[self.NamedEntries.d2Psi_d01] *= (ikx * iky)
        vectorOnGridVertex[self.NamedEntries.d2Psi_d11] *= (iky * iky)

        # phi_k in Mpc/h -> dijPsi is in (h/Mpc).
        
        # Verify that the picture looks ok
        waveNumberMod2 = 0
        zero = True
        for k in waveNumbers:
            zero = zero and k == 0
            waveNumberMod2 += k * k
        
        waveNumberMod2 *= (self._unitWaveNumber * self._unitWaveNumber)
        if not zero:
            vectorOnGridVertex[self.NamedEntries.verificationImage] *= - waveNumberMod2 / self.eightPiGNewtonOverC2


    def PostpareFields(self, vectorOnGridVertex, xy):
        """If prepare exists, then afterwards you do a postpare..."""
        # for one thing, here we can compute what the scale is at which a point is exactly
        # on a caustic: mu^-1 = |delta^i_j - \partial_{i, j} psi| == 0
        d00psi = vectorOnGridVertex[self.NamedEntries.d2Psi_d00]
        d01psi = vectorOnGridVertex[self.NamedEntries.d2Psi_d01]
        d11psi = vectorOnGridVertex[self.NamedEntries.d2Psi_d11]

        # Solve[Expand[(1 - a f00) (1 - a f11) - a a f01 f10] == 0, a]

        discriminant = pow(d00psi,2) + 4*pow(d01psi,2) - 2*d00psi*d11psi + pow(d11psi,2)
        if ( discriminant < 0):
            result = [-1, -1]
        else:
            det = -pow(d01psi,2) + d00psi*d11psi
            if ( det == 0 ):
                result = [0, 0] # already a caustic right now.
            else:
                center = (d00psi + d11psi) / (2 * det)
                sqDisc = math.sqrt(discriminant) / (2 * det)
                result = [center + sqDisc, center - sqDisc]
                if ( result[0] > result[1]):
                    result = [result[1], result[0]]
                if ( result[0] < 0 and result[1] >= 0 ):
                    result = [result[1], result[0]]
#                if result[0] > 0 and result[1] > 0:
#                    print("Congratulations! Weird result: two positive values for the caustic scale:", result)
                # Verify that we did not implement mistakes:
                for scale in result:
                    term1 = (1 - scale * d00psi) * (1 - scale * d11psi)
                    term2 = scale*scale*d01psi*d01psi
                    if abs( term1 - term2) / max(1, abs(scale)) > 1e-5:
                        for scale2 in result:
                            print("# ABC formula is broken, sol ", scale2, ":", abs( (1 - scale2 * d00psi) * (1 - scale2 * d11psi) - scale2*scale2*d01psi*d01psi), "at xy", xy, "d00, d01, d11:", d00psi, d01psi, d11psi, "det:", det, "sqDisc:", sqDisc, "equation:", "(1 - scale *",  d00psi, ") * (1 - scale * ", d11psi, ") - scale*scale*", d01psi, "*", d01psi, " = ", (1 - scale2 * d00psi) * (1 - scale2 * d11psi), "-",  scale2*scale2*d01psi*d01psi," vs ", term1, " - ", term2, " = ", term1 - term2 )
                        quit()
                if result[0] > 0 and result[0] < self.recordedMinCausticScale:
                    self.recordedMinCausticScale = result[0]
    
        # d00psi in units h/Mpc, scale * d00psi is dimensionless,
        # scale is hence in units Mpc/h.
        vectorOnGridVertex[self.NamedEntries.causticScale0] = result[0]
        vectorOnGridVertex[self.NamedEntries.causticScale1] = result[1]

    def PrintSimpleReverseLensingPlane(self):

        def doit(entry, position):
            print( position[1], position[0], entry[self.NamedEntries.gradPsi_1], entry[self.NamedEntries.gradPsi_0])

        self._allFields.RodLoopWithWaveNumbers(doit)


    def LensOneSourcePlane(self, lensingKernel, sourceLayout = [], bands = [[0, 0], [0, 0]]):
        """Return a source plane, with discrete layout sourceLayout = [xSize, ySize] (default value is [], meaning you get same as the density field), with for each point in that plane a list of positions on which it appears in the image plane with associated magnification. lensingKernel is Dl Dls / Ds; the term which defines the strength of
            a lens system, in units Mpc/h.
        
            Optionally use bands, if you do not want the sourceLayout to be interpreted as laying over the entire lens plane. Positive values in bands mean that the source plane is smaller than the lense plane (ranging {bands[0,0], end - bands[0,1]} x {bands[1,0], end - bands[1,1]}) and negative values mean larger than the lens plane, in units of the lens plane indices.
        """

        if self.enableProfiling:
            pr = cProfile.Profile()
            pr.enable()

        if len(sourceLayout) < 2:
            sourceLayout = [self._layout[0], self._layout[1]]


        # The distance factor in the lensing potential:
        lensStrength = lensingKernel

        # The spatial derivatives we took with the FFT above,
        # rho is a surface density M_sun / (Mpc/h)^2
        # so 8 pi G_N / c^2 rho is in units of (h/Mpc)
        # the lens kernel Dls Dl / Ds is in units Mpc/h

        # note about dimensions of the gradient:
        # in memory, dipsi is normalized,
        # meaning it is 8piGN/c^2 k_i / k^2 rho,
        # this means it is in units of [ 8piGN/c^2 rho * Mpc/h ] = dimless.
        # lensStrength is in units (Mpc/h)
        # so oneDerivToIndexSpace must convert [lensStrength * grad] = Mpc/h to unit index.

        # Actually, we only need to know the deviation in units of index
        oneDerivToIndexSpace = [self._layout[i] / self._lensModel.comovingSideLength for i in range(2)]

#        print("Starting analysis of the full source plane with lens strength", lensStrengthFirstDeriv, lensStrengthSecondDeriv, ". poissonNorm * lensStrength * derivativeRescaling: ", poissonNorm, " * ", lensStrength, " * ", derivativeRescaling)

        result = LensOneSourcePlaneResult.LensOneSourcePlaneResult(self._lensModel, self._layout, sourceLayout, bands)

        # simply walk the entire lens plane, and see where the center of each cell came from.
        
        interpolatorRangeX = self._interpolator.range2d[0]
        interpolatorXSize = interpolatorRangeX[1] - interpolatorRangeX[0]
        interpolatorRangeY = self._interpolator.range2d[1]
        interpolatorYSize = interpolatorRangeY[1] - interpolatorRangeY[0]
        
        # if the interpolator has a range like [-2, 3], the zero lies at index 2.
        lxBase = -interpolatorRangeX[0]
        lyBase = -interpolatorRangeY[0]

        lensInfoGradIndices = [self.NamedEntries.gradPsi_0, self.NamedEntries.gradPsi_1]
        lensInfoJacIndices = [self.NamedEntries.d2Psi_d00, self.NamedEntries.d2Psi_d01, self.NamedEntries.d2Psi_d11]

        # prepare muInv on the entire grid, with the current lensStrength, so we can interpolate that.
        # Why now and not globally? Because it depends non-linearly on
        # lensStrength, which is a free parameter in this function call.
        muInvGrid = np.ndarray(tuple(self._layout))
        timeArrivalGrid = np.ndarray(tuple(self._layout))
        lensPot = np.ndarray(tuple(self._layout))
        filteredRho = np.ndarray(tuple(self._layout))
        result.setMuInvGrid(muInvGrid)
        result.setLensPot(lensPot)
        result.setFilteredImage(filteredRho)
        noNegativeMusFound = True
        for lx in range(self._layout[0]):
            for ly in range(self._layout[1]):
                # note about dimensions:
                # in memory, dijpsi is normalized,
                # meaning it is 8piGN/c^2 k_i k_j / k^2 rho,
                # this means it is in units of [ 8piGN/c^2 rho ] = (Mpc/h)^-1.
                # lensStrength is in units (Mpc/h)
                #
                d00psi = lensStrength * self._allFields._realView[lx, ly, lensInfoJacIndices[0]]
                d01psi = lensStrength * self._allFields._realView[lx, ly, lensInfoJacIndices[1]]
                d11psi = lensStrength * self._allFields._realView[lx, ly, lensInfoJacIndices[2]]
                muInv = ((1 - d00psi) * (1 - d11psi) - d01psi * d01psi)
#                if (abs(muInv) < 1e-3) :
#                   print(muInv, "=", "(1 - ", d00psi, ") * (1 - ", d11psi, ") - ", d01psi, " * ", d01psi)
                #DH I THINK I WANT TO CHANGE THIS TO TIME OF ARRIVAL
                
                #The gradient of Psi is dimensionless 
                
                
                #Psi is in units of Mpc /h

                #BARTELMMAN 2010 STATES THAT THIS
                #The GradientPsi herei
                #the defelction in one direction is. 
                deflectionX =  self._allFields._realView[lx, ly, lensInfoGradIndices[0]]
                deflectionY =  self._allFields._realView[lx, ly, lensInfoGradIndices[1]]
                totalDeflection = np.sqrt(deflectionX**2+deflectionY**2) #which ithink is in radians
              
                  
                muInvGrid[lx, ly] = muInv
                noNegativeMusFound = noNegativeMusFound and muInv >= 0
                lensPot[lx, ly] = lensStrength * self._allFields._realView[lx, ly, self.NamedEntries.psi]

                #make psi unitless from Bartlemann 2010 equation 76
                unitless = 2.*self.distances.Dls / (self.distances.Ds*self.distances.Dl)
                
                fermatPotential =  0.5*totalDeflection**2 - \
                  self._allFields._realView[lx, ly, self.NamedEntries.psi]*unitless
                  
                cInMpcPerSecond = 9.7156e-15
                timeDelayDistance = (1.+self._lensModel.zLens)*\
                  self.distances.Dl*self.distances.Ds/self.distances.Dls/cInMpcPerSecond
                
                timeArrivalGrid[lx, ly] = fermatPotential*timeDelayDistance
                
                              
                filteredRho[lx, ly] = self._allFields._realView[lx, ly, self.NamedEntries.verificationImage]
#                muInv = abs(((1 - d00psi) * (1 - d11psi) - d01psi * d01psi))
#                if ( muInv > 0 ) :
#                    muInvGrid[lx, ly] = math.log(muInv)
#                else:
#                    muInvGrid[lx, ly] = -300

        if noNegativeMusFound:
            print("Beware: no negative values of mu encountered, so probably no multiple images (unless you have an exact analytical SIS...).")
        print("# Max/min of time arrival surface:", np.max(timeArrivalGrid), np.min(timeArrivalGrid))
        print("# Max/min of mu^(-1):", np.max(muInvGrid), np.min(muInvGrid))
        print("# Max/min of lensing potential psi:", np.max(lensPot), np.min(lensPot))
        print("# lensStrength:", lensStrength)

        for lx in range(interpolatorRangeX[0], self._layout[0] - interpolatorRangeX[1]):
            print(lx)
            for ly in range(interpolatorRangeY[0], self._layout[1] - interpolatorRangeY[1]):
             
#                print(lx, ly)
                lensInfoAtXY = self._allFields._realView[ lx + interpolatorRangeX[0] : lx + interpolatorRangeX[1],
                                                          ly + interpolatorRangeY[0] : ly + interpolatorRangeY[1],
                                                          :]

                muInvSubgridAtXY = muInvGrid[lx + interpolatorRangeX[0] : lx + interpolatorRangeX[1],
                                             ly + interpolatorRangeY[0] : ly + interpolatorRangeY[1] ]

                timeArrSubgridAtXY = timeArrivalGrid[lx + interpolatorRangeX[0] : lx + interpolatorRangeX[1],
                                                ly + interpolatorRangeY[0] : ly + interpolatorRangeY[1] ]
                # now we obtained the right slice, and we are going to see which index source points (point sources!)
                # end up in the range [ix:ix+1][iy:iy+1]
                
                # this gives [[[0, 0], [0, 0]], [[0, 0], [0, 0]]] for the CIC interpolator
                lensCorners2source = np.array([[[0., 0.]] * interpolatorYSize] * interpolatorXSize)

                for dlx in range(interpolatorRangeX[0], interpolatorRangeX[1]):
                    sx = result.lens2Source(lx + dlx, 0)
                    for dly in range(interpolatorRangeY[0], interpolatorRangeY[1]):
                        sy = result.lens2Source(ly + dly, 1)
                        sxy = [sx, sy]
                        for i in range(2):
                            lensCorners2source[dlx, dly, i] = sxy[i] - result.lens2SourceScaling( oneDerivToIndexSpace[i] * lensStrength * lensInfoAtXY[dlx, dly, lensInfoGradIndices[i]], i)
                # note about dimensions:
                # in memory, dpsi/dx is normalized,
                # meaning it is 8piGN/c^2 k_i / k^2 rho,
                # this means it is in units of [ 8piGN/c^2 rho * Mpc/h ] = dimless.
                # lensStrength is in units (Mpc/h)
                # so oneDerivToIndexSpace must convert Mpc/h to unit index.

                # which range of integer values in the source-indices is
                # spanned by our range of lens-plane indices?
                sxRange = self._interpolator.FunctionRangeOnDomain(lensCorners2source[:, :, 0], np.array([[lxBase, lxBase + 2], [lyBase, lyBase + 2]]))
                syRange = self._interpolator.FunctionRangeOnDomain(lensCorners2source[:, :, 1], np.array([[lxBase, lxBase + 2], [lyBase, lyBase + 2]]))

                # we take the ceilings of the ranges, because the lowest value MUST be inside, as must be the highest value - 1.
#                sxRange = math.ceil(sxRange[0]), math.ceil(sxRange[1])
#                syRange = math.ceil(syRange[0]), math.ceil(syRange[1])

                # now we have a list of source-plane vertices, each of which has a single image
                # in the image plane (lens plane) inside the domain [[lxBase, lxBase + 2], [lyBase, lyBase + 2]]
                xyRanger = DiamondCutter(lensCorners2source)
                for sx in xyRanger.xRange():
                    # let's be less stupid, and compute the exact syRange for the current value of sx.
                    for sy in xyRanger.yRange(sx):
                        # is this actually a pixel we care about?
                        if sx < 0 or sx >= sourceLayout[0] or sy < 0 or sy >= sourceLayout[1]:
                            continue
                    
                        # is the source pixel really inside the projected lens-pixel on the source plane?
                        # Added an epsilon allowance in the calculation that tests if
                        # a point is inside. That is because the definitive test is simply
                        # if we can invert the interpolation equation.
                        # It can happen that machine precision errors, lead us to
                        # discard a point here, while the inversion would have succeeded, and
                        # vice-versa. By only letting one of the two decide, we have a bigger chance
                        # of finding the multiple images at least on one of the neighbouring pixels.
                        
#                        if ( sx == 204 and sy == 204 ):
#                            print("sx, sy", sx, sy, "inside with epsilon 0:",  PointInsideFourPoints([sx, sy], lensCorners2source, 0e0))
#                            print("sx, sy", sx, sy, "inside with epsilon -0.01:",  PointInsideFourPoints([sx, sy], lensCorners2source, -0.01))
#                        if sx == 199 and sy == 199 and lx == 199 and ly == 199:
#                            print("hoi", PointInsideFourPoints([sx, sy], lensCorners2source, 1e-8))
#                            CloudInCell.globalFeedBackHack = True
#                        else:
#                            CloudInCell.globalFeedBackHack = False

#                        if not PointInsideFourPoints([sx, sy], lensCorners2source, 1e-8):
#                            print("This should never happen anymore.");
#                            exit()
#                            continue

                        # we can invert, because we have two constraints (sx, sy) and want two parameters (lx, ly).
                        # note: the inversion gives a delta in the lens plane.
                        lensedSourceXYList = self._interpolator.Invert(lensCorners2source[:, :, 0], sx, lensCorners2source[:, :, 1], sy)
 
                        # actually, our routine PointInsideFourPoints is very reliable. So if we got no results,
                        # maybe the analytical formulae contain large cancellations. Try numerically before we
                        # really give up.
                        # This seems to really work: removes the last few spots in the image.
                        if len(lensedSourceXYList) < 1:
#                            CloudInCell.globalFeedBackHack = True
                            lensedSourceXYList = self._interpolator.InvertNumerically(lensCorners2source[:, :, 0], sx, lensCorners2source[:, :, 1], sy)
#                            CloudInCell.globalFeedBackHack = False

#                        if ( sx == 203 and sy == 204 ) or ( sx == 204 and sy == 204 ) or ( sx == 205 and sy == 204 ):
#                        if ( sx == 204 and sy == 204 ):
#                                print("lens pixel x: [", lx - interpolatorRangeX[0], ", ", lx + interpolatorRangeX[1] - 1, ")", ", lens pixel y: [", ly - interpolatorRangeY[0], ", ", ly + interpolatorRangeY[1] - 1, "). sxRange:", sxRange, "syRange:", syRange)
#                                print(" Source pixel [", sx, ", ", sy, "], might land on lens plane coordinate: [", lx, " + ", lensedSourceXYList, ", ", ly, " + ", lensedSourceXYList, "\nlensCorners2source: ", lensCorners2source, "\nmuInv:", muInv, "\n\n")
#                                CloudInCell.globalFeedBackHack = True
#                                print ("this should not be nan or inf:", self._interpolator.Invert(lensCorners2source[:, :, 0], sx, lensCorners2source[:, :, 1], sy));
#                                CloudInCell.globalFeedBackHack = False

                        # the interpolator already verified that the values are correct, returns None if anything is wrong. Otherwise, trust that these values are good.
                        if (len(lensedSourceXYList) > 0):
                            # get the magnification.
                            # We could say that d_0 psi and d_1 psi are defined by the interpolator,
                            # and hence that d_{i,j} should be taken from the interpolator.
                            # But the physics are better approximated by the actual
                            # d_{i, j} psi at the gripoints, so we interpolate those instead.
                            for lensedPos_delta in lensedSourceXYList:
    #                            d00psi = lensStrength * self._interpolator.Interpolate( lensInfoAtXY[:, :, lensInfoJacIndices[0]], lensedPos_delta)
    #                            d01psi = lensStrength * self._interpolator.Interpolate( lensInfoAtXY[:, :, lensInfoJacIndices[1]], lensedPos_delta)
    #                            d11psi = lensStrength * self._interpolator.Interpolate( lensInfoAtXY[:, :, lensInfoJacIndices[2]], lensedPos_delta)
    #
    #                            muInv = ((1 - d00psi) * (1 - d11psi) - d01psi * d01psi)
                                if np.isfinite(lensedPos_delta[0]) and np.isfinite(lensedPos_delta[1]):
                                    muInv = self._interpolator.Interpolate( muInvSubgridAtXY, lensedPos_delta)
                                    timeArrival = self._interpolator.Interpolate( timeArrSubgridAtXY, lensedPos_delta)
                                    
                                    if abs(muInv) < 1e-9:
                                        print("Got a tiny 1/mu:", muInv, "at interpolated distance:", lensedPos_delta, "on grid:", muInvSubgridAtXY)
    #                                result.Set(sx, sy,[lx + lensedPos_delta[0], ly + lensedPos_delta[1], math.exp(muInv)])
                                    result.Set(sx, sy,[lx + lensedPos_delta[0], ly + lensedPos_delta[1], muInv, timeArrival])

#                                if ( sx == 203 and sy == 204 ) or ( sx == 205 and sy == 204 ) or ( sx == 204 and sy == 204 )  :
#                                if ( sx == 204 and sy == 204 )  :
#                                    print("lens pixel x: [", lx - interpolatorRangeX[0], ", ", lx + interpolatorRangeX[1] - 1, ")", ", lens pixel y: [", ly - interpolatorRangeY[0], ", ", ly + interpolatorRangeY[1] - 1, "). sxRange:", sxRange, "syRange:", syRange)
#                                    print(" Source pixel [", sx, ", ", sy, "], lands on lens plane coordinate: [", lx + lensedPos_delta[0], ", ", ly + lensedPos_delta[1], "]\ndeltas in lens plane (inverted interpolation): ", lensedPos_delta[0], lensedPos_delta[1], "\nlensCorners2source: ", lensCorners2source, "\nlensInfoAtXY: ", lensInfoAtXY[dlx, dly, tuple(lensInfoGradIndices)], "\nmuInv:", muInv, "\nmuInvSubgridAtXY:", muInvSubgridAtXY,  "\n\n")

#                break
#            break

        if self.enableProfiling:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        return result

    def DrawOneSourcePlane(self, lensStrength = 1, lensColor = [0.22, 0.6, 1, 0.5], sourceColor = [1, 0, 0, 0.9], imageColor = [1, 0.76, 0.23, 0.9], partOfN = None, sourceSize = 0.05):
        """Animates a circular source moving behind this lens, based on the output of this.LensOneSourcePlane."""
        
        print("Starting animation.")
        
        srcRadius = int( self._layout[0] * sourceSize);
        srcRadius2 = srcRadius * srcRadius
        softRadius = srcRadius // 3
        hardRadius = srcRadius - softRadius
        hardRadius2 = hardRadius * hardRadius
        imgXMax = self._layout[0] - 2 * srcRadius
        imgYMax = self._layout[1] - 2 * srcRadius
#        numImgs = imgXMax * imgYMax

        lensColor = rgba.RGBAPixel(lensColor)
        sourceColor = rgba.RGBAPixel(sourceColor)
        imageColor = rgba.RGBAPixel(imageColor)
        white = rgba.RGBAPixel([1, 1, 1, 1])

        minDens, maxDens = self.GetDensityRange()
        densScale = maxDens - minDens
        densScale = 1 if densScale == 0 else densScale

        lensInfoGradIndices = [self.NamedEntries.gradPsi_0, self.NamedEntries.gradPsi_1]
        lensInfoJacIndices = [self.NamedEntries.d2Psi_d00, self.NamedEntries.d2Psi_d01, self.NamedEntries.d2Psi_d11]

        # create an image:
        # (lens) density in color a
        # source in color b
        # image in color c
    
        def CleanSlate():
            # put the density
            img = rgba.RGBAImage(self._layout)
            for lx in range(self._layout[0]):
                for ly in range(self._layout[1]):
                    nextD = (self._densityField[lx, ly] - minDens) / densScale
                    img[lx, ly].SetTo(nextD * lensColor)
            return img

        # Prepare the source template
        def GetSourcePixel(dx, dy,  color):
#            for dx in range(-srcRadius, srcRadius + 1):
#                for dy in range(-srcRadius, srcRadius + 1):
            result = None
            r2 = dx * dx + dy * dy
            if dx*dx + dy*dy < srcRadius2 :
                r = math.sqrt(r2)
                if ( dx*dx + dy*dy > hardRadius2 ):
                    weight = max(0,  (srcRadius - r ) / softRadius);
                    result = color *  weight
                else:
                    # max => and, min => or
                    distFromGalaxy = max([
                        r - hardRadius * 0.9,
                        r - hardRadius * 0.9,
                        min([(abs(dy / hardRadius - 0.5 * math.sin(math.pi * pow(abs(dx / hardRadius), 0.5 ) * np.sign(dx) ) )) - 0.1,
                            pow(dx / hardRadius, 2) + pow(dy / hardRadius, 2) - 0.1
                        ])
                    ])
#                    if abs(dx) < hardRadius and abs(dy) < hardRadius and (abs(dy / hardRadius - 0.5 * math.sin(math.pi * pow(abs(dx / hardRadius), 0.5 ) * np.sign(dx) ) ) < 0.1 or pow(dx / hardRadius, 2) + pow(dy / hardRadius, 2) < 0.15) :
                    if ( distFromGalaxy < 0):
                        result = white
                    elif ( distFromGalaxy < 0.1):
                        # remember: + is alpha blending, no need to re-scale the background color!
                        result = white * (1 - abs(10 * distFromGalaxy)) + color
                    else:
                        result = color
            return result


#        for sx in range(srcRadius, srcRadius + imgXMax):
#            for sy in range(srcRadius, srcRadius + imgYMax):
        for spx in range(srcRadius + imgXMax//2, srcRadius + imgXMax//2 + 1):
            if ( partOfN != None):
                yrange = partOfN
            else:
                yrange = range(srcRadius, srcRadius + imgYMax)
            for spy in yrange:

                print(spx, spy)

                nextImg = CleanSlate()
                
                for lx in range(self._layout[0]):
                    for ly in range(self._layout[1]):
                
                        # where are we? Where did we come from?
                        lensInfoAtXY = self._allFields._realView[ lx, ly ]
            
                        sx = lx - lensStrength * lensInfoAtXY[lensInfoGradIndices[0]]
                        sy = ly - lensStrength * lensInfoAtXY[lensInfoGradIndices[1]]

                        # The drawing: is the source here?
                        spdx = spx - lx
                        spdy = spy - ly
                        srcPixel = GetSourcePixel(spdx, spdy, sourceColor)
                        if ( srcPixel != None):
                            nextImg[lx, ly] += srcPixel # behind!

                        # the drawing: does this light ray originate from somewhere near the source?
                        lnsPixel = GetSourcePixel(spx - sx, spy - sy, imageColor)
                        if ( lnsPixel != None):
                            d00psi = lensInfoAtXY[lensInfoJacIndices[0]]
                            d01psi = lensInfoAtXY[lensInfoJacIndices[1]]
                            d11psi = lensInfoAtXY[lensInfoJacIndices[2]]

                            muInv = ((1 - d00psi) * (1 - d11psi) - d01psi * d01psi)
                            if ( muInv == 0 ) :
                                muInv = 1.e-3 # this is the drawing routine, don't get scared: this limiting does not end up in physics
                            weight = 1. / abs(muInv) # yes, abs. Magnification is scalar, not caring about orientation. Negative mu means a flipped image!
                            # Note: screen brightness of a pixel is not linearly related to the rgb value,
                            # but is given by: r_brightness = r_value^2.2
                            weigt = pow(weight, 1/2.2)
                            nextImg[lx, ly] = weight * lnsPixel + nextImg[lx, ly] # before!


                                
#                        print(dx, dy, imagedTemplate[srcRadius + dx, srcRadius + dy], nextImg[sx + dx, sy + dy])

                nextImg.WritePNG("/Users/wesselvalkenburg/Desktop/pics2/image_" + "{:04d}".format(spx) +"_{:04d}.png".format(spy), background=[0, 0, 0, 1])


if __name__ == "__main__":
    dimBase = 200
    lensLayout = [dimBase, dimBase]
    sourceLayout = [2*dimBase, 3*dimBase]
    #myMock = SingleLensProcessor(MockLens.MockLens(lensLayout, dimBase//10, dimBase//4, MockLens.W))
    myMock = SingleLensProcessor(MockDensity.LensDensity(lensLayout, MockDensity.SIS))
#    myMock.PrintSimpleReverseLensingPlane()

#    myMock = SingleLensProcessor(MockLens.MockLens([dimBase, dimBase], dimBase//10, 0, MockLens.I))

#    myMock = SingleLensProcessor(MockLens.MockLens([12, 12], 3, 3))

    if ( len(sys.argv) > 2):
        multiImageTot = lensLayout[1] // 2
        multiImageStart = lensLayout[1] // 4
        thID = int(sys.argv[1])
        thNum = int(sys.argv[2])
        startOfN = multiImageStart + (multiImageTot * thID) // thNum
        endOfN = multiImageStart + (multiImageTot * (thID + 1)) // thNum
        if ( thID == thNum - 1 ):
            endOfN = multiImageStart + multiImageTot
        partOfN = range(startOfN, endOfN)
    else:
        partOfN = range(lensLayout[0] // 4, 3 * lensLayout[0] // 4)
    myMock.DrawOneSourcePlane(lensStrength = 100, sourceSize = 0.1, partOfN = partOfN)


