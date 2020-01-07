import numpy as np
import random
import PlaneLenser.CloudInCell
import matplotlib.pyplot as plt
import math
#from scipy.misc import imresize # NOPE, scipy.misc.imresize screws up the normalization. No go.
from matplotlib.colors import LogNorm
import PlaneLenser.MathematicaHorror
from astropy.io import fits



class LensModel:
    def __init__(self, dimensions, Shift = 0.5, ComovingSideLength = 1., extraSurface = -1):
        self.dims = dimensions  # dimensions [length, heigth] lens plane
        self.shift = Shift     # shift from the center of the lens plane of the central position of the lensing object
        self.comovingSideLength = ComovingSideLength   # ......????? real length of the lens plane
        
        self.indexToComovingLengthScale = self.comovingSideLength / self.dims[0]      # ?????? length one pixel -> only square domain!!!!!!??????

        self.PosObjIndex = [(self.dims[0]/2 - self.shift), (self.dims[1]/2 - self.shift)]    # central position of the lensing object

        self.PosObj = [self.PosObjIndex[0] * self.indexToComovingLengthScale, self.PosObjIndex[1] * self.indexToComovingLengthScale]    # central position of the lensing object
        print('# Center Main Lens on real sky: ', self.PosObj)
        print('# Center Main Lens on pixel plane: ', self.dims[0]/2 - self.shift, self.dims[1]/2 - self.shift)
        
        self.xi = self.GetXi()       # distances to the lens center
    
        self.surface = self.LensDensity()   # surface density of the lens
        self.bands = [[0, 0], [0, 0]]
        if extraSurface > 0:
            self.AddBands(extraSurface)


    def AddBands(self, extraSurface):
        offset = [int(self.surface.shape[0] * (extraSurface)), int(self.surface.shape[1] * ( extraSurface))]
        self.bands = [offset, offset]
        newShape = [int(self.surface.shape[0] * ( 1 + 2 * extraSurface)), int(self.surface.shape[1] * ( 1 + 2 * extraSurface))]
        newSurface = np.zeros(newShape)
        avg = np.sum(self.surface) / ( self.surface.shape[0] * self.surface.shape[1] )
        for x in range(newShape[0]):
            for y in range(newShape[1]):
                newSurface[x, y] = avg

        for x in range(self.surface.shape[0]):
            for y in range(self.surface.shape[1]):
                newSurface[x + offset[0], y + offset[1]] = self.surface[x, y]

        self.surface = newSurface
        self.comovingSideLength *= ( 1 + 2 * extraSurface)

    def ShowLens(self):
        print('Max and min of field: ',np.amax(self.surface),np.amin(self.surface))
        plt.pcolor(self.surface)
        plt.colorbar()
        plt.title('Lens surface field')
        plt.show()

    def LensDensity(self):
        surface = np.zeros(self.dims)
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                surface[x, y] = self.Value([x, y], [x * self.indexToComovingLengthScale, y * self.indexToComovingLengthScale])
        return surface

    def Value(self, posIndex, pos):
        return None

    def GetXi(self):
        xi = np.indices(self.dims)
        xi = (xi[0] * self.indexToComovingLengthScale - self.PosObj[0])**2 + (xi[1] * self.indexToComovingLengthScale - self.PosObj[1])**2
        return np.sqrt(xi)

    def EightPiGNewtonOverC2(self):
        # Wolfram Alpha:
        # https://www.wolframalpha.com/input/?i=8+pi+G+%2F+c%5E2+solar+mass+in+Mpc
        # gives 8 pi G / c^2 solar mass in Mpc
        # = 1.203×10^-18 Mpc (megaparsecs)
        # So:
        GN = 6.67408e-11 # m^3 kg^-1 s^-2
        fourPi = 4 * math.pi
        eightPi = 2 * fourPi
        c = 299792458 # m / s
        Msun = 1.98847e30 # kg
        Mpc = 3.08567758128e22 # m
        eightPiGNoverC2_m_per_kg = eightPi * GN / (c*c) # m / kg
        eightPiGNoverC2_m_per_MSun = eightPiGNoverC2_m_per_kg * Msun # m / MSun
        eightPiGNoverC2_Mpc_per_MSun = eightPiGNoverC2_m_per_MSun / Mpc # Mpc / MSun
        print("# 8 Pi G_N / c^2 = ", eightPiGNoverC2_Mpc_per_MSun, "Mpc / MSun")
#        quit()
        return eightPiGNoverC2_Mpc_per_MSun #1.203e-18 # If you use Solar mass and Mpc as mass and distance units, this is your number. 8piG/c^2

    def BetasEta(self, distances, wantEtas = False):
        thetas = self.xi / distances.Dl        # angles of positions on lens plane
        betas = np.array(thetas)    # angles of positions on source plane
        eta = betas * distances.Ds        # distances to center in source plane
        if ( wantEtas):
            return betas, eta
        return betas

    def ReportEinsteinRadius(self, distances):
        return
    
    def toString(self):
        return "super class LensModel, you should override the toString(self) method."

class SISModel(LensModel):
    """ Singular Isothermal Sphere Model"""
    
    def __init__(self, dimensions, velDisp, Shift = 0.5, ComovingSideLength = 1.):
        self.sigma = velDisp     # velocity dispersion proportional to c, i.e. sigma/c
        
        LensModel.__init__(self, dimensions, Shift, ComovingSideLength)
        
#        print('FactorSIS: ',self.factor, 'xiE: ',self.xiE)

        #self.analyticRatio, self.xiImgs = self.GetAnalyticRatio()   # analytical ratio and xi which have multiple images
    
    def ThetaEAndXiE(self, distances):
        thetaE = self.GetEinsteinRadius(distances)   # Einstein radius as an angle
        xiE = distances.Dl * thetaE             # Eistein radius as a distance
        print ("thetaE:", thetaE, "xiE:", xiE)
        return thetaE, xiE
    
    
    def GetEinsteinRadius(self, distances):
        return 4 * np.pi * self.sigma**2 * distances.Dls / distances.Ds
    
    def LensDensity(self):
        surface = 0.5*self.sigma**2/self.xi
        return surface

    def Value(self, posIndex, pos):
        xi2 = (pos[0] - self.PosObj[0])**2 + (pos[1] - self.PosObj[1])**2
        return 0.5*self.sigma**2/(np.sqrt(xi2))

    def EightPiGNewtonOverC2(self):
        # for the density, we take sigma in units of c (sigma == sigma/c^2),
        # and we omitted the 1/GN which cancels out in the Poisson equation.
        # So: poisson equation becomes, (note: 8 pi vs 4 pi, because of the surface projection of 3d density)
        # nabla^2 psi = 8 pi GN/c^2 sigma^2/(2GN) 1/theta = 4 pi sigma^2/c^2 1/theta
        # hence the normalization that multiplies the surface density that the
        # poisson solver has in its hands, is 8 pi.
        return 8.* math.pi

#    def Factor(self, distances):
#        return distances.Dl * self.GetEinsteinRadius(distances) / 2

    def GetAnalyticRatio(self, distances, sourcePlane):

        thetaE, xiE = self.ThetaEAndXiE(distances)
        
        betas = self.BetasEta(distances)

        thetaPlus = betas + thetaE
        thetaMinus = betas - thetaE
        muPlus = 1. + thetaE/betas
        muMinus = 1. - thetaE/betas
        muTot = np.abs(muPlus) + np.abs(muMinus)
        muTot[betas>thetaE] = muPlus[betas>thetaE]
        ratio = np.abs(muPlus/muMinus)
        ratio[betas>thetaE] = 0.0
        betasImg = np.array(betas)
        betasImg[betasImg>thetaE] = 0.0
        return ratio, muTot, betasImg

    def Plots(self, distances, dimSource):
        ratio, muTot, betasImg = self.GetAnalyticRatio(distances, np.zeros(dimSource))
        # ratio as function of radius analytically -> for SIS only
        plt.figure()
        plt.semilogy(betasImg[betasImg>0.0], ratio[ratio>0.0],'+')
        plt.xlabel('beta', fontsize=16)
        plt.ylabel('ratio_analytic', fontsize=16)
        plt.title(' Analyical ratio(beta)')
        # histogram of analytical ratios -> for SIS only
        plt.figure()
        plt.hist(ratio[ratio>0.0],100,log=True)
        plt.title('Analytical hist')
        # field of magnitude analytical ratios
        plt.figure()
        plt.imshow(ratio, norm=LogNorm(vmin=0.1, vmax=np.amax(ratio)))
        plt.colorbar()
        plt.title('Analytical field ratios')
        # field of muTot
        plt.figure()

        plt.imshow(muTot, norm=LogNorm(vmin=0.1, vmax=np.amax(muTot)))
        plt.colorbar()
        plt.title('Analytical muTot')
        plt.show()
        return ratio, muTot, betasImg

    def toString(self):
        return "SIS, sigma: %7.1e, resolution %i" % (self.sigma, self.dims[0])


class NSISModel(LensModel):
    """ Non-Singular Isothermal Sphere Model"""
    
    def __init__(self, dimensions, velDisp, epsilon, Shift = 0.5, ComovingSideLength = 1.):
        self.sigma = velDisp     # velocity dispersion proportional to c, i.e. sigma/c
        self.epsilon = epsilon   # The softening radius of the central singularity

        LensModel.__init__(self, dimensions, Shift, ComovingSideLength)
    
    
    def ThetaEAndXiE(self, distances):
        thetaE = self.GetEinsteinRadius(distances)   # Einstein radius as an angle
        xiE = distances.Dl * thetaE             # Eistein radius as a distance
        print ("thetaE:", thetaE, "xiE:", xiE)
        return thetaE, xiE
    
    
    def GetEinsteinRadius(self, distances):
        return 4 * np.pi * self.sigma**2 * distances.Dls / distances.Ds

    def Value(self, posIndex, pos):
        xi2 = (pos[0] - self.PosObj[0])**2 + (pos[1] - self.PosObj[1])**2
        return 0.5*self.sigma**2/(np.sqrt(xi2 + self.epsilon**2))

    def EightPiGNewtonOverC2(self):
        # for the density, we take sigma in units of c (sigma == sigma/c^2),
        # and we omitted the 1/GN which cancels out in the Poisson equation.
        # So: poisson equation becomes, (note: 8 pi vs 4 pi, because of the surface projection of 3d density)
        # nabla^2 psi = 8 pi GN/c^2 sigma^2/(2GN) 1/theta = 4 pi sigma^2/c^2 1/theta
        # hence the normalization that multiplies the surface density that the
        # poisson solver has in its hands, is 8 pi.
        return 8.* math.pi

#    def Factor(self, distances):
#        return distances.Dl * self.GetEinsteinRadius(distances) / 2

    def GetAnalyticRatio(self, distances, sourcePlane):

        thetaE, xiE = self.ThetaEAndXiE(distances)

        betas = self.xi / distances.Dl
        
        thetas4BetasDims = [self.dims[0], self.dims[1], 2]
        thetas4Betas = np.zeros(thetas4BetasDims)
        ratio = np.zeros(self.dims)
        # for non-sis, the deflection angle is no longer constant,
        # so the inversion beta(theta) -> theta(beta) is no
        # longer trivial.
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                fourPossibleThetas = MathematicaHorror.NSISThetasFromBeta(betas[x, y], self.epsilon**2, thetaE)
                if ( x == self.dims[0] // 2 ) and (y == self.dims[1] // 2):
                    print(x, y, betas[x, y], fourPossibleThetas, betas[x, y], self.epsilon**2, thetaE)
                counter = 0
                firstTheta = 0
                thetas4Betas[x, y, 0] = None
                thetas4Betas[x, y, 1] = None
                haveTwoImages = False
                for i in range(4):
                    if np.isfinite(fourPossibleThetas[i]):
                        if ( counter > 0):
                            # discard this value if it is the same as the first one we took
                            if np.isclose(firstTheta, fourPossibleThetas[i]):
                                continue
                        thetas4Betas[x, y, counter] = fourPossibleThetas[i]
                        counter += 1
                        if ( counter > 1):
                            haveTwoImages = True
                            break
                        else:
                            firstTheta = fourPossibleThetas[i]

                if haveTwoImages:
                    ratio[x, y] = 1
                else:
                    ratio[x, y] = 0

        return ratio, betas, None

    def Plots(self, distances, dimSource):
        r, etaImg = self.GetAnalyticRatio(distances, np.zeros(dimSource))
        # ratio as function of radius analytically -> for SIS only
    
        mask1 = etaImg > 0.0
        mask2 = r > 0.0
        mask = [ [mask1[x, y] and mask2[x, y] for y in range(self.dims[1])] for x in range(self.dims[0]) ]
        plt.figure()
        plt.semilogy(etaImg[mask], r[mask],'+')
        plt.xlabel('eta', fontsize=16)
        plt.ylabel('ratio_analytic', fontsize=16)
        plt.title(' Analyical ratio(eta)')

#        # histogram of ratios -> for SIS only
#        plt.figure()
#        plt.hist(r[r>0.0],100,log=True)
#        plt.title('Analytical hist')
        # field of magnitude analytical ratios
        plt.figure()
        plt.imshow(r, norm=LogNorm(vmin=0.1, vmax=np.amax(r)))
        plt.colorbar()
        plt.title('Analytical field ratios')

#        # field of muTot
#        plt.figure()
#        plt.imshow(mu, norm=LogNorm(vmin=0.1, vmax=np.amax(mu)))
#        plt.colorbar()
#        plt.title('Analytical muTot')
#        plt.show()
        return r, etaImg

    def toString(self):
        return "NSIS, sigma: %7.1e, resolution %i" % (self.sigma, self.dims[0])

class PointMassModel(LensModel): # not over
    """
    Point Mass Model: one pixel large. That's a delta function convolved with one tophat in x and one in y -> nearest point binning.
    """
    
    def __init__(self, dimensions, mass, Shift = 0.0, ComovingSideLength = 1.):
        self.mass = mass
        self.pmPixelSurface = ComovingSideLength**2 / (dimensions[0] * dimensions[1])
        self.singlePixelDensity = mass / self.pmPixelSurface
        LensModel.__init__(self, dimensions, Shift, ComovingSideLength)

    def Value(self, posIndex, pos):
#        deltas = [abs(posIndex[i] - self.PosObjIndex[i]) for i in range(len(posIndex))]
        weight = 1
        for i in range(len(posIndex)):
            if not int(math.floor(posIndex[i])) == self.PosObjIndex[i]:
                weight = 0
        if not weight == 0:
            print("Point mass at", posIndex, "with weight", weight)
        return weight * self.singlePixelDensity

    def ReportEinsteinRadius(self, distances):
        er = math.sqrt(self.EightPiGNewtonOverC2()*0.5 * self.mass * distances.Dls / (distances.Ds * distances.Dl))
        erMpc = er * distances.Dl
        erPix = erMpc / self.indexToComovingLengthScale
        print("Analytical einstein radius:\n", er, "radians\n", erMpc, "Mpc\n", erPix, "pixels\n\n")

    def Factor(self):
        return 0.0

    def toString(self):
        return "Point mass of %7.1e, centered at %.2f %.2f" % (self.mass, self.PosObjIndex[0], self.PosObjIndex[1])

class NFWModel(LensModel): # not over
    """ NFW Model"""
    
    def __init__(self, dimensions, rs=1., Shift = 0.5, ComovingSideLength = 1.):
        self.rs = rs        #scale radius
        
        LensModel.__init__(self, dimensions, Shift, ComovingSideLength)
    
    def Value(self, posIndex, pos):
        xi = np.sqrt((pos[0]-PosObj[0])**2 + (pos[1]-PosObj[1])**2)
        Xi = 0.
        if (xi<self.rs):
            Xi = 2.*self.rs/np.sqrt(self.rs*self.rs-xi*xi)*np.arctanh(np.sqrt((self.rs-xi)/(self.rs+xi)))
        elif (xi>self.rs):
            Xi = 2.*self.rs/np.sqrt(-self.rs*self.rs+xi*xi)*np.arctan(np.sqrt((-self.rs+xi)/(self.rs+xi)))
        else:
            Xi = 2.*self.rs/np.sqrt(-self.rs*self.rs+xi*xi)*np.arctan(np.sqrt((-self.rs+xi)/(self.rs+xi)))
            return self.rs**3./(-self.rs*self.rs+xi*xi)*(1.-Xi)
        return rs**3/(-rs*rs+xi*xi)*(1.-Xi)

    def Factor(self):
        return 0.0

class SersicModel(LensModel): # not over
    """ Sersic Model"""
    
    def __init__(self, dimensions, xie=1., n=1., Shift = 0.5, ComovingSideLength = 1.):
        self.xie = xie      # effective radius
        self.n = n          # Sersic shape parameter
        
        LensModel.__init__(self, dimensions, Shift, ComovingSideLength)
    
    def Value(self, posIndex, pos):
        xi = np.sqrt((pos[0]-PosObj[0])**2 + (pos[1]-PosObj[1])**2)
        if ((0.5<self.n) and (self.n<10)):
            bn = 2*self.n - 1/3 + 4/(405*self.n)
        return np.exp(bn*(1-(xi/self.xie)**self.n))
    
    def Factor(self):
        return 0.0

class FitsModel(LensModel):

    def __init__(self, defs):
        filename = defs["fits"]["file"]
        self.filename = filename
        hdulist = fits.open(filename)
        self.data_dm = hdulist[0].data
        self.head_dm = hdulist[0].header
        print("# Loaded", filename, "with layout", self.data_dm.shape)
        hdulist.close()
        if defs["fits"]["massUnit"] == "solar mass":
            pass
        else:
            raise Exception("Please, just use solar mass as mass unit in your fits file.")

        if defs["fits"]["lengthUnit"] == "Mpc/h":
            pass
        else:
            raise Exception("Please, just use Mpc/h as length units.")

        self.h = defs["H0"] * 1.e-2
    
        def getMassStatsNow():
            l = defs["comovingSideLength"]
#            print("# Computing mass, l, shape(2), sum:", l, self.data_dm.shape, np.sum(self.data_dm))
            nowPixelSurface = l * l / (self.data_dm.shape[0] * self.data_dm.shape[1])
            return "total: %7.1e M_sun, min rho: %7.1e M_Sun / (Mpc/h)^2 , max rho: %7.1e M_Sun / (Mpc/h)^2" % (np.sum(self.data_dm) * nowPixelSurface,
                     np.min(self.data_dm) * nowPixelSurface,
                     np.max(self.data_dm) * nowPixelSurface)

#        nowPixelSurface = defs["comovingSideLength"] * defs["comovingSideLength"] / (self.data_dm.shape[0] * self.data_dm.shape[1])
#        self.data_dm /= nowPixelSurface
        if "fixedSampleSize" in defs:

            massBefore = getMassStatsNow()

            newSize = defs["fixedSampleSize"]
            
            oldInvPixelSurface = np.shape(self.data_dm)[0] * np.shape(self.data_dm)[1]

            newInvPixelSurface = newSize * newSize

            self.Resize((newSize, newSize))

            # Now we resized, integrating
            # the values on all pixels.
            # But they were densities and they need to
            # become densities again. So rescale with the
            # pixel rescaling.
            pixResize = newInvPixelSurface / oldInvPixelSurface
            self.data_dm *= pixResize


            massAfter2 = getMassStatsNow()

            print("# Total mass before and after resizing:", massBefore, massAfter2)
            print("# New shape of Fits data:", self.data_dm.shape)
        else:
            print("# Total mass in fits image, min, max:", getMassStatsNow())


        print("# Min/max densities in fits file:", np.min(self.data_dm), np.max(self.data_dm))
    
    
        extraSurface = defs["extraSurface"] if "extraSurface" in defs else -1

        LensModel.__init__(self, np.shape(self.data_dm), ComovingSideLength = defs["comovingSideLength"], extraSurface = extraSurface, Shift = 0)

    def Resize(self, newLayout):
        """
        Resize the data using binning with CIC mass window function.
        For deconvolvability, we use particle sizes as large as the new
        pixel size. This means there is a big smoothing happening,
        treating the old pixels as pixels as large as oldlayout/newlayout.
        Bad. But for testing.
        Assumes periodic boundary conditions!!
        """
        oldSizes = self.data_dm.shape

        newData_dm = np.zeros(newLayout)

        def CIC(x):
            return 1 - abs(x) if x < 1 else 0

        def fpToPixPair(fp):
            # floor(x) + 1 != ceil(x), namely if ceil(x) == x == floor(x).
            return [int(math.floor(fp)), int(math.floor(fp) + 1)]
 
        def pixFix(i, iend):
            while i < 0 :
                i += iend
            while i >= iend:
                i -= iend
            return i
 
        for y in range(oldSizes[0]):
            ny = y / oldSizes[0] * newLayout[0]
            y_pix = fpToPixPair(ny)
            for x in range(oldSizes[1]):
                nx = x / oldSizes[1] * newLayout[1]
                x_pix = fpToPixPair(nx)

                for py in y_pix:
                    for px in x_pix:
                        weight = CIC(ny - py) * CIC(nx - px)
                        newData_dm[pixFix(py, newLayout[0]), pixFix(px, newLayout[1])] += weight * self.data_dm[y, x]
                        

        self.data_dm = newData_dm


    def LensDensity(self):
        return self.data_dm

    def EightPiGNewtonOverC2(self):
        """
        EightPiGNewtonOverC2 * rho must be in units (h/Mpc)^2.
        We have solar masses. Plain [EightPiGNewtonOverC2 * 1] gives
        something in units Mpc.
        If rho is solar mass / (Mpc/h)^3, we need to put
        EightPiGNewtonOverC2 * 1 in unit Mpc/h.
        """
        return LensModel.EightPiGNewtonOverC2(None) * self.h # so this thing is in unit Mpc / (h M_sun)

    def toString(self):
        return self.filename


class DMModel(LensModel): # center assume to be center of image
    """Dark Matter Model from simulations data"""
    
    def __init__(self, ComovingSideLength, Shift=0.0, filepath = '../../DwarfGals/WDM/cluster_0_total_sph.fits', RestrictionSize = False, FWHMTrue = False, Size = 100, h = 0.7): # sidelength in Mpc in the given images!! -> lets give it just in Mpc and change it after....
        
        hdulist = fits.open(filepath)
        self.data_dm = hdulist[0].data
        self.head_dm = hdulist[0].header
        hdulist.close()
        
        self.h = h
        
        LensModel.__init__(self, np.shape(self.data_dm), Shift, ComovingSideLength)
        # Now assuming that the input length was in Mpc/h.
    
        self.surface = (self.data_dm/self.h)
        self.FWHM = np.amax(self.surface)/2
        if (RestrictionSize):
            self.surface = self.surface[np.int(self.dims[0]/2-Size/2):np.int(self.dims[0]/2+Size/2),np.int(self.dims[1]/2-Size/2):np.int(self.dims[1]/2+Size/2)]
            self.dims = np.shape(self.surface)

        if (FWHMTrue):
            print(len(self.surface[self.surface>self.FWHM]))
            print("\n\nNote from WV: No, we should not limit the mass, we should still include it, but create an array of booleans that flag the lens-plane pixels whose image must NOT be included in the statistics.\n\n");
            self.surface[self.surface>self.FWHM] = self.FWHM
    
    def LensDensity(self):
       return None

    def PlotCuttedDensityOnMiddleAxes(self):
        plt.figure()
        plt.plot(self.surface[0,:])
        plt.xlabel('pos', fontsize=16)
        plt.ylabel('surface density', fontsize=16)
        plt.title(' Surface density cut for x=0')
        plt.show()

    def FWHMPlot(self):
        plt.figure()
        plt.semilogy(np.ndarray.flatten(self.xi), np.ndarray.flatten(self.surface), '+')
        plt.xlabel('radius to center', fontsize=16)
        plt.ylabel('surface density', fontsize=16)
        plt.title(' Surface density as function of radius')
        print('change?')
        plt.show()

if __name__ == "__main__":
    dim = 100
    #SIS = SISModel(dimensions=[dim,dim], zLens=0.3, zSource=0.5, velDisp=0.1, ComovingSideLength=100, Shift=0.5, epsilon=0.)
#    SIS = SISModel(dimensions=[dim,dim], velDisp=0.1, ComovingSideLength=1., Shift=0.5)
    #SIS.ShowLens()
#    ratio, muTot, betasImg = SIS.Plots(Distances(zLens=0.0118, zSource=0.0236, OmegaM = 0.3, OmegaK = 0), [dim,dim])
    #WDM = DMModel(ComovingSideLength = 0.010000, filepath = '../../DwarfGals/WDM/cluster_0_total_sph.fits') # ComovingSideLength in Mpc over h????????, Masses in solar masses
    CDM = DMModel(ComovingSideLength = 0.010000, filepath = '/Users/wesselvalkenburg/Downloads/z_0.20/HIRES_MAPS/cluster_0_total.fits', FWHMTrue = False)










def LensAndTitleFromDefs(defs):
    """
    Given your parameters, sets up the right lens and returns it.
    """

    lens = None
    if "fits" in defs:
        lens = FitsModel(defs)
    elif "SIS" in defs:
        lens = SISModel((defs["SIS"]["resolution"], defs["SIS"]["resolution"]), defs["SIS"]["velocityDispersion"], Shift = 0.5, ComovingSideLength = defs["comovingSideLength"])
    elif "pointLens" in defs:
        lens = PointMassModel((defs["pointLens"]["resolution"], defs["pointLens"]["resolution"]), defs["pointLens"]["mass"], Shift = 0, ComovingSideLength = defs["comovingSideLength"])

    return lens















