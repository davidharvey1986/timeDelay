import numpy as np
from matplotlib import pyplot as plt
from astro_tools_extra import *
from scipy.optimize import curve_fit

def main():

    densityProfiles = np.linspace(1.5, 3, 7)
    projected = []
    for iDensity in densityProfiles:
        iIndex = projectDensityProfile( iDensity)
        print iDensity, iIndex
        projected.append(iIndex)

    plt.plot(densityProfiles,projected)
    popt, pcov = curve_fit(func, densityProfiles,projected)
    plt.plot(densityProfiles, func( densityProfiles, *popt))

    print popt

    plt.show()


def projectDensityProfile( powerLawIndex, nBins=10, nPixels=600, zrange=50. ):

    radius = radius_cube( 0., 0., 0., z_range=[-zrange,zrange], nPixels=nPixels )

    zPixelSize = 2.*zrange/nPixels
    pixelSize = 2./nPixels
    pixelSize = 2./nPixels

    #plt.imshow(radius[50,:,:])
    #plt.show()
    densityProfile = radius**(-powerLawIndex)*pixelSize**3

    
    projectedDensityProfile = np.nansum(densityProfile, axis=2)*zPixelSize

    rGrid = radius_square( 0., 0., nPixels=nPixels)

    radialBins = 10**np.linspace(-1, 0, nBins+1)

    profile = np.zeros(nBins)
    centralBins = (radialBins[1:] + radialBins[:-1])/2.

    for iBin in xrange(nBins):
        inBin = (rGrid>radialBins[iBin]) & \
          (rGrid<radialBins[iBin+1])

            
        area = np.pi*(radialBins[iBin+1]**2 - radialBins[iBin]**2)
        
        profile[iBin] = np.sum( projectedDensityProfile[inBin] ) / area

        
    
    popt, pcov = curve_fit(func, np.log10(centralBins), np.log10(profile))
    error = np.sqrt(np.diag(pcov))
    index = popt[1]
    return index
    #plt.plot( np.log10(centralBins), np.log10(profile))
    #plt.show()

def func(x, a, b):
    return a  + x*b
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()
