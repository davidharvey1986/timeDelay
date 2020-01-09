import numpy as np
from astropy.io import  fits

def getHaloMass( filename, radialGrid=None):
    if radialGrid is None:
        radialGrid = getRadGrid()
        
    data = fits.open(filename)[0].data
    dPix = 1e-4*1e-4
    haloMass = np.sum( data[ radialGrid < 500] )*dPix
    print('Halo mass is ', haloMass  / 1e12)
    return np.log10(haloMass)

def getRadGrid():
    xy = np.arange(0,1000) - 500


    xGrid, yGrid = np.meshgrid(xy, xy)

    rGrid = np.sqrt(xGrid**2+yGrid**2)
    
    return rGrid
