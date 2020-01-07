import lensing as l
from matplotlib import pyplot as plt
import numpy as np
import pyfits as fits
def main( m200=1e14, redshift=0.3):
    '''
    Create a 2D projected density map of an NFW, 
    with a concentration along the m-c relation
    '''


    concentration = 10.
    scaleRadius = l.profiles.nfw.scale_radius( m200, concentration, redshift)
    print scaleRadius
    dPix = 1e-4
    xVector = (np.arange(1000) - 500.)*dPix #so that dpix = 1e-4 Mpc
    yVector = (np.arange(1000) - 500.)*dPix
    xGrid, yGrid = np.meshgrid( xVector, yVector )
    radius = np.sqrt(xGrid**2 + yGrid**2)

    kappa2dim = np.zeros(radius.shape)
    
    for iDim in xrange(radius.shape[0]):

        kappa1dim = \
          l.profiles.nfw.kappa( radius[iDim,:], \
                concentration, scaleRadius, redshift)

        density = kappa1dim * l.critical_kappa( z_lens=redshift) * 1e12
        density[ np.isfinite(density) == False ] = np.max(density[ np.isfinite(density)])
        kappa2dim[iDim,:] = density

    outputfilename = r'../data/NFW_%0.2f.fits' % np.log10(m200)
    fits.writeto(  outputfilename, kappa2dim, clobber=True)
        

if __name__ == '__main__':
    main()
