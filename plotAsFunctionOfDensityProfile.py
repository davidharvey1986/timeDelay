
from convolveDistributionWithLineOfSight import *
from astropy.io import fits
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
from powerLawFit import *

def main( ):
    '''
    Loop through each halo and get the density profile 
    and then plot the distribution as a function of the powerlaw index
    of the density profile
    '''

    
    allFiles = glob.glob('../output/CDM/z_*/B*cluster_0_*_total*.json')
    rGrid = getRadGrid()
    
    #For aesthetics                                                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=2.8, vmax=3.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    lineStyles = ['--',':','-.']
    #####

    beta = []
    betaError = []
    powerLaw = []
    powerLawError = []

    for iHalo in allFiles:
        pdf = combineJsonFiles([iHalo])

        densityProfileIndex, densityProfileIndexError = \
          getDensityProfileIndex( iHalo, rGrid=rGrid)
        powerLaw.append(densityProfileIndex)
        powerLawError.append(densityProfileIndexError)

        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = powerLawFit( pdf, yMin=1e-5, curveFit=True )
        beta.append(powerLawFitClass.params['params'][1])
        betaError.append( powerLawFitClass.params['error'][1])
        #################
        

        plt.plot(pdf['x']-pdf['x'][np.argmax(pdf['y'])],pdf['y'], color=scalarMap.to_rgba(np.abs(densityProfileIndex)))

        #if 1.14 > powerLawFitClass.params['params'][1]:
        #    pdb.set_trace()
    plt.show()

    plt.errorbar(beta,powerLaw,yerr=powerLawError, xerr=betaError, fmt='o')
    plt.show()
    
def getDensityProfileIndex( jsonFileName, rGrid=None, nRadialBins=10):
    if rGrid is None:
        rGrid = getRadGrid()
        
    radial, density = getDensityProfile( jsonFileName, rGrid=rGrid, nRadialBins=nRadialBins)

    popt, pcov = curve_fit(func, radial, density)
    error = np.sqrt(np.diag(pcov))
    index = (density[-1] - density[0])/(radial[-1] - radial[0])

    return index, error[1]

def func(x, a, b):
    return a  + x*b

def getDensityProfile( jsonFileName, rGrid=None, nRadialBins=20):
    dataDir = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections/'
    
    haloName = jsonFileName.split('/')[-1].split('_')[0]
    projection = jsonFileName.split('/')[-1].split('_')[3]
    redshift = jsonFileName.split('/')[-2]
    fitsFileName = 'cluster_0_'+projection+'_total_sph.fits'
    dataFileName = dataDir+haloName+'_EAGLE_CDM/'+redshift+'/HIRES_MAPS/'+fitsFileName
    data = fits.open(dataFileName)[0].data
    radialBins = 10**np.linspace(-1, 2, nRadialBins+1)/0.1

    density = np.zeros(nRadialBins)
    for iRadBin in range(nRadialBins):
        inBin = (rGrid  > radialBins[iRadBin]) & (rGrid < radialBins[iRadBin+1])
        area = np.pi*(radialBins[iRadBin+1]**2 - radialBins[iRadBin]**2)
        density[ iRadBin ] =  np.mean( data[inBin] ) / area

    radialBinCenters = (radialBins[:-1] + radialBins[1:])/2.
    return np.log10(radialBinCenters), np.log10(density)
if __name__ == '__main__':
    main()
