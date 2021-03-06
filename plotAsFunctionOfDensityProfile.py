
from convolveDistributionWithLineOfSight import *
from astropy.io import fits
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
from powerLawFit import *
import generateDifferentSelectionFunctions as selectFunct
import matplotlib
font = { 'size'   : 15}

matplotlib.rc('font', **font)

def main( ):
    '''
    Loop through each halo and get the density profile 
    and then plot the distribution as a function of the powerlaw index
    of the density profile
    '''
    fig = plt.figure(figsize=(7,7))

    
    allFiles = glob.glob('../output/CDM/z_0.*/B*cluster_0_*_total*.json')
    
    #For aesthetics                                                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=2.8, vmax=3.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    lineStyles = ['--',':','-.']
    #####

    tPeak = []
    tPeakError = []
    beta = []
    betaError = []
    powerLaw = []
    powerLawError = []
    nHalos = []
    RMS = []
    zLens = []
    allDistributionsPklFile = \
              "../output/CDM/selectionFunction/"+\
              "sparselyPopulatedParamSpace.pkl"

    allDistributions = \
      pkl.load(open(allDistributionsPklFile,'rb'))
    fiducialCosmology = \
          {'H0':70., 'OmegaM':0.3, 'OmegaL':0.7, 'OmegaK':0.}
    cosmoKeys = fiducialCosmology.keys()
    for iHalo in allDistributions:
        doNotTrainThisSample =  \
              np.any(np.array([fiducialCosmology[iCosmoKey] != \
              iHalo['cosmology'][iCosmoKey] \
              for iCosmoKey in cosmoKeys]))

        if doNotTrainThisSample:
            continue

                           
        zLens.append( np.float(iHalo['fileNames'][0].split('/')[-2].split('_')[1]))
        nHalosInField =substructure( iHalo['fileNames'][0] ) 
        nHalos.append(nHalosInField)
     
        densityProfileIndex, densityProfileIndexError = \
          getDensityProfileIndex( iHalo['fileNames'][0])
        powerLaw.append(densityProfileIndex)
        powerLawError.append(densityProfileIndexError)
        inputPDF = {'y':iHalo['y'], 'x':iHalo['x'], 'yError':iHalo['yError'],}

        #####FIT POWER LAW TO THE DISTRIBUTION##############
        
        powerLawFitClass = powerLawFit( inputPDF, yMin=1.e-2, \
                        curveFit=True, inputYvalue='y' )
        
        beta.append(powerLawFitClass.params['params'][1])

        betaError.append( powerLawFitClass.params['error'][1])
        RMS.append( np.sqrt(np.sum((powerLawFitClass.getPredictedProbabilities()-powerLawFitClass.yNoZeros)**2)))
        #dX = pdf['x'][1] - pdf['x'][0]
        #pdf['y'] /= np.sum(pdf['y'])*dX
        #cumPDF = np.cumsum(pdf['y']*dX)
        #medTime = pdf['x'][np.argmin(np.abs(cumPDF - 0.5))]
        #medTimeError = pdf['x'][np.argmin(np.abs(cumPDF - 0.84))] - medTime
        #print(medTime)
        tPeak.append( powerLawFitClass.params['params'][0])
        tPeakError.append(  powerLawFitClass.params['error'][0])
        #################
        
        #if   (powerLawFitClass.params['params'][1] > 5):
            
        #    plt.plot(pdf['x'], pdf['y'])
        #    plt.plot(pdf['x'], powerLawFitClass.getPredictedProbabilities(pdf['x']), ls='-')
        
        #    plt.yscale('log')
        #    plt.show()
        
    powerLaw = np.array(powerLaw)
    beta  = np.array(beta)
    tPeak = np.array(tPeak)
    nHalos=  np.array(nHalos)
    betaError = np.array(betaError)
    tPeakError = np.array(tPeakError)
    powerLawError = np.array(powerLawError)
    #For aesthetics                                                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=2.8, vmax=3.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    lineStyles = ['--',':','-.']
    alpha = 1.
    #####

    ax = plt.gca()
    nHalos = np.array(nHalos)

    


    zLens = np.array(zLens)
    color = ['blue','red','green','cyan','orange']
    for i, iz in enumerate(np.unique(zLens)):

        ax.errorbar(powerLaw[ (zLens==iz)], \
            beta[(zLens==iz)], \
            xerr=powerLawError[ (zLens==iz)],\
            yerr=betaError[(zLens==iz)], \
            fmt='o', color=color[i], \
            alpha=alpha, label='z='+str(iz))


    ax.plot(np.linspace(-2.1,-1.6), np.linspace(-2.1,-1.6)*-1, 'k--', label=r'Eqn (10)')
    #ax.errorbar(0,0,0,0,fmt='o',label='No Substructure', color='black')
    #ax.errorbar(0,0,0,0,fmt='^',label='Substructure', color='black')
    ax.legend()

    getAndPlotTrend(powerLaw, beta, ax, '-', color='blue')
    #getAndPlotTrend(powerLaw[nHalos > 1], beta[nHalos > 1], ax, '-', color='red')
    ax.set_xlim(-2.1,-1.6)
    #popt, pcov = curve_fit(func, powerLaw[nHalos == 1], beta[nHalos == 1])
    #ax.plot( powerLaw, func( powerLaw, *popt), color='blue')
    #popt, pcov = curve_fit(func, powerLaw[nHalos > 1], beta[nHalos > 1])
    #ax.plot( powerLaw, func( powerLaw, *popt), color='red')
    
    #popt, pcov = curve_fit(func, powerLaw, tPeak)
    #axarr[1].plot( powerLaw, func( powerLaw, *popt))
    ax.set_xlabel('Density Profile Power Law Index')
    ax.set_ylabel('PDF Power Law Index')
    plt.savefig('../plots/densityProfile.pdf')
    plt.show()

def saveAllDensityProfileIndexes(load=True):
    #So i dont have to recall the functiuon which takes a while
    #in the hubble interpolator
    
    allJsonFiles = glob.glob('/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/z*/*.json')

    densityProfileIndex = np.zeros((len(allJsonFiles),2))
    rGrid = getRadGrid()
    for i, iJson in enumerate(allJsonFiles):
        indexAndError =  getDensityProfileIndex( iJson, rGrid=None, nRadialBins=10, load=load)
        densityProfileIndex[i,:] = indexAndError
    
    pkl.dump([allJsonFiles,densityProfileIndex], \
                 open('pickles/densityProfileIndexes.pkl', 'wb'), 2)
    
def getDensityProfileIndex( jsonFileName, rGrid=None, nRadialBins=10):

    if os.path.isfile( 'pickles/densityProfileIndexes.pkl'):
        jsonFiles, indexes = \
          pkl.load(open('pickles/densityProfileIndexes.pkl','rb'))

        matchedJson = '/'.join(jsonFileName.split('/')[-2:])
        matchedJsonToThese = \
          np.array([ '/'.join(i.split('/')[-2:]) for i in jsonFiles])
        indexes =  indexes[matchedJsonToThese == matchedJson, :]
        if len(indexes) == 0:
            print('cant find file name %s' % jsonFileName)
            pdb.set_trace()

        return indexes[0]
    if rGrid is None:
        rGrid = getRadGrid()
        
    radial, density = getDensityProfile( jsonFileName, rGrid=rGrid, nRadialBins=nRadialBins)

    popt, pcov = curve_fit(func, radial, density)
    error = np.sqrt(np.diag(pcov))
    index = popt[1] #(density[-1] - density[0])/(radial[-1] - radial[0])
    #if index < -0.94:
    #plt.plot(radial, density)
    #plt.plot(radial, func( radial, *popt))
    #plt.show()
    Correction = 0.96761529 #from my deprojection code
    return index-Correction, error[1]

def func(x, a, b):
    return a  + x*b

def getDensityProfile( jsonFileName, rGrid=None, nRadialBins=10):
    dataDir = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections/'
    
    haloName = jsonFileName.split('/')[-1].split('_')[0]
    projection = jsonFileName.split('/')[-1].split('_')[3]
    redshift = jsonFileName.split('/')[-2]
    fitsFileName = 'cluster_0_'+projection+'_total_sph.fits'
    dataFileName = dataDir+haloName+'_EAGLE_CDM/'+redshift+'/HIRES_MAPS/'+fitsFileName
    data = fits.open(dataFileName)[0].data
    radialBins = 10**np.linspace(1.,2., nRadialBins+1)
    density = np.zeros(nRadialBins)
    for iRadBin in range(nRadialBins):
        inBin = (rGrid  > radialBins[iRadBin]) & (rGrid < radialBins[iRadBin+1])
        area = np.pi*(radialBins[iRadBin+1]**2 - radialBins[iRadBin]**2)
        density[ iRadBin ] =  np.sum( data[inBin] ) / area

    radialBinCenters = (radialBins[:-1] + radialBins[1:])/2.

    return np.log10(radialBinCenters), np.log10(density)


def substructure( jsonFileName, minLogMass = 7 ):
    '''
    see if there is a halo in the projcted distance
    '''
    dataDir = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections/'
    
    haloName = jsonFileName.split('/')[-1].split('_')[0]
    projection = jsonFileName.split('/')[-1].split('_')[3]
    redshift = jsonFileName.split('/')[-2]
    dataFileName = \
      dataDir+haloName+'_EAGLE_CDM/'+redshift+\
      '/GALAXY_CATALOGS/cluster_0_'+projection+'_galaxy_catalog.dat'

    data = np.loadtxt(dataFileName, \
            dtype=[('id',int), ('x', float), ('y', float), ('z', float), \
                       ('mass', float), ('central', int)])

    distance = np.sqrt( (data['x']-500)**2 +  (data['y']-500)**2)

    distanceCutPixels = 200.
    nHalos = len(distance[(distance < distanceCutPixels) & (data['mass'] > 7) ])
    
    return nHalos
    
def getAndPlotTrend( x, y, axis, fmt, color='grey', pltX=None, sigma=None):
    
    trendParams, cov = \
      curve_fit( func, x, y, p0=[1.,1.], sigma=sigma)
    pError = np.sqrt(np.diag(cov))
    
    percentInError = 1.
    
    while percentInError > 0.68:
        
    
        pUpper = [trendParams[0]+pError[0], \
                    trendParams[1]-pError[1]]
        pLower = [trendParams[0]-pError[0], \
                    trendParams[1]+pError[1]]

        lower = func( x, *pLower)
        upper = func( x, *pUpper)

        percentInError = len(y[ (y - lower > 0) & (y - upper < 0) ])/len(y)
        pError *= 0.9
        
    if pltX is None:
        pltX = np.linspace(-2.1,-1.6)
    axis.plot(  pltX, func(  pltX, *trendParams), fmt, color=color)

    axis.fill_between( pltX, func( pltX, *pUpper), \
                           func( pltX, *pLower), \
                         alpha=0.3, color=color )
    
    

if __name__ == '__main__':
    #saveAllDensityProfileIndexes()
    main()

