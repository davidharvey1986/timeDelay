
from plotAsFunctionOfDensityProfile import *
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
    totalMass = []
    nHalos = []
    RMS = []
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

 
 
        nHalosInField = substructure( iHalo['fileNames'][0] ) 
        nHalos.append(nHalosInField)
     
        totalMassForHalo = getTotalMass( iHalo['fileNames'][0])

        totalMass.append(totalMassForHalo)

        #####FIT POWER LAW TO THE DISTRIBUTION##############
        
 
        inputPDF = {'y':iHalo['y'], 'x':iHalo['x'], 'yError':iHalo['yError'],}

        powerLawFitClass = powerLawFit( inputPDF, yMin=1.5e-2, curveFit=True, inputYvalue='y' )
        
        #beta.append(powerLawFitClass.params['params'][1])
        #betaError.append( powerLawFitClass.params['error'][1])
        #RMS.append( np.sqrt(np.sum((powerLawFitClass.getPredictedProbabilities()-powerLawFitClass.yNoZeros)**2)))
        #dX = pdf['x'][1] - pdf['x'][0]
        #pdf['y'] /= np.sum(pdf['y'])*dX
        #cumPDF = np.cumsum(pdf['y']*dX)
        #medTime = pdf['x'][np.argmin(np.abs(cumPDF - 0.5))]
        #medTimeError = pdf['x'][np.argmin(np.abs(cumPDF - 0.84))] - medTime
        #print(medTime)
        tPeak.append( powerLawFitClass.getFittedPeakTimeAndError()[0])
        tPeakError.append(  powerLawFitClass.getFittedPeakTimeAndError()[1])
        
        #################
        #if  ( powerLawFitClass.getFittedPeakTimeAndError()[0] > 0.8) & (totalMassForHalo<11.5) & (nHalosInField==1):
            
        #    plt.plot(pdf['x'], pdf['y'])
        #    plt.plot(pdf['x'], powerLawFitClass.getPredictedProbabilities(xInput=pdf['x']), ls='-')

        #    plt.yscale('log')
        #    plt.show()
        
    totalMass = np.array(totalMass)
    beta  = np.array(beta)
    tPeak = np.array(tPeak)
    nHalos=  np.array(nHalos)
    betaError = np.array(betaError)
    tPeakError = np.array(tPeakError)
    #For aesthetics                                                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=2.8, vmax=3.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    lineStyles = ['--',':','-.']
    alpha = 0.2
    #####

    ax = plt.gca()
    nHalos = np.array(nHalos)
    
    ax.errorbar(totalMass, tPeak, \
                yerr=tPeakError, fmt='o', \
                          color='blue', alpha=alpha)
    '''
    ax.errorbar(totalMass[nHalos > 1], tPeak[nHalos > 1],\
                yerr=tPeakError[nHalos > 1], \
            fmt='o', alpha=alpha, color='red', label='Substrucutre')
    '''
    
    #axarr[1].errorbar(powerLaw[nHalos == 1], tPeak[nHalos == 1], \
    #        xerr=powerLawError[nHalos == 1], yerr=tPeakError[nHalos == 1], fmt='o', color='red')


    #axarr[1].errorbar(powerLaw[nHalos > 1], tPeak[nHalos > 1], \
    #        xerr=powerLawError[nHalos > 1], yerr=tPeakError[nHalos > 1], fmt='o')
            
    
    #axarr[0].scatter(powerLaw, beta, c=RMS)
    #axarr[1].scatter(powerLaw, tPeak, c=RMS)
    pltX = np.linspace(10.8, 11.4)

    
    ax.plot( pltX, 2*(pltX-11) + 1.6, 'k--', label=r'$\Delta t (most prob) \propto M(<5kpc)^2$')

    getAndPlotTrend(totalMass, tPeak, ax, '-', color='blue', \
                        pltX=pltX, sigma=tPeakError)
    #getAndPlotTrend(totalMass[nHalos > 1], tPeak[nHalos > 1], ax, '-', color='red', \
    #                    pltX=pltX, sigma=tPeakError[nHalos > 1])

    #popt, pcov = curve_fit(func, powerLaw[nHalos == 1], beta[nHalos == 1])
    #ax.plot( powerLaw, func( powerLaw, *popt), color='blue')
    #popt, pcov = curve_fit(func, powerLaw[nHalos > 1], beta[nHalos > 1])
    #ax.plot( powerLaw, func( powerLaw, *popt), color='red')
    ax.legend()

    #popt, pcov = curve_fit(func, powerLaw, tPeak)
    #axarr[1].plot( powerLaw, func( powerLaw, *popt))
    ax.set_xlabel('log($M(<5kpc)/M_\odot$)')
    ax.set_ylabel('$\log(\Delta t$ [most prob] )')
    ax.set_xlim(pltX[0],pltX[-1])
    plt.savefig('../plots/totalMass.pdf')
    plt.show()
    
def func(x, a, b):
    return a  + x*b

def getTotalMass( jsonFileName, rGrid=None, radialCut=50.):
    if os.path.isfile( 'pickles/totalMasses.pkl'):
        jsonFiles, masses = \
          pkl.load(open('pickles/totalMasses.pkl','rb'))

        matchedJson = '/'.join(jsonFileName.split('/')[-2:])
        matchedJsonToThese = \
          np.array([ '/'.join(i.split('/')[-2:]) for i in jsonFiles])
        masses =  masses[matchedJsonToThese == matchedJson]
        if len(masses) == 0:
            print('cant find file name %s' % jsonFileName)
            pdb.set_trace()
        return masses[0]
    
    dataDir = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections/'
    
    haloName = jsonFileName.split('/')[-1].split('_')[0]
    projection = jsonFileName.split('/')[-1].split('_')[3]
    redshift = jsonFileName.split('/')[-2]
    fitsFileName = 'cluster_0_'+projection+'_total_sph.fits'
    dataFileName = dataDir+haloName+'_EAGLE_CDM/'+redshift+'/HIRES_MAPS/'+fitsFileName
    data = fits.open(dataFileName)[0].data
    dPix = 1e-4*1e-4
    return np.log10(np.sum( data[ rGrid < radialCut ])*dPix)
  
def saveAllTotalMass(load=True):
    #So i dont have to recall the functiuon which takes a while
    #in the hubble interpolator
    
    allJsonFiles = glob.glob('/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/z*/B**cluster_0_*_total_*.json')

    totalMass = np.zeros(len(allJsonFiles))
    rGrid = getRadGrid()
    for i, iJson in enumerate(allJsonFiles):
        iTotalMass =  getTotalMass( iJson, rGrid=rGrid)
        totalMass[i] = iTotalMass
    
    pkl.dump([allJsonFiles,totalMass], \
                 open('pickles/totalMasses.pkl', 'wb'), 2)

if __name__ == '__main__':
    saveAllTotalMass()
    main()

