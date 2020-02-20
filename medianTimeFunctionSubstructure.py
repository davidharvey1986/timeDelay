
from plotAsFunctionOfDensityProfile import *

def medianTimeFunctionSubstructure():
    

    plt.figure(figsize=(8,6))
    #For aesthetics                                                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=2.8, vmax=3.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    lineStyles = ['--',':','-.']
    #####


    pklFile = 'pickles/medianTimeFunctionSubstructure.pkl'

    if not os.path.isfile(pklFile):
        getHaloStats(pklFile)

    nHalos, tPeak, tPeakError, medTime, maxTime = \
          pkl.load(open(pklFile, 'rb'))
    nHalos = np.array(nHalos)
    tPeak =   np.array(tPeak)
    tPeakError  = np.array(tPeakError)
    medTime  = np.array(medTime)
    maxTime = np.array(maxTime)

    maxTimeAv = []
    maxTimeAvErr = []
    medTimeAv = []
    medTimeAvErr = []
    
    for i in np.unique(np.array(nHalos)):
        
        medTimeAv.append(np.mean(medTime[i==nHalos]))
        medTimeAvErr.append(np.std(medTime[i==nHalos])/np.sqrt(len(medTime[i==nHalos])))

        
        
        maxTimeAv.append(np.mean(maxTime[i==nHalos]))
        maxTimeAvErr.append(np.std(maxTime[i==nHalos])/np.sqrt(len(maxTime[i==nHalos])))
        
    plt.errorbar( np.unique(np.array(nHalos))-1, medTimeAv, yerr=medTimeAvErr, fmt='o', color='black', label=r'$\log(\Delta t)_{\rm med}$', capsize=2.)
    plt.errorbar( np.unique(np.array(nHalos))-1, maxTimeAv, yerr=maxTimeAvErr, fmt='o', color='red', label=r'$\log(\Delta t)_{\rm max}$', capsize=2.)
    plt.legend()


    
    plt.xlabel(r'Number Substructures > $10^7M_\odot$')
    plt.ylabel(r'$\log(\Delta t)$')
    plt.savefig('../plots/FunctionOfSubstructure.pdf')
    plt.show()
def getNumHalos( minLogMass):
    allFiles = glob.glob('../output/CDM/z_0.*/B*cluster_0_*_total*.json')

    return np.array([ substructure(i, minLogMass=minLogMass) for i in allFiles])
def getHaloStats(pklFile):
    tPeak = []
    tPeakError = []
    medTime = []
    maxTime = []
    nHalos = []
    allFiles = glob.glob('../output/CDM/z_0.*/B*cluster_0_*_total*.json')
    for iHalo in allFiles:
 
        pdf = selectFunct.selectionFunction( [iHalo], newHubbleParameter=70., useLsst=True )
                           
        nHalosInField =substructure( iHalo ) 
        nHalos.append(nHalosInField)
        medTime.append( pdf['x'][np.argmin(np.abs(np.cumsum(pdf['y'])/np.sum(pdf['y'])-0.5))])
        maxTime.append( pdf['x'][np.argmax(pdf['y'])])
                            
        #####FIT POWER LAW TO THE DISTRIBUTION##############
        
        powerLawFitClass = powerLawFit( pdf, yMin=1.e-2, \
                        curveFit=True, inputYvalue='y' )
        

        tPeak.append( powerLawFitClass.params['params'][0])
        tPeakError.append(  powerLawFitClass.params['error'][0])

        
    pkl.dump([nHalos, tPeak, tPeakError, medTime, maxTime], open(pklFile, 'wb'))



if __name__ == '__main__':
    medianTimeFunctionSubstructure()

