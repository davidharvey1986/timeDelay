from getStatisticalErrorOnHubble import *
from powerLawFit import *

def main():
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''
    #fig = plt.figure(figsize=(5,5))


    #gs = gridspec.GridSpec(10,1)

    #axisA = plt.subplot( gs[0:6,0])
    #axisB = plt.subplot( gs[7:,0])
    #axisC = axisB.twinx()
    axisA = plt.gca()
    colors = ['r','b','g','c','orange','k','m','y','pink']


    haloNames = ['B002','B005','B008','B009']
    ratioYearToMonth = []
    ratioYearToMonthError = []
    beta = []
    betaError = []

    for iColor, iHalo in enumerate(haloNames):
        pklFileName = '../output/CDM/selectionFunction/SF_%s_70_lsst.pkl' % iHalo
    
        finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        for i in finalMergedPDFdict.keys():
            if 'y' in i:
                finalMergedPDFdict[i] =  1. - np.cumsum(finalMergedPDFdict[i])/np.sum(finalMergedPDFdict[i])
        #finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])
        #finalMergedPDFdict['yError'] /= np.max(finalMergedPDFdict['y'])
        #finalMergedPDFdict['yLensPlane'] /= np.max(finalMergedPDFdict['yLensPlane'])

        #noLogErrorBar = finalMergedPDFdict['yError'] >  finalMergedPDFdict['y']
        #finalMergedPDFdict['yError'][noLogErrorBar] = \
        #  finalMergedPDFdict['y'][noLogErrorBar]*0.99
          
        axisA.fill_between(finalMergedPDFdict['x'], \
                    finalMergedPDFdict['y'] - finalMergedPDFdict['yError']/2., \
                    finalMergedPDFdict['y'] + finalMergedPDFdict['yError']/2., \
                    color=colors[iColor], alpha=0.5)
          
        axisA.errorbar(finalMergedPDFdict['x'],finalMergedPDFdict['y'],\
                    label=r"%s" % (iHalo), \
                        color=colors[iColor])              
       

      

        #axisB.errorbar( iColor+0.9, powerLawFitClass.params['params'][1], \
        #                    yerr=powerLawFitClass.params['error'][1], \
        #                    fmt='o', color=colors[iColor])

        #axisC.errorbar( iColor+1.1,  \
        #               powerLawFitClass.getFittedPeakTimeAndError()[0],
        #                    yerr= powerLawFitClass.getFittedPeakTimeAndError()[1], \
        #                    fmt='*', color=colors[iColor])

    #axisB.set_xticks([1, 2, 3, 4])
    #axisC.set_ylabel(r'log$(\Delta t_{\rm peak}$ / days )')

    #axisB.set_xticklabels(haloNames)
    axisA.legend()
    #axisA.set_yscale('log')
 
    axisA.set_xlabel(r'log($\Delta T$/ days)', labelpad=-1)
    axisA.set_ylabel(r'P(>log($\Delta T$/ days))')
    axisA.set_xlim(-1.,3.)
    #axisB.set_xlabel('Halo Name')
    #axisB.set_ylabel(r'$\beta$')
    #axisA.set_ylim(1e-2,2.)
    

    plt.savefig('../plots/haloToHaloVariance.pdf')
    plt.show()


    

def oplotEnsembleDistribution(axisA, axisB):
    pklFileName = '../output/CDM/combinedPDF_100.0.pkl'
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
    finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])
    axisA.fill_between(finalMergedPDFdict['x'], \
                    finalMergedPDFdict['y'] - finalMergedPDFdict['yError']/2., \
                    finalMergedPDFdict['y'] + finalMergedPDFdict['yError']/2., \
                    color='k', alpha=0.5)
                    
    axisA.errorbar(finalMergedPDFdict['x'],finalMergedPDFdict['y'],\
                    label=r"Total", color='k')


    #####FIT POWER LAW TO THE DISTRIBUTION##############
    peakTime = np.max( finalMergedPDFdict['y'] )
    fittedProbability =  \
      (finalMergedPDFdict['y'] <peakTime) & (finalMergedPDFdict['y']>1e-2)
    params, error = \
      sis.fitPowerLaw( finalMergedPDFdict['x'][fittedProbability], \
                            finalMergedPDFdict['y'][fittedProbability])
                         
    axisA.plot( finalMergedPDFdict['x'], \
                10**sis.straightLine( finalMergedPDFdict['x'], *params), \
                    '--', color='black')

    ratio, ratioError = \
          getTheoreticalRatioAndError( params[1], lsstError=3., timeA = 365, timeB=30. )
    if axisB is not None:
        axisB.errorbar( 4, ratio, yerr=ratioError, fmt='o', color='k')
    #########  #########  #########  #########  #########

    
if __name__ == '__main__':

    main()
