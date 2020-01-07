'''
analyse the mass sheet test
'''

from plotHubbleDistributions import *

def main():

    '''
    Create a plot of the PDF for the same lens but  for different mass sheets
    '''
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(15,1)

    axisA = plt.subplot( gs[0:11,0])
    axisB = plt.subplot( gs[12:,0])
    
    
    massSheets = [1., 2., 5., 10., 100.]
    colors = ['r','b','g','c','orange','k']

    allBeta = []
    allBetaError = []
    peakTimeDelay = []
    peakTimeDelayError = []

    for iColor, iMassSheet in enumerate(massSheets):
        pklFileName = '/Users/DavidHarvey/Documents/Work/TimeDelay/'+\
          'output/CDM/massSheetTest/z_0.37/massSheet_'+str(iMassSheet)+'.json.pkl'
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            raise ValueError("No pickle file found (%s) "%pklFileName)

        finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])
        finalMergedPDFdict['yError'] /= np.max(finalMergedPDFdict['y'])

        plotPDF( finalMergedPDFdict, colors[iColor], r"H0=%i kms/Mpc" % iMassSheet, axisA, yType='y' )

        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = powerLawFit( finalMergedPDFdict )
        
        allBeta.append(powerLawFitClass.params['params'][1])
        allBetaError.append(powerLawFitClass.params['error'][1])

        peakTimeDelay.append( finalMergedPDFdict['x'][np.argmax(finalMergedPDFdict['y'])] + 2.56)
        peakTimeDelayError.append( finalMergedPDFdict['x'][1] - finalMergedPDFdict['x'][0])
        ######################################################

    axisB.errorbar( massSheets, allBeta, yerr=allBetaError, fmt='.', color='k')
    axisB.set_ylabel(r'$\beta$')
    params, cov = \
      curve_fit( straightLine, massSheets, allBeta, p0=[1.,1.], sigma=allBetaError)

    axisB.plot( massSheets, straightLine( np.array(massSheets), *params),'k--', ms=5)

    print("Beta = %0.5E + %0.2E H0" % tuple(params))
    print("BetaErrors = %0.5E + %0.2E H0" % tuple(np.sqrt(np.diag(cov))))
    
    axisC = axisB.twinx()
    axisC.errorbar( massSheets, peakTimeDelay, yerr=peakTimeDelayError, fmt='*', color='k')
    params, cov = \
      curve_fit( oneOverTime, massSheets, peakTimeDelay, p0=[1.,1.], sigma=peakTimeDelayError)
    axisC.plot( massSheets, oneOverTime( np.array(massSheets), *params),'k--')


    axisC.set_ylabel(r'log$(\Delta t_{\rm peak}$ / days )')


    print("Tpeak = %0.5E + %0.2E H0" % tuple(params))
    print("TpeakErrors = %0.5E + %0.2E H0" % tuple(np.sqrt(np.diag(cov))))



    ######################################################
    axisB.set_xlabel('Mass Sheet /$10^{13}M_\odot$')

    
    axisA.legend()
    axisA.set_yscale('log')
    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    axisA.set_ylabel(r'P(log($\Delta t$/ days))')
    axisA.set_xlim(-1.4,3.5)
    axisA.set_ylim(1e-2,1.2)
    plt.savefig('../plots/massSheetDegeneracyTest.pdf')
    plt.show()
    
if __name__ == '__main__':
    main()
