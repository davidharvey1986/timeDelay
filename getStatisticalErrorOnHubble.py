

from plotHubbleDistributions import *
from powerLawFit import *
from scipy.stats import powerlaw
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc

def statisticalErrorOnBeta():
    

    hubbleParameters, powerLawIndex, powerLawIndexError = \
      getPowerLawForDifferentHubbleParameters()

    hubbleToPowerLawCorrelation, hubbleToPowerLawCorrelationError = \
      curve_fit( straightLine, hubbleParameters, powerLawIndex, \
                         p0=[1.,1.], sigma=powerLawIndexError)


    ax = plt.gca()
    
    paramErrorList = np.array([ 0., 1.0, 2.0, 5.0, 10.])
    totalNumberQuasars = 10**np.linspace(2., 5., 10)
    errorOnHubble = np.zeros(len(totalNumberQuasars))

    nIterations = 200
    hubbleParameters = [70.]
    timeDelayRange = [ 3., 365.]
    nYears = 5
    listOfYears = [1., 3., 5.]
    
    iHubbleParameter = 70.
    
    #For aesthetics                                                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=np.min(paramErrorList), vmax=np.max(paramErrorList))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    lineStyles = ['--',':','-.']
    #####   

    logChoiceTimeDelays = np.linspace(  np.log10(10.), np.log10(nYears*365.), 1000000)

    testHubblePowerLawIndex = straightLine( iHubbleParameter, *hubbleToPowerLawCorrelation)

    hubbleForNumberQuasars = []
        #for iIteration in xrange(nIterations):

    for iLineStyle, nYears in enumerate(listOfYears):
        bins = np.linspace(  np.log10(10.), np.log10(nYears*365.), np.max([10, np.int(totalNumberQuasars[-1]/1000)]) + 1)
        
        timeDelays = (bins[1:] + bins[:-1])/2.

        choiceTimeDelays = 10**logChoiceTimeDelays
            
        probabilityOfLogTimeDelay = choiceTimeDelays**(testHubblePowerLawIndex)

        
        selectedTimeDelays = np.random.choice( logChoiceTimeDelays, \
                                            p=probabilityOfLogTimeDelay/np.sum(probabilityOfLogTimeDelay), \
                                                size= np.int(totalNumberQuasars[-1]))

            
        pdf, x = np.histogram(selectedTimeDelays, bins=bins, density=True)
        error = np.sqrt(pdf*totalNumberQuasars[-1])/totalNumberQuasars[-1]
        xc = (x[1:] + x[:-1])/2.
                
        pdf /= np.max(pdf)
        error /= np.max(pdf)
        inputPDF = {'x':xc, 'y':pdf, 'yError':error}
    
        powerLawFitClass = powerLawFit( inputPDF,  yMin=0. )

            #print params[1]
        iHubble =  (powerLawFitClass.params['params'][1]- hubbleToPowerLawCorrelation[0])/hubbleToPowerLawCorrelation[1]
        #print iIteration,'/',nIterations
                #print iHubble
        hubbleForNumberQuasars.append(iHubble)

        statErrorOnHubble = powerLawFitClass.params['error'][1]/powerLawFitClass.params['params'][1]*100.  *np.sqrt(totalNumberQuasars[-1])/np.sqrt(totalNumberQuasars)


        for iColor, iParamError in enumerate(paramErrorList): 
        
            errorOnHubble =  np.sqrt(statErrorOnHubble**2 + (iParamError)**2)
            
          #np.sqrt(np.sum( (np.array(hubbleForNumberQuasars)-iHubbleParameter)**2) \
          #                /nIterations )/iHubbleParameter*100.


            if iLineStyle == 0:
                ax.plot( 0, 0, \
                            label=r'$\sigma^{sys}=%i$' % (iParamError), \
                          color=scalarMap.to_rgba(iParamError),\
                            ls='-')
            
            ax.plot( totalNumberQuasars, errorOnHubble, \
                          color=scalarMap.to_rgba(iParamError),\
                            ls=lineStyles[iLineStyle])
                            
    for iLineStyle, nYears in enumerate(listOfYears):
        ax.plot( 0, 0, \
                     label=r'$N_{\rm yrs}=%i$' % (nYears), \
                     color='k',\
                     ls=lineStyles[iLineStyle])
                     
    ax.plot([3000, 3000], [0.,20],'k')
    ax.fill_between([np.min(totalNumberQuasars),np.max(totalNumberQuasars) ], [0., 0.], [1.,1.],color='grey', alpha=0.75)
    ax.fill_between([np.min(totalNumberQuasars),np.max(totalNumberQuasars) ], [1., 1.], [2.5,2.5],color='grey', alpha=0.5)
    ax.set_xlim(np.min(totalNumberQuasars),np.max(totalNumberQuasars))

    ax.text( 3000, 1., 'Expected number of Lenses from LSST', rotation=90.,\
                 verticalalignment='bottom', horizontalalignment='right')
    ax.set_ylim(0.,15)
    ax.set_xscale('log')
    ax.legend()
    ax.set_xlabel(r'Number Measured Time Delays')
    ax.set_ylabel(r'Perecentage Error on $H_0$')
    plt.savefig('../plots/statisticalErrorOnHubble.pdf')
    plt.show()
    
def statisticalErrorOnRatio():

    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''


    axisA = plt.gca()


    

    hubbleParameters = np.array([50., 60., 70., 80., 90., 100.])
    colors = ['r','b','g','c','orange','k']

    ratioYearToMonth = []
    ratioYearToMonthError = []
    
    for iColor, iHubbleParameter in enumerate(hubbleParameters):
        pklFileName = '../output/CDM/combinedPDF_'+str(iHubbleParameter)+'.pkl'
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            raise ValueError("No pickle file found (%s) "%pklFileName)
        finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])

        #####FIT POWER LAW TO THE DISTRIBUTION##############


        powerLawFitClass = powerLawFit( finalMergedPDFdict )

        ######################################################In :
        #Get the ratio of time delays between 1 year and 1 month
        
        ratio, ratioError = \
          getTheoreticalRatioAndError( powerLawFitClass.params['params'][1], lsstError=3., timeA = 365, timeB=30. )
       
        ratioYearToMonth.append(ratio)
        ratioYearToMonthError.append(ratioError)
        ######################################################

    
    #axisA.errorbar( hubbleParameters, ratioYearToMonth, yerr=ratioYearToMonthError, color='k', fmt='o' )
    #Fit a function that determines the ratio as a function of hubble constant
    #Assuming that ratio = A*10**(B*Hubble)

    totalNumberQuasars = 10**np.linspace(1., 3., 10)
    plotHubbleParameters = 70. #np.linspace(65,75,100)
    paramErrorList = np.array([ 0., 1.0, 2.0, 5.0, 10.])/100.
    print paramErrorList
    nIterations = 1000
    for iParamError in paramErrorList:
        errorOnHubble = []
        print iParamError
        for iTotalNumberQuasars in totalNumberQuasars:
            iErrorOnHubble = []
            for iIteration in xrange(nIterations):
                inputRatio = np.random.normal(np.array(ratioYearToMonth), np.array(ratioYearToMonth)*iParamError)
                    
                params, cov = \
                  curve_fit( straightLine, hubbleParameters, np.log10(inputRatio), \
                         p0=[1.,1.])
                         

                paramsError = np.sqrt(np.diag(cov))
                
                inputParams = [params[0], params[1]*np.random.normal( 1. , iParamError)]
                                                            
                estimatedRatios = 10**sis.straightLine( plotHubbleParameters, *inputParams)
                estimatedHubble = np.log10( estimatedRatios ) / params[1] - params[0]
                trueRatios = 10**sis.straightLine( plotHubbleParameters, *params)
                theoreticalError = estimatedHubble - plotHubbleParameters

                statisticalErrorRatio = \
                    np.sqrt( iTotalNumberQuasars*(1.+trueRatios)/(trueRatios*iTotalNumberQuasars**2))

                totalErrorOnHubble = np.sqrt( theoreticalError**2 + getErrorOnHubbleParameter( trueRatios, statisticalErrorRatio, \
                                                        params, paramsError)**2)
                
                iErrorOnHubble.append(totalErrorOnHubble)

            errorOnHubble.append( np.nanmean(np.array(iErrorOnHubble)) / plotHubbleParameters*100)
        iParamErrorStr = iParamError 
        axisA.plot(totalNumberQuasars, errorOnHubble, label=r'$\sigma_{\beta}/\beta=%0.2f$' % iParamErrorStr)
        
    axisA.set_xscale('log')
    

    axisA.set_ylim(0.,22.)
    #axisA.set_ylabel(r'$N_{365}/N_{30}$')
    axisA.legend(loc=1)
    #axisA.set_xlabel(r'$H_0$ (km/s/Mpc)')
    axisA.set_xlabel(r'Number Measured Time Delays')
    axisA.set_ylabel(r'Perecentage Error on $H_0$')
    plt.savefig('../plots/statisticalErrorOnHubble.pdf')
    
    plt.show()

def getPowerLawForDifferentHubbleParameters():
    powerLawIndex = []
    powerLawIndexError = []
    hubbleParameters = np.array([50., 60., 70., 80., 90., 100.])

    
    for iColor, iHubbleParameter in enumerate(hubbleParameters):
        print iHubbleParameter
        pklFileName = '../output/CDM/combinedPDF_'+str(iHubbleParameter)+'.pkl'
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            raise ValueError("No pickle file found (%s) "%pklFileName)
        finalMergedPDFdict['yError'] /= np.max(finalMergedPDFdict['y'])
        finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])


        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = powerLawFit( finalMergedPDFdict )


        powerLawIndex.append( powerLawFitClass.params['params'][1] )
        powerLawIndexError.append( powerLawFitClass.params['error'][1] )

    return hubbleParameters, powerLawIndex, powerLawIndexError

def getTheoreticalRatioAndError( beta, lsstError=3., timeA = 365, timeB=30. ):
    '''
    The total observational error in the ratio is the error in the theoretical
    parameters from the distribution of time delays (error in beta)
    and the stastical error in the number of halos
    '''
    NumberOfTimesA = \
      (timeA+lsstError)**(beta+1) - (timeA-lsstError)**(beta+1)
      
    theoreticalErrorTa = \
       (beta+1.)*( (timeA+lsstError)**beta - (timeA-lsstError)**beta )

    NumberOfTimesB = \
      (timeB+lsstError)**(beta+1) - (timeB-lsstError)**(beta+1)
      
    theoreticalErrorTb = \
       (beta+1.)*( (timeA+lsstError)**beta - (timeB-lsstError)**beta )

    ratio = NumberOfTimesA / NumberOfTimesB
    
    totalTheoreticalError = \
      np.sqrt( (theoreticalErrorTa/NumberOfTimesA)**2 + \
                   (theoreticalErrorTa/NumberOfTimesB))*ratio


    return ratio, totalTheoreticalError

    
def getEmpiricalRatioAndError( PDFdict, lsstError=3., timeA = 365, timeB=30. ):
    '''
    The total observational error in the ratio is the error in the theoretical
    parameters from the distribution of time delays (error in beta)
    and the stastical error in the number of halos
    '''
        
    NumberOfTimesA = np.interp(  np.log10(timeA), PDFdict['x']+2.56, \
                                    np.log10(PDFdict['y']))

  
    
    ErrorInA = np.interp( np.log10(timeA), PDFdict['x']+2.56,PDFdict['yError'] )
    
   
    NumberOfTimesB = np.interp( np.log10(timeB), PDFdict['x']+2.56, np.log10(PDFdict['y']))

    ErrorInB = np.interp( np.log10(timeB), PDFdict['x']+2.56,PDFdict['yError'])

    ratio = 10**NumberOfTimesA / 10**NumberOfTimesB
    
    totalTheoreticalError = \
      np.sqrt( (ErrorInA/NumberOfTimesA)**2 + \
                   (ErrorInB/NumberOfTimesB))*ratio


    return ratio, totalTheoreticalError

    
def getErrorOnHubbleParameter( ratio, errorRatio, params, paramsError):
    '''
    Given that I assume
    
    ratio = A log( H0) + B

    then 

    H0 = 10**(A/ratio - B)

    then error propagation is as follows


    '''
    
    sigmaRatio = 1. / ( params[1]*ratio * np.log(10.)) * errorRatio
    sigmaParamsA = paramsError[0]
    sigmaParamsB = np.log10(ratio)/params[1]**2 *paramsError[1]


    return np.sqrt(sigmaRatio**2 + sigmaParamsA**2 + sigmaParamsB**2 )




if __name__ == '__main__':
    statisticalErrorOnBeta()
