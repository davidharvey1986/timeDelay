'''
I want to prediuct the constraints on teh hubble parameter from the
estiamted PDFs generated

I will fit a double powerLaw

'''
import plotAsFunctionOfDensityProfile as getDensity
from powerLawFit import *
import fitHubbleParameter as fitHubble
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis
import corner as corner
from hubbleInterpolatorClass import *
from scipy.stats import norm
import matplotlib.lines as mlines
from matplotlib import rcParams
rcParams["font.size"] = 16
from scipy.ndimage import gaussian_filter as gauss


    
def plotCornerPlot( sampleSize=1000,samplesPerIteration = 30000,
                        trueHubble=True):

    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$']
    ndim = 3
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(8,8))
    color = ['blue','red','green','cyan']
    
    for icolor, sampleSize in enumerate([10000]):
        samples = getMCMCchainForSamplesSize(sampleSize, 10, 100., None, trueHubble=trueHubble)
        
        if (not trueHubble):
            truths  = [0.7, 0.4, -1.75]
        else:
            truths = None
            for i in axarr[:,0]:
                i.plot([0.7,.7],[-2,10],'k--')

        nsamples = samples.shape[0]
                    
        corner.corner(samples , \
                      bins=100, smooth=True, \
                      plot_datapoints=False,
                      fig=figcorner, \
                      labels=labels, plot_density=True, \
                      truths=truths,\
                      weights=np.ones(nsamples)/nsamples,\
                    color=color[icolor],\
                          levels=(0.68,0.95), labelsize=15,\
                          truth_color='black')

        
    if  trueHubble:
        reportParameters(samples)
        axarr[1,1].set_xlim( 0.45, 0.53)
        axarr[2,2].set_xlim( -1.9, -1.6)
        axarr[1,0].set_ylim( 0.44, 0.53)
        axarr[2,0].set_ylim( -1.9, -1.6)
        axarr[2,1].set_ylim( -1.9, -1.6)
        axarr[2,1].set_xlim( 0.44, 0.53)

    else:
        axarr[1,1].set_xlim( 0.3, 0.58)
        axarr[2,2].set_xlim( -1.86, -1.68)
        axarr[1,0].set_ylim( 0.3, 0.58)
        axarr[2,0].set_ylim( -1.86, -1.68)
        axarr[2,1].set_ylim( -1.86, -1.68)
        axarr[2,1].set_xlim( 0.3, 0.58)
            
    axarr[0,0].set_xlim( 0.65, 0.74)
    axarr[1,0].set_xlim( 0.65, 0.74)
    axarr[2,0].set_xlim( 0.65, 0.74)
    

    #hundreds = mlines.Line2D([], [], color='blue', label=r'$10^2$ Lenses')
    #thousand = mlines.Line2D([], [], color='red', label='$10^3$ Lenses')
    #tenthous = mlines.Line2D([], [], color='green', label='$10^4$ Lenses')
    
    ##axarr[0,1].legend(handles=[hundreds,thousand,tenthous], \
      #  bbox_to_anchor=(0., 0.25, 1.0, .0), loc=4)
    if trueHubble:
        plt.savefig('../plots/degenerciesTrueHubble.pdf')
    else:
        plt.savefig('../plots/degenercies.pdf')
    plt.show()

    
    
def nonPerfectFittingFunction(nComponents=5):
    fig = plt.figure(figsize=(10,6))

    inputHubbleParameter = 70.
    #####
    #sampleSizes, estimates = \
    #  getPredictedConstraints(inputHubbleParameter)
    #estimates /=inputHubbleParameter/100.
    #plt.plot( sampleSizes, estimates,\
    #              label="Exact PDF")

    #####
    

    sampleSizes, estimates = \
     getPredictedConstraints(inputHubbleParameter, trueHubble=False)
    estimates /=inputHubbleParameter/100.

    
    plt.errorbar( sampleSizes, estimates, \
                  label=r'$\Delta t_{\rm min} = 0$ days', color='red')
    constraints3000 = np.interp( 3000, sampleSizes, estimates)
    constraints400 = np.interp( 400, sampleSizes, estimates)

    ########
    sampleSizes, estimates = \
      getPredictedConstraints(inputHubbleParameter, minimumTimeDelay=10)
      
    plt.errorbar( sampleSizes, estimates/inputHubbleParameter*100., \
                label=r'$\Delta t_{\rm min} = 10$ days', color='blue')
    #######




    
    plt.plot( sampleSizes, np.ones(len(sampleSizes))+1, 'c--', \
                  label='Current systematic limit')

    
    plt.plot( [3000,3000], [0,10], 'k--', label='LSST (Optimistic)')
    plt.plot( sampleSizes, np.zeros(len(sampleSizes))+constraints3000,\
                  'k--')
    plt.plot( [400,400], [0,10], 'k:', label='LSST (Conservative)')
    plt.plot( sampleSizes, np.zeros(len(sampleSizes))+constraints400,\
                  'k:')
    #plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylim(0., 6)
    plt.xlim(1e2, 1e4)

    plt.xlabel('nSamples')
    plt.ylabel(r'Percentage error on $H_0$')
    plt.savefig('../plots/statisticalErrorOnHubble.pdf')
    plt.show()

def getPredictedConstraints(inputHubbleParameter, \
                                nIterations = 10,\
                                nSampleSizes = 11,\
                                exactIndex=False,\
                                trueHubble=False,\
                                differentHalo=None,\
                                minimumTimeDelay=0.):



    hubbleInterpolaterClass = hubbleInterpolator( minimumTimeDelay=minimumTimeDelay)
    if minimumTimeDelay > 0:
        hubbleInterpolaterClass.getTrainingData(pklFile='picklesMinimumDelay/trainingData.pkl')
    else:
        hubbleInterpolaterClass.getTrainingData('exactPDFpickles/trainingData.pkl')
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()


    sampleSizes = 10**np.linspace(2,4,nSampleSizes)
    
    estimates = np.zeros(nSampleSizes)


    #Loop through each sample size
    for i, iSampleSize in enumerate([300]):
        print("Sample Size: %i" % (iSampleSize))
        
        samples = \
          getMCMCchainForSamplesSize(iSampleSize, nIterations, \
                        inputHubbleParameter, hubbleInterpolaterClass,\
                                         trueHubble=trueHubble,\
                                         differentHalo=differentHalo,\
                                         minimumTimeDelay=minimumTimeDelay)

        if exactIndex:
            truth = \
              (samples[:,1] > 0.19) & (samples[:,1] < 0.21) &\
              (samples[:,2] > -1.95) & (samples[:,2] < -1.5)
            #estimates[i] = \
            #  norm.fit(samples[truth,0])[1]/np.sqrt(nIterations)*100

            median, error = \
              getModeAndError( samples[truth, 0])

        else:
            #estimates[i] = \
            #  norm.fit(samples[:,0])[1]/np.sqrt(nIterations)*100
            
            median, error =  getModeAndError( samples[:,0])

        estimates[i] = error


    return sampleSizes, estimates


    
def getMCMCchainForSamplesSize( iSampleSize, nIterations,\
                    inputHubbleParameter, hubbleInterpolaterClass,\
                    trueHubble=False, differentHalo=None,\
                    minimumTimeDelay=0.):
    samples = None

    if trueHubble:
        pklFile = 'picklesTrueHubble/multiFitSamples_%i.pkl' % iSampleSize
    elif differentHalo is not None:
        pklFile = 'picklesDifferentHalo/multiFitSamples_%s_%i.pkl' % (differentHalo, iSampleSize)
    elif minimumTimeDelay > 0:
        pklFile = 'picklesMinimumDelay/multiFitSamples_%i.pkl' % (iSampleSize)
    else:
        pklFile = 'exactPDFpickles/multiFitSamples_%i.pkl' % iSampleSize
    

    if os.path.isfile(pklFile):
        return pkl.load(open(pklFile, 'rb'))
    
    logProb = np.array([])
    for iIteration in range(nIterations):
        print("Iteration: %i/%i" % (iIteration+1, nIterations))
        selectedTimeDelays = \
              selectRandomSampleOfTimeDelays( iSampleSize, \
                inputHubbleParameter, hubbleInterpolaterClass,\
                    trueHubble=trueHubble,differentHalo=differentHalo,\
                                minimumTimeDelay=minimumTimeDelay)
      
        fitHubbleClass = \
              fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)

        y, x = np.histogram(fitHubbleClass.samples[:,0])
        xc = (x[1:] + x[:-1])/2.
            
        if samples is None:
            samples = fitHubbleClass.samples
        else:
            samples = np.vstack( (samples, fitHubbleClass.samples))


        pdb.set_trace()
    pkl.dump(samples, open(pklFile, 'wb'))
        
    return samples


def selectRandomSampleOfTimeDelays( nSamples, hubbleParameter, \
                        hubbleInterpolaterClass, trueHubble=False,\
                        differentHalo=None, minimumTimeDelay=0.):
    '''
    From a given pdf randomly select some time delays

    '''
    
    if minimumTimeDelay == 0:
        minimumTimeDelay = 1e-3
        
    logMinimumTimeDelay = np.log10(minimumTimeDelay)
    
    pklFileName = \
      '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
      % (hubbleParameter)
      
    if differentHalo is not None:
      pklFileName = \
        '../output/CDM/selectionFunction/SF_%s_%i_lsst.pkl' \
        % (differentHalo, hubbleParameter)
      
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

        
    interpolateToTheseTimes= np.linspace(np.log10(minimumTimeDelay), 3, 1e6)
      
    if trueHubble | ( differentHalo is not None) :

        interpolatedProbClass = CubicSpline( finalMergedPDFdict['x'], \
                                             finalMergedPDFdict['y'])

        interpolatedProb = interpolatedProbClass( interpolateToTheseTimes)
    else:
        theta = np.array([0.7, 0.5, -1.75] )
        interpolatedCumSum = hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, theta )
        interpolatedProb = undoCumSum( interpolatedCumSum)
      
      
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)
    
    #Interpolate the p
    
    
    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))
    
    bins = np.max([20, np.int(nSamples/100)])
    y, x = np.histogram(logTimeDelays, \
                    bins=np.linspace(-1,3,bins), density=True)
                    
    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
    if trueHubble:
        xcentres += 0.0404
        
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.sqrt(np.cumsum(error**2)/(np.arange(len(error))+1))
    #cumsumYError = np.ones(len(y))/nSamples/2.
    xcentres+=dX/2.


    return {'x':xcentres, 'y':cumsumY,  'error':cumsumYError}

def getModeAndError( samples ):

    itPs = 10000
    nIterations = np.int(samples.shape[0] / itPs)
    
    maxLike = []
    indivError = []
    for i in range(nIterations):
        iSample = samples[i*itPs:(i+1)*itPs]
        y, x = \
          np.histogram( iSample[(iSample>0.65) &(iSample<0.8)], \
                            bins=np.linspace(0.5, 1., 50),  \
                        density=True)
        
        xc = (x[1:]+x[:-1])/2.
        #dX =  x[1] - x[0]
        maxLike.append(xc[np.argmax(y)])
        indivError.append( np.std( iSample[(iSample>0.65) &(iSample<0.8)] ))



    #error = np.sqrt(np.sum((np.array(maxLike)-0.7)**2)/len(maxLike))

    maxLikeMean = np.median(np.array(maxLike))
    
    #error  = np.std(samples[(samples>0.65) & (samples<0.8)])
    error = np.nanmedian(np.array(indivError))
    print(error)

    if np.isfinite(error) == False:
        pdb.set_trace()
    errorOnError = np.std(np.array(indivError)) /np.sqrt(len(indivError))
    return maxLikeMean, error*100

def undoCumSum(cumulative):
    output = [0] * len(cumulative)
    for i in range(len(cumulative)-1):
        output[i+1] = cumulative[i+1] - cumulative[i]
    output[0] = cumulative[0]

    return np.array(output)

def reportParameters( samples ):
    for i in range(samples.shape[1]):
        print("Assuming Gaussian Posterior: ", norm.fit(samples[:,i]))
        
if __name__ == '__main__':
    #plotCornerPlot()
    
    nonPerfectFittingFunction()
##
