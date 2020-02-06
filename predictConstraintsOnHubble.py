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
                        trueHubble=False, differentHalo='B002'):

    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$']
    ndim = 3
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(12,12))
    color = ['blue','red','green','cyan']
    
    for icolor, sampleSize in enumerate([100,1000,10000]):
        samples = getMCMCchainForSamplesSize(sampleSize, 10, 70., None, differentHalo=differentHalo, trueHubble=trueHubble)
        
        if (not trueHubble) & ( differentHalo is None):
            truths  = [0.7, 0.4, -1.75]
        else:
            truths = None
            for i in axarr[:,0]:
                i.plot([0.7,.7],[-2,10],'k--')

        nsamples = samples.shape[0]                     
        corner.corner(samples, \
                      bins=100, smooth=True, \
                      plot_datapoints=False,
                      fig=figcorner, \
                      labels=labels, plot_density=True, \
                      truths=truths,\
                      weights=np.ones(nsamples)/nsamples,\
                    color=color[icolor],\
                          levels=(0.68,), labelsize=15,\
                          truth_color='black')

    if  trueHubble:
        axarr[1,1].set_xlim( 0.3, 0.5)
        axarr[2,2].set_xlim( -1.9, -1.6)
        axarr[1,0].set_ylim( 0.3, 0.5)
        axarr[2,0].set_ylim( -1.9, -1.6)
        axarr[2,1].set_ylim( -1.9, -1.6)
        axarr[2,1].set_xlim( 0.3, 0.5)
    
    elif ( differentHalo is not None) :
        axarr[1,1].set_xlim( 0.3, 0.5)
        axarr[2,2].set_xlim( -1.96, -1.8)
        axarr[1,0].set_ylim( 0.3, 0.5)
        axarr[2,0].set_ylim( -1.96, -1.8)
        axarr[2,1].set_ylim( -1.96, -1.8)
        axarr[2,1].set_xlim( 0.3, 0.5)

    else:
        axarr[1,1].set_xlim( 0.3, 0.58)
        axarr[2,2].set_xlim( -1.86, -1.68)
        axarr[1,0].set_ylim( 0.3, 0.58)
        axarr[2,0].set_ylim( -1.86, -1.68)
        axarr[2,1].set_ylim( -1.86, -1.68)
        axarr[2,1].set_xlim( 0.3, 0.58)
            
    axarr[0,0].set_xlim( 0.6, 0.8)
    axarr[1,0].set_xlim( 0.6, 0.8)
    axarr[2,0].set_xlim( 0.6, 0.8)
    

    hundreds = mlines.Line2D([], [], color='blue', label=r'$10^2$ Lenses')
    thousand = mlines.Line2D([], [], color='red', label='$10^3$ Lenses')
    tenthous = mlines.Line2D([], [], color='green', label='$10^4$ Lenses')
    
    axarr[0,1].legend(handles=[hundreds,thousand,tenthous], \
        bbox_to_anchor=(0., 0.25, 1.0, .0), loc=4)
    if trueHubble:
        plt.savefig('../plots/degenerciesTrueHubble.pdf')
    elif  ( differentHalo is not None):
        plt.savefig('../plots/degenerciesDifferentHalo.pdf')
    else:
        plt.savefig('../plots/degenercies.pdf')
    plt.show()

    
    
def nonPerfectFittingFunction(nComponents=5):

    inputHubbleParameter = 70.
      
    sampleSizes, estimates = \
      getPredictedConstraints(inputHubbleParameter, differentHalo='B009')
    estimates /=inputHubbleParameter/100.

    fig = plt.figure(figsize=(10,6))
    plt.plot( sampleSizes, estimates,\
                  label="Ensemble samples")
    
    #sampleSizes, estimates = \
    #  getPredictedConstraints(inputHubbleParameter,trueHubble=False)
      
    #plt.plot( sampleSizes, estimates/inputHubbleParameter*100., \
    #            label='Exact CDF')
    plt.plot( sampleSizes, np.ones(len(sampleSizes)), '--')
    plt.plot( [3162,3162], [0,10], 'k--', label='Expected from LSST')
    
    constraints = estimates[ np.argmin(np.abs(sampleSizes-3000))]
    plt.plot( sampleSizes, np.zeros(len(sampleSizes))+constraints, 'k--')

    #plt.yscale('log')
    plt.xscale('log')
    #plt.legend()
    plt.ylim(0., 8)
    plt.xlim(1e2, 1e4)

    plt.xlabel('nSamples')
    plt.ylabel(r'Percentage error on $H_0$')
    plt.savefig('../plots/statisticalErrorOnHubble.pdf')
    plt.show()
    
def perfectFittingFunction(nComponents=5):
    '''
    This is done via a cubic spline interpolator between
    each pdf
    '''

    inputHubbleParameter = 70.

    pklFile = 'pickles/perfectFittingFunction.pkl'

    sampleSizes, estimates = pkl.load(open(pklFile,'rb'))
    

    nIterations = estimates.shape[1]
    print(nIterations)
    plt.plot( sampleSizes[:-1], np.std(estimates, axis=1)[:-1]/inputHubbleParameter*100.)

    
    plt.yscale('log')
    plt.xscale('log')


    plt.xlabel('nSamples')
    plt.ylabel(r'Percentage error on $H_0$)')
    plt.show()

def getPredictedConstraints(inputHubbleParameter, \
                                nIterations = 10,\
                                nSampleSizes = 3,\
                                exactIndex=False,\
                                trueHubble=False,\
                                differentHalo=None):


    #Set up the hubble interpolator class to train and predict
    #the PDF
    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()

    #predictFeatures = hubbleInterpolaterClass.reshapedFeatures
    #print(predictFeatures[:,0])
    #predictFeatures = predictFeatures[ (predictFeatures[:,0] == 0.7) & (prepdbictFeatures[:,1] == 0.74), :]

    #predictFeatures[:,2] = np.linspace(-1.,-2.,predictFeatures.shape[0])
    #hubbleInterpolaterClass.plotPredictedPDF( predictFeatures )

    sampleSizes = 10**np.linspace(2,4,nSampleSizes)
    
    estimates = np.zeros(nSampleSizes)
    
    #Loop through each sample size
    for i, iSampleSize in enumerate(sampleSizes):
        print("Sample Size: %i" % (iSampleSize))
        
        samples = \
          getMCMCchainForSamplesSize(iSampleSize, nIterations, \
                        inputHubbleParameter, hubbleInterpolaterClass,\
                                         trueHubble=trueHubble,\
                                         differentHalo=differentHalo)
  
        if exactIndex:
            truth = \
              (samples[:,1] > 0.19) & (samples[:,1] < 0.21) &\
              (samples[:,2] > -1.95) & (samples[:,2] < -1.5)
            #estimates[i] = \
            #  norm.fit(samples[truth,0])[1]/np.sqrt(nIterations)*100

            median, upper, lower = \
              getModeAndError( samples[truth, 0])

        else:
            #estimates[i] = \
            #  norm.fit(samples[:,0])[1]/np.sqrt(nIterations)*100
            
            median, upper, lower =  getModeAndError( samples[:,0])

        estimates[i] = upper


    return sampleSizes, estimates


    
def getMCMCchainForSamplesSize( iSampleSize, nIterations,\
                    inputHubbleParameter, hubbleInterpolaterClass,\
                                    trueHubble=False, differentHalo=None):
    samples = None

    if trueHubble:
        pklFile = 'picklesTrueHubble/multiFitSamples_%i.pkl' % iSampleSize
    elif differentHalo is not None:
        pklFile = 'picklesDifferentHalo/multiFitSamples_%s_%i.pkl' % (differentHalo, iSampleSize)
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
                    trueHubble=trueHubble,differentHalo=differentHalo)
      
        fitHubbleClass = \
              fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)

        if samples is None:
            samples = fitHubbleClass.samples
        else:
            samples = np.vstack( (samples, fitHubbleClass.samples))


    pkl.dump(samples, open(pklFile, 'wb'))
        
    return samples


def selectRandomSampleOfTimeDelays( nSamples, hubbleParameter, \
                        hubbleInterpolaterClass, trueHubble=False,\
                                        differentHalo=None):
    '''
    From a given pdf randomly select some time delays

    '''

    pklFileName = \
      '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
      % (hubbleParameter)
    if differentHalo is not None:
      pklFileName = \
        '../output/CDM/selectionFunction/SF_%s_%i_lsst.pkl' \
        % (differentHalo, hubbleParameter)
      
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

    interpolateToTheseTimes=  \
      np.linspace(np.min(finalMergedPDFdict['x']), \
            np.max(finalMergedPDFdict['x']),nSamples*100)
    interpolateToTheseTimes= np.linspace(-1, 3, nSamples*100)
      
    if trueHubble | ( differentHalo is not None) :

        interpolatedProbClass = CubicSpline( finalMergedPDFdict['x'], \
                                             finalMergedPDFdict['y'])

        interpolatedProb = interpolatedProbClass( interpolateToTheseTimes)
    else:
        theta = np.array([0.7, 0.4, -1.75] )
        interpolatedProb = hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, theta )
  
      
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)
    
    #Interpolate the p
    
    
    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))
    
    bins = np.max([10, np.int(nSamples/100)])
    y, x = np.histogram(logTimeDelays, \
                    bins=np.linspace(-1,3,100), density=True)
                    
    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
    if trueHubble:
        xcentres += 2.5*dX
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.sqrt(np.cumsum(error**2)/(np.arange(len(error))+1))

    return {'x':xcentres, 'y':cumsumY, 'error':cumsumYError}

def getModeAndError( samples ):

    nIterations = np.int(samples.shape[0]/22000)

    print(nIterations)

    itPs = np.int(samples.shape[0]/nIterations)
    maxLike = []
    for i in range(nIterations):
        iSample = samples[i*itPs:(i+1)*itPs]
        y, x = \
          np.histogram( iSample[(iSample>0.6) &(iSample<0.8)], \
                            bins=np.linspace(0.5, 1., 50),  \
                        density=True)

        xc = (x[1:]+x[:-1])/2.
        dX =  x[1] - x[0]
        maxLike.append(xc[np.argmax(y)])
        #maxLike.append(np.percentile( iSample[(iSample>0.6) &(iSample<0.8)], 50))


        #yCumSum = np.cumsum(y)*dX
    
        #upper = xc[ np.argmin( np.abs(yCumSum[np.argmax(y)] + 0.34 - yCumSum))]
        #lower = xc[ np.argmin( np.abs(yCumSum[np.argmax(y)] - 0.34 - yCumSum))]
        #upper = xc[ np.argmin( np.abs( yCumSum -0.84) )]
        #lower = xc[ np.argmin( np.abs(yCumSum - 0.16) )]

        #error = np.sqrt(np.sum( y*(xc-maxLike)**2*dX ))
        #print(error)
    error = np.sqrt(np.sum((np.array(maxLike)-0.7)**2)/len(maxLike))
   

    maxLikeMean = np.median(np.array(maxLike))
    
    error  = np.std(samples[(samples>0.6) & (samples<0.8)])   
    return maxLikeMean, error*100, error*100

    
if __name__ == '__main__':
    #plotCornerPlot()
    nonPerfectFittingFunction()
