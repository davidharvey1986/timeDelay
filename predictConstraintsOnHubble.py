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
import hubbleInterpolatorClass as hubbleModel
from scipy.stats import norm
import matplotlib.lines as mlines
from matplotlib import rcParams
rcParams["font.size"] = 16
from scipy.ndimage import gaussian_filter as gauss

import time

    
    
def nonPerfectFittingFunction(nComponents=5, inputHubbleParameter=0.7):
    fig = plt.figure(figsize=(10,6))

    #####
    #sampleSizes, estimates = \
    #  getPredictedConstraints(inputHubbleParameter)
    #estimates /=inputHubbleParameter/100.
    #plt.plot( sampleSizes, estimates,\
    #              label="Exact PDF")

    #####
    '''
    '''
    sampleSizesNoMin, estimatesNoMin = getPredictedConstraints()

    plt.errorbar( sampleSizesNoMin, estimatesNoMin/inputHubbleParameter*100., \
                  label=r'$\Delta t_{\rm min} = 0$ days', color='red')
                  
    constraints3000 = np.interp( 3000, sampleSizesNoMin, estimatesNoMin)
    constraints400 = np.interp( 400, sampleSizesNoMin, estimatesNoMin)
    '''
    ########
    sampleSizes, estimates = \
      getPredictedConstraints(minimumTimeDelay=10)
      
    plt.errorbar( sampleSizes, estimates/inputHubbleParameter*100., \
                label=r'$\Delta t_{\rm min} = 10$ days', color='blue')
    #######
    
    
    '''
    
    
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

def getPredictedConstraints(nIterations = 10,\
                                nSampleSizes = 11,\
                                minimumTimeDelay=0.):



    hubbleInterpolaterClass = \
          hubbleModel.hubbleInterpolator( minimumTimeDelay=minimumTimeDelay)
    
    if minimumTimeDelay > 0:
        hubbleInterpolaterClass.getTrainingData(pklFile='picklesMinimumDelay/trainingDataWithMass.pkl')
    else:
        hubbleInterpolaterClass.getTrainingData('pickles/trainingDataWithMass.pkl')

    hubbleInterpolaterClass.getTimeDelayModel()
  
    sampleSizes = 10**np.linspace(2,4,nSampleSizes)
    
    estimates = np.zeros(nSampleSizes)


    #Loop through each sample size
    for i, iSampleSize in enumerate(sampleSizes):
        print("Sample Size: %i" % (iSampleSize))

        samples = \
          getMCMCchainForSamplesSize(iSampleSize, nIterations, \
                                         hubbleInterpolaterClass,\
                                         minimumTimeDelay=minimumTimeDelay)

                
        median, error =  getModeAndError( samples[:,0])

        estimates[i] = error
        
    
    return sampleSizes, estimates


    
def getMCMCchainForSamplesSize( iSampleSize, nIterations,\
                    hubbleInterpolaterClass,\
                    minimumTimeDelay=0.):
    samples = None
    
    if minimumTimeDelay > 0:
        pklFile = 'picklesMinimumDelay/multiFitSamples_withMass_%i.pkl' \
          % (iSampleSize)
    else:
        pklFile = 'exactPDFpickles/multiFitSamples_withMass_%i.pkl' \
          % iSampleSize
    

    if os.path.isfile(pklFile):
        return pkl.load(open(pklFile, 'rb'))
    logProb = np.array([])
    for iIteration in range(nIterations):
        print("Iteration: %i/%i" % (iIteration+1, nIterations))
        selectedTimeDelays = \
              selectRandomSampleOfTimeDelays( iSampleSize, \
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



    pkl.dump(samples, open(pklFile, 'wb'))
        
    return samples


def selectRandomSampleOfTimeDelays( nSamples, minimumTimeDelay=0.):
    '''
    From a given pdf randomly select some time delays

    '''
    #Real world inerpolator
    realWorldInterpolator = \
      hubbleModel.hubbleInterpolator( )
    
    realWorldInterpolator.getTrainingData('pickles/trainingDataForObsWithMass.pkl')

    realWorldInterpolator.getTimeDelayModel()

    if minimumTimeDelay == 0:
        minimumTimeDelay = 1e-3
        
    logMinimumTimeDelay = np.log10(minimumTimeDelay)

    interpolateToTheseTimes= np.linspace(-3, 3, 1e6)
      
    inputParams = {'H0':0.7, 'OmegaM':0.3, 'OmegaK':0., 'OmegaL':0.7, \
        'zLens':0.4, 'densityProfile':-1.75, 'totalMass':11.5}
        
    interpolatedCumSum = \
          realWorldInterpolator.predictCDF( interpolateToTheseTimes, inputParams )
          
    interpolatedProb = undoCumSum( interpolatedCumSum)
      
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)

    #Interpolate the p
    

    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))

    logTimeDelays = logTimeDelays[ logTimeDelays>logMinimumTimeDelay ]
    
    bins = np.linspace(-1,4,1000)
    y, x = np.histogram(logTimeDelays, \
                    bins=bins, density=True)

    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
        
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.sqrt(np.cumsum(error**2)/(np.arange(len(error))+1))

    xcentres += dX/2.

    return {'x':xcentres, 'y':cumsumY,  'error':cumsumYError}

def getModeAndError( samples ):

    

    itPs = 22000 # np.int(samples.shape[0]/nIterations)
    nIterations = np.int(samples.shape[0] / itPs)
    
    maxLike = []
    meanLike = []
    indivError = []
    for i in range(nIterations):
        iSample = samples[i*itPs:(i+1)*itPs]
        y, x = \
          np.histogram( iSample, \
                            bins=np.linspace(0.6, 0.8, 50),  \
                        density=True)
        
        xc = (x[1:]+x[:-1])/2.
        #dX =  x[1] - x[0]
        maxLike.append(xc[np.argmax(y)])
        meanLike.append(np.mean(iSample))

        
        indivError.append( np.std( iSample ))



    #error = np.sqrt(np.sum((np.array(maxLike)-0.7)**2)/len(maxLike))

    maxLikeMean = np.mean(np.array(maxLike))
    
    #error  = np.std(samples[(samples>0.65) & (samples<0.8)])
    error = np.nanmedian(np.array(indivError))

    if np.isfinite(error) == False:
        pdb.set_trace()
    
    errorOnError = np.std(np.array(indivError)) /np.sqrt(len(indivError))
    

    return np.mean(maxLike), np.median(indivError)

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
    nonPerfectFittingFunction()
##
