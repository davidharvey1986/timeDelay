


'''
I want to look at the sensitiit of the predictor to a range in masses by tranining on a mass cut
and sampling away from that

Assumpyion is that the training sample is smaller mass than the true sample

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

    
    
def sensivitiyToDifferentHalo(nComponents=5, inputHubbleParameter=0.7):
    
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
    sampleSizesNoMin, estimatesNoMin = getPredictedConstraints(massCut=11.1)


    plt.errorbar( sampleSizesNoMin, estimatesNoMin/inputHubbleParameter*100., \
                  label=r'$\Delta t_{\rm min} = 0$ days', color='red')
                  

    plt.xscale('log')
    plt.legend()
    plt.ylim(0., 6)
    plt.xlim(1e2, 1e4)

    plt.xlabel('nSamples')
    plt.ylabel(r'Percentage error on $H_0$')
    plt.savefig('../plots/statisticalErrorOnHubble.pdf')
    plt.show()

def getPredictedConstraints(nIterations = 2,nSampleSizes = 3,massCut=0.):

    #this is what i will fit with, so this will training on things less than
    #the mass cut
    hubbleInterpolaterClass = hubbleModel.hubbleInterpolator( massCut=[0., massCut] )
    
    hubbleInterpolaterClass.getTrainingData('pickles/trainingDataMassCutLower_%0.1f.pkl' % massCut )

    hubbleInterpolaterClass.getTimeDelayModel()
    
    sampleSizes = 10**np.linspace(2,4,nSampleSizes)
    
    estimates = np.zeros(nSampleSizes)


    #Loop through each sample size
    for i, iSampleSize in enumerate([10000]):
        print("Sample Size: %i" % (iSampleSize))

        samples = \
          getMCMCchainForSamplesSize(iSampleSize, nIterations, \
                                         hubbleInterpolaterClass,\
                                         massCut=massCut)

                
        median, error =  getModeAndError( samples[:,0])

        estimates[i] = error
        
    
    return sampleSizes, estimates


    
def getMCMCchainForSamplesSize( iSampleSize, nIterations,\
                    hubbleInterpolaterClass, massCut=0.):
    samples = None

    if massCut > 0:
        pklFile = 'picklesMassCut/multiFitSamples_%i_%0.1f.pkl' \
          % (iSampleSize, massCut)
    else:
        pklFile = 'exactPDFpickles/multiFitSamples_%i.pkl' \
          % iSampleSize
    

    if os.path.isfile(pklFile):
        return pkl.load(open(pklFile, 'rb'))
    
    for iIteration in range(nIterations):
        print("Iteration: %i/%i" % (iIteration+1, nIterations))
        selectedTimeDelays = \
              selectRandomSampleOfTimeDelays( iSampleSize, massCut=massCut)
      
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


def selectRandomSampleOfTimeDelays( nSamples, massCut=0.):
    '''
    From a given pdf randomly select some time delays

    '''
    #Real world inerpolator so it neeeds a mass cut of >massCut
    #arbitatiry large upper bound to make sure all are included (i.e 13)
    realWorldInterpolator = \
      hubbleModel.hubbleInterpolator( massCut=[massCut, 13.])
    
    realWorldInterpolator.getTrainingData('pickles/trainingDataMassCutUpper_%0.1f.pkl' % massCut)

    realWorldInterpolator.getTimeDelayModel()

    
    interpolateToTheseTimes= np.linspace(-3, 3, 1e6)

    theta = {'H0':0.7, 'OmegaM':0.3, 'OmegaK':0., \
    'OmegaL':0.7, 'zLens':0.56, 'densityProfile':-1.75, 'totalMass':11.2}
    interpolatedCumSum = \
          realWorldInterpolator.predictCDF( interpolateToTheseTimes, \
                                                  theta )
    interpolatedProb = undoCumSum( interpolatedCumSum)
      
      
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)

    #Interpolate the p
    

    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))

        
    bins = np.linspace(-1,4,1000)
    y, x = np.histogram(logTimeDelays, \
                    bins=bins, density=True)

    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
        
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.sqrt(np.cumsum(error**2)/(np.arange(len(error))+1))

    xcentres += dX/2.
    pdb.set_trace()
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
    
    pdb.set_trace()
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
    sensivitiyToDifferentHalo()
##
