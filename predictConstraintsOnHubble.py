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
rcParams["font.size"] = 12
from scipy.ndimage import gaussian_filter as gauss
from matplotlib import gridspec
import time

    
    
def main():
    
    
    fig = plt.figure(figsize=(5,13))
    
    labels = \
        [r'$H_0',r'$z_{lens}$',\
            r'$\alpha$', r'$log(M)$',\
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$' ]

    gs = None

    inputPar = [0.7, 0.4, 1.75, 11.05, 0.3, 0.7]
    colors = ['red','blue']

    lims = [(0.,6.), (0.,40.), (0., 8.), (0.,1.0), (5.,11.5), (1.5,5.)]
    
    for iColor, minTime in enumerate([0., 10.]):
        sampleSizes, estimates = \
            getPredictedConstraints(minimumTimeDelay=minTime)

        if gs is None:
            gs = gridspec.GridSpec( estimates.shape[0], 1)
        for iPar in range(estimates.shape[0]):
            ax = plt.subplot(gs[iPar,0])

            ax.errorbar( sampleSizes*(1.+iColor/50.), \
                    estimates[iPar, 0,:]/inputPar[iPar]*100., \
                    yerr=estimates[iPar, 1,:]/inputPar[iPar]*100.,\
                  label=r'$\Delta t_{\rm min} = 0$ days', \
                      color=colors[iColor], fmt='o', capsize=3)

            if minTime == 10:
                constraints3000 = \
                  np.interp( 3000, sampleSizes, \
                                 estimates[iPar, 0,:]/inputPar[iPar]*100.)
                constraints400 = \
                  np.interp( 400, sampleSizes, \
                                 estimates[iPar, 0,:])/inputPar[iPar]*100.
     
  
                ax.plot( [400,400], [0,100], 'k:', \
                             label='LSST (Conservative)')
                ax.plot( [1e1,2e4], [constraints400,constraints400], 'k:')

    
                ax.plot( [3000,3000], [0,100], 'k--', \
                             label='LSST (Optimistic)')
                ax.plot( [1e1,2e4], [constraints3000,constraints3000],\
                             'k--')

            if iPar == 0:
                ax.plot( [1e1,2e4], [4.,4.], 'c--', label='This work')

                ax.plot(  [1e1,1e4], [1.5,1.5], 'r--', \
                        label='Limit of current model')

            
        #plt.yscale('log')
            ax.set_xscale('log')
            #ax.legend()
            ax.set_ylim(lims[iPar])
            ax.set_xlim(50, 4.5e3)
            if iPar != estimates.shape[0] -1:
                ax.set_xticklabels([])
            ax.set_xlabel('nSamples')
            ax.set_ylabel(r'$\Delta$ %s/%s' % (labels[iPar],labels[iPar]))
    fig.subplots_adjust(hspace=0)
    
    
    fig.align_ylabels()
    plt.savefig('../plots/statisticalErrorOnHubble.pdf')
    plt.show()

def getPredictedConstraints(nIterations = 100,\
                                nSampleSizes = 16,\
                                minimumTimeDelay=0.):


    hubbleInterpolaterClass = \
          hubbleModel.hubbleInterpolator()
    
    if minimumTimeDelay > 0:
        hubbleInterpolaterClass.getTrainingData(pklFile='picklesMinimumDelay/trainingDataWithMass.pkl')
    else:
        hubbleInterpolaterClass.getTrainingData('pickles/trainingDataWithMass.pkl')

    hubbleInterpolaterClass.getTimeDelayModel()
  
    sampleSizes = 10**np.linspace(1,4,nSampleSizes)
    
    estimates = None 


    #Loop through each sample size
    for i, iSampleSize in enumerate(sampleSizes[:-1]):
        print("Sample Size: %i" % (iSampleSize))

        samples = \
          getMCMCchainForSamplesSize(iSampleSize, nIterations, \
                                         hubbleInterpolaterClass,\
                                         minimumTimeDelay=minimumTimeDelay)

        if estimates is None:
            estimates = np.zeros((samples.shape[1], 2, nSampleSizes))

        #for iPar in range(samples.shape[1]):
        #    median, error =  getModeAndError( samples[:,iPar])
            
        estimates[:, 0, i] = np.std(samples, axis=0)   
        estimates[:, 1, i] = np.std(samples, axis=0)*0.1


    return sampleSizes, estimates


    
def getMCMCchainForSamplesSize( iSampleSize, nIterations,\
                    hubbleInterpolaterClass,\
                    minimumTimeDelay=0.):
    samples = None
    
    if minimumTimeDelay > 0:
        pklFile = 'picklesMinimumDelay/maxLike_%i.pkl' \
          % (iSampleSize)
    else:
        pklFile = 'exactPDFpickles/maxLike_%i.pkl' \
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
       
       
            
        if samples is None:
            samples = fitHubbleClass.maxLikeParams
        else:
            samples = np.vstack( (samples, fitHubbleClass.maxLikeParams))



    pkl.dump(samples, open(pklFile, 'wb'))
        
    return samples


def selectRandomSampleOfTimeDelays( nSamples, minimumTimeDelay=0.):
    '''
    From a given pdf randomly select some time delays

    '''
    #Real world inerpolator
              
    realWorldInterpolator = \
          hubbleModel.hubbleInterpolator()
    realWorldInterpolator.getTrainingData('pickles/trainingDataWithMass.pkl')
    realWorldInterpolator.getTimeDelayModel()

    if minimumTimeDelay == 0:
        minimumTimeDelay = 1e-3
        
    logMinimumTimeDelay = np.log10(minimumTimeDelay)

    interpolateToTheseTimes= np.linspace(-3, 3, np.int(1e6))
      
    inputParams = {'H0':0.7, 'OmegaM':0.3, 'OmegaK':0., 'OmegaL':0.7, \
        'zLens':0.40, 'densityProfile':-1.75, 'totalMass':11.1}
        
    interpolatedCumSum = \
          realWorldInterpolator.predictCDF( interpolateToTheseTimes, inputParams )
          
    interpolatedProb = undoCumSum( interpolatedCumSum)
      
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)

    #Interpolate the p
    

    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))

    logTimeDelays = logTimeDelays[ logTimeDelays > logMinimumTimeDelay ]
    
    bins = np.linspace(-1,4,1000)
    
    y, x = np.histogram(logTimeDelays, bins=bins, density=True)

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
    #print("nIteration %i" % nIterations)
    maxLike = []
    meanLike = []
    indivError = []
    for i in range(nIterations):
        iSample = samples[i*itPs:(i+1)*itPs]
        y, x = np.histogram( iSample, bins=50,  density=True)
        
        xc = (x[1:]+x[:-1])/2.
        #dX =  x[1] - x[0]
        maxLike.append(xc[np.argmax(y)])
        meanLike.append(np.median(iSample))

        
        indivError.append( np.std( iSample ))
        

    
    #error = np.sqrt(np.sum((np.array(maxLike)-0.7)**2)/len(maxLike))

    maxLikeMean = np.mean(np.array(maxLike))
    
    #error  = np.std(samples[(samples>0.65) & (samples<0.8)])
    error = np.nanmedian(np.array(indivError))

   
    errorOnError = np.std(np.array(indivError))


    return np.std(maxLike), errorOnError

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
    main()
##
