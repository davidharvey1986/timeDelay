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
rcParams["font.size"] = 12
from scipy.ndimage import gaussian_filter as gauss
import lensing_parameters as lensing
from matplotlib import colors as mcolors
from fitDataToModel import getObservations
from matplotlib import gridspec
def main( nMonteCarlo=20, nNoise=8 ):

    params = np.array([])

    allSamples  = None
    maxLike = []
    labels = \
        [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',\
            r'$\alpha$', r'$log(M(<5kpc)/M_\odot)$',\
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$' ]
    noiseLevel = np.logspace(-4,-1,nNoise)
    HubbleEstimate = np.zeros((nNoise, 3))
    
    for iNoise, iNoiseLevel in enumerate(noiseLevel):
        for iMonteCarlo in np.arange(nMonteCarlo):
            pklFile = '%i_monteCarlo_noise_%0.2f.pkl' % \
              (iMonteCarlo, iNoiseLevel)

            samples = monteCarlo(pklFile=pklFile, monteCarlo=True, \
                                 noiseLevel=iNoiseLevel)

            if allSamples is None:
                allSamples = samples
            else:
                allSamples = np.vstack((allSamples, samples))
            
                         

        lo, med, hi = np.percentile(allSamples[:,0],[16,50,84], axis=0)

        HubbleEstimate[iNoise, 0] = med
        HubbleEstimate[iNoise, 1] = med -lo
        HubbleEstimate[iNoise, 2] =  hi-med

    gs = gridspec.GridSpec(2,1)
    ax = plt.subplot(gs[0,0])
    ax.errorbar( noiseLevel, HubbleEstimate[:,0]/0.7, \
                      yerr=HubbleEstimate[:,1:].T/0.7, fmt='o', capsize=2)
    ax.set_xlabel('Kernel Noise Level')
    ax.set_ylabel(r'$H_0/70$ km/s/Mpc')
    
    ax.set_xscale('log')
    plt.savefig('../plots/dependenceOnNoise.pdf')
    plt.show()
        
def monteCarlo(pklFile='fitDataToModel_withMass_LCDMk_noZ0.6.pkl', \
             monteCarlo=False, noiseLevel=4e-3):
    
    if  not os.path.isfile( 'pickles/monteCarlosOfData/%s' % pklFile ):
        
        selectedTimeDelays = getObservations(monteCarlo=monteCarlo)

        dataSelectionFunction = '../output/CDM/selectionFunction/SF_data.pkl'
          
        hubbleInterpolaterClass = \
          hubbleInterpolator(allDistributionsPklFile=dataSelectionFunction,\
                                 regressorNoiseLevel=noiseLevel)

        hubbleInterpolaterClass.getTrainingData('pickles/trainingDataForObsWithMass.pkl')
        
        hubbleInterpolaterClass.getTimeDelayModel()
    

    
        fitHubbleClass = \
          fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)

        pkl.dump(fitHubbleClass.samples,open('pickles/monteCarlosOfData/%s' % pklFile,'wb'))
        samples = fitHubbleClass.samples
    else:
        
        samples = pkl.load(open( 'pickles/monteCarlosOfData/%s' % pklFile,'rb'))

    if monteCarlo:
        return samples
    
    ndim = samples.shape[1]
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(8,8))

    parRange = [[0.55,0.8],[0.45,0.6],[-2.,-1.6]]
    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$', r'$log(M(<5kpc)/M_\odot)$'\
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$', r'$Sigma' ]
    nsamples = samples.shape[0]
    #plotMockedData( figcorner)
    
    corner.corner(samples,  levels=[0.68], bins=20,\
                    plot_datapoints=False, labels=labels, fig=figcorner,\
                      weights=np.ones(nsamples)/nsamples, color='black', \
                      hist_kwargs={'linewidth':3.},\
                      contour_kwargs={'linewidths':3.}, smooth=True)
  
    for i in range(samples.shape[1]):
        lo, med, hi = np.percentile(samples[:,i],[16,50,84])
        print( med, med-lo, hi-med)

    plt.savefig('../plots/fitDataToModel.pdf')

    plt.show()


 
if __name__ == '__main__':
    main()
    
    
    
