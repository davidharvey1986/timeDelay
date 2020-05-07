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
import lensing_parameters as lensing
from matplotlib import colors as mcolors


def main(pklFile='pickles/fitDataToModelPoisson.pkl'):
        

    ndim = 6
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(8,8))

    selectedTimeDelays = getObservations()

    hubbleInterpolaterClass = hubbleInterpolator()

    hubbleInterpolaterClass.getTrainingData('exactPDFpickles/noCosmology.pkl')
        
    hubbleInterpolaterClass.getTimeDelayModel(modelFile='pickles/noCosmology.pkl')
    

    if not os.path.isfile( pklFile ):
        fitHubbleClass = \
          fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)

        pkl.dump(fitHubbleClass.samples,open(pklFile,'wb'))
        samples = fitHubbleClass.samples
    else:
        
        samples = pkl.load(open(pklFile,'rb'))

    parRange = [[0.55,0.8],[0.45,0.6],[-2.,-1.6]]
    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$', \
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$' ]
    nsamples = samples.shape[0]
    #plotMockedData( figcorner)
    
    corner.corner(samples,  levels=[0.68], bins=20,\
                    plot_datapoints=False, labels=labels, fig=figcorner,\
                      weights=np.ones(nsamples)/nsamples, color='black', \
                      hist_kwargs={'linewidth':3.},\
                      contour_kwargs={'linewidths':3.}, smooth=True)
    '''
    axarr[1,1].set_xlim( 0.3, 1.0)
    axarr[2,2].set_xlim( -2, -1.5)
    axarr[1,0].set_ylim( 0.3, 1.0)
    axarr[2,0].set_ylim(  -2, -1.5)
    axarr[2,1].set_ylim(  -2, -1.5)
    axarr[2,1].set_xlim( 0.3, 1.0)
    axarr[0,0].set_xlim( 0.6, 0.8)
    axarr[1,0].set_xlim( 0.6, 0.8)
    axarr[2,0].set_xlim( 0.6, 0.8)
    axarr[0,0].set_xlim(0.6, 0.8)
    axarr[1,1].set_xlim(0.3, 1.0)
    axarr[2,2].set_xlim(-2, -1.5)
    axarr[0,0].set_ylim(0., 0.1)
    axarr[1,1].set_ylim(0., 0.3)
    axarr[2,2].set_ylim(0., 0.3)
    '''
    for i in range(samples.shape[1]):
        lo, med, hi = np.percentile(samples[:,i],[16,50,84])
        print( med, med-lo, hi-med)

    pdb.set_trace()
    plt.savefig('../plots/fitDataToModel.pdf')

    plt.show()


  
    

def plotMockedData(fig, nIterations=10):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]


    pklFile = 'exactPDFpickles/multiFitSamples_30.pkl'
    samples = pkl.load(open(pklFile,'rb'))
    nsamples = samples.shape[0]
    nSamplePerIt = np.int(nsamples / nIterations)
    error = []
    mean  = []
    for i in range(nIterations):
        error.append(np.std(samples[i*nSamplePerIt:(i+1)*nSamplePerIt,0]))
        mean.append(np.mean(samples[i*nSamplePerIt:(i+1)*nSamplePerIt,0]))
        corner.corner(samples[i*nSamplePerIt:(i+1)*nSamplePerIt,:], levels=(0.68,), smooth=True,\
                          fig=fig, bins=25, density=True,plot_datapoints=False,\
                            weights=np.ones(nSamplePerIt)/nSamplePerIt, \
                          color=sorted_names[np.random.randint(0,len(sorted_names))])
                          
    corner.corner(samples, levels=(0.68,), smooth=True,\
                     fig=fig, bins=25, density=True,plot_datapoints=False,\
                            weights=np.ones(nsamples)/nsamples, \
                          color='red', \
                      hist_kwargs={'linewidth':2.},\
                      contour_kwargs={'linewidths':2.})
    pdb.set_trace()

    
def getObservations(nBins=20):
    data = np.loadtxt( '../data/oguriTimeDelays.txt',\
                           dtype=[('name',object), ('zs', float), \
                                    ('zl', float),\
                                      ('timeDelays', float)])

    logTimeDelays = np.log10(np.sort(data['timeDelays']))
    nSamples = len(data)
    y, x = np.histogram(logTimeDelays, \
        bins=np.linspace(-1,3,nBins+1), density=True)
                    
    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
    #xcentres += 1.3*dX
        
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.ones(nBins)/nSamples/2. #np.sqrt(np.cumsum(error**2)/(np.arange(len(error))+1))


    
    #y = np.cumsum(np.ones(len(data)))/len(data)
   # error  = np.ones(len(data))/len(data)
   # cumsumYError = np.sqrt(np.cumsum(error**2)/(np.arange(len(error))+1))

    ind = data['zl'] > 0
    dl = np.array([ lensing.ang_distance(i) for i in data['zl'][ind]])
    dls = \
        np.array([lensing.ang_distance(data['zs'][ind][i], \
                                z0=data['zl'][ind][i])\
               for i in range(len(data[ind]))])
    ds = np.array([ lensing.ang_distance(i) for i in data['zs'][ind]])
    print("GAUSS PRIOR",norm.fit(data['zl']))

    #-1.8*dX
    return {'x':xcentres, 'y':cumsumY, 'error':cumsumYError}


if __name__ == '__main__':
    main()
    
    
    
