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


def main( dmModel='CDM', nBootStraps=1000 ):

    labels = \
        [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',\
            r'$\alpha$', r'$log(M(<5kpc)/M_\odot)$',\
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$' ]
  
    
    pklFile = 'pickles/%s_monteCarlo_withMass_LCDM.pkl' % dmModel
    if os.path.isfile( pklFile ):
        allSamples = pkl.load(open(pklFile,'rb'))
    else:
        allSamples = bootStrapData( dmModel='CDM', nBootStraps=nBootStraps )
        pkl.dump(allSamples, open(pklFile, 'wb'))
        
    nsamples = allSamples.shape[0]
    #plotMockedData( figcorner)
    ndim = allSamples.shape[1]
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(12,12))

    corner.corner(allSamples,  levels=[0.68], bins=20,\
                    plot_datapoints=False, labels=labels, fig=figcorner,\
                      weights=np.ones(nsamples)/nsamples, color='black', \
                      hist_kwargs={'linewidth':3.},\
                      contour_kwargs={'linewidths':3.}, smooth=True)
                      
    for i in range(allSamples.shape[1]):
        lo, med, hi = np.percentile(allSamples[:,i],[16,50,84])
        print( med, med-lo, hi-med)


    figcorner.savefig('../plots/%s_LCDMresults.pdf' % dmModel)
    plt.show()

def bootStrapData( dmModel='CDM', nBootStraps=10):
    dataSelectionFunction = \
          '../output/%s/selectionFunction/SF_data.pkl' % dmModel
          
    hubbleInterpolaterClass = \
          hubbleInterpolator(allDistributionsPklFile=dataSelectionFunction,\
                                 regressorNoiseLevel=1e-3, dmModel=dmModel)

    hubbleInterpolaterClass.getTrainingData('%spickles/trainingDataForObsWithMass.pkl'\
                                                % dmModel)
        
    hubbleInterpolaterClass.getTimeDelayModel()
    allSamples  = None
        
    for iBootStrap in np.arange(nBootStraps):
        print("%i/%i" % (iBootStrap+1, nBootStraps))
        iParams = fitModelToDataBootStrap(hubbleInterpolaterClass)

        if allSamples is None:
            allSamples = iParams
        else:
            allSamples = np.vstack((allSamples, iParams))

    return allSamples
            
def fitModelToDataBootStrap( model ):
    
        
    selectedTimeDelays = getObservations(monteCarlo=True)

    fitHubbleClass = \
          fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    model)

    return fitHubbleClass.maxLikeParams
  
    

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


    
def getObservations(nBins=20, monteCarlo=False):
    data = np.loadtxt( '../data/oguriTimeDelays.txt',\
                           dtype=[('name',object), ('zs', float), \
                                    ('zl', float),\
                                      ('timeDelays', float), \
                                      ('timeDelayErr', float) ])

    if monteCarlo:
        timeDelays = np.random.randn( len(data))*data['timeDelayErr']+data['timeDelays']
    else:
        timeDelays = data['timeDelays']

    logTimeDelays = np.log10(np.sort(timeDelays))
    nSamples = len(data)
    y, x = np.histogram(logTimeDelays, \
        bins=np.linspace(-1,3,nBins+1), density=True)
                    
    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
        
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

    return {'x':xcentres, 'y':cumsumY, 'error':cumsumYError}

def writeResultsToTex(nMonteCarlo=100):


    models = ['Fixed', '$\Lambda$CDM', '$\Lambda$CDMk']
    pklEndings = ['', '_LCDM','_LCDMk']

    labels = \
        ['Model', r'$H_0/($km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$', r'log$(M(<5$kpc$)/M_\odot)$',\
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$' ]

    resultsFile = open('../plots/results.tex', 'w')
    resultsFile.write('\\begin{table*}\n')
    resultsFile.write('\\begin{center}\n')
    resultsFile.write('\caption{\label{tab:haloStats}}\n')
    resultsFile.write('\\begin{tabular}{cccccccc}\n')
    resultsFile.write('\hline\n')
    resultsFile.write('&'.join(labels)+'\\\\\n')
    resultsFile.write('\hline\n')
    
    for iModel in np.arange(len(models)):
        allSamples = None

        line = models[iModel]
        for iMonteCarlo in np.arange(nMonteCarlo):
            pklFile = '%i_monteCarlo_withMass%s.pkl' \
              % (iMonteCarlo,pklEndings[iModel])

            samples = main(pklFile=pklFile, monteCarlo=True)

            if allSamples is None:
                allSamples = samples
            else:
                allSamples = np.vstack((allSamples, samples))

        for i in range(allSamples.shape[1]):
            
            lo, med, hi = np.percentile(allSamples[:,i],[16,50,84])
            if i == 0 :
                lo *= 100
                med *= 100
                hi *= 100
            
            line = line+'& $ %0.2g_{-%0.1g}^{+%0.1g} $ ' % \
              (med, med-lo, hi-med)
        line = line+'\\\\ \n'
        resultsFile.write(line)

    resultsFile.write('\hline\n')
    resultsFile.write('\end{tabular}\n')
    resultsFile.write('\end{center}\n')
    resultsFile.write('\\end{table*}\n')

    resultsFile.close()
if __name__ == '__main__':
    #writeResultsToTex()
    main(dmModel='CDM')
    main(dmModel='L11p2')
    main(dmModel='L8')
    #monteCarlo()
    
    
    
