
import fitHubbleParameter as fitHubble
from hubbleInterpolatorClass import *
import predictConstraintsOnHubble as predictHubble
predictHubble.rcParams["font.size"] = 12
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
def main(nBins=6):

    powerLawIndex = np.linspace(-2,-1.,nBins)
    redshift = np.linspace(0.22,1.02,nBins) 
    hubbleParam = np.linspace(.5,1.,nBins)
    interpolateToTheseTimes=   np.linspace(0, 3, 1000)
    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData(pklFile='picklesMinimumDelay/trainingData.pkl')
    hubbleInterpolaterClass.extractPrincipalComponents()

    #for i in 10**np.linspace(-4,4, 10):
     #   hubbleInterpolaterClass.learnPrincipalComponents(weight=i)
    hubbleInterpolaterClass.learnPrincipalComponents()
    for i, iPL in enumerate(powerLawIndex):
        print(i)
        for j, iRedshift in enumerate(redshift):
            for k, iHubble in enumerate(hubbleParam):
                interpolatedProb = \
                  hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, \
                                                np.array([iHubble,iRedshift, iPL] ))

                plt.plot(interpolateToTheseTimes, interpolatedProb)
    plt.show()


def plotPCAanalysis(nComponents=7):
    
    #For aesthetics                                                         
    jet = cm = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=1, vmax=10)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #####
    
    fig, axarr = plt.subplots( 2, 2, figsize=(14,6) )
    box = dict(pad=0, color='white')
    plt.subplots_adjust(hspace=0)

    principalComponents = np.arange(2,10)
    AICc = []
    
    for iNumPrincipalComponents in principalComponents:
        print(iNumPrincipalComponents)

        hubbleInterpolaterClass = \
          hubbleInterpolator(nPrincipalComponents=np.int(iNumPrincipalComponents))
        hubbleInterpolaterClass.getTrainingData(pklFile='exactPDFpickles/trainingData.pkl')
        hubbleInterpolaterClass.extractPrincipalComponents()
        hubbleInterpolaterClass.learnPrincipalComponents()
        AICc.append(hubbleInterpolaterClass.getGaussProcessLogLike())
        timeDelays = hubbleInterpolaterClass.timeDelays
        dX = timeDelays[1] - timeDelays[0]

        diffPredict = \
          np.zeros( ( hubbleInterpolaterClass.features.shape[0],\
                        len(timeDelays)))
        diffPCA = \
          np.zeros( ( hubbleInterpolaterClass.features.shape[0],\
                            len(timeDelays)))
                            
        for i in range(hubbleInterpolaterClass.features.shape[0]):
       
            inputFeatures = hubbleInterpolaterClass.reshapedFeatures[i,:]
        
            predictPDF = \
              hubbleInterpolaterClass.predictPDF( timeDelays, inputFeatures)
          
            truePDF = hubbleInterpolaterClass.pdfArray[i,:]

            trueComponents = \
              hubbleInterpolaterClass.principalComponents[i,:]
            pcaPDF =  \
              hubbleInterpolaterClass.pca.inverse_transform( trueComponents )
        
            truePDF = np.cumsum(truePDF)/np.sum(truePDF)
            predictPDF = np.cumsum(predictPDF)/np.sum(predictPDF)
            pcaPDF = np.cumsum(pcaPDF)/np.sum(pcaPDF)

            diffPredict[i,:] = truePDF - predictPDF
            diffPCA[i, :] =  truePDF - pcaPDF
            if iNumPrincipalComponents==6:
                axarr[1,0].plot( timeDelays, truePDF - predictPDF, \
                           alpha=0.1, color='grey')

        
                axarr[0,0].plot( timeDelays, truePDF - pcaPDF, \
                                     alpha=0.1, color='grey')

        axarr[0,0].plot( timeDelays,  np.mean( diffPCA, axis=0), \
                color=scalarMap.to_rgba(iNumPrincipalComponents))
                
        axarr[1,0].plot( timeDelays, np.mean( diffPredict, axis=0), \
                color=scalarMap.to_rgba(iNumPrincipalComponents))

        meanDiff = np.abs(np.mean( diffPCA, axis=0))*10000
        
        axarr[0,1].plot( timeDelays[:-1], meanDiff[:-1],\
                label='%i PCs' % iNumPrincipalComponents,\
                    color=scalarMap.to_rgba(iNumPrincipalComponents))
        meanDiffPred = np.abs(np.mean( diffPredict, axis=0))*10000

        axarr[1,1].plot( timeDelays[:-1], meanDiffPred[:-1], \
                label='%i PCs' % iNumPrincipalComponents,\
                color=scalarMap.to_rgba(iNumPrincipalComponents))

    axarr[0,1].legend(bbox_to_anchor=(1.25, 1.))
    axarr[0,0].set_xticklabels([])
    axarr[0,1].set_xticklabels([])

    axarr[0,0].set_ylabel(r'CDF$_T$ - CDF$_{PCA}$', bbox=box)
    axarr[0,1].set_ylabel(r'|$\langle\Delta$CDF$\rangle$| / $10^{-4}$', bbox=box)
    axarr[1,0].set_ylabel(r'CDF$_T$ - CDF$_{\bar{T}}$', bbox=box)
    axarr[1,1].set_ylabel(r'$\langle\Delta$CDF$\rangle$ / $10^{-4}$', bbox=box)
    axarr[1,1].set_xlabel(r'log($\Delta t$/ days)')
    axarr[1,0].set_xlabel(r'log($\Delta t$/ days)')
    axarr[0,1].set_yscale('log')
    axarr[1,1].set_yscale('log')
    fig.align_ylabels(axarr)
    plt.savefig('../plots/pcaAnalysis.pdf')
    plt.show()

    fig = plt.figure(figsize=(8,4))
    plt.plot(principalComponents, np.array(AICc)-np.max(np.array(AICc)))
    plt.ylabel('Change in Information Criterion ($\Delta$AIC)')
    plt.xlabel('Number of Principal Components')
    plt.show()
        
def plotFinalHubbleModel( sampleSize=10000,hubbleParameter=70):
    '''
    plot the fitted hubble model from predictConstraints
    '''
    
    pklFileName = \
      '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
      % (hubbleParameter)

      
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
    
    samples = \
      predictHubble.getMCMCchainForSamplesSize(sampleSize, 10, \
                                    hubbleParameter, \
                                    None, trueHubble=True)

    dX = finalMergedPDFdict['x'][1] - finalMergedPDFdict['x'][0]

    bestFitParams = getBestFitParameter( samples )

    timeDelays = np.linspace(-1,3,100)
    dXtime = timeDelays[1]-timeDelays[0]

    hubbleInterpolaterClass = hubbleInterpolator(nPrincipalComponents=4)
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()
    timeDelays = hubbleInterpolaterClass.timeDelays

    predictPDF = \
          hubbleInterpolaterClass.predictPDF( finalMergedPDFdict['x'], \
                                                bestFitParams)


    #predictPDF /= np.sum(predictPDF*dX)
    finalMergedPDFdict['y'] /= np.sum(finalMergedPDFdict['y']*dX)
    plt.plot( finalMergedPDFdict['x'] + 1.8*dXtime,\
                  1-np.cumsum(finalMergedPDFdict['y'])/np.sum(finalMergedPDFdict['y']))
    plt.plot( finalMergedPDFdict['x'],1.-predictPDF)
    plt.show()

def getBestFitParameter( samples ):
    nSamples = samples.shape[0]
    bestFitParams = []
    for iDim in range(samples.shape[1]):
        y, x = np.histogram(samples[:,iDim], \
            bins=np.int(nSamples/1000), density=True)
        xc = (x[1:]+x[:-1])/2.
        bestFitParams.append( xc[np.argmax(y)])

    return np.array(bestFitParams)

        
if __name__ == '__main__':
    #main()
    #plotPCAanalysis()
    plotFinalHubbleModel()
