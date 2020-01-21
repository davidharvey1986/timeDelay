'''
I want to prediuct the constraints on teh hubble parameter from the
estiamted PDFs generated

I will fit a double powerLaw

'''
import plotAsFunctionOfDensityProfile as getDensity
from powerLawFit import *
from interpolateSourcePlane import *
import fitHubbleParameter as fitHubble
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis
import corner as corner

def nonPerfectFittingFunction(nComponents=4):
    inputHubbleParameter = 70.
    pklFile = 'multiParameterFit.pkl'
    if os.path.isfile(pklFile):
        sampleSizes, estimates = pkl.load(open(pklFile, 'rb'))
    else:
        sampleSizes, estimates = \
          getPredictedConstraints(inputHubbleParameter)
        pkl.dump([sampleSizes, estimates], open(pklFile, 'wb'))
      
    plt.plot( sampleSizes[:-1], np.std(estimates, axis=1)[:-1]/inputHubbleParameter*100.)
    pdb.set_trace()
    plt.yscale('log')
    plt.xscale('log')


    plt.xlabel('nSamples')
    plt.ylabel(r'$\sigma_{H_0}/H_0$')
    plt.show()
    
def perfectFittingFunction(nComponents=5):


    inputHubbleParameter = 70.

    pklFile = 'perfectFittingFunction.pkl'

    sampleSizes, estimates = pkl.load(open(pklFile,'rb'))
    

    nIterations = estimates.shape[1]
    print(nIterations)
    plt.plot( sampleSizes[:-1], np.std(estimates, axis=1)[:-1]/inputHubbleParameter*100.)
    plt.yscale('log')
    plt.xscale('log')


    plt.xlabel('nSamples')
    plt.ylabel(r'$\sigma_{H_0}/H_0$')
    plt.show()

def getPredictedConstraints(inputHubbleParameter, \
                                nIterations = 10,\
                                nSampleSizes = 11):
                                
    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()

    predictFeatures = hubbleInterpolaterClass.reshapedFeatures
    #print(predictFeatures[:,0])
    #predictFeatures = predictFeatures[ (predictFeatures[:,0] == 0.7) & (predictFeatures[:,1] == 0.74), :]

    #predictFeatures[:,2] = np.linspace(-1.,-2.,predictFeatures.shape[0])
    #hubbleInterpolaterClass.plotPredictedPDF( predictFeatures )

    sampleSizes = 10**np.linspace(2,4,nSampleSizes)
    color=['blue','red','green']

    
    estimates = np.zeros((nSampleSizes, nIterations))
    
    for i, iSampleSize in enumerate(sampleSizes):
        samples = None
        for iIteration in range(nIterations):
            print("Sample Size: %i/%i, iteration: %i/%i" %\
                      (i+1, nSampleSizes, iIteration+1, nIterations))
            selectedTimeDelays = \
              selectionFunctionZmed( iSampleSize, inputHubbleParameter, hubbleInterpolaterClass)
      
            fitHubbleClass = \
              fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)

            if samples is None:
                samples = fitHubbleClass.samples
            else:
                samples = np.vstack( (samples, fitHubbleClass.samples))
                
            estimates[i,iIteration] = \
              fitHubbleClass.params['params'][0]
        pkl.dump(samples, open('pickles/multiFitSamples_%i.pkl' % iSampleSize))
        
    return sampleSizes, estimates


def selectionFunctionZmed( nSamples, hubbleParameter, hubbleInterpolaterClass ):
    '''
    From a given pdf randomly select some time delays
    '''

    pklFileName = \
      '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
      % (hubbleParameter)


    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))



    #interpolatedProbClass = CubicSpline( finalMergedPDFdict['x'], p)
    interpolateToTheseTimes=  \
      np.linspace(np.min(finalMergedPDFdict['x']), np.max(finalMergedPDFdict['x']),nSamples*100)
    interpolateToTheseTimes=  \
      np.linspace(-3, 4, nSamples*100)
    interpolatedProb = \
      hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, np.array([0.7, 0.2, -1.9] ))

    #interpolatedProb = interpolatedProbClass( interpolateToTheseTimes)
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)
    
    #Interpolate the p
    
    
    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))
    
    bins = np.max([10, np.int(nSamples/100)])
    y, x = np.histogram(logTimeDelays, bins=np.linspace(-3,4,100), density=True)
    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2. + dX/2.
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.sqrt(np.cumsum(error**2))/np.sqrt(np.arange(len(error))+1) 

    
    return {'x':xcentres, 'y':cumsumY, 'error':cumsumYError}

class hubbleInterpolator:
    '''
    The idea is to model the PDF and interpolate between source
    planes
    '''

    def __init__( self, nPrincipalComponents=6, selectionFunctionZmed=2.0):
        
        self.nPrincipalComponents = nPrincipalComponents
        self.SFzMed = selectionFunctionZmed
        
    def getTrainingData( self, pklFile = 'trainingData.pkl' ):
        '''
        Import all the json files from a given lens redshift
        '''
        
        self.hubbleParameters = \
          np.array([50., 60., 70., 80., 90., 100.])
          
        if os.path.isfile( pklFile):
            self.features,  self.timeDelays, self.pdfArray =  \
              pkl.load(open(pklFile, 'rb'))
            self.nFeatures = len(self.features.dtype)
            self.features['hubbleParameter'] /= 100.
            return
        
        self.timeDelayDistClasses = []

        features = np.array([], dtype=[('hubbleParameter', float), ('zLens',float), ('densityProfile', float)])
        
        self.nFeatures = len(features.dtype)


        self.pdfArray = None
        dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'

        allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
        rGrid = getDensity.getRadGrid()
        
        for iColor, iHubbleParameter in enumerate(self.hubbleParameters):
            for iFile in allFiles:
                fileName = iFile.split('/')[-1]
                zLensStr = fileName.split('/')[-2]
                zLens = np.float(zLensStr.split('_')[1])

                pklFileName = \
                  '../output/CDM/selectionFunction/SF_%s_%s_%i_lsst.pkl' \
                  % (zLensStr,fileName,iHubbleParameter )
              
            
                finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
                finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])
            
                if self.pdfArray is None:
                    self.pdfArray = finalMergedPDFdict['y']
                else:
                    self.pdfArray = \
                      np.vstack((self.pdfArray, finalMergedPDFdict['y']))

                densityProfile = getDensity.getDensityProfileIndex(iFile, rGrid=rGrid)[0]
                print(densityProfile)
                
                iFeature = np.array( [(iHubbleParameter, zLens, densityProfile )], \
                    dtype = features.dtype)

                
                features = np.append( features, iFeature)
            
        self.features = features
        self.timeDelays =  finalMergedPDFdict['x']

        pkl.dump([self.features,  self.timeDelays, self.pdfArray], \
                     open(pklFile, 'wb'))
                     
    def extractPrincipalComponents( self ):
        '''
        Now extract the principal components for each halo
        '''
        self.pca = PCA(n_components=self.nPrincipalComponents)
        self.pca.fit( self.pdfArray )
        self.principalComponents = self.pca.transform( self.pdfArray )
            
    def learnPrincipalComponents( self ):
        '''
        Using a mixture of Gaussian Processes 
        predict the distributions of compoentns
        It will return a list of nPrincipalComponent models that try to
        learn each component to the data
        '''
        
        #Now regress each component to any hubble value
        
        #self.learnedGPforPCAcomponent = []
        self.predictor = []

        kernel =  0.608**2*Matern(length_scale=0.484, nu=3/2) + \
          WhiteKernel(noise_level=1e3)
        
        #kernel = RationalQuadratic()

        self.nPDF = len(self.features)
        
        self.reshapedFeatures = \
          self.features.view('<f8').reshape((self.nPDF,self.nFeatures))
          
        for i in range(self.nPrincipalComponents):
            
            gaussProcess = \
              GaussianProcessRegressor( alpha=10, kernel=kernel)

            print(self.features.shape)
            gaussProcess.fit( self.reshapedFeatures, self.principalComponents[:,i])

            
            self.predictor.append(gaussProcess)
            
            #cubicSpline = CubicSpline(self.hubbleParameters, self.principalComponents[:,i])
            #self.cubicSplineInterpolator.append(cubicSpline)
       
           
            
    def predictPDF( self, timeDelays, inputFeatures ):
        '''
        For now compare to the trained data
        '''
        
        predictedComponents = \
          np.zeros(self.nPrincipalComponents)

        
        for iComponent in range(self.nPrincipalComponents):
            
            predictedComponents[iComponent] = \
              self.predictor[iComponent].predict(inputFeatures.reshape(1,-1))

        predictedTransform = \
          self.pca.inverse_transform(  predictedComponents )

        #Just interpolate the x range for plotting
        pdfInterpolator = \
          CubicSpline( self.timeDelays, predictedTransform)
          
        predicted =   pdfInterpolator(timeDelays)

        if np.any(predicted < -0.1):
            print(inputFeatures, np.min(predicted))
            #raise ValueError("Predicted probabilites negative")
        
        predicted[ predicted< 0] = 0
        return predicted

    def plotPredictedPDF( self, inputFeatures):
        
        #plot the now predicted PDF
        plt.figure(0)
        for i, iFeature in enumerate(inputFeatures):

            
            predictedTransform = \
              self.predictPDF( np.linspace(-2, 3, 100), iFeature )
            
            print(predictedTransform)
            plt.plot( np.linspace(-2, 3, 100), predictedTransform, \
                          label=str(iFeature))

        
        plotPDF = self.pdfArray[ (self.reshapedFeatures[:,0] == 0.7) & (self.reshapedFeatures[:,1] == 0.74),:]
        
        for iHubblePar in range(plotPDF.shape[0]):

            plt.plot( self.timeDelays, plotPDF[iHubblePar,:],  alpha=0.3)
            
        plt.xlim(-1,3)
        plt.legend()
        plt.show()

        
if __name__ == '__main__':
    nonPerfectFittingFunction()
