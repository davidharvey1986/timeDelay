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


def nonPerfectFittingFunction(nComponents=4):
    inputHubbleParameter = 70.

    sampleSizes, estimates = \
      getPredictedConstraints(inputHubbleParameter)


def perfectFittingFunction(nComponents=4):


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
                                nIterations = 100,\
                                nSampleSizes = 10):
                                
    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()

    predictFeatures = hubbleInterpolaterClass.reshapedFeatures
    print(predictFeatures[:,0])
    #predictFeatures = predictFeatures[ (predictFeatures[:,0] == 70) & (predictFeatures[:,1] == 0.74), :]

    
    #hubbleInterpolaterClass.plotPredictedPDF( predictFeatures )

    sampleSizes = 10**np.linspace(2,4,nSampleSizes)
    color=['blue','red','green']

    
    estimates = np.zeros((nSampleSizes, nIterations))
    for i, iSampleSize in enumerate(sampleSizes):
        
        for iIteration in range(nIterations):
            print("Sample Size: %i/%i, iteration: %i/%i" %\
                      (i+1, nSampleSizes, iIteration+1, nIterations))
            selectedTimeDelays = \
              selectionFunctionZmed( iSampleSize, inputHubbleParameter)
      
            fitHubbleClass = \
              fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)
            estimates[i,iIteration] = \
              fitHubbleClass.params['params'][0]


    pkl.dump([sampleSizes, estimates], \
        open('perfectFittingFunction.pkl','wb'))

    return sampleSizes, estimates


def selectionFunctionZmed( nSamples, hubbleParameter ):
    '''
    From a given pdf randomly select some time delays
    '''

    pklFileName = \
      '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
      % (hubbleParameter)
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

    #Interpolate the p
    p = finalMergedPDFdict['y'] / np.sum(finalMergedPDFdict['y'])
    interpolatedProbClass = CubicSpline( finalMergedPDFdict['x'], p)
    interpolateToTheseTimes=  np.linspace(np.min(finalMergedPDFdict['x']),np.max(finalMergedPDFdict['x']),nSamples*100)
    interpolatedProb = interpolatedProbClass( interpolateToTheseTimes)
    interpolatedProb[interpolatedProb<0] = 0
    interpolatedProb /= np.sum(interpolatedProb)
    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb, size=np.int(nSamples))
    
    bins = np.max([10, np.int(nSamples/100)])
    y, x = np.histogram(logTimeDelays, bins=np.int(bins), density=True)
    xcentres = (x[1:] + x[:-1])/2.
    error = np.sqrt(y*nSamples)/nSamples
    return {'x':xcentres, 'y':y, 'error':error}

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
                pklFileName = \
                  '../output/CDM/selectionFunction/SF_%s_%i_lsst.pkl' \
                  % (fileName, iHubbleParameter )
              
            
                finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
                finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])
            
                if self.pdfArray is None:
                    self.pdfArray = finalMergedPDFdict['y']
                else:
                    self.pdfArray = \
                      np.vstack((self.pdfArray, finalMergedPDFdict['y']))

                densityProfile = getDensity.getDensityProfileIndex(iFile, rGrid=rGrid)[0]
                zLens = np.float(iFile.split('/')[-2].split('_')[1])
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

        kernel = ConstantKernel() + \
          0.608**2*Matern(length_scale=1.484, nu=3/2) + \
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
        return predicted / np.max(predicted)

    def plotPredictedPDF( self, inputFeatures):
        
        #plot the now predicted PDF
        plt.figure(0)
        for i, iFeature in enumerate(inputFeatures):

            
            predictedTransform = \
              self.predictPDF( np.linspace(-2, 3, 100), iFeature )
            
            print(predictedTransform)
            plt.plot( np.linspace(-2, 3, 100), predictedTransform, \
                          label=str(iFeature))

        
        plotPDF = self.pdfArray[ (self.reshapedFeatures[:,0] == 70) & (self.reshapedFeatures[:,1] == 0.74),:]
        
        for iHubblePar in range(plotPDF.shape[0]):

            plt.plot( self.timeDelays, plotPDF[iHubblePar,:],  alpha=0.3)
            
        plt.xlim(-1,3)
        plt.legend()
        plt.show()

        
if __name__ == '__main__':
    nonPerfectFittingFunction()
