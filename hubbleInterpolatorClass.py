import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis
import plotAsFunctionOfDensityProfile as getDensity
import os
import pickle as pkl
from interpolateSourcePlane import *
from sklearn.gaussian_process.kernels import   DotProduct  as DotProduct 
from sklearn.gaussian_process.kernels import   RationalQuadratic  as RationalQuadratic
from sklearn.gaussian_process.kernels import   RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import  WhiteKernel
from sklearn.gaussian_process.kernels import  Matern
from sklearn.gaussian_process.kernels import   ExpSineSquared

class hubbleInterpolator:
    '''
    The idea is to model the PDF and interpolate between source
    planes
    '''

    def __init__( self, nPrincipalComponents=6, selectionFunctionZmed=2.0):
        
        self.nPrincipalComponents = nPrincipalComponents
        self.SFzMed = selectionFunctionZmed
        
    def getTrainingData( self, pklFile = 'pickles/trainingData.pkl' ):
        '''
        Import all the json files from a given lens redshift
        '''
        
        self.hubbleParameters = \
          np.array([50., 60., 70., 80., 90., 100.])

        if pklFile is not None:
            if os.path.isfile( pklFile):
                self.features,  self.timeDelays, self.pdfArray =  \
                pkl.load(open(pklFile, 'rb'))
                self.nFeatures = len(self.features.dtype)
                return
        
        self.timeDelayDistClasses = []

        features = np.array([], dtype=[('hubbleParameter', float), \
                        ('zLens',float), ('densityProfile', float)])
        
        self.nFeatures = len(features.dtype)


        self.pdfArray = None
        dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'

        allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
        rGrid = getDensity.getRadGrid()
        
        for iColor, iHubbleParameter in enumerate(self.hubbleParameters):
            for iFile in allFiles:
                fileName = iFile.split('/')[-1]
                

                zLensStr = iFile.split('/')[-2]
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
                
                iFeature = np.array( [(iHubbleParameter, zLens, densityProfile )], \
                    dtype = features.dtype)

                
                features = np.append( features, iFeature)
            
        self.features = features
        self.features['hubbleParameter'] /= 100.
        #self.features['densityProfile'] *= -0.5

        self.timeDelays =  finalMergedPDFdict['x']
        if pklFile is not None:
            pkl.dump([self.features,  self.timeDelays, self.pdfArray], \
                     open(pklFile, 'wb'))
                     
    def extractPrincipalComponents( self ):
        '''
        Now extract the principal components for each halo
        '''
        self.pca = PCA(n_components=self.nPrincipalComponents)
        self.pca.fit( self.pdfArray )
        self.principalComponents = self.pca.transform( self.pdfArray )
            
    def learnPrincipalComponents( self, weight=1. ):
        '''
        Using a mixture of Gaussian Processes 
        predict the distributions of compoentns
        It will return a list of nPrincipalComponent models that try to
        learn each component to the data
        '''
        
        #Now regress each component to any hubble value
        
        #self.learnedGPforPCAcomponent = []
        self.predictor = []

        #kernel =  Matern(length_scale=1., nu=3/2) + \
        #  WhiteKernel(noise_level=1e3)
        
        #kernel =  ExpSineSquared(length_scale=2)
        kernel =  RBF(length_scale=weight) + \
          WhiteKernel(noise_level=1e3)

        self.nPDF = len(self.features)
        
        self.reshapedFeatures = \
          self.features.view('<f8').reshape((self.nPDF,self.nFeatures))
          
        for i in range(self.nPrincipalComponents):
            
            gaussProcess = \
              GaussianProcessRegressor( alpha=1., kernel=kernel)

            gaussProcess.fit( self.reshapedFeatures, self.principalComponents[:,i])

            
            self.predictor.append(gaussProcess)
            
            #cubicSpline = CubicSpline(self.hubbleParameters, self.principalComponents[:,i])
            #self.cubicSplineInterpolator.append(cubicSpline)
        print("log likelihood of predictor is %0.3f" %\
                  self.getGaussProcessLogLike())
    def getGaussProcessLogLike( self ):

        logLike = 0
        for iGaussProcess in self.predictor:
            logLike += iGaussProcess.log_marginal_likelihood()
        return logLike
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

        #if np.any(predicted < -0.1):
         #   print(inputFeatures, np.min(predicted))
            #raise ValueError("Predicted probabilites negative")
        
        predicted[ predicted< 0] = 0
        return predicted

    def plotPredictedPDF( self, inputFeatures):
        
        #plot the now predicted PDF
        plt.figure(0)
        for i, iFeature in enumerate(inputFeatures):

            
            predictedTransform = \
              self.predictPDF( np.linspace(-2, 3, 100), iFeature )
            

            plt.plot( np.linspace(-2, 3, 100), predictedTransform, \
                          label=str(iFeature))

        
        plotPDF = self.pdfArray[ (self.reshapedFeatures[:,0] == 0.7) & (self.reshapedFeatures[:,1] == 0.74),:]
        
        for iHubblePar in range(plotPDF.shape[0]):

            plt.plot( self.timeDelays, plotPDF[iHubblePar,:],  alpha=0.3)
            
        plt.xlim(-1,3)
        plt.legend()
        plt.show()

