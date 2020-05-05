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
import time
class hubbleInterpolator:
    '''
    The idea is to model the PDF and interpolate between source
    planes
    '''

    def __init__( self, nPrincipalComponents=7, minimumTimeDelay=0.):
        
        self.logMinimumTimeDelay = np.log10(np.max([minimumTimeDelay, 0.1]))
        self.nPrincipalComponents = nPrincipalComponents
        
    def getTrainingData( self, pklFile = 'exactPDFpickles/trainingData.pkl' ):
        '''
        Import all the json files from a given lens redshift
        '''
        
        self.hubbleParameters = \
          np.linspace(60,80,21)

        if pklFile is not None:
            if os.path.isfile( pklFile):
                self.features,  self.timeDelays, self.pdfArray =  \
                pkl.load(open(pklFile, 'rb'))
                self.nFeatures = len(self.features.dtype)
                return
        
        self.timeDelayDistClasses = []
        


        self.pdfArray = None

        rGrid = getDensity.getRadGrid()

        allDistributionsPklFile = \
          "../output/CDM/selectionFunction/"+\
          "sparselyPopulatedParamSpace.pkl"
          
        allDistributions = pkl.load(open(allDistributionsPklFile,'rb'))

        cosmoKeys =  allDistributions[0]['cosmology'].keys()
        
        featureDtype = [ (i, float) for i in cosmoKeys]
        featureDtype.append( ('zLens',float) )
        featureDtype.append( ('densityProfile', float))
        features = np.array([], dtype=featureDtype)
        self.nFeatures = len(features.dtype)

        for finalMergedPDFdict in allDistributions:
            fileName = finalMergedPDFdict['fileNames'][0]
            print(fileName)
            zLensStr = fileName.split('/')[-2]
            zLens = np.float(zLensStr.split('_')[1])
            print(zLens)
            finalMergedPDFdict['y'] = \
              finalMergedPDFdict['y'][ finalMergedPDFdict['x'] > \
                                           self.logMinimumTimeDelay]
            finalMergedPDFdict['x'] = \
              finalMergedPDFdict['x'][ finalMergedPDFdict['x'] > \
                                           self.logMinimumTimeDelay]

            finalMergedPDFdict['y'] = \
              np.cumsum(finalMergedPDFdict['y'])/\
              np.sum(finalMergedPDFdict['y'])
            
            if self.pdfArray is None:
                self.pdfArray = finalMergedPDFdict['y']
            else:
                self.pdfArray = \
                  np.vstack((self.pdfArray, finalMergedPDFdict['y']))

            densityProfile = \
              getDensity.getDensityProfileIndex(fileName, rGrid=rGrid)[0]

            allPars = [ finalMergedPDFdict['cosmology'][i] for i in cosmoKeys]
            allPars.append(zLens)
            allPars.append(densityProfile)

            iFeature = np.array(allPars, dtype=features.dtype)

            #TO do this allPars needs to be a tuple!
            features = np.append( features, np.array(tuple(allPars), dtype = featureDtype))
            


        self.features = features
        self.features['H0'] /= 100.
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

            
    def learnPrincipalComponents( self, weight=1.):
        '''
        Using a mixture of Gaussian Processes 
        predict the distributions of compoentns
        It will return a list of nPrincipalComponent models that try to
        learn each component to the data
        '''
        #Now regress each component to any hubble value
        
        #self.learnedGPforPCAcomponent = []
        self.predictor = []

        kernel =  Matern(length_scale=1., nu=3./2.) + \
          WhiteKernel(noise_level=1e3)
        
        #kernel =  ExpSineSquared(length_scale=2)
        #kernel =  RBF(length_scale=1.) + \
        #  WhiteKernel(noise_level=1.)

        self.nPDF = len(self.features)
        
        self.reshapedFeatures = \
          self.features.view('<f8').reshape((self.nPDF,self.nFeatures))
        
        for i in range(self.nPrincipalComponents):

            testSamples =  \
              ((10**np.linspace(-2,0,5))*\
              len(self.principalComponents[:,i])).astype(int)
            pdb.set_trace()
            for iTester in testSamples:
                start = time.time()
                gaussProcess = \
                  GaussianProcessRegressor( alpha=1e-3, kernel=kernel)

                gaussProcess.fit( self.reshapedFeatures[:iTester,:], \
                                      self.principalComponents[:iTester,i])

                finish = time.time()
                print("Time to train is %0.2f" % (finish-start))
                pdb.set_trace()
            self.predictor.append(gaussProcess)
            
            #cubicSpline = CubicSpline(self.hubbleParameters, self.principalComponents[:,i])
            #self.cubicSplineInterpolator.append(cubicSpline)

       

        
            
        print("log likelihood of predictor is %0.3f" %\
                  self.getGaussProcessLogLike())

    def getTimeDelayModel( self, modelFile=None ):
        '''
        Run the two programs to learn and inteprolate the pdf
        i will save it in a pkl file as it might take some time.
        '''

        if modelFile is None:
            modelFile = 'pickles/hubbleInterpolatorModel.pkl'
        if os.path.isfile(modelFile):
            self.predictor = pkl.load(open(modelFile, 'rb'))
        else:
            self.extractPrincipalComponents()
            self.learnPrincipalComponents()
            pkl.dump( self.predictor, open(modelFile, 'wb'))

    def getGaussProcessLogLike( self ):

        logLike = 0
        for iGaussProcess in self.predictor:

            logLike += iGaussProcess.log_marginal_likelihood()
            
        return logLike
    def predictPDF( self, timeDelays, inputFeatureDict ):
        '''
        For now compare to the trained data
        '''

        inputFeatures = np.array([ inputFeatureDict[i] for i in self.features.dtype.names])
        
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
        #    print(inputFeatures, np.min(predicted))
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

