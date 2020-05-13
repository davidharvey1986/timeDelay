import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis
import plotAsFunctionOfDensityProfile as getDensity
import plotAsFuncTotalMass as getMass 
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
from scipy.interpolate import LinearNDInterpolator
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter as gauss
from sklearn import preprocessing



class hubbleInterpolator:
    '''
    The idea is to model the PDF and interpolate between source
    planes
    '''

    def __init__( self, nPrincipalComponents=9, minimumTimeDelay=0.001,\
                      allDistributionsPklFile=None, massCut=[0., 16], \
                      regressorNoiseLevel=1e-10):
        '''
        inputTrainFeatures: a list of the cosmology keys to train over
        minimmumTimeDelay: should we expect a minimum possibly observed time delay 
        omitHalo: a list of halo names to be omitted from the fitting
        
        '''
        self.regressorNoiseLevel = regressorNoiseLevel
        self.massCut = massCut
        self.logMinimumTimeDelay = np.log10(np.max([minimumTimeDelay, 0.001]))
        self.nPrincipalComponents = nPrincipalComponents

        # if i dont want to train all the cosmologies

        #if so i need a defaul cosmology
        self.fiducialCosmology = \
          {'H0':70., 'OmegaM':0.3, 'OmegaL':0.7, 'OmegaK':0.}
        self.cosmoKeys = self.fiducialCosmology.keys()

        
        #How to split up the trianing sample to speed it up
        if allDistributionsPklFile is None:
            self.allDistributionsPklFile = \
              "../output/CDM/selectionFunction/"+\
              "sparselyPopulatedParamSpace.pkl"
        else:
            self.allDistributionsPklFile =\
              allDistributionsPklFile
              
        self.interpolatorDistributions = \
          "../output/CDM/selectionFunction/"+\
          "sparselyPopulatedParamSpace.pkl"
    
    def getTrainingData( self, pklFile = 'exactPDFpickles/trainingData.pkl' ):
        '''
        Import all the json files from a given lens redshift
        Train only the fiducial cosmology and then interpolate a shift
        for the cosmology

        '''
      

        if pklFile is not None:
            if os.path.isfile( pklFile):
                self.features,  self.timeDelays, self.pdfArray =  \
                pkl.load(open(pklFile, 'rb'))
                self.nFeatures = len(self.features.dtype)
                return
    

        self.pdfArray = None

     
        
        allDistributions = pkl.load(open(self.allDistributionsPklFile,'rb'))

        cosmoKeys =  allDistributions[0]['cosmology'].keys()
        
        
        rGrid = getMass.getRadGrid()
        
        featureDtype = \
          [( ('zLens',float) ), ('densityProfile', float), ('totalMass', float) ]

        features = np.array([], dtype=featureDtype)
        self.nFeatures = len(features.dtype)

        for finalMergedPDFdict in allDistributions:
          
            doNotTrainThisSample =  \
              np.any(np.array([self.fiducialCosmology[iCosmoKey] != \
              finalMergedPDFdict['cosmology'][iCosmoKey] \
              for iCosmoKey in cosmoKeys]))

            if doNotTrainThisSample:
                continue

            

            fileName = finalMergedPDFdict['fileNames'][0]
            
            totalMassForHalo = \
              getMass.getTotalMass( fileName, rGrid=rGrid)

            if (totalMassForHalo < self.massCut[0]) |\
               (totalMassForHalo > self.massCut[1]):
               continue

            print(totalMassForHalo)
              
            zLensStr = fileName.split('/')[-2]
            zLens = np.float(zLensStr.split('_')[1])
            finalMergedPDFdict['y'] = \
              finalMergedPDFdict['y'][ finalMergedPDFdict['x'] > \
                                           self.logMinimumTimeDelay]
            finalMergedPDFdict['x'] = \
              finalMergedPDFdict['x'][ finalMergedPDFdict['x'] > \
                                           self.logMinimumTimeDelay]

            #By predicting the PDF i will havea  more regular CDF
            
            finalMergedPDFdict['y'] = \
              np.cumsum(finalMergedPDFdict['y'])/\
              np.sum(finalMergedPDFdict['y'])
            
            if self.pdfArray is None:
                self.pdfArray = finalMergedPDFdict['y']
            else:
                self.pdfArray = \
                  np.vstack((self.pdfArray, finalMergedPDFdict['y']))

            densityProfile = \
              getDensity.getDensityProfileIndex(fileName)[0]

            allPars = [ zLens, densityProfile, totalMassForHalo]

            iFeature = np.array(allPars, dtype=features.dtype)

            #TO do this allPars needs to be a tuple!
            features = np.append( features, np.array(tuple(allPars), \
                                dtype = featureDtype))
            


        self.features = features
        

        self.timeDelays =  \
          finalMergedPDFdict['x'][ finalMergedPDFdict['x'] > \
                                       self.logMinimumTimeDelay]

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

            
    def learnPrincipalComponents( self, length_scale=1., nu=3./2.):
        '''
        Using a mixture of Gaussian Processes 
        predict the distributions of compoentns
        It will return a list of nPrincipalComponent models that try to
        learn each component to the data
        '''
        #Now regress each component to any hubble value
        
        #self.learnedGPforPCAcomponent = []
        self.predictor = []

        kernel =  Matern(length_scale=length_scale, nu=nu)
        
        #kernel =  ExpSineSquared(length_scale=1)
        #kernel =  RBF(length_scale=1.) + \
         # WhiteKernel(noise_level=1.)

        self.nPDF = len(self.features)
        self.reshapedFeatures = \
          self.features.view('<f8').reshape((self.nPDF,self.nFeatures))

        for i in range(self.nPrincipalComponents):

            gaussProcess = \
              GaussianProcessRegressor( alpha=self.regressorNoiseLevel, kernel=kernel,\
                                            n_restarts_optimizer=10)

            gaussProcess.fit(self.reshapedFeatures, self.principalComponents[:,i])

                
            self.predictor.append(gaussProcess)

    def interpolateCosmologyShift( self ):

        #I need points and values, points are a list of ndarrays of
        #the cosmology

        #Values will be the difference in the median of the distribution which will shift linearly with the cosmology
        
        #get the pickle file
        allDistributions = \
          pkl.load(open(self.interpolatorDistributions,'rb'))

        values = np.array([])

        cosmoKeys = self.fiducialCosmology.keys()
        points = None
        defaultFileName = None
        featureDtype = [ (i, float) for i in cosmoKeys]
        self.cosmologyFeatures = np.array([], dtype=featureDtype)
        
        for iDistribution in allDistributions:

            
            fileName = iDistribution['fileNames'][0]
            
            if defaultFileName is None:
                #assume cosmology shift is not dependent on the halo
                defaultFileName = fileName

            if defaultFileName != fileName:
                continue
            zLensStr = fileName.split('/')[-2]
            zLens = np.float(zLensStr.split('_')[1])
            
            densityProfile = \
              getDensity.getDensityProfileIndex(fileName)[0]

            totalMassForHalo = \
              getMass.getTotalMass( fileName, rGrid=rGrid)

            defaultDistributionIndex = \
              ( self.features['densityProfile'] == densityProfile ) &\
              ( self.features['zLens'] == zLens ) &\
              ( self.features['totalMass'] == totalMass)

            defaulfDistributionPDF = \
              self.pdfArray[ defaultDistributionIndex,:][0]
            defaulfDistribution = \
              np.cumsum(defaulfDistributionPDF)/np.sum(defaulfDistributionPDF)
              
            newCosmoDist = iDistribution['y'][ iDistribution['x'] > \
                                       self.logMinimumTimeDelay]
            
            newCosmoDistCumSum = np.cumsum(newCosmoDist)/np.sum(newCosmoDist)
            distributionShift = \
              np.interp( 0.5, newCosmoDistCumSum, self.timeDelays) - \
              np.interp( 0.5, defaulfDistribution, self.timeDelays)

            
            values = np.append(values, distributionShift)
            iPoint = \
              np.array([ iDistribution['cosmology'][i] \
                             for i in cosmoKeys])
            iPoint[0] /= 100.
            if points is None:
                points = iPoint
            else:
                points = np.vstack((points, iPoint))
                
                
            #TO do this allPars needs to be a tuple!

            self.cosmologyFeatures = \
              np.append( self.cosmologyFeatures, np.array(tuple(iPoint),dtype = featureDtype))

        
        self.interpolatorFunction = LinearRegression()
        self.interpolatorFunction.fit(points, values)
        

    def getTimeDelayModel( self ):
        '''
        Run the two programs to learn and inteprolate the pdf
        i will save it in a pkl file as it might take some time.
        '''


        self.extractPrincipalComponents()
        self.learnPrincipalComponents()

        interpolatorFunction = 'pickles/cosmoInterpolator.pkl'
        
        if os.path.isfile( interpolatorFunction ):
            self.interpolatorFunction = \
              pkl.load(open(interpolatorFunction,'rb'))
            self.cosmologyFeatures = \
              pkl.load(open('pickles/cosmologyFeatures.pkl','rb'))
        else:
            self.interpolateCosmologyShift()
            pkl.dump(self.cosmologyFeatures, \
                         open('pickles/cosmologyFeatures.pkl','wb'))
            pkl.dump(self.interpolatorFunction, \
                        open(interpolatorFunction, 'wb'))

        
        self.nFreeParameters = len(self.fiducialCosmology.keys())+\
          len(self.features.dtype)

        #+2 for the widht of the distributions

    def getGaussProcessLogLike( self, theta=None ):

        logLike = 0
        for iGaussProcess in self.predictor:

            logLike += iGaussProcess.log_marginal_likelihood()
            
        return logLike
    
    def predictCDF( self, timeDelays, inputFeatureDict ):
        '''
        For now compare to the trained data
        '''
        
        #parse the input parameters
        

        inputFeatureNames = ['zLens','densityProfile','totalMass']
        
        inputFeatures = \
          np.array([ inputFeatureDict[i] for i in inputFeatureNames ])

        features = inputFeatures.reshape(1,-1)
        
        predictedComponents = \
              np.zeros(self.nPrincipalComponents)
              
        interpolateThisCosmology = \
              np.array([ inputFeatureDict[i] \
                    for i in self.fiducialCosmology.keys()])

        
        for iComponent in range(self.nPrincipalComponents):
            #there are now many preditors for the subsamples
            #So the predictor for this subsample is
            predictor = self.predictor[iComponent]


            predictedComponents[iComponent] = \
                  predictor.predict(features)

        self.predictedComponents = predictedComponents

       
        
        #predictedTransformCDF = \
        #  np.cumsum(  predictedTransformPDF)/np.sum(predictedTransformPDF)
          

        cal = 0
        predictedCosmoShift = \
          self.interpolatorFunction.predict(interpolateThisCosmology.reshape(1,-1)) + cal
          

        #Just interpolate the x range for plotting

        predictedTransformCDF =  \
               self.pca.inverse_transform( predictedComponents)
        pdfInterpolator = \
              CubicSpline( self.timeDelays+predictedCosmoShift, \
                               predictedTransformCDF)
        
            
        predicted =   pdfInterpolator(timeDelays)

        #if np.any(predicted < -0.1):
        #    print(inputFeatures, np.min(predicted))
            #raise ValueError("Predicted probabilites negative")
        
        predicted[ predicted< 0] = 0

        #account for distriunbtion of models
        #dX = timeDelays[1] - timeDelays[0]
        #Ensure the same dt 
        #variancePixels = inputFeatureDict['variance']/dX
        #predictedConvolved = gauss(predicted, variancePixels)

        return predicted


    def plotPredictedPDF( self, inputFeatures):
        
        #plot the now predicted PDF
        plt.figure(0)
        for i, iFeature in enumerate(inputFeatures):

            
            predictedTransform = \
              self.predictCDF( np.linspace(-2, 3, 100), iFeature )
            

            plt.plot( np.linspace(-2, 3, 100), predictedTransform, \
                          label=str(iFeature))

        
        plotPDF = self.pdfArray[ (self.reshapedFeatures[:,0] == 0.7) & (self.reshapedFeatures[:,1] == 0.74),:]
        
        for iHubblePar in range(plotPDF.shape[0]):
            plt.plot( self.timeDelays, plotPDF[iHubblePar,:],  alpha=0.3)
            
        plt.xlim(-1,3)
        plt.legend()
        plt.show()

