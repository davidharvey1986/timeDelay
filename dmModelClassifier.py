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
from medianTimeFunctionSubstructure import substructure
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

class dmClassifier:
    '''
    Build up a array of features for the three dm models
    and then use it to be able to classify a cdf
    '''

    def __init__( self, nPrincipalComponents=2, minimumTimeDelay=0.001,\
                      massCut=[0., 16], \
                      regressorNoiseLevel=1e-2, \
                      dmModels=['CDM','L8','L11p2']):
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

        self.dmModels = dmModels

              
        self.interpolatorDistributions = \
          "../output/CDM/selectionFunction/"+\
          "sparselyPopulatedParamSpace.pkl"
    
    def getTrainingData( self, \
        pklFile = 'pickles/trainingDataForClassifier.pkl',\
                             minLogMass=7):
        '''
        Import all the json files from a given lens redshift
        Train only the fiducial cosmology and then interpolate a shift
        for the cosmology

        '''
      

        if pklFile is not None:
            if os.path.isfile( pklFile):
                self.features,  self.timeDelays, self.pdfArray, self.dmLabel =  \
                pkl.load(open(pklFile, 'rb'))
                self.nFeatures = len(self.features.dtype)

                return
    

        self.pdfArray = None

     
        

        
        featureDtype = \
          [( ('zLens',float) ), ('densityProfile', float), \
           ('totalMass', float), ('nSubstructure', float) ]

        self.dmLabel = []
        features = np.array([], dtype=featureDtype)
        self.nFeatures = len(features.dtype)

        for iDMmodel in self.dmModels:
            
            #How to split up the trianing sample to speed it up
            allDistributionsPklFile = \
              "../output/%s/selectionFunction/SF_fiducialCosmo.pkl" \
              % iDMmodel
            
            allDistributions = \
              pkl.load(open(allDistributionsPklFile,'rb'))
            cosmoKeys =  allDistributions[0]['cosmology'].keys()

            
            for finalMergedPDFdict in allDistributions:
          
                doNotTrainThisSample =  \
                  np.any(np.array([self.fiducialCosmology[iCosmoKey] != \
                    finalMergedPDFdict['cosmology'][iCosmoKey] \
                    for iCosmoKey in cosmoKeys]))

                if doNotTrainThisSample:
                    continue

            

                fileName = finalMergedPDFdict['fileNames'][0]
            
                totalMassForHalo = \
                  getMass.getTotalMass( fileName, dmModel=iDMmodel)

                if (totalMassForHalo < self.massCut[0]) |\
                  (totalMassForHalo > self.massCut[1]):
                    continue

                nSubstructure = \
                  substructure( fileName, minLogMass=minLogMass)
              
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

                allPars = \
                  [ zLens, densityProfile, totalMassForHalo,nSubstructure]

                iFeature = np.array(allPars, dtype=features.dtype)

                #TO do this allPars needs to be a tuple!
                features = np.append( features, np.array(tuple(allPars), \
                                dtype = featureDtype))
            
                self.dmLabel.append(iDMmodel)

        self.features = features
        

        self.timeDelays =  \
          finalMergedPDFdict['x'][ finalMergedPDFdict['x'] > \
                                       self.logMinimumTimeDelay]

        if pklFile is not None:
            pkl.dump([self.features,  self.timeDelays, self.pdfArray, self.dmLabel], \
                     open(pklFile, 'wb'))
                     
    def extractPrincipalComponents( self ):
        '''
        Now extract the principal components for each halo
        '''
        self.pca = PCA(n_components=self.nPrincipalComponents)
        self.pca.fit( self.pdfArray )
        self.principalComponents = self.pca.transform( self.pdfArray )

        
    def learnAllFeatures( self, length_scale=1., nu=3./2.):
        '''
        Using a mixture of Gaussian Processes 
        predict the distributions of compoentns
        It will return a list of nPrincipalComponent models that try to
        learn each component to the data
        '''
        
        randomForest = RandomForestClassifier(1000)
        randomForest.fit(self.trainingSet['features'], self.trainingSet['label'])
        

        self.classifier = randomForest

  

    def getTimeDelayModel( self ):
        '''
        Run the two programs to learn and inteprolate the pdf
        i will save it in a pkl file as it might take some time.
        '''


        self.learnAllFeatures()

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
    
    def classifyCDF( self, CDF, zLens ):
        '''
        For now compare to the trained data
        '''

        #ensure the CDF is at the correct time delays
        pdfInterpolator = \
              CubicSpline( CDF['x'], CDF['y'])
        
        compressThisCDF =   pdfInterpolator(self.timeDelays)
        

        principalComponents = self.pca.transform( compressThisCDF.reshape(1,-1) )

        label = self.classifier.predict(np.append(principalComponents, zLens).reshape(1,-1))

        return label


    def generateTestAndTrainingSets( self, proportionOfTrainingSet=0.2 ):
        '''
        split the pdfArray in to a training and test set

        proportionOfTrainingSet is the proportion of the trianing that i 
        will use to test

        '''
        nTotalSamples = self.pdfArray.shape[0]

        nTestSamples = np.int(nTotalSamples*proportionOfTrainingSet)
        nTrainSamples = nTotalSamples - nTestSamples
        print("%i Train and %i Test" %(nTrainSamples,nTestSamples))
        testSamplesIndex = np.arange(nTotalSamples)
        trainingSamplesIndex = []
        
        for iSample in range(nTrainSamples):
            index = np.random.randint( 0, len(testSamplesIndex))
            trainingSamplesIndex.append( testSamplesIndex[index])
            testSamplesIndex = np.delete(testSamplesIndex, index)
        
        
        trainingPrincipalComponents = self.principalComponents[trainingSamplesIndex,:]
        trainRedshiftLens = self.features['zLens'][trainingSamplesIndex]
    
        trainFeatures = \
          self.appendRedshiftToPrincipalComponents( trainingPrincipalComponents, \
                                                   trainRedshiftLens)
                                                   
        trainLabel = np.array(self.dmLabel)[trainingSamplesIndex]
        self.trainingSet = {'features': trainFeatures, 'label':trainLabel}
        
        testPrincipalComponents = self.principalComponents[testSamplesIndex,:]
        testRedshiftLens = self.features['zLens'][testSamplesIndex]
        
        testFeatures = \
          self.appendRedshiftToPrincipalComponents( testPrincipalComponents, \
                                                   testRedshiftLens)
        testLabel = np.array(self.dmLabel)[testSamplesIndex]
        self.testSet = {'features': testFeatures, 'label':testLabel}
        self.testSamplesIndex = testSamplesIndex

        

        
    def appendRedshiftToPrincipalComponents( self, prinComp, redshifts):


        allFeatures = \
          np.zeros((prinComp.shape[0], prinComp.shape[1]+1))
        allFeatures[:,:-1] = prinComp
        allFeatures[:,-1] = redshifts

        return allFeatures
