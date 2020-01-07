#!/usr/local/bin/python3

import json
import numpy
from matplotlib import pyplot as plt
from convolveDistributionWithLineOfSight import *
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
import glob
import get_natural_cubic_spline_model as spline
from sklearn.gaussian_process.kernels import   DotProduct  as DotProduct 
from sklearn.gaussian_process.kernels import   RationalQuadratic  as RationalQuadratic
from sklearn.gaussian_process.kernels import   RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import  WhiteKernel
from sklearn.gaussian_process.kernels import  Matern



    
def fitMultLensSingleRedshift(nComponents=1):
    sourcePlane = sourcePlaneInterpolator(  nPrincipalComponents=nComponents)
    sourcePlane.getTrainingData()
    sourcePlane.generateFeatureArray()
    sourcePlane.extractPrincipalComponents()
    sourcePlane.learnPrincipalComponents()

    nRedshifts = 20
    predictFeatures = np.zeros((nRedshifts,2))
    predictFeatures[:,1] = sourcePlane.timeDelayDistClasses[0].mass
    predictFeatures[:,0] = np.linspace(sourcePlane.zLens, 10., nRedshifts)

    sourcePlane.predictPDF(predictFeatures)
    
def fitSingleLens( nComponents=2 ):


    jsonFile = '../output/CDM/z_0.25/sourcePlaneInterpolation.json'

    #Get a single piece of data (to be expanded)
    dist = timeDelayDistribution(jsonFile)

    #Set up the PCA 
    pca = PCA(n_components=nComponents)

    #set up the feature array which is basically just a nSourceZ x nBins array with the pdf in them
    #when expanded the lens redshfit will need to be considered
    featureArray = \
      np.zeros((len(dist.finalPDF['finalLoS']),len(dist.finalPDF['finalLoS'][0].timeDelayPDF['x'])))
    
    #also get an array of redshifts
    redshifts = np.array([])
    for iZ, iSourcePlane in enumerate(dist.finalPDF['finalLoS']):
        featureArray[iZ,:] = iSourcePlane.timeDelayPDF['y']
        redshifts = np.append(redshifts, iSourcePlane.data['z'])
        
    #Extract the PCA from the PDFs
    pca.fit( featureArray )

    transform = pca.transform( featureArray )
    #Get a prediction of the first nComponents of the PCA
    predictedY = pca.inverse_transform( transform)

    #Now regress each component to any source redshift
    nRedshifts =10
    redshiftsArr = np.zeros((len(redshifts),1))
    print(redshifts)
    redshiftsArr[:,0] = redshifts
    predictOnTheseRedshifts = np.linspace(0.3, 10., nRedshifts)
    predictedPCAtransform = np.zeros((len(redshifts), nComponents))
    
    for i in range(nComponents):
        gaussProcess = GaussianProcessRegressor()
        gaussProcess.fit( redshiftsArr, transform[:,i])
        predictions = gaussProcess.predict( redshiftsArr)
        predictedPCAtransform[:,i] = predictions

    
    predictedTransform = pca.inverse_transform( predictedPCAtransform )
    
    #Plot the true against the predicted nComponents of the PCA
    for iZ, iSourcePlane in enumerate(dist.finalPDF['finalLoS']):

        plt.plot(iSourcePlane.timeDelayPDF['x'],predictedTransform[iZ,:]/iSourcePlane.timeDelayPDF['y'],':', label=iSourcePlane.data['z'])
        
    plt.show()



class sourcePlaneInterpolator:
    '''
    The idea is to model the PDF and interpolate between source
    planes
    '''

    def __init__( self, nPrincipalComponents=2):
        
        self.nPrincipalComponents = nPrincipalComponents

    def getTrainingData( self ):
        '''
        Import all the json files from a given lens redshift
        '''
        self.timeDelayDistClasses = []

        jsonFiles = glob.glob('../output/CDM/z_0.25/B*cluster_*_*_total*.json')
        jsonFiles = ['../output/CDM/z_0.25/B009_cluster_0_1_total_sph.fits.py.raw.json']
        for iJsonFile in jsonFiles:
            dist = \
              timeDelayDistribution(iJsonFile, \
                        timeDelayBins=np.linspace(-2,2,100))
            dist.getHaloMass()
            self.zLens = dist.zLens

            self.timeDelayDistClasses.append(dist)

    def generateFeatureArray( self, smoothingKernel=1 ):
        '''
        Generate a nSourceRedshift x nHalo array of features
        '''
        #keep track of the features to be used
        features = np.array([], dtype=[('zSource',float), ('mass', float)])
        self.nFeatures = len(features.dtype)
        #an array of the pdf values for each feature combination
        pdfArray = None
        self.clusterID = []
        for i, iPDF in enumerate(self.timeDelayDistClasses):
            for iSourcePlane in iPDF.finalPDF['finalLoS']:

                iFeature = np.array( [(iSourcePlane.data['z'], iPDF.mass)], \
                  dtype = features.dtype)
                features = np.append( features, iFeature)
                if smoothingKernel is not None:
                    x = iSourcePlane.timeDelayPDF['x']
                    y = iSourcePlane.timeDelayPDF['y']
                    totalY = y*(x[1]-x[0])

                    totalImages = np.int((1./np.min(totalY[totalY>0])))
                    y = y*(x[1]-x[0])*totalImages
                    nKnots = np.min([np.max([2,totalImages/5.]),100])
                    print('nKnots is ',nKnots)
                    smoothedPDFmodel = \
                      spline.get_natural_cubic_spline_model(x, y,minval=min(x), maxval=max(x), \
                                                        n_knots=nKnots)
                    
                    smoothePDF = smoothedPDFmodel.predict(x)
                    smoothePDF /= np.sum(smoothePDF*(x[1]-x[0]))
                    
                else:
                    smoothePDF = iSourcePlane.timeDelayPDF['y']
                if pdfArray is None:
                    pdfArray = smoothePDF
                else:
                    pdfArray = np.vstack((pdfArray, smoothePDF))
                self.clusterID.append(i)
        self.featureLabels = features
        self.featureArray = pdfArray

        self.nPDF = len(features)

        
    def extractPrincipalComponents( self ):
        '''
        Now extract the principal components for each halo
        '''
        self.principalComponents = np.zeros( (self.nPDF,self.nPrincipalComponents))
        for i in range(self.nPDF):
            index = i==np.array(self.clusterID)
            if self.featureArray[index,:].shape[0] < 4:
                continue

            

            self.pca = PCA(n_components=self.nPrincipalComponents)
        

        #Extract the PCA from the PDFs
            self.pca.fit( self.featureArray[index,:] )
            iPrincipalComponents = self.pca.transform(  self.featureArray[index,:] )
            self.principalComponents[index,:] = iPrincipalComponents
              
            
    def learnPrincipalComponents( self ):
        '''
        Using a mixture of Gaussian Processes 
        predict the distributions of compoentns
        It will return a list of nPrincipalComponent models that try to
        learn each component to the data
        '''
        
        #Now regress each component to any source redshift
        predictedPCAtransform = np.zeros((6, self.nPrincipalComponents))
        
        self.reshapedFeatures = \
          self.featureLabels.view('<f8').reshape((self.nPDF,self.nFeatures))

        self.learnedGPforPCAcomponent = []
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1e2)
        kernel = RationalQuadratic() 

        for i in range(self.nPrincipalComponents):
            
            gaussProcess = GaussianProcessRegressor( alpha=10, kernel=kernel)
            gaussProcess.fit( self.reshapedFeatures, self.principalComponents[:,i])
            self.learnedGPforPCAcomponent.append(gaussProcess)


    def predictPDF( self, features ):
        '''
        For now compare to the trained data
        '''

        predictedComponents = np.zeros((features.shape[0], self.nPrincipalComponents))
        for iComponent in range(self.nPrincipalComponents):

            predictedComponents[:,iComponent] = \
              self.learnedGPforPCAcomponent[iComponent].predict( features )
        plt.figure(0)
        for i, iFeature in enumerate(features):

            predictedTransform = self.pca.inverse_transform(  predictedComponents[i,:] )

            plt.plot( predictedTransform, label=str(iFeature))
            index = np.argmin(np.abs(self.featureLabels['zSource'] - iFeature[0] ))
            plt.plot( self.featureArray[index,:], ls=':',label=str(self.featureLabels['zSource'][index]))

        plt.legend()
        plt.show()

            

        for i in range(10):
            
             plt.plot( self.featureArray[i,:], label=str(self.featureLabels['zSource'][i]))
        plt.legend()
        plt.show()
if __name__ == '__main__':
    
    fitMultLensSingleRedshift()
    
