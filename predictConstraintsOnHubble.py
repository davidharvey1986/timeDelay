'''
I want to prediuct the constraints on teh hubble parameter from the
estiamted PDFs generated

I will fit a double powerLaw

'''

from powerLawFit import *
from interpolateSourcePlane import *

from scipy.interpolate import CubicSpline

def main(nComponents=4):

    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()
    hubbleInterpolaterClass.predictPDF( hubbleInterpolaterClass.hubbleParameters[:-1] + 5.)
    inputHubbleParameter = 70.
    selectedTimeDelays = \
      selectionFunctionZmed( 1e3, inputHubbleParameter, 2.5)
    plt.hist(selectedTimeDelays)
    plt.show()

def selectionFunctionZmed( nSamples, hubbleParameter, \
                        selectionFunctionZMed ):
    '''
    From a given pdf randomly select some time delays
    '''

    pklFileName = \
      '../output/CDM/selectionFunction/SF_%i_%0.2f.pkl' \
      % (hubbleParameter, selectionFunctionZMed)
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
    finalMergedPDFdict['y'] /= np.sum(finalMergedPDFdict['y'])
    logTimeDelays = \
      np.random.choice(finalMergedPDFdict['x'], \
                    p=finalMergedPDFdict['y'],\
                    size=np.int(nSamples))
    return logTimeDelays
class hubbleInterpolator:
    '''
    The idea is to model the PDF and interpolate between source
    planes
    '''

    def __init__( self, nPrincipalComponents=6, selectionFunctionZmed=2.0):
        
        self.nPrincipalComponents = nPrincipalComponents
        self.SFzMed = selectionFunctionZmed
        
    def getTrainingData( self ):
        '''
        Import all the json files from a given lens redshift
        '''
        self.timeDelayDistClasses = []

        self.hubbleParameters = \
          np.array([50., 60., 70., 80., 90., 100.])

        self.pdfArray = None

        for iColor, iHubbleParameter in enumerate(self.hubbleParameters):
            pklFileName = \
              '../output/CDM/selectionFunction/SF_%i_%0.2f.pkl' \
              % (iHubbleParameter, self.SFzMed)
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
            
            if self.pdfArray is None:
                self.pdfArray = finalMergedPDFdict['y']
            else:
                self.pdfArray = \
                  np.vstack((self.pdfArray, finalMergedPDFdict['y']))
                
        
        self.timeDelays =  finalMergedPDFdict['x']
        
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
        self.cubicSplineInterpolator = []

        kernel = ConstantKernel() + \
          Matern(length_scale=2, nu=3/2) + \
          WhiteKernel(noise_level=1e2)
          
        kernel = RationalQuadratic() 

        for i in range(self.nPrincipalComponents):
            
            #gaussProcess = \
            #  GaussianProcessRegressor( alpha=10, kernel=kernel)
              
            #gaussProcess.fit( self.hubbleParameters.reshape(-1,1), \
            #                self.principalComponents[:,i])

            cubicSpline = CubicSpline(self.hubbleParameters, self.principalComponents[:,i])
            self.cubicSplineInterpolator.append(cubicSpline)
       
           
            
    def predictPDF( self, features ):
        '''
        For now compare to the trained data
        '''
        
        predictedComponents = \
          np.zeros((features.shape[0], self.nPrincipalComponents))

        
        for iComponent in range(self.nPrincipalComponents):
            
            #predictedComponents[:,iComponent] = \
            #  self.learnedGPforPCAcomponent[iComponent].predict( features )
            predictedComponents[:,iComponent] = \
              self.cubicSplineInterpolator[iComponent](features)


        #plot the now predicted PDF
        plt.figure(0)
        for i, iFeature in enumerate(features):

            predictedTransform = self.pca.inverse_transform(  predictedComponents[i,:] )
            print( predictedTransform[0])
            plt.plot( self.timeDelays, predictedTransform, label=str(iFeature))
            
        for iHubblePar in range(self.pdfArray.shape[0]):

            plt.plot( self.timeDelays, self.pdfArray[iHubblePar,:], \
                    label=str(self.hubbleParameters[iHubblePar]), alpha=0.3)
            
        plt.xlim(-1,3)
        plt.legend()
        plt.show()

        
if __name__ == '__main__':
    main()
