import hubbleInterpolatorClass as hubbleModel

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
import ipdb as pdb
import os
def main():
    '''
    The new ensemble interpolatr averages the models from many subsamples
    of data and then finding yhe mean of all of them, like a random forest

    As a function of subsamples i want to see the difference between the predicted and true
    
    '''
    #For aesthetics                                                            \
                                                                                
    jet = cm = plt.get_cmap('rainbow')
    cNorm  = colors.Normalize(vmin=0.65, vmax=0.75)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #


    
    axarr = plt.gca()
    theta = {'H0':0.7, 'OmegaM':0.3, 'OmegaK':0., \
                  'OmegaL':0.7, 'zLens':0.5, 'densityProfile':-1.75}

    interpolateToTheseTimes= np.linspace(0, 3, 1e6)

    #the first ensemble to test which is looking at H0 and OmegaM

    #hubble interpolator over a small number o
    hubbleInterpolatorJustH0 = \
      hubbleModel.hubbleInterpolator( inputFeaturesToTrain=['H0'], \
                                        nSubSamples=1)
    hubbleInterpolatorJustH0.getTrainingData('exactPDFpickles/trainingDataJustH0.pkl')

    hubbleInterpolatorJustH0.getTimeDelayModel(modelFile='pickles/justH0.pkl')
    interpolatedCumSumJustH0 = \
      hubbleInterpolatorJustH0.predictPDF( interpolateToTheseTimes, theta )
    nSamplesToCheck = 1000


    subSampleList = [1000]
    ls = ['-','--']
    componentList = np.arange(1,7)[::-1]
    for iLine, iNumSubSamples in enumerate(subSampleList):
        for nPrincipalComponents in componentList:
        
            #for each samplesize, i want to check the 
            print(iNumSubSamples)
        

        #second ensemble tester which is just H0 and no ensemble
            pklFile = 'pickles/diffArray_%i_nComp_%i.pkl' % (iNumSubSamples, nPrincipalComponents)
            if os.path.isfile( pklFile ):
                differenceArray = pkl.load(open(pklFile,'rb'))
            else:
                hubbleInterpolaterClass = hubbleModel.hubbleInterpolator()
                hubbleInterpolaterClass.getTrainingData('exactPDFpickles/allTrainingData.pkl')
                if iNumSubSamples == subSampleList[0]:
                    featureNumbers = np.random.randint( 0, hubbleInterpolaterClass.features.shape[0], size=nSamplesToCheck)
                    differenceArray = getDiffArray( iNumSubSamples, featureNumbers, nSamplesToCheck = nSamplesToCheck, nPrincipalComponents=nPrincipalComponents )
                pkl.dump(differenceArray, open(pklFile,'wb'))
            
            variance = np.sqrt(np.sum(differenceArray**2, axis=0)/differenceArray.shape[0])
            axarr.plot(hubbleInterpolatorJustH0.timeDelays, variance, ls =ls[iLine], \
                        label='Ncomp %i, Nsub: %i' % (nPrincipalComponents,iNumSubSamples) )
        
    
    axarr.legend()
    
    plt.show()
    
def getDiffArray( iNumSubSamples, featureNumbers, nSamplesToCheck = 2, nPrincipalComponents=7 ):


    hubbleInterpolaterClass = \
      hubbleModel.hubbleInterpolator(nSubSamples=iNumSubSamples, nPrincipalComponents=nPrincipalComponents )

    hubbleInterpolaterClass.getTrainingData('exactPDFpickles/allTrainingData.pkl')
    hubbleInterpolaterClass.getTimeDelayModel('pickles/hubbleInterpolatorModel%iSubSamples.pkl' % iNumSubSamples)
    
    differenceArray = np.zeros( (nSamplesToCheck, len(hubbleInterpolaterClass.timeDelays)))


    


    for iFeatureIndex, iFeatureNumber in enumerate(featureNumbers):
            
        print('%i, %i/%i' % (iFeatureNumber, iFeatureIndex+1, nSamplesToCheck))
        iFeatureSet =  hubbleInterpolaterClass.features[iFeatureNumber]

        theta = {}
        for iTheta in iFeatureSet.dtype.names:
            theta[ iTheta ]  = iFeatureSet[iTheta]
            
        interpolatedCumSum = \
              hubbleInterpolaterClass.predictPDF( hubbleInterpolaterClass.timeDelays, theta )

        truth = hubbleInterpolaterClass.pdfArray[iFeatureNumber, :]
            
        differenceArray[iFeatureIndex,:] = interpolatedCumSum - truth

    return differenceArray

def principalComponentChecker():
    '''
    The new ensemble interpolatr averages the models from many subsamples
    of data and then finding yhe mean of all of them, like a random forest
    
    '''
    #params to test over
    
    theta = {'H0':0.7, 'OmegaM':0.3, 'OmegaK':0., \
                  'OmegaL':0.7, 'zLens':0.5, 'densityProfile':-1.75}

    interpolateToTheseTimes= np.linspace(0, 3, 1e6)

    #the first ensemble to test which is looking at H0 and OmegaM

    #hubble interpolator over a small number o
    hubbleInterpolatorJustH0 = \
      hubbleModel.hubbleInterpolator( inputFeaturesToTrain=['H0'], \
                                        nSubSamples=1)

     
    hubbleInterpolatorJustH0.getTrainingData('exactPDFpickles/trainingDataJustH0.pkl')
    hubbleInterpolatorJustH0.getTimeDelayModel(modelFile='pickles/justH0.pkl')
    interpolatedCumSumJustH0 = \
      hubbleInterpolatorJustH0.predictPDF( interpolateToTheseTimes, theta )


    fig, axarr = plt.subplots( hubbleInterpolatorJustH0.nPrincipalComponents, 1, figsize=(12,6))
    fig.subplots_adjust(hspace=1)
    for iNumSubSamples in [1000]:
        for i in ['','Shuffled']:
            print(i)
            hubbleInterpolaterClass = \
              hubbleModel.hubbleInterpolator(nSubSamples=iNumSubSamples )

            hubbleInterpolaterClass.getTrainingData('exactPDFpickles/allTrainingData.pkl')
            hubbleInterpolaterClass.getTimeDelayModel('pickles/hubbleInterpolatorModel%iSubSamples%s.pkl' % (iNumSubSamples,i))
     
            interpolatedCumSum = \
            hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, theta )

            #second ensemble tester which is just H0 and no ensemble
        
    
            for i in np.arange(hubbleInterpolaterClass.nPrincipalComponents):
                axarr[i].hist(hubbleInterpolaterClass.predictedComponents[i,:], density=True)
                ylims = axarr[i].get_ylim()
                axarr[i].plot(hubbleInterpolatorJustH0.predictedComponents[i]*[1,1], [0,100], 'k-')
                axarr[i].set_ylim(ylims)
    plt.show()
    

if __name__ == '__main__':
    #principalComponentChecker()
    main()
