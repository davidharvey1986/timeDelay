
import hubbleInterpolatorClass as hubbleModel

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
import ipdb as pdb
import os
from scipy.interpolate import CubicSpline

import plotAsFunctionOfDensityProfile as getDensity
from matplotlib import gridspec 


def main():

    '''
    New method, test the GPR for the default cosmology
    and then lineraly interpolate this to a new cosmology
    '''

    gs = gridspec.GridSpec( 5, 1)
    axA = plt.subplot( gs[:3,0])
    axB = plt.subplot( gs[3:,0])

    #the first ensemble to test which is looking at H0 and OmegaM

    #hubble interpolator over a small number o
    hubbleInterpolator = \
      hubbleModel.hubbleInterpolator( )
      
    hubbleInterpolator.getTrainingData('exactPDFpickles/noCosmology.pkl')

    hubbleInterpolator.getTimeDelayModel(modelFile='pickles/noCosmology.pkl')
    

    allDistributions = \
      pkl.load(open(hubbleInterpolator.allDistributionsPklFile,'rb'))

    #Check the interpolator for this number of samples
    nSamples = 1000

    #set up an array
    diffArray = np.zeros((nSamples, len(hubbleInterpolator.timeDelays)))
    
    cosmoKeys = hubbleInterpolator.fiducialCosmology.keys()
    doneInts = []
    for i in np.arange(nSamples):
        randInt = np.random.randint(0, len(allDistributions))

        if randInt in doneInts:
            continue
        doneInts.append(randInt)
        
        iDist = allDistributions[randInt]

        print("%i/%i" % (i, nSamples))

        truth = iDist['y'][ iDist['x'] > \
                    hubbleInterpolator.logMinimumTimeDelay]
                    
        truthCumSum = np.cumsum(truth)/np.sum(truth)
        params = iDist['cosmology']
      
        fileName = iDist['fileNames'][0]
        zLensStr = fileName.split('/')[-2]
        zLens = np.float(zLensStr.split('_')[1])
        densityProfile = \
          getDensity.getDensityProfileIndex(fileName)[0]
  
        defaultDistributionIndex = \
              ( hubbleInterpolator.features['densityProfile'] == densityProfile ) &\
              ( hubbleInterpolator.features['zLens'] == zLens )

        truthCDF = hubbleInterpolator.pdfArray[defaultDistributionIndex,:][0]
        
    
        params['zLens'] = zLens
        params['densityProfile'] = densityProfile
        params['H0'] /= 100.

        
        
        trueComponents = hubbleInterpolator.principalComponents[defaultDistributionIndex,:]

        pcaCDF =   hubbleInterpolator.pca.inverse_transform( trueComponents[0] )

        
        
        
        interpolateThisCosmology = \
          np.array([ iDist['cosmology'][i] for i in \
                         hubbleInterpolator.fiducialCosmology.keys()])
                         
        cosmoShift = \
          hubbleInterpolator.interpolatorFunction.predict(interpolateThisCosmology.reshape(1,-1))
       
        spline =    CubicSpline( hubbleInterpolator.timeDelays+cosmoShift, pcaCDF)
        
        shiftPCACDF = spline( hubbleInterpolator.timeDelays)
        
        predictCDF = hubbleInterpolator.predictCDF( hubbleInterpolator.timeDelays, params)

        diff = truthCumSum - predictCDF


        diffArray[i,:] = diff
        
        axA.plot( hubbleInterpolator.timeDelays, truthCumSum - predictCDF, color='grey')
        #if np.max(np.abs(diff)) > 0.1:
         #   pdb.set_trace()

    axB.plot(hubbleInterpolator.timeDelays, np.sqrt(np.sum(diffArray**2, axis=0)/nSamples))
    plt.show()
    pdb.set_trace()

def getBestParsForGPR():
  
    hubbleInterpolator = \
      hubbleModel.hubbleInterpolator( preprocess=True)
      
    hubbleInterpolator.getTrainingData('exactPDFpickles/noCosmology.pkl')

    hubbleInterpolator.getTimeDelayModel(modelFile='pickles/noCosmology.pkl')
    hubbleInterpolator.learnPrincipalComponents()
    defaultLog = hubbleInterpolator.getGaussProcessLogLike()

 
    print(defaultLog)
    '''
    length_scale = 10**np.linspace(0.,4, 5)
    nuList = np.linspace(0.1, 2., 5)
    noise_level = 10**np.linspace(0.,4, 5)
    alphaList = 10**np.linspace(-4.,2, 6)
    logLike = []
    params = []
    for iL in length_scale:
        for nu in nuList:
            for noise in noise_level:
                for alpha in alphaList:
    '''
    maternWeightList = 10**np.linspace(-1,3,10)
    whiteKernelList = 10**np.linspace(-1,3,10)
    logLike = np.zeros([10,10])
    for i, maternWeight in enumerate(maternWeightList):
        for j, whiteKernel in enumerate(whiteKernelList):
            hubbleInterpolator.learnPrincipalComponents( maternWeight=maternWeight, whiteKernelWeight= whiteKernel)
            logLike[ i, j] = hubbleInterpolator.getGaussProcessLogLike()
            if logLike[ i, j] > defaultLog:
                print(maternWeight,whiteKernel, logLike[ i, j] - defaultLog)
                pdb.set_trace()
    plt.imshow(logLike)
            
                    
    
    pdb.set_trace()
if __name__ == '__main__':
    main()
