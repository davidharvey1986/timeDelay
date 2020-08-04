
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
from copy import deepcopy as cp
import plotAsFunctionOfDensityProfile as getDensity
from matplotlib import gridspec 
import plotAsFuncTotalMass as getMass 

def main():
    fig = plt.figure( figsize=(10,6) )
    gs = gridspec.GridSpec( 6, 1)
    axAa = plt.subplot( gs[:3,0])
    axAb = plt.subplot( gs[3:,0])
    
   # axBa = plt.subplot( gs[:3,1])
   # axBb = plt.subplot( gs[3:,1])
    
    fig.subplots_adjust(hspace=0) 

    principalComponentList = np.linspace(3, 9, 7)
    nSamples = 10000
    for iPrincipalComponent in principalComponentList:
        pklFile = 'pickles/newHubbleInterpolatorTest_%i.pkl' % iPrincipalComponent
        
        if os.path.isfile( pklFile):
            diffArrays = pkl.load(open(pklFile, 'rb'))
        else:
            diffArrays = getDiffArraysForNumPrincpalComponents( np.int(iPrincipalComponent), nSamples=nSamples )
            pkl.dump(diffArrays, open(pklFile, 'wb'))
  
        #if iPrincipalComponent==9:
         #   for i in np.arange(nSamples):
          #      #axAa.plot( diffArrays['x'], diffArrays['diffPCA'][i,:], color='grey', alpha=0.5)
           #     axAa.plot( diffArrays['x'], diffArrays['diffPredict'][i,:], color='grey', alpha=0.1)

        #axAb.plot(diffArrays['x'], np.sqrt(np.mean(diffArrays['diffPCA']**2, axis=0)), label='%i' % iPrincipalComponent)
        axAa.plot(diffArrays['x'], np.mean(diffArrays['diffPredict'], axis=0)*100, label='%i' % iPrincipalComponent)
        variance =  np.sqrt(np.mean(diffArrays['diffPredict']**2, axis=0))
        axAb.plot(diffArrays['x'], variance)
        
    axAa.legend(ncol=3, prop={'size': 10}, title='# Principal Comp', fontsize=6)
    
    axAb.set_xlabel(r'$log(\Delta t$ /days)')
    #axBb.set_xlabel(r'$log(\Delta t$ /days)')
    
    axAa.set_ylabel(r'$\langle \bar{CDF} - CDF_{T}\rangle$ $/10^2$')
    axAb.set_ylabel(r'$\sqrt{\langle (\bar{CDF} - CDF_{T})^2\rangle}$')
    #axBa.set_ylabel(r'$CDF_{\tilde{T}} - CDF_{T}$')

    axAa.set_xlim(-1,3)
    axAb.set_xlim(-1,3)
    #axBa.set_xlim(-1,3)
    #axBb.set_xlim(-1,3)
    print('Maximum statistical limit is %0.3f' % np.max(variance)) 
    axAa.set_xticklabels([])
    #axBa.set_xticklabels([])
    fig.align_ylabels()
    plt.savefig('../plots/gprWithInterpolator.pdf')
    plt.show()


    
def getDiffArraysForNumPrincpalComponents( nComponents, nSamples=100 ):

    '''
    New method, test the GPR for the default cosmology
    and then lineraly interpolate this to a new cosmology
    '''

    #hubble interpolator over a small number o
    hubbleInterpolator = \
      hubbleModel.hubbleInterpolator( nPrincipalComponents=nComponents )
      
    hubbleInterpolator.getTrainingData('exactPDFpickles/trainingDataWithMass.pkl')

    hubbleInterpolator.getTimeDelayModel()
    

    allDistributions = \
      pkl.load(open(hubbleInterpolator.allDistributionsPklFile,'rb'))


    #set up an array
    #This one is truth - predictedCDF
    diffArray = np.zeros((nSamples, len(hubbleInterpolator.timeDelays)))
    
    #this one is truth - true pca with shifted cosmology
    diffPCA= np.zeros((nSamples, len(hubbleInterpolator.timeDelays)))

    #Cosmology labels
    cosmoKeys = hubbleInterpolator.fiducialCosmology.keys()
    
    doneInts = []
    for i in np.arange(nSamples):
        
        #Cant do all of them so randomly select one
        randInt = np.random.randint(0, len(allDistributions))

        #Makes sure i dont re-do one
        if randInt in doneInts:
            continue
        doneInts.append(randInt)

        #Get the raw distriubtion
        iDist = allDistributions[randInt]

        #and determine the cdf
        truth = iDist['y'][ iDist['x'] > \
                hubbleInterpolator.logMinimumTimeDelay]
                
        truthCumSum = np.cumsum(truth)/np.sum(truth)

        #Get the params of this distribution
        params = iDist['cosmology']
      
        fileName = iDist['fileNames'][0]
        zLensStr = fileName.split('/')[-2]
        zLens = np.float(zLensStr.split('_')[1])
        densityProfile = \
          getDensity.getDensityProfileIndex(fileName)[0]

        totalMassForHalo = getMass.getTotalMass( fileName )

        defaultDistributionIndex = \
              ( hubbleInterpolator.features['densityProfile'] == densityProfile ) &\
              ( hubbleInterpolator.features['zLens'] == zLens ) & \
              ( hubbleInterpolator.features['totalMass'] == totalMassForHalo )
              

        params['zLens'] = zLens
        params['densityProfile'] = densityProfile
        params['totalMass'] = totalMassForHalo
        params['H0'] /= 100.
        
        #and get the principal components that describe this
        truePrincpalComponents = \
          hubbleInterpolator.principalComponents[defaultDistributionIndex,:]

        #and then the distriubtion described by the PCA in the default cosmology
        pcaCDFinDefaultCosmology =  hubbleInterpolator.pca.inverse_transform( truePrincpalComponents[0] )
        
        #Get the cosmological interpolated shift and shift it               
        interpolateThisCosmology = \
          np.array([ iDist['cosmology'][i] for i in \
                         hubbleInterpolator.fiducialCosmology.keys()])
                         
        cosmoShift = \
          hubbleInterpolator.interpolatorFunction.predict(interpolateThisCosmology.reshape(1,-1))

          
        spline = CubicSpline( hubbleInterpolator.timeDelays+cosmoShift, pcaCDFinDefaultCosmology)
        
        #So this is the PDF of the true components, interpolated to the new cosmology
        pcaPDFinShiftedCosmology = spline( hubbleInterpolator.timeDelays)
        

        diffPCA[i,:] = pcaPDFinShiftedCosmology - truthCumSum
    
        #This is the predicted CDF from GPR interpolated to the new cosmology
        predictCDF = hubbleInterpolator.predictCDF( hubbleInterpolator.timeDelays, params)
        diffArray[i,:] =  predictCDF - truthCumSum

    return {'x':  hubbleInterpolator.timeDelays, 'diffPredict': diffArray, 'diffPCA':diffPCA}

def getBestParsForGPR():
  
    hubbleInterpolator = \
      hubbleModel.hubbleInterpolator( )
      
    hubbleInterpolator.getTrainingData('pickles/trainingDataWithMass.pkl')

    hubbleInterpolator.getTimeDelayModel()

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

    nLength=10
    nWhite = 100
    maternWeightList = [1.5] #10**np.linspace(-1,3,nLength)
    whiteKernelList = 10**np.linspace(-5,0,nWhite)

    logLike = np.zeros([nLength,nWhite])
    for i, maternWeight in enumerate(maternWeightList):
        for j, whiteKernel in enumerate(whiteKernelList):
            hubbleInterpolator.regressorNoiseLevel= whiteKernel
            hubbleInterpolator.learnPrincipalComponents( length_scale=maternWeight)
            logLike[ i, j] = hubbleInterpolator.getGaussProcessLogLike()
            #if logLike[ i, j] > defaultLog:
            #    print(maternWeight,whiteKernel, logLike[ i, j] - defaultLog)

            #pdb.set_trace()
    #plt.imshow(logLike, extent=[-5, 0, -1, 3], origin='lower')

    plt.figure(figsize=(8,6))
    gs =  gridspec.GridSpec(2,1)
    ax = plt.subplot(gs[0,0])
    ax.set_xlabel(r'log(Noise Level)')
    ax.set_ylabel(r'log likelihood')
    #plt.ylabel(r'log(Matern Length Scale)')
    print(whiteKernelList[np.argmax(logLike[0,:])])
    ax.plot(whiteKernelList, logLike[0,:], '-')
    ax.set_xscale('log')
    
    plt.savefig('../plots/GPRparams.pdf')
    plt.show()
                    
    
    pdb.set_trace()

def plotAllDefaultCosmologies():

    hubbleInterpolator = hubbleModel.hubbleInterpolator( regressorNoiseLevel=1e0)
      
    hubbleInterpolator.getTrainingData('exactPDFpickles/noCosmologyWithMass.pkl')

    hubbleInterpolator.getTimeDelayModel()
    for i in np.arange(hubbleInterpolator.pdfArray.shape[0]):
        plt.plot(hubbleInterpolator.timeDelays, \
                hubbleInterpolator.pdfArray[i,:], color='grey', alpha=0.5)
        inputParams = cp(hubbleInterpolator.fiducialCosmology)
       
        inputParams['H0'] /= 100.
        inputParams['zLens'] = hubbleInterpolator.features['zLens'][i]
        inputParams['densityProfile'] = \
          hubbleInterpolator.features['densityProfile'][i]
        inputParams['totalMass'] = \
          hubbleInterpolator.features['totalMass'][i]   
        predict = \
          hubbleInterpolator.predictCDF(hubbleInterpolator.timeDelays, inputParams)
        plt.plot(hubbleInterpolator.timeDelays, predict, color='red', alpha=0.5)

                
    pklFileName = \
      '../output/CDM/selectionFunction/'+\
      'allHalosFiducialCosmology.pkl'
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
    cumsum = np.cumsum(finalMergedPDFdict['y'])/np.sum(finalMergedPDFdict['y'])
    plt.plot(finalMergedPDFdict['x'],cumsum, 'b')

    plt.plot(hubbleInterpolator.timeDelays,\
                np.mean(hubbleInterpolator.pdfArray, axis=0), 'g')

    plt.show()
                
    
if __name__ == '__main__':
    main()
    #plotAllDefaultCosmologies()
    #getBestParsForGPR()
