'''
This script fits a power law to the input pdf

It assumes that we want to fit between the peak and the
probability > 1e-2


'''
import pickle as pkl
from scipy.optimize import curve_fit
import emcee
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import poisson

import numpy as np
import os
import ipdb as pdb
from astropy.modeling import models
from matplotlib import pyplot as plt


def lnprob( theta, xTrue, yTrue, error, hubbleInterpolator ):


    #maxProbCum =  np.cumsum(hubbleInterpolator.predictPDF( xTrue, np.array([0.7,0.3,-1.9]) ))
    #maxProbCum /= np.max(maxProbCum)

    
    #maxProb = 1./np.sum((maxProbCum - yTrue)**2)
    thetaDict = \
      {'H0':theta[0], 'zLens':theta[1], 'densityProfile':theta[2], 'totalMass':theta[3], \
        'OmegaM':0.3, 'OmegaL':0.7, 'OmegaK':0}

    cumsumYtheory = \
      hubbleInterpolator.predictCDF( xTrue, thetaDict )

      
   
    
    prior = priorOnParameters( thetaDict, hubbleInterpolator )
    
    prob = 1./np.sum((cumsumYtheory - yTrue)**2)

    if np.isnan(prob):
        pdb.set_trace()
        return -np.inf

   
    return prob*prior
    
def priorOnParameters( thetaDict, hubbleInterpolator ):

  
   
    for iThetaKey in hubbleInterpolator.features.dtype.names:
        if iThetaKey == 'totalMass':
            if (thetaDict['totalMass'] < 11) |\
              (thetaDict['totalMass'] > 13):
              return -np.inf
        else:
            if (thetaDict[iThetaKey] < \
                    np.min(hubbleInterpolator.features[iThetaKey])) | \
                (thetaDict[iThetaKey] > \
                    np.max(hubbleInterpolator.features[iThetaKey])):

                return -np.inf


    for iCosmoKey in hubbleInterpolator.cosmologyFeatures.dtype.names:
        if iCosmoKey == 'H0':
            continue
       
        if (thetaDict[iCosmoKey] < \
                np.min(hubbleInterpolator.cosmologyFeatures[iCosmoKey])) | \
          (thetaDict[iCosmoKey] > \
               np.max(hubbleInterpolator.cosmologyFeatures[iCosmoKey])):
            return -np.inf
    
    if (thetaDict['H0'] < 0.6) | (thetaDict['H0'] > 0.8):
        
        return -np.inf
    #if (thetaDict['variance'] < 0) | (thetaDict['variance'] > 0.5):
    #    return -np.inf

    variancePrior = norm.pdf(thetaDict['zLens'], loc=0.55, scale=0.4)
 
    return 1
class fitHubbleParameterClass:
    
    def __init__( self, pdf, hubbleInterpolator,\
                      yMax=None, yMin=1e-2, inputYvalue='y'):
        '''
        Init the pdf 
          
        the pdf is a dict of 'x', 'y', 'yLensPlane', 'yError','yLensPlaneError'
        
        '''
        self.hubbleInterpolator = hubbleInterpolator
                   
        self.pdf = pdf
       
        self.fitHubble()



    def fitHubble( self, nthreads=4  ):

  
        nwalkers = 20

        ndim = self.hubbleInterpolator.nFreeParameters  - 3

        burn_len=100
        chain_len=1000
        pos0 = np.random.rand(nwalkers,ndim)
        pos0[:,0] = np.random.uniform( 0.6, 0.8, nwalkers) 
        pos0[:,1] =  np.random.uniform( 0.2, 0.74, nwalkers)
        pos0[:,2] =  np.random.uniform( -1.6,-2., nwalkers)
        pos0[:,3] =  np.random.uniform( 11, 13., nwalkers)
        #pos0[:,4] =  np.random.uniform( 0.25, 0.35, nwalkers)
        #pos0[:,5] =  np.random.uniform( 0.65, 0.75, nwalkers)
        #pos0[:,6] =  np.random.uniform( -0.02, 0.02, nwalkers)
        

        args = (self.pdf['x'], self.pdf['y'], \
                    self.pdf['error'], self.hubbleInterpolator )

        dmsampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                            lnprob, \
                                          args=args, \
                                          threads=nthreads)
                                          
        pos, prob, state  = dmsampler.run_mcmc(pos0, burn_len, progress=True)


    
        pos, prob, state  = dmsampler.run_mcmc(pos, chain_len,\
                                        progress=True)
        self.samples = dmsampler.flatchain

        errorLower, median, errorUpper = \
          np.percentile(self.samples, [16, 50, 84], axis=0)

        error = np.mean([median - errorLower, errorUpper - median], axis=0)

        self.params = {'params':median, 'error':error}


    def getPredictedProbabilities( self, xInput=None ):
        if xInput is None:
            xInput = self.xNoZeros

        return 10**self.fitFunc( xInput, *self.params['params'])
    
    def getFittedPeakTimeAndError( self ):
        '''
        Return the fitted peak time
        '''
        
        peakTime =   -self.params['params'][0]/self.params['params'][1]
        error = peakTime * np.sqrt( np.sum( (self.params['error']/self.params['params'])**2))
        return peakTime, error

    def saveSamples( self, pklFile):
        pkl.dump( self.samples, open(pklFile, 'wb'))
