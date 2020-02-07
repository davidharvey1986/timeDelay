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

import numpy as np
import os
import ipdb as pdb
from astropy.modeling import models
from matplotlib import pyplot as plt


def lnprob( theta, xTrue, yTrue, error, hubbleInterpolator ):


    #maxProbCum =  np.cumsum(hubbleInterpolator.predictPDF( xTrue, np.array([0.7,0.3,-1.9]) ))
    #maxProbCum /= np.max(maxProbCum)

    
    #maxProb = 1./np.sum((maxProbCum - yTrue)**2)

    
    cumsumYtheory = hubbleInterpolator.predictPDF( xTrue, theta )
    


    #if np.any(np.isfinite(yTheory) == False):
    #    return -np.inf

    #if np.any(yTheory < 0):
     #   return -np.inf
    if (theta[0] < 0.55) | (theta[0] > 0.85):
        return -np.inf
    if (theta[1] < 0.) | (theta[1] > 1.0):
        return -np.inf
    if (theta[2] < -2.1) | (theta[2] > -1.0):
        return -np.inf
    
    #cumsumYtheory = np.cumsum( yTheory )/np.sum(yTheory)

    prob = np.sum(norm.logpdf( cumsumYtheory[error!=0], yTrue[error!=0], scale=error[error!=0]))
    
    #prob = 1./np.sum((cumsumYtheory - yTrue)**2)
    
    #if (prob > maxProb):

    if np.isnan(prob):
        pdb.set_trace()
        return -np.inf
    return prob
    


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

        ndim = 3
        burn_len=100
        chain_len=1000
        pos0 = np.random.rand(nwalkers,ndim)
        pos0[:,0] = np.random.rand( nwalkers) * 0.05 + 0.7
        pos0[:,1] =  np.random.randn( nwalkers) * 0.1 + 0.4
        pos0[:,2] =  np.random.randn( nwalkers) * 0.1 - 1.75
        
        args = (self.pdf['x'], self.pdf['y'], \
                    self.pdf['error'], self.hubbleInterpolator )

        dmsampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                            lnprob, \
                                          args=args, \
                                          threads=nthreads)
                                          
        pos, prob, state  = dmsampler.run_mcmc(pos0, burn_len)


    
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
