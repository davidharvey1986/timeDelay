'''
This script fits a power law to the input pdf

It assumes that we want to fit between the peak and the
probability > 1e-2


'''
import pickle as pkl
from scipy.optimize import curve_fit
import emcee
from scipy.stats import norm
import numpy as np
import os
import ipdb as pdb
from astropy.modeling import models
from matplotlib import pyplot as plt


def lnprob( theta, xTrue, yTrue, error, hubbleInterpolator ):

    yTheory = hubbleInterpolator.predictPDF( xTrue, theta )
    
    if np.any(np.isfinite(yTheory) == False):
        return -np.inf

    if np.any(yTheory < 0):
        return -np.inf

    
    prob = np.sum(norm.logpdf( yTheory[error!=0], \
                    yTrue[error!=0], error[error!=0]))
    
    if np.isnan(prob):
        pdb.set_trace()
        return -np.inf
    return prob
    


class fitHubbleParameterClass:
    
    def __init__( self, pdf, hubbleInterpolator,\
                      yMax=None, yMin=1e-2, inputYvalue='y', \
                      loadPklFile='loadPklFile.pkl'):
        '''
        Init the pdf 
          
        the pdf is a dict of 'x', 'y', 'yLensPlane', 'yError','yLensPlaneError'
        
        '''
        self.hubbleInterpolator = hubbleInterpolator
       
        self.loadPklFile = loadPklFile
            
        self.pdf = pdf
       
        self.fitHubble()



    def fitHubble( self, nthreads=4  ):

        if (os.path.isfile(self.loadPklFile )):
            self.samples = pkl.load(open(self.loadPklFile, 'rb'))
        else:

        #No zeros
            nwalkers = 20

            ndim = 1
            burn_len=20
            chain_len=50
            pos0 = np.random.rand(nwalkers,ndim)*2.+70.

            args = (self.pdf['x'], self.pdf['y'], \
                    self.pdf['error'], self.hubbleInterpolator )

            dmsampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                          lnprob, \
                                          args=args, \
                                          threads=nthreads,\
                                        )
                                          
            pos, prob, state  = dmsampler.run_mcmc(pos0, burn_len)


    
            pos, prob, state  = dmsampler.run_mcmc(pos, chain_len)
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
