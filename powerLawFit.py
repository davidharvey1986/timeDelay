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
def lnprob( theta, xTrue, yTrue, error ):

    yTheory = straightLine( xTrue, *theta)

    return np.sum(norm.logpdf( yTheory, yTrue, error))
    
    
def straightLine( x, *p):
    p1, p2 = p
    #p2=2.
    return p1 +x*p2

def brokenPowerLaw( x, *p):
    p1, p2, p3, p4, p5= p
    print(p5)
    if (p4 < 0.001) | (p5 < 0) | (p5 > 1.):
        return np.zeros(len(x))+np.inf
    f = models.SmoothlyBrokenPowerLaw1D(amplitude=p5, x_break=p1, alpha_1=p2, alpha_2=p3)

    f.delta = p4

    return np.log10(f(10**x))

class powerLawFit:
    
    def __init__( self, pdf, yMax=None, yMin=1e-2, inputYvalue='y', \
                      loadPklFile='loadPklFile.pkl', curveFit=False):
        '''
        Init the pdf 
          
        the pdf is a dict of 'x', 'y', 'yLensPlane', 'yError','yLensPlaneError'
        
        '''
        self.loadPklFile = loadPklFile
        self.pdf = pdf
        if yMax is None:
            yMax = np.max( pdf[inputYvalue] )

        for iKey in pdf.keys():
            if 'y' in iKey:
                pdf[iKey] /= yMax

        index =  \
          (pdf[inputYvalue]>yMin) & \
          (pdf['x'] < pdf['x'][np.argmax(pdf[inputYvalue])])
              
        self.pdf = pdf
        self.xNoZeros = pdf['x'][index]
        self.yNoZeros = np.log10(pdf[inputYvalue][index])
        self.yErrorInLog =  pdf[inputYvalue+'Error'][index]/np.sqrt(40.) / (pdf[inputYvalue][index]*np.log(10.))
        if curveFit:
            self.fitPowerLawCuveFit()
        else:
            self.fitPowerLaw()
        
    def fitPowerLawCuveFit( self  ):

        #No zero

        coeffs, var = curve_fit( straightLine, self.xNoZeros, self.yNoZeros, p0=[1.,1.], maxfev=10000)
        error = np.sqrt(np.diag(var))

        self.params = {'params':coeffs, 'error':error}


    def fitPowerLaw( self, nthreads=4  ):

        if (os.path.isfile(self.loadPklFile )):
            self.samples = pkl.load(open(self.loadPklFile, 'rb'))
        else:

        #No zeros
            nwalkers = 10
            ndim = 2
            burn_len=100
            chain_len=1000

            pos0 = np.random.rand(nwalkers,ndim)*2.
            args = (self.xNoZeros, self.yNoZeros, self.yErrorInLog )
    
            dmsampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                          lnprob, \
                                          args=args, \
                                          threads=nthreads,\
                                        )
                                          
            pos, prob, state  = dmsampler.run_mcmc(pos0, burn_len)


    
            pos, prob, state  = dmsampler.run_mcmc(pos, chain_len)
            self.samples = dmsampler.flatchain
        
        errorLower, median, errorUpper = np.percentile(self.samples, [16, 50, 84], axis=0)
        error = np.mean([median - errorLower, errorUpper - median], axis=0)

        self.params = {'params':median, 'error':error}


    def getPredictedProbabilities( self, xInput=None ):
        if xInput is None:
            xInput = self.xNoZeros

        return 10**straightLine( xInput, *self.params['params'])
    
    def getFittedPeakTimeAndError( self ):
        '''
        Return the fitted peak time
        '''
        
        peakTime =   -self.params['params'][0]/self.params['params'][1]
        error = peakTime * np.sqrt( np.sum( (self.params['error']/self.params['params'])**2))
        return peakTime, error

    def saveSamples( self, pklFile):
        pkl.dump( self.samples, open(pklFile, 'wb'))
