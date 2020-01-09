import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
from cosmolopy import distance
from timeDelayDistributionClass import *

def plotMagBiasedSISexample():
    jsonFile = '../output/SISexample/SIS_example_z0.2_400_4.SISexample.json'
    cluster = timeDelayDistribution(jsonFile, zLens=0.2)

    plt.plot(cluster.finalPDF['finalLoS'][0].timeDelayPDF['x'],\
                 cluster.finalPDF['finalLoS'][0].timeDelayPDF['y'])
                 
    plt.plot(cluster.finalPDF['finalLoS'][0].biasedTimeDelayPDF['x'],\
              cluster.finalPDF['finalLoS'][0].biasedTimeDelayPDF['y'], label='Biased')
    plt.show()
    
def plotMagBiasedPDF():
    pklFile = '../output/CDM/combinedPDF_100.0.pkl'
    
    pdf = pkl.load(open(pklFile, 'rb'))
    print(pdf.keys())
    dx = pdf['x'][1] - pdf['x'][0]
    yLens = pdf['yLensPlane']/np.sum(pdf['yLensPlane'])/dx
    yB = pdf['yBiasedLens']/np.sum(pdf['yBiasedLens'])/dx
    yLoS = pdf['y']/np.sum(pdf['y'])/dx

    plt.plot(pdf['x'],yLens )
    plt.plot(pdf['x'],yB, label='Bias')
    plt.plot(pdf['x'],yLoS, label='LoS')

    plt.legend()
    #plt.yscale('log')
    plt.show()


def plotMagnificationBias():
    redshifts = np.linspace(20., 28., 10)
    magnification = 1.5
    bias = []
    for i in redshifts:
        bias.append( magnificationBias( 5., [magnification], limitingObsMag=i ))

    plt.plot(redshifts, bias)
    plt.show()
    
def magnificationBias( redshift, magnification, limitingObsMag=27, maxMemory=10000 ):
    '''
    Get the magnification bias for a given geodesic

    I have changed it so it doesn loop anymore and puts it in a an array
    This way it is significatly quicker, but heavy on the memory
    So if magnificaito is very large and blows the computer up, i have a limit
    and cut it into bits and do in chunks

    '''
    bias = []
    lumFuncRaw = \
      luminosityFunction( redshift, limitingObsMag=limitingObsMag  )

    nMagnifications = len(magnification)

    if nMagnifications <= maxMemory:
        biasMagnitudes = np.matrix(np.tile( lumFuncRaw.magnitudes, (nMagnifications,1)))
    else:
        #Memory fail
        #do it in bits
        nIterations = np.int(np.ceil(len(magnification )/maxMemory))
        bias = np.array([])
        print("Protecting memory doing in %i bits" % nIterations)
        for i in range(nIterations):
            
            print("%i/%i" % (i+1, nIterations))
            iMagnifications = magnification[i*maxMemory:(i+1)*maxMemory]
            iBias = magnificationBias(redshift, iMagnifications)
            bias =np.append( bias, iBias)
         
        return bias
    '''
    m - mf = -2.5( log(m) - log(m/mu))
    m - mf = -2.5( log(m) - log(m) + log(mu))
    m - mf = -2.5logmu
    mf = m + 2.5logmu
    '''
    matrixMags = np.matrix(2.5*np.log10(magnification))

    biasMagnitudes += matrixMags.T

    biasedLumFuncs = \
        getLuminosityFunctionMatrix( np.array(biasMagnitudes), \
                                     lumFuncRaw.magStar, lumFuncRaw.lumStar)

    newBias = np.sum(biasedLumFuncs['y'], axis=1) / \
        np.sum(lumFuncRaw.luminosityFunction['y'])
    
    return newBias

class luminosityFunction():
    '''
    In order to calcualte the magnification bias 
    i need to the quasar lumonisoity funciton
    
    This class does this using the maths
    laid out in 
    https://arxiv.org/pdf/1001.2037.pdf

    using fitting coefficients from
    https://arxiv.org/pdf/1612.01544.pdf

    '''

    def __init__( self, redshift, limitingObsMag = 27):
        cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
        cosmo = distance.set_omega_k_0(cosmo)    

        distancePc = distance.luminosity_distance(redshift, **cosmo)*1e6
        
        limitingAbsoluteMag = limitingObsMag - \
          5.*np.log10(distancePc) - 5
        self.magnitudes = np.linspace(-28, limitingAbsoluteMag, 10000)
        
        self.redshift=redshift
        self.getLuminosityStar()
        self.getMagnitudeStar()
        self.getLuminosityFunction()

    def getLuminosityFunction( self, alpha=-3.23, beta=-1.35):

        brightEndPower = 0.4*(alpha+1)*(self.magnitudes - self.magStar)
        faintEndPower = 0.4*(beta+1)*(self.magnitudes - self.magStar)
        #Equation 10
        self.luminosityFunction = {'x':self.magnitudes,
          'y':self.lumStar/(10**brightEndPower + 10**faintEndPower)}



    def getLuminosityStar( self):
        '''
        get the normalisation of the luminosity function
        https://arxiv.org/pdf/1612.01544.pdf

        '''
        a = -6.0991
        b = 0.0209
        c = 0.0171
        
        logPhiStar = a +\
          b*self.redshift +\
          c*self.redshift**2

        self.lumStar = np.exp(logPhiStar)

    def getMagnitudeStar( self, h=0.7):
        '''
        get the normalisation of the luminosity function
        https://arxiv.org/pdf/1612.01544.pdf

        '''
        a = -22.5216
        b = -1.6510
        c = 0.2869
        
        self.magStar = a +\
          b*self.redshift +\
          c*self.redshift**2 

def getLuminosityFunctionMatrix( magnitudes, magStar, lumStar, \
                                 alpha=-3.23, beta=-1.35):

    brightEndPower = 0.4*(alpha+1)*(magnitudes - magStar)
    faintEndPower = 0.4*(beta+1)*(magnitudes - magStar)
    #Equation 10
    luminosityFunction = {'x':magnitudes,
       'y':lumStar/(10**brightEndPower + 10**faintEndPower)}
    return luminosityFunction

if __name__ == '__main__':
    plotMagBiasedPDF()
