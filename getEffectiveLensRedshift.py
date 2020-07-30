
from PlaneLenser import Distances
import lensing_parameters as lens
import numpy as np
import ipdb as pdb
from fitDataToModel import main as fitDataToModel
from fitDataToModel import getObservations
from fitDataToModel import getMCMCchain
from matplotlib import pyplot as plt
from matplotlib import gridspec
import hubbleInterpolatorClass as hubbleModel
from itertools import product
from scipy.interpolate import interpn
from scipy.interpolate import NearestNDInterpolator
import pickle as pkl
from matplotlib import gridspec
from fitHubbleParameter import lnprob as getPosterior

def main():
    data = np.loadtxt( '../data/oguriTimeDelays.txt',\
                           dtype=[('name',object), ('zs', float), \
                                    ('zl', float),\
                                      ('timeDelays', float), \
                                      ('timeDelayErr', float) ])

    
    timeDelays = data['timeDelays']
    allSamples = getMCMCchain('LCDM')

    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(9,1)
    axis = plt.subplot( gs[0:4,0])
    axisA = plt.subplot( gs[5:,0])
    
    
    zSource = data['zs'][data['zl'] > 0]
    zLens = data['zl'][data['zl'] > 0]
    
    plotHistogram( zLens, axis, color='red', label='Data',bins=8 )
    plotHistogram( allSamples[:,1], axis, color='blue', label='Estimate' )

    plotHistogram( allSamples[:,3], axisA, color='blue', label='Estimate', **{'lw':2} )

    masses = getEstmatedMass( zLens, zSource, rsep=1.5 )
    plotHistogram( masses, axisA, color='red', label=r'Data ($\bar{r}_{\rm sep}=1.5$")',bins=10 )
    
    masses = getEstmatedMass( zLens, zSource, rsep=1. )
    plotHistogram( masses, axisA, color='green', label=r'Data ($\bar{r}_{\rm sep}=1$")',bins=10 )

    
    
    
    axis.set_xlabel('Lens Redshift')

    axisA.set_xlabel(r'log($M(<5$kpc$)/M_\odot$)')
    axisA.set_ylabel(r'Probability Density')
    axis.set_ylabel(r'Probability Density')
    axis.legend()
    axisA.legend()

    plt.savefig('../plots/redshiftDistribution.pdf')
    plt.show()

def getEstmatedMass( zLens, zSource, rsep=1.5 ):
    #assuming an einstein radius of 5kpc

    c = 3e5 
    dlsOverdsdl = \
      np.array([ 1./lens.ang_distance(zLens[i])*\
                lens.ang_distance(zSource[i],z0=zLens[i])/\
            lens.ang_distance(zSource[i]) for i in np.arange(len(zLens))])
    dlsOverdsdl = dlsOverdsdl.T
    einsteinRadiusRad = rsep/2./206265.

    einsteinRadiusRadKpc = \
      np.array([einsteinRadiusRad*lens.ang_distance(i)*1e3 for i in zLens])

    ratio = (5./einsteinRadiusRadKpc)

    ratioMass = np.log10(einsteinRadiusRad**2/4.3e-3/4*c**2/(dlsOverdsdl[0]/1e6)*ratio)
    mass = np.log10(einsteinRadiusRad**2/4.3e-3/4*c**2/(dlsOverdsdl[0]/1e6))
    #pdb.set_trace()
    

    
    return ratioMass
    
    

def plotHistogram( samples, axis, color='red', label=None, weight=None,\
                bins=10, **kwargs ):
    
    y, x = np.histogram( samples,  bins=bins, density=True, \
                                 weights=weight)
    dX = (x[1] - x[0])/2.
    xcentres = (x[:-1]+x[1:])/2.
                
    xcentres = np.append( x[0]-dX, xcentres)
    y = np.append(0, y)

    xcentres = np.append(xcentres, xcentres[-1])
    y = np.append(y, 0)
            
    axis.step( xcentres, y, color=color, \
                   label=label, **kwargs)
    axis.fill_between( xcentres, \
                        np.zeros(len(y)), y, step='pre', \
                        color=color, alpha=0.5)

def posteriorGradientForMassAndHubble(nPoints=10):
    '''
    What is the gradient of the posterior dH0/dlogM?
    '''
    
    hubbleInterpolaterClass = \
          hubbleModel.hubbleInterpolator( )
    
  
    hubbleInterpolaterClass.getTrainingData('pickles/trainingDataForObsWithMass.pkl')

    hubbleInterpolaterClass.getTimeDelayModel()

    
    data = getObservations()
    
    
    posteriorPoints = [list(np.linspace(0.65,0.75,nPoints))]
    for iParam in hubbleInterpolaterClass.features.dtype.names:
        minParam =np.min(hubbleInterpolaterClass.features[iParam])
        maxParam =np.max(hubbleInterpolaterClass.features[iParam])
                             
        posteriorPoints.append(list(np.linspace(minParam, maxParam, nPoints)))
        
    nParams = len(posteriorPoints)
    restOfCosmology = [0.3, 0.7, 0.]
    prob = np.array([])
    for progress, iParamCombination in enumerate(product(*posteriorPoints)):

     
        allParams = list(iParamCombination)+restOfCosmology

        prob = np.append(prob, getPosterior( allParams, data['x'], data['y'], None, hubbleInterpolaterClass ))

    probArrayShape = tuple(np.array([ 0 for i in range(nParams) ]) +nPoints)
    probArray = prob.reshape(probArrayShape)
    probArray[np.isfinite(probArray) == False ] = np.nan
    
    probArrayJustH0Mass = np.nansum(np.nansum(probArray, axis=1),axis=1)

    probArrayJustH0Mass /= np.max(probArrayJustH0Mass)

    extent = [0.65,0.75,np.min(hubbleInterpolaterClass.features['totalMass']),\
                  np.max(hubbleInterpolaterClass.features['totalMass'])]

    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
    plt.imshow(probArrayJustH0Mass, origin='lower',extent=extent, aspect=aspect)

    plt.show()
    pdb.set_trace()
        
if __name__ == '__main__':
    main()
