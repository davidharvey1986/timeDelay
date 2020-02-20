'''
Quickly plot the distributions of time delays from the same simulations
for two different H0

'''
import fitDataToModel as fdm
import json as json
from matplotlib import pyplot as plt
import glob
import numpy as np
import ipdb as pdb

import os
import pickle as pkl
from matplotlib import gridspec
#import determineHaloToHaloVariance as h2h
import matplotlib as mpl
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
from powerLawFit import *
from scipy.ndimage import gaussian_filter as gauss
import analyseSISexample as SISexample

from matplotlib import rcParams
rcParams["font.size"] = 16
    
    
def main():
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''

    plt.figure(figsize=(8,6))

    axisA = plt.gca()
    
    
    hubbleParameters = [50., 60., 70., 80., 90., 100.]
    colors = ['r','b','g','c','orange','k']

    allBeta = []
    allBetaError = []
    peakTimeDelay = []
    peakTimeDelayError = []

    for iColor, iHubbleParameter in enumerate(hubbleParameters):
        pklFileName = '../output/CDM/selectionFunction/SF_%i_lsst.pkl' % iHubbleParameter
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            raise ValueError("No pickle file found (%s) "%pklFileName)

        for i in finalMergedPDFdict.keys():
            if 'y' in i:
                finalMergedPDFdict[i] =  1. - np.cumsum(finalMergedPDFdict[i])/np.sum(finalMergedPDFdict[i])


     
        plotPDF( finalMergedPDFdict, colors[iColor], \
            r"H0=%i kms/Mpc" % iHubbleParameter, axisA, \
                    yType='y', nofill=False )

    
    axisA.legend()
    #axisA.set_yscale('log')
    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
    axisA.set_ylabel(r'P(>log[$\Delta t$])')

    axisA.set_xlim(-1,2.75)
    #axisA.set_ylim(1e-2,2.)
    pdf = fdm.getObservations()
    axisA.plot(pdf['x'],1.-pdf['y'])
    plt.savefig('../plots/allLensesDifferentHubbleValues.pdf')
    plt.show()
   
    
def plotPDF( PDF, color, label, axisA, yType='yBiasedLens', nofill=False ):

      


    bigError = (PDF[yType+'Error'] > PDF[yType]) 
    
    PDF[yType][ PDF[yType] == 0] = 1e-9
    PDF[yType+'Error'][ PDF[yType+'Error'] == 0] = 1e-10
         
    PDF[yType+'Error'][bigError] = \
      PDF[yType][bigError]*0.99

    if not nofill:
        axisA.fill_between( PDF['x'], \
                            PDF[yType] + PDF[yType+'Error']/2.,  \
                            PDF[yType] - PDF[yType+'Error']/2.,
                            color=color, alpha=0.5)

    #calibration
    axisA.plot(PDF['x'],PDF[yType],label=label, color=color)


    

def oneOverTime( x, *p):

    p1, p2 = p

    return p1 + p2/x

def getAndPlotTrend( x, y, axis, fmt):
  

    trendParams, cov = \
      curve_fit( straightLine, x, y, p0=[1.,1.])
    pError = np.sqrt(np.diag(cov))
    axis.plot(  x, straightLine(  x, *trendParams), fmt)

    
    pLower = [trendParams[0]+pError[0], \
                  trendParams[1]-pError[1]]
    pUpper = [trendParams[0]-pError[0], \
                  trendParams[1]+pError[1]]

    axis.fill_between( x, straightLine( x, *pUpper), \
                           straightLine( x, *pLower), \
                         alpha=0.3, color='grey' )
    
    

    
if __name__ == '__main__':
    main()



