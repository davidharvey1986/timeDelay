'''
Quickly plot the distributions of time delays from the same simulations
for two different H0

'''
import json as json
from matplotlib import pyplot as plt
import glob
import numpy as np
import ipdb as pdb

import os
import pickle as pkl
from matplotlib import gridspec
#import determineHaloToHaloVariance as h2h
from lensing import *
import matplotlib as mpl
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
from powerLawFit import *
from scipy.ndimage import gaussian_filter as gauss

def plotMassBins():
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
    allMassBins = [ i.split('/')[-1].split('_')[1].split('p')[0][0:-1] for i in glob.glob(dirD+'/CDM/massBin*.pkl')]

    colors = ['r','b','g','c','orange','k','grey']



    gs = gridspec.GridSpec(10,1)

    axisA = plt.subplot( gs[0:6,0])
    axisB = plt.subplot( gs[7:,0])
    axisC = axisB.twinx()
    allBeta = []
    allPeakTimes = []
    for iColor, iMassBin in enumerate(allMassBins):
        
        pklFileName = '../output/CDM/massBin_'+iMassBin+'.pkl'
        if not os.path.isfile( pklFileName):
            print('No filename %s' % pklFileName)
            continue
        
            
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            print("No pickle file found (%s) "%pklFileName)
            continue
        #finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])

        plotPDF( finalMergedPDFdict, colors[iColor], r"log($M/M_\odot$)=%0.2f" % np.float(iMassBin), axisA, yType='y', nofill=True )
            
        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = powerLawFit( finalMergedPDFdict )
                                             
        #axisA.plot( finalMergedPDFdict['x'], powerLawFitClass.getPredictedProbabilities(),\
         #               '--', color=colors[iColor])   
        ######################################################

        axisB.errorbar( np.float( iMassBin)-0.01, powerLawFitClass.params['params'][1], \
                            yerr=powerLawFitClass.params['error'][1], fmt='o', color=colors[iColor])


        

        axisC.errorbar( np.float( iMassBin)+0.01,  \
                             powerLawFitClass.getFittedPeakTimeAndError()[0],
                            yerr= powerLawFitClass.getFittedPeakTimeAndError()[1], \
                            fmt='*', color=colors[iColor])

        allPeakTimes.append(powerLawFitClass.getFittedPeakTimeAndError()[0])
        allBeta.append(powerLawFitClass.params['params'][1])

    getAndPlotTrend( np.array(allMassBins).astype(float), allBeta, axisB, '-')
    getAndPlotTrend( np.array(allMassBins).astype(float), allPeakTimes, axisC, '--')

    trendParams, cov = \
      curve_fit( straightLine,  np.array(allMassBins).astype(float), allPeakTimes, p0=[1.,1.])
    pkl.dump( trendParams, open('peakTimeMassParams.pkl','wb'))
    
    axisA.legend()
    axisB.legend()
    axisA.set_yscale('log')

    axisC.set_ylabel(r'log[$\Delta t_{\rm peak}$ / days ]')

    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
    axisA.set_ylabel(r'P(log[$\Delta t$]) / P(log[$\Delta t_{\rm peak}$])')

    axisA.set_xlim(-1.,2.75)
    axisA.set_ylim(1e-2,2.)
    axisB.set_xlabel(r'log$(M_{\rm lens}/M_\odot$)')
    axisB.set_ylabel(r'$\beta$')
    plt.savefig('../plots/massBins.pdf')
    plt.show()




def differentLensRedshiftsSingleHubble():
    '''
    Create a plot for different redshifts
    '''

    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
    allRedshifts = [ i.split('/')[-1] for i in glob.glob(dirD+'/CDM/z*')]
    colors = ['r','b','g','c','orange','k']



    gs = gridspec.GridSpec(10,1)

    axisA = plt.subplot( gs[0:6,0])
    axisB = plt.subplot( gs[7:,0])
    axisC = axisB.twinx()
    
    for iColor, iRedshift in enumerate(allRedshifts):
        
        
        pklFileName = '../output/CDM/'+iRedshift+'/combinedPDF.pkl'
        
            
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            print("No pickle file found (%s) "%pklFileName)
            continue
        
        axisA.plot(finalMergedPDFdict['x'],finalMergedPDFdict['yBiased'], \
                     label=r"%s" % iRedshift.split('_')[1], color=colors[iColor])
        #axisA.plot(integratedFinalMergedPDFdict['x']+2.56,integratedFinalMergedPDFdict['y'], \
        #              ls='--', color=colors[iColor])

        
        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = powerLawFit( finalMergedPDFdict, inputYvalue='y', yMin=0 )
                                             
        ######################################################

        axisB.errorbar( np.float( iRedshift.split('_')[1]) -0.01, \
                            powerLawFitClass.params['params'][1], \
                            yerr=powerLawFitClass.params['error'][1], \
                            fmt='o', color=colors[iColor])


                            
    axisB.plot( 0, 0,'o', color='orange', label='Change lens and kernel')
    axisB.plot( 0, 0,'*', color='orange', label='Changing Kernel Only')
    allRedshiftsFloat = np.array([ np.float(i.split('_')[-1]) for i in allRedshifts if not '25' in i])
    zs=3.
    allDistances = np.array([ ang_distance( i)/ang_distance( zs) for i in allRedshiftsFloat])
    

    #axisB.plot( plotRedshifts, plotPrediction, 'k--', label=r'$\Gamma \propto D_{\rm S}/D_{\rm L}$')
    #h2h.oplotEnsembleDistribution(axisA, None)
    axisB.set_ylim(0.8,1.4)
    axisA.legend(title=r"$z_{\rm lens}$")
    #axisB.legend()
    axisA.set_yscale('log')


    axisC.set_ylabel(r'log[$\Delta t_{\rm peak}$ / days ]')

    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
    axisA.set_ylabel(r'P(log[$\Delta t$]) / P(log[$\Delta t_{\rm peak}$])')

    axisA.set_xlim(-2.,3.)
    axisB.set_xlim(0.1,0.8)

    axisA.set_ylim(1e-2,2.)
    axisB.set_xlabel('Lens Redshift')
    axisB.set_ylabel(r'$\beta$')
    plt.savefig('../plots/differentLensRedshiftsSingleHubble.pdf')
    plt.show()
    
def compareConvolvedToUnconvolvedLoS():
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''

    gs = gridspec.GridSpec(10,1)

    axisA = plt.subplot( gs[0:7,0])
    axisB = plt.subplot( gs[7:,0])

    colors = ['r','b','g','c','orange','k']

    pklFileName = '../output/CDM/combinedPDF_100.0.pkl'
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

    dx = finalMergedPDFdict['x'][1] - finalMergedPDFdict['x'][0]
    finalMergedPDFdict['y'] /= np.sum(finalMergedPDFdict['y'])*dx
    finalMergedPDFdict['yLensPlane'] /= np.sum(finalMergedPDFdict['yLensPlane'])*dx
    
    pdfInLinearTime = finalMergedPDFdict['y'] / ( 10**finalMergedPDFdict['x']*np.log(10.) )
    colors = ['orange','black']
    microLensing = [1.,2]
    for i, iMicro in enumerate(microLensing):
        pdfInLinearTimeSmoothed = gauss( pdfInLinearTime, iMicro)
        pdfInLogTimeSmoothed = pdfInLinearTimeSmoothed * ( 10**finalMergedPDFdict['x']*np.log(10.) )
        finalMergedPDFdict['microLensing'] = pdfInLogTimeSmoothed
    
        axisA.plot( finalMergedPDFdict['x'], finalMergedPDFdict['microLensing'], \
                        label='Microlensing (%i day(s))' % iMicro, color=colors[i])

        axisB.plot(finalMergedPDFdict['x'], finalMergedPDFdict['microLensing']-finalMergedPDFdict['yLensPlane'],\
                    color=colors[i])
    
    plotPDF( finalMergedPDFdict, 'red', 'Without LoS', axisA, yType='yLensPlane', nofill=True )
    plotPDF( finalMergedPDFdict, 'green', 'With LoS', axisA, yType='y', nofill=True )


    ratio = finalMergedPDFdict['y']-finalMergedPDFdict['yLensPlane']
#    ratioError = np.sqrt( (finalMergedPDFdict['yError']/finalMergedPDFdict['y'])**2 + \
 #     (finalMergedPDFdict['yLensPlaneError']/finalMergedPDFdict['yLensPlane'])**2 ) *ratio

    
    axisB.plot(finalMergedPDFdict['x'], ratio, color='green')


      
    #axisB.fill_between( finalMergedPDFdict['x']+2.56,\
    #                      ratio + ratioError/2., \
    #                      ratio - ratioError/2.,
    #                            color='red', alpha=0.5)


       
    axisB.plot([-1,1.5],[0,0],'k--')
    axisA.legend()
    
 
    axisB.set_xlabel(r'log($\Delta t$/ days)')
    axisA.set_ylabel(r'P(log[$\Delta t$])')

    axisB.set_ylabel(r'$P-P_{\rm int}$')

    axisA.set_xlim(-1.,1.5)
    axisB.set_xlim(-1.,1.5)
    axisA.set_ylim(1e-2,1.5)
    axisB.set_ylim(-0.3,0.3)
    #axisB.set_yscale('log')
    axisA.set_xticklabels([])

    plt.savefig('../plots/compareConvolveWithLoS.pdf')
    plt.show()
    
    
    
def allLensesDifferentHubbleValues():
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(15,1)

    axisA = plt.subplot( gs[0:11,0])
    axisB = plt.subplot( gs[12:,0])
    
    
    hubbleParameters = [50., 60., 70., 80., 90., 100.]
    colors = ['r','b','g','c','orange','k']

    allBeta = []
    allBetaError = []
    peakTimeDelay = []
    peakTimeDelayError = []

    for iColor, iHubbleParameter in enumerate(hubbleParameters):
        pklFileName = '../output/CDM/combinedPDF_'+str(iHubbleParameter)+'.pkl'
        if os.path.isfile(pklFileName):
            finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        else:
            raise ValueError("No pickle file found (%s) "%pklFileName)

        for i in finalMergedPDFdict.keys():
            if 'y' in i:
                finalMergedPDFdict[i] /= np.sum( finalMergedPDFdict[i])*(finalMergedPDFdict['x'][1]- finalMergedPDFdict['x'][0])

     
        plotPDF( finalMergedPDFdict, colors[iColor], \
            r"H0=%i kms/Mpc" % iHubbleParameter, axisA, \
                    yType='y', nofill=False )

        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = \
          powerLawFit( finalMergedPDFdict, inputYvalue='y', yMin=1e-2  )
        
        allBeta.append(powerLawFitClass.params['params'][1])
        allBetaError.append(powerLawFitClass.params['error'][1])

        peakTimeDelay.append(powerLawFitClass.getFittedPeakTimeAndError()[0])
        peakTimeDelayError.append(powerLawFitClass.getFittedPeakTimeAndError()[1])
        ######################################################

    axisB.errorbar( hubbleParameters, allBeta, yerr=allBetaError, fmt='.', color='k')
    axisB.set_ylabel(r'$\beta$')
    params, cov = \
      curve_fit( straightLine, hubbleParameters, allBeta, p0=[1.,1.], sigma=allBetaError)

    axisB.plot( hubbleParameters, straightLine( np.array(hubbleParameters), *params),'k--', ms=5)

    print("Beta = %0.5E + %0.2E H0" % tuple(params))
    print("BetaErrors = %0.5E + %0.2E H0" % tuple(np.sqrt(np.diag(cov))))
    
    axisC = axisB.twinx()
    axisC.errorbar( hubbleParameters, peakTimeDelay, yerr=peakTimeDelayError, fmt='*', color='k')
    params, cov = \
      curve_fit( oneOverTime, hubbleParameters, peakTimeDelay, p0=[1.,1.], sigma=peakTimeDelayError)
    axisC.plot( hubbleParameters, oneOverTime( np.array(hubbleParameters), *params),'k--')



    axisC.set_ylabel(r'log[$\Delta t_{\rm peak}$ / days]')


    print("Tpeak = %0.5E + %0.2E H0" % tuple(params))
    print("TpeakErrors = %0.5E + %0.2E H0" % tuple(np.sqrt(np.diag(cov))))



    ######################################################
    axisB.set_xlabel('Hubble Constant (km/s/Mpc)')

    
    axisA.legend()
    #axisA.set_yscale('log')
    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
    axisA.set_ylabel(r'P(log[$\Delta t$])')

    axisA.set_xlim(-1,2.75)
    #axisA.set_ylim(1e-2,2.)
    plt.savefig('../plots/allLensesDifferentHubbleValues.pdf')
    plt.show()

    tableOfParameters = open('../plots/hubbleParameters.tab','wb')
    tableOfParameters.write("\\begin{table}\n \centering \n \\begin{tabular}{|c|c|c|} \n \hline \\\\ \n")
    tableOfParameters.write("$H_0$ & $ \beta $ & $ \Delta t_{\\rm peak} $ \\\\ \n")
    for i in xrange(len(hubbleParameters)):
        tableOfParameters.write("%i & $ %0.2f \pm %0.2f $ & $ %0.2f \pm %0.2f $ \\\\ \n" % \
                            (hubbleParameters[i], allBeta[i],allBetaError[i],peakTimeDelay[i],peakTimeDelayError[i]))
    tableOfParameters.write("\hline \end{tabular} \n \caption{\label{tab:data}} \n \end{table}")
    tableOfParameters.close()
   
    
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
    #plotMassBins()
    #compareConvolvedToUnconvolvedLoS()
    allLensesDifferentHubbleValues()
    #idfferentLensRedshiftsSingleHubble()


