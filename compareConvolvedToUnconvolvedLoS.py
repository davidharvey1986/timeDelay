from convolveDistributionWithLineOfSight import *
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter as gauss

def compareConvolvedToUnconvolvedLoS(hubbleParameter=70.):
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''
    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/z_0.25'
    jsonFile = dataDir+'/B002_cluster_0_2_total_sph.fits.py.raw.json'

    jsonFile = "../output/SISexample/SIS_example_z0.2_250_5.0_4.0.json"


    gs = gridspec.GridSpec(10,1)

    axisA = plt.subplot( gs[0:7,0])
    axisB = plt.subplot( gs[7:,0])

    colors = ['r','b','g','c','orange','k']

    pklFile = \
      '../output/CDM/singleSourcePlane/singleSourcePlaneSIS_70.pkl'
      
    cluster = timeDelayDistribution( jsonFile, \
            newHubbleParameter=hubbleParameter, \
            timeDelayBins=np.linspace(1,2,200), \
            outputPklFile=pklFile, zLens=0.2)
            
    sourcePlane = cluster.finalPDF['finalLoS'][-1]
    
    
    interpolatePDF = \
      sourcePlane.interpolateGivenPDF( sourcePlane.timeDelayWithLineOfSightPDF['x'], sourcePlane.timeDelayPDF)
    interpolateCDF = 1. - np.cumsum(interpolatePDF)/np.sum(interpolatePDF)  
    finalMergedPDFdict = \
      {'x':sourcePlane.timeDelayWithLineOfSightPDF['x'],\
        'yLensPlane':interpolateCDF, \
        'y':sourcePlane.timeDelayWithLineOfSightPDF['y']}


    dx = finalMergedPDFdict['x'][1] - finalMergedPDFdict['x'][0]
    finalMergedPDFdict['y'] /= np.sum(finalMergedPDFdict['y'])*dx
          
    pdfInLinearTime = finalMergedPDFdict['y'] / ( 10**finalMergedPDFdict['x']*np.log(10.) )

    colors = ['orange','black']
    microLensing = [1.,2.]
    
    for i, iMicro in enumerate(microLensing):
        pdfInLinearTimeSmoothed = gauss( pdfInLinearTime, iMicro)
        
        pdfInLogTimeSmoothed = \
          pdfInLinearTimeSmoothed * ( 10**finalMergedPDFdict['x']*np.log(10.) )
        finalMergedPDFdict['microLensing'] =  pdfInLogTimeSmoothed
        microLensingCumSum = 1. - np.cumsum(finalMergedPDFdict['microLensing'])/\
          np.sum(finalMergedPDFdict['microLensing'])
        
        axisA.plot( finalMergedPDFdict['x'], microLensingCumSum, \
                        label='Microlensing (%i day(s))' % iMicro, color=colors[i])

                        
        ratio = microLensingCumSum-finalMergedPDFdict['yLensPlane']

        axisB.plot(finalMergedPDFdict['x'], ratio, color=colors[i])


    axisA.plot( finalMergedPDFdict['x'], finalMergedPDFdict['yLensPlane'], \
                    color='red', label='Without LoS')
    yCumSum = 1.-np.cumsum(finalMergedPDFdict['y'])/np.sum(finalMergedPDFdict['y'])
    
    axisA.plot( finalMergedPDFdict['x'], yCumSum, color='green', \
                    label='With LoS' )


    ratio = yCumSum-finalMergedPDFdict['yLensPlane']

    axisB.plot(finalMergedPDFdict['x'], ratio, color='green')

    
   
    

       
    axisB.plot([-1,3.],[0,0],'r-')
    axisA.legend()
    
 
    axisB.set_xlabel(r'log($\Delta t$/ days)')
    axisA.set_ylabel(r'$p$(>log[$\Delta t$])')

    axisB.set_ylabel(r'$p-p_{\rm int}$')

    axisA.set_xlim(1.5,2.2)
    axisB.set_xlim(1.5,2.2)
    #axisA.set_ylim(-0.1,1.1)
    #axisB.set_ylim(-0.02,0.02)
    #axisB.set_yscale('log')
    axisA.set_xticklabels([])

    plt.savefig('../plots/compareConvolveWithLoS.pdf')
    plt.show()
    

if __name__ == '__main__':
    compareConvolvedToUnconvolvedLoS()
