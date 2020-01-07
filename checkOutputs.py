#!/usr/local/bin/python3
from convolveDistributionWithLineOfSight import *


def main():
    '''
    Check the outputs of the new models
    to older ones
    '''

    fileName = 'z_0.20/B002_cluster_0_1_total_sph.fits.py.raw.json'
    newHubbleParameter=100.
    combinedOldPDF = \
      timeDelayDistribution('../output/CDM_5_sourceplane/'+fileName,newHubbleParameter=newHubbleParameter)
    combinedNewPDF = \
      timeDelayDistribution('../output/CDM/'+fileName, newHubbleParameter=newHubbleParameter)

    print(combinedOldPDF.finalPDF['finalLoS'][-1].data['z'])
    print(combinedNewPDF.finalPDF['finalLoS'][-1].data['z'])
    
    
    plt.plot(combinedOldPDF.finalPDF['finalLoS'][-1].timeDelayWithLineOfSightPDF['x'], combinedOldPDF.finalPDF['finalLoS'][-1].timeDelayWithLineOfSightPDF['y'])
    
    plt.plot(combinedOldPDF.finalPDF['finalLoS'][-1].timeDelayPDF['x'], combinedOldPDF.finalPDF['finalLoS'][-1].timeDelayPDF['y'])
    
    plt.plot(combinedNewPDF.finalPDF['finalLoS'][-1].timeDelayWithLineOfSightPDF['x'], combinedNewPDF.finalPDF['finalLoS'][-1].timeDelayWithLineOfSightPDF['y'])
    plt.plot(combinedNewPDF.finalPDF['finalLoS'][-1].biasedTimeDelayWithLineOfSightPDF['x'], combinedNewPDF.finalPDF['finalLoS'][-1].biasedTimeDelayWithLineOfSightPDF['y'])
    plt.show()

if __name__ == '__main__':
    main()
    
    
