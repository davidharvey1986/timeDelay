
from convolveDistributionWithLineOfSight import *

def main():

    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM.old/z_0.25'
    jsonFile = dataDir+'/B002_cluster_0_2_total_sph.fits.py.raw.json'


    hubbleParameters = [100., 50., 60., 70., 80., 90.]
    ax = plt.gca()
    for iHubbleParameter in hubbleParameters:

        cluster = timeDelayDistribution( jsonFile, \
                    newHubbleParameter=iHubbleParameter, \
                    timeDelayBins=np.linspace(1,2,50), \
                    outputPklFile='../output/CDM.old/singleSourcePlane/singleSourcePlane_%i.pkl' % iHubbleParameter)
        sourcePlane = cluster.finalPDF['finalLoS'][-1]


        plt.plot(sourcePlane.timeDelayPDF['x'], sourcePlane.timeDelayPDF['y'], label=r'$H_0$=%i kms$^{-1}Mpc$^{-1}$' % iHubbleParameter)
    

    ax.set_xlabel(r'log($\Delta t$/ days)')
    ax.set_ylabel(r'P(log[$\Delta t$])')
    ax.set_ylim(0., 8.)
    ax.set_xlim(0.85, 2.3)
    
    ax.legend()
    plt.show()

    




if __name__ == '__main__':
    main()
