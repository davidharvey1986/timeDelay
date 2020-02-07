
from convolveDistributionWithLineOfSight import *

def main():

    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/z_0.25'
    jsonFile = dataDir+'/B002_cluster_0_2_total_sph.fits.py.raw.json'


    
    ax = plt.gca()
    
    hubbleParameter = 70.
    cluster = timeDelayDistribution( jsonFile, \
            newHubbleParameter=hubbleParameter, \
            timeDelayBins=np.linspace(1,2,50), \
            outputPklFile='../output/CDM/singleSourcePlane/singleSourcePlane_%i.pkl' % hubbleParameter)
    sourcePlane = cluster.finalPDF['finalLoS'][-1]

    
    plt.plot(sourcePlane.timeDelayPDF['x'], sourcePlane.timeDelayPDF['y'],\
                 label='Lens Plane')
        
    plt.plot(sourcePlane.timeDelayWithLineOfSightPDF['x'], \
            sourcePlane.timeDelayWithLineOfSightPDF['y'], \
                 label='los')


    ax.set_xlabel(r'log($\Delta t$/ days)')
    ax.set_ylabel(r'P(log[$\Delta t$])')
    ax.set_ylim(0., 8.)
    ax.set_xlim(0.85, 2.3)
    
    ax.legend()
    plt.show()


    
    z = [i.data['z'] for i in  cluster.finalPDF['finalLoS']]
    w = [i.data['weight'] for i in  cluster.finalPDF['finalLoS']]
    
    
    plt.plot(z,w)
    plt.yscale('log')
    plt.show()



if __name__ == '__main__':
    main()
