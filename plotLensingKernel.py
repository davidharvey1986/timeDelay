'''
Quick Plot to show source redshifts
'''
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
from timeDelayDistributionClass import *

def getLensingKernel( zs, zl=0.3 ):
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
    cosmo = dist.set_omega_k_0(cosmo)

    dl =  dist.angular_diameter_distance(zl, **cosmo)
    ds =  dist.angular_diameter_distance(zs,  **cosmo)
    dls =  dist.angular_diameter_distance(zs, z0=zl, **cosmo)


    return dl*dls/ds

def main():
    zLenses = np.array([0.20, 0.25, 0.37, 0.50, 0.74])
    fig, axarr = plt.subplots( len(zLenses), figsize=(12, 4))
    plt.subplots_adjust(hspace=0)
    for iLensIndex, iLens in enumerate(zLenses):

        plotLensConfiguration(iLens, ax=axarr[iLensIndex],\
                    nBins=100, iHubbleParameter=70.)
        if iLensIndex != len(zLenses) -1:
            axarr[iLensIndex].set_xticklabels([])

        else:
            axarr[iLensIndex].set_xlabel('Redshift')
        axarr[iLensIndex].set_yticklabels([])
        if iLensIndex == 2:
            print(iLensIndex)
            axarr[iLensIndex].set_ylabel('Lensing Kernel (Arbitary Units)')

    plt.savefig('../plots/lensSourceConfiguration.pdf')
    plt.show()


        
def plotLensConfiguration(zLens, nBins=100, ax=None, iHubbleParameter=70.):
       

    
    zs = np.linspace(zLens, 8, nBins)
    if ax is None:
        ax = plt.gca()
    lensingKernel = getLensingKernel(zs, zl=zLens)
    lensingKernel /= np.max(lensingKernel)
    ax.plot(zs,lensingKernel, 'k-' )

    ax.plot([zLens, zLens], [0, np.max(lensingKernel)], 'r-')
    print(zLens)
    #For aesthetics                                                         
    jet = cm = plt.get_cmap('rainbow')
    cNorm  = colors.Normalize(vmin=np.log10(zs[0]), vmax=np.log10(zs[-1]))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #####
    
    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/z_%0.2f' % zLens
    jsonFile = dataDir+'/B002_cluster_0_2_total_sph.fits.py.raw.json'
    cluster = timeDelayDistribution( jsonFile, \
                    newHubbleParameter=iHubbleParameter, \
                    timeDelayBins=np.linspace(1,2,50), \
                    outputPklFile='../output/CDM/singleSourcePlane/singleSourcePlane_%0.2f_%i.pkl' % (zLens, iHubbleParameter))
    z0 = zLens
    for i, iSourcePlane in enumerate(cluster.finalPDF['finalLoS']):
        color = scalarMap.to_rgba( np.log10(z0))
        ax.fill_between( [z0, iSourcePlane.data['z']],\
                        [np.max(lensingKernel), np.max(lensingKernel)], \
                        color=color,alpha=0.8)
        z0 = iSourcePlane.data['z']
    ax.set_xscale('log')

    ax.set_xlim(0.1, 8)
    ax.set_ylim(0, np.max(lensingKernel))



if __name__ == '__main__':
    main()
