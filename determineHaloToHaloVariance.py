import pickle as pkl
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams["font.size"] = 16
import ipdb as pdb
def main():
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''

    plt.figure(figsize=(8,6))


    axisA = plt.gca()
    colors = ['r','b','g','c','orange','k','m','y','pink']
    allDistributionsPklFile = \
      "../output/CDM/selectionFunction/"+\
      "sparselyPopulatedParamSpace.pkl"
    allHalos = pkl.load(open(allDistributionsPklFile,'rb'))
    
    haloDistributions = {'B002':None,\
                         'B005':None,\
                          'B008':None,\
                            'B009':None}

    fiducialCosmology = \
       {'H0':70., 'OmegaM':0.3, 'OmegaL':0.7, 'OmegaK':0.}
    cosmoKeys = fiducialCosmology.keys()
    for iColor, iHalo in enumerate(allHalos):
        doNotTrainThisSample =  \
              np.any(np.array([fiducialCosmology[iCosmoKey] != \
              iHalo['cosmology'][iCosmoKey] \
              for iCosmoKey in cosmoKeys]))

        if doNotTrainThisSample:
            continue

        
        haloName = iHalo['fileNames'][0].split('/')[-1].split('_')[0] 

        iHalo['y'] =  1. - np.cumsum(iHalo['y'])/np.sum(iHalo['y'])
        if haloDistributions[haloName] is None:
            haloDistributions[haloName] = iHalo['y']
        else:
            haloDistributions[haloName] = \
              np.vstack((haloDistributions[haloName], iHalo['y']))
        

    for iColor, iHaloName in enumerate(haloDistributions.keys()):

        median, low, high = np.percentile( haloDistributions[iHaloName], [50, 16, 84], axis=0)

        axisA.plot(iHalo['x'],median, '--', \
                    label=r"%s" % (iHaloName), \
                        color=colors[iColor], lw=2)              
        axisA.plot(iHalo['x'],low, '-', \
                        color=colors[iColor], lw=1)
        axisA.plot(iHalo['x'],high, '-', \
                        color=colors[iColor], lw=1)  
        axisA.fill_between(  iHalo['x'], low, high, \
                                color=colors[iColor], alpha=0.1)
    axisA.legend()

    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    axisA.set_ylabel(r'P(>log($\Delta t$))')
    axisA.set_xlim(-1.,3.)
   

    plt.savefig('../plots/haloToHaloVariance.pdf')
    plt.show()


    
if __name__ == '__main__':

    main()
