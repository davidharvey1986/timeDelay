import numpy as np
from plotHubbleDistributions import plotPDF
import os
import pickle as pkl
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
from matplotlib import pyplot as plt
import ipdb as pdb
from matplotlib import rcParams
rcParams["font.size"] = 16

def main():
    
    hubbleParameters = np.linspace(60,80,11)
    omegaMatter = np.linspace(0.25, 0.35, 11)
    omegaK = np.linspace(-0.02, 0.02, 11)
    omegaL = np.linspace(0.65,0.75,11)
    cosmologyList = \
      {'H0':hubbleParameters, 'OmegaM':omegaMatter, \
           'OmegaK':omegaK, 'OmegaL':omegaL}
    colorMaps = ['Reds','Greens','Blues','Purples','Oranges']

    labels = {'H0':'$H_0$', 'OmegaM':r'$\Omega_M$', \
                  'OmegaK':r'$\Omega_K$', 'OmegaL':r'$\Omega_\Lambda$'}
    fig, axarr = plt.subplots( len(cosmologyList.keys()), figsize=(12,12))
    
    fig.subplots_adjust(hspace=0)
    
    
    for iColorMap, iCosmoPar in enumerate(cosmologyList.keys()):
        
        #For aesthetics                                               
        jet = cm = plt.get_cmap(colorMaps[iColorMap])
        cNorm  = colors.Normalize(vmin=0, vmax=11)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        #####
        axisA = axarr[iColorMap]
        cosmology = {'H0':70., 'OmegaM':0.3, 'OmegaK':0, 'OmegaL':0.7}

        defaultCosmologyPkl = \
              "../output/CDM/combinedPDF_h%0.2f_oM%0.4f_oK%0.4f_%0.4f.pkl" \
              % (cosmology['H0'],cosmology['OmegaM'],cosmology['OmegaK'], \
                               cosmology['OmegaL'])
        defaultCosmology = pkl.load(open(defaultCosmologyPkl,'rb'))
        for i in defaultCosmology.keys():
                if 'y' in i:
                    defaultCosmology[i] =  1. - \
                      np.cumsum(defaultCosmology[i])/\
                      np.sum(defaultCosmology[i])
        
        for jColorInMap, iParInCosmoList in \
          enumerate(cosmologyList[iCosmoPar]):
            
            cosmology[iCosmoPar] = iParInCosmoList
            pklFileName = \
              "../output/CDM/combinedPDF_h%0.2f_oM%0.4f_oK%0.4f_%0.4f.pkl" \
              % (cosmology['H0'],cosmology['OmegaM'],cosmology['OmegaK'], \
                               cosmology['OmegaL'])



            if os.path.isfile(pklFileName):
                finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
            else:
                continue
                raise ValueError("No pickle file found (%s) "%pklFileName)


            for i in finalMergedPDFdict.keys():
                if 'y' in i:

                    finalMergedPDFdict[i] =  1. - \
                      np.cumsum(finalMergedPDFdict[i])/\
                      np.sum(finalMergedPDFdict[i])
                      
                    finalMergedPDFdict[i] -= defaultCosmology[i]


            label = r"%s: %0.2f" % (labels[iCosmoPar],iParInCosmoList)
            plotPDF( finalMergedPDFdict, scalarMap.to_rgba(jColorInMap), \
                label, axisA, yType='y', nofill=False )


            axisA.set_xlim(0.5,3)
           
        if iColorMap != len(cosmologyList.keys()) -1 :
            axisA.set_xticklabels([])
           
        scalarMap.set_array([])

        
        cax = fig.add_axes([0.8, 0.70-iColorMap/5.2, 0.025, 0.14])
        cbar = plt.colorbar( scalarMap, cax=cax)
        cbar.set_ticks([0,5,10])
        cbar.set_ticklabels([cosmologyList[iCosmoPar][i] \
                                 for i in cbar.get_ticks()])
        cbar.ax.set_title(labels[iCosmoPar])
    #axisA.set_yscale('log')
    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
            
    axisA.text(0.03, 0.5, \
                r'P(>log[$\Delta t$]) - P(>log[$\Delta t$])$_{\rm \Lambda CDM}$', \
                va='center',transform=fig.transFigure, rotation=90)
    plt.savefig('../plots/cosmoDependence.pdf')
    plt.show()
    
if __name__ == '__main__':
    main()
