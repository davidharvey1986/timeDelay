from predictConstraintsOnHubble import *

def plotDegenerciesOfDifferentHalos(sampleSize=10000):

    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$']
    ndim = 3
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(12,12))
    color = ['blue','red','green','cyan']

    for icolor, differentHalo in enumerate(['B002','B008']):
        samples = getMCMCchainForSamplesSize(sampleSize, 10, 70., None, differentHalo=differentHalo)
        

        for i in axarr[:,0]:
            i.plot([0.7,.7],[-2,10],'k--')

        nsamples = samples.shape[0]                     
        corner.corner(samples, \
                      bins=100, smooth=True, \
                      plot_datapoints=False,
                      fig=figcorner, \
                      labels=labels, plot_density=True, \
                      weights=np.ones(nsamples)/nsamples,\
                    color=color[icolor],\
                          levels=(0.68,), labelsize=15,\
                          truth_color='black')

    axarr[1,1].set_xlim( 0.4, 0.5)
    axarr[2,2].set_xlim( -1.9, -1.8)
    axarr[1,0].set_ylim( 0.4, 0.5)
    axarr[2,0].set_ylim( -1.9, -1.8)
    axarr[2,1].set_ylim( -1.9, -1.8)
    axarr[2,1].set_xlim( 0.4, 0.5)
    
    axarr[0,0].set_xlim( 0.65, 0.75)
    axarr[1,0].set_xlim( 0.65, 0.75)
    axarr[2,0].set_xlim( 0.65, 0.75)
    

    hundreds = mlines.Line2D([], [], color='blue', label=r'Subsample 1')
    thousand = mlines.Line2D([], [], color='red', label='Subsample 2')
    
    axarr[0,1].legend(handles=[hundreds,thousand], \
        bbox_to_anchor=(0., 0.25, 1.0, .0), loc=4)
   
    plt.savefig('../plots/degenerciesDifferentHalo.pdf')
    plt.show()

if __name__ == '__main__':
    plotDegenerciesOfDifferentHalos()
