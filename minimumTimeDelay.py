from predictConstraintsOnHubble import *

def minimumTimeDelay( sampleSize=7943):

    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',r'$\alpha$']
    ndim = 3
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(12,12))
    color = ['blue','red','green','cyan']
    minimumTimeDelay = [0., 10.]
    for icolor, iMinimumTimeDelay in enumerate(minimumTimeDelay):
        samples = getMCMCchainForSamplesSize(sampleSize, 10, 70., None,  minimumTimeDelay=iMinimumTimeDelay )#trueHubble=trueHubble)
        
    
        truths  = [0.7, 0.4, -1.75]
    

        nsamples = samples.shape[0]                     
        corner.corner(samples, \
            bins=100, smooth=True, \
            plot_datapoints=False,
            fig=figcorner, \
            labels=labels, plot_density=True, \
            truths=truths,\
            weights=np.ones(nsamples)/nsamples,\
            color=color[icolor],\
            levels=(0.68,), labelsize=15,\
            truth_color='black')


    axarr[1,1].set_xlim( 0.3, 0.58)
    axarr[2,2].set_xlim( -1.86, -1.68)
    axarr[1,0].set_ylim( 0.3, 0.58)
    axarr[2,0].set_ylim( -1.86, -1.68)
    axarr[2,1].set_ylim( -1.86, -1.68)
    axarr[2,1].set_xlim( 0.3, 0.58)
            
    axarr[0,0].set_xlim( 0.6, 0.8)
    axarr[1,0].set_xlim( 0.6, 0.8)
    axarr[2,0].set_xlim( 0.6, 0.8)
    

    hundreds = mlines.Line2D([], [], color='blue', label=r'$10^2$ Lenses')
    thousand = mlines.Line2D([], [], color='red', label='$10^3$ Lenses')
    tenthous = mlines.Line2D([], [], color='green', label='$10^4$ Lenses')
    
    axarr[0,1].legend(handles=[hundreds,thousand,tenthous], \
        bbox_to_anchor=(0., 0.25, 1.0, .0), loc=4)


    plt.show()

if __name__ == '__main__':
   minimumTimeDelay()
