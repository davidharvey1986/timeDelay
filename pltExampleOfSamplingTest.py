from predictConstraintsOnHubble import *

def plotCornerPlot( sampleSize=100, trueDistribution=False):

    labels = \
      [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',\
           r'$\alpha$', r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$',r'$\Sigma_\alpha$',r'$\Sigma_z$' ]
    ndim = 5
            
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(15,15))
    color = ['blue','red','green','cyan']
    
    for icolor, sampleSize in enumerate([0.,10]):
        samples = getMCMCchainForSamplesSize(100, 10,  None, minimumTimeDelay=sampleSize)
        if (not trueDistribution):
            truths  = [0.7, 0.5, -1.75, 0.3, 0.7, 0.]
        else:
            truths = None
            for i in axarr[:,0]:
                i.plot([0.7,.7],[-2,10],'k--')

        nsamples = samples.shape[0]
                   
        corner.corner(samples , \
                      bins=20, smooth=True, \
                      plot_datapoints=False,
                      fig=figcorner, \
                      labels=labels, plot_density=True, \
                      truths=truths,\
                      weights=np.ones(nsamples)/nsamples,\
                    color=color[icolor],\
                          levels=(0.68,0.95), labelsize=15,\
                          truth_color='black')

        
    if  trueDistribution:
        reportParameters(samples)
        #axarr[1,1].set_xlim( 0.45, 0.53)
        #axarr[2,2].set_xlim( -1.9, -1.6)
        #axarr[1,0].set_ylim( 0.44, 0.53)
        #axarr[2,0].set_ylim( -1.9, -1.6)
        #axarr[2,1].set_ylim( -1.9, -1.6)
        #axarr[2,1].set_xlim( 0.44, 0.53)

    else:
        axarr[1,1].set_xlim( 0.3, 0.58)
        axarr[2,2].set_xlim( -1.86, -1.68)
        axarr[1,0].set_ylim( 0.3, 0.58)
        axarr[2,0].set_ylim( -1.86, -1.68)
        axarr[2,1].set_ylim( -1.86, -1.68)
        axarr[2,1].set_xlim( 0.3, 0.58)

    for i in np.arange(ndim):
        axarr[i,0].set_xlim( 0.6, 0.8)


    #hundreds = mlines.Line2D([], [], color='blue', label=r'$10^2$ Lenses')
    #thousand = mlines.Line2D([], [], color='red', label='$10^3$ Lenses')
    #tenthous = mlines.Line2D([], [], color='green', label='$10^4$ Lenses')
    
    ##axarr[0,1].legend(handles=[hundreds,thousand,tenthous], \
      #  bbox_to_anchor=(0., 0.25, 1.0, .0), loc=4)
    if trueDistribution:
        plt.savefig('../plots/degenerciesTrueDistribution.pdf')
    else:
        plt.savefig('../plots/degenercies.pdf')
    plt.show()

    
if __name__ == '__main__':
    plotCornerPlot()
