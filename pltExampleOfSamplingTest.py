from predictConstraintsOnHubble import *
from matplotlib import rcParams
rcParams["font.size"] = 12
def plotCornerPlot( sampleSize=1000):

    labels = \
        [r'$H_0/ (100 $km s$^{-1}$Mpc$^{-1}$)',r'$z_{lens}$',\
            r'$\alpha$', r'$log(M(<5kpc)/M_\odot)$',\
           r'$\Omega_M$',r'$\Omega_\Lambda$',r'$\Omega_K$' ]
           
    ndim = 6
            
    figcorner, axarr = plt.subplots(ndim,ndim,figsize=(12,12))
    color = ['black','red','green','cyan']

    for icolor, minTimeDelay in enumerate([0.]):
        allSamples = \
          getMCMCchainForSamplesSize(sampleSize, 100,  None, \
                            minimumTimeDelay=minTimeDelay)
        itPerSample = np.int(allSamples.shape[0]/(100))
        truths  = [0.7, 0.4,  -1.75, 11.05, 0.3, 0.7, 0.]
        
        #Remove more of the burn in
        #sampleIndex = np.array([])
        #for iIteration in range(100):
        #    nKeep = np.int(0.5*itPerSample)
            
       #     sampleIndex = \
        #          np.append(sampleIndex, \
        #    np.arange((iIteration)*itPerSample+nKeep,\
        #                        (iIteration+1)*itPerSample))

        #samples = allSamples[sampleIndex.astype(int), :]
        #samples = checkConvergence( samples)
        nsamples = allSamples.shape[0]
        #maxLikes = getMaxLikeFromSamples(samples)

        #nsamples = maxLikes.shape[0]
        print(nsamples)
        corner.corner(allSamples , \
                      bins=40, smooth=2,smooth1d=1, \
                      plot_datapoints=False,
                      fig=figcorner, \
                      labels=labels, plot_density=True, \
                      truths=truths,\
                      weights=np.ones(nsamples)/nsamples,\
                      color=color[icolor],  \
                      hist_kwargs={'linewidth':3.},\
                      contour_kwargs={'linewidths':3.}, \
                      levels=(0.68,0.4), \
                      truth_color='black')

        

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

    plt.savefig('../plots/degenercies.pdf')
    plt.show()

def getMaxLikeFromSamples( samples,nIterations =100 ):
    itPs = np.int(samples.shape[0]/nIterations)
    
    maxLikes = np.zeros( (nIterations, samples.shape[1]))

 
    for iPar in range(samples.shape[1]):
        for jIt in range(nIterations):

            iSample = samples[jIt*itPs:(jIt+1)*itPs, iPar]
            y, x = \
              np.histogram( iSample, bins=50, density=True)
        
            xc = (x[1:]+x[:-1])/2.
            maxLikes[jIt, iPar] = xc[np.argmax(y)]
            print(maxLikes[jIt, iPar])

    return maxLikes



def checkConvergence( samples,nIterations = 100 ):
    nChains = 20
    itPs = np.int(samples.shape[0]/nIterations)
    

    nSamplesPerChain = np.int(itPs / nChains)
    convergence = np.zeros((samples.shape[1], nIterations))

    for iPar in range(samples.shape[1]):
        for jIt in range(nIterations):

            iSample = samples[jIt*itPs:(jIt+1)*itPs, iPar]

            #split out sample in to chains
            chains = np.zeros((nChains, nSamplesPerChain))

            for iChain in range(nChains):
                indexes = np.arange(nSamplesPerChain)*nChains+iChain

                chains[iChain, : ] = iSample[indexes]

            
            M = 20. #N walkers
            N = itPs/M #numnber of samples per


            B = \
              N / (M-1)*np.sum( (np.mean(chains,axis=1) - \
                                     np.mean(iSample))**2)
            W = 1./M*np.sum(np.std(chains,axis=1)**2)
            Vhat = (N-1)/N*W+(M+1)/(M*N)*B

                
            convergence[iPar, jIt] =  np.sqrt(Vhat/W)

    maxConvergence = np.max(convergence, axis=0)       
    #return cutSamples
if __name__ == '__main__':
    plotCornerPlot()
