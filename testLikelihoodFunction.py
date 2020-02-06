
def testLikelihood(nBins=21, nSamples=1000):

    
    powerLawIndex = np.linspace(-2,-1.,nBins)
    redshift = np.linspace(0.22,1.02,nBins) 
    hubbleParam = np.linspace(0.5,1.,nBins)
    interpolateToTheseTimes=   np.linspace(-1, 3, 1000)

    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents(weight=1.)
    
    interpolatedProb = hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, \
                                              np.array([0.7,0.3, -1.9] ))
    

 
    logTimeDelays = \
      np.random.choice(interpolateToTheseTimes, \
                    p=interpolatedProb/np.sum(interpolatedProb), size=np.int(nSamples))
    
    bins = np.max([10, np.int(nSamples/100)])
    y, x = np.histogram(logTimeDelays, \
                    bins=np.linspace(-1,3,100), density=True)
    dX = (x[1] - x[0])
    xcentres = (x[1:] + x[:-1])/2.
    error = np.sqrt(y*nSamples)/nSamples

    cumsumY = np.cumsum( y )  / np.sum(y)
    cumsumYError = np.sqrt(np.cumsum(error**2))
    
    
    #cumsumYError = np.sqrt(np.cumsum(error**2))
    #cumsumYError[ error == 0] == 1.

    totalProb = np.zeros((nBins,nBins,nBins))
    maxProb              = fitHubble.lnprob( np.array([0.7, 0.3, -1.9]), \
                                            xcentres, \
                                          cumsumY, \
                                          cumsumYError,\
                                          hubbleInterpolaterClass)
    prb = hubbleInterpolaterClass.predictPDF( xcentres, np.array([0.7, 0.3, -1.9]))
    
    for i, iPL in enumerate(powerLawIndex):
        for j, iRedshift in enumerate(redshift):
            for k, iHubble in enumerate(hubbleParam):
                theta =  np.array([iHubble, iRedshift, iPL])
                pr = hubbleInterpolaterClass.predictPDF( xcentres, theta)

                probability = fitHubble.lnprob(theta, \
                                          xcentres, \
                                          cumsumY, \
                                          cumsumYError,\
                                          hubbleInterpolaterClass)
                if probability > maxProb:
                    print(iHubble,iRedshift,iPL)
                    plt.plot(xcentres, cumsumY-cumsumY,label='data')
                    plt.plot(xcentres, np.cumsum(pr)/np.sum(pr) - cumsumY,label='what it thinks')
                    plt.plot(xcentres, np.cumsum(prb)/np.sum(prb) - cumsumY, ':', label='what i think')
                    plt.legend()
                    plt.show()
                    
                pdb.set_trace()
                totalProb[i,j,k] = probability
        print(i+1)
   
    fig, ax = plt.subplots(3, 1)
    indexPL = np.argmin(np.abs(powerLawIndex - -1.9))
    indexZ =  np.argmin(np.abs(redshift - 0.3))
    indexH =  np.argmin(np.abs(hubbleParam - 0.7))
    print(hubbleParam[indexH], redshift[indexZ], powerLawIndex[indexPL])

    ax[0].plot( powerLawIndex, totalProb[:, indexZ, indexH])
    ax[1].plot( redshift, totalProb[indexPL, :, indexH])
    ax[2].plot( hubbleParam, totalProb[indexPL, indexZ, :])
    plt.show(block=False)
        

    pdb.set_trace()
