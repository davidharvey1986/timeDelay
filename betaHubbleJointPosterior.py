'''
Richard wants a joint posterior of the beta and the hubble

to do this i will need a function that returns a posterior for a given number of lenses


'''


from getStatisticalErrorOnHubble import *
from powerLawFit import *
import corner
def main():
    #getSamplePickles()
    plotSamplePickles()
def plotSamplePickles():
    
    totalNumberQuasars = 10**np.linspace(2., 5., 4)

    #figcorner, axarr = plt.subplots(2,2)

    #For aesthetics                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=0., \
                            vmax=len(totalNumberQuasars))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #####                                                               

    for iNumQuasarIndex, nQuasars in enumerate(totalNumberQuasars):
        outputPickle = '../output/CDM/statReach/statisticalReach_%i.pkl' % nQuasars
        samples = pkl.load(open(outputPickle,'rb'))
        errorLower, median, errorUpper = np.percentile(samples, [16, 50, 84], axis=0)

        samples[:,0] -= median[0]
        samples[:,0] += 70.
        print samples.shape
        plt.hist(samples[:,0], bins=20)


        
    
    plt.show()
def getSamplePickles():
    totalNumberQuasars = 10**np.linspace(2., 5., 4)

    for i in totalNumberQuasars:
        fitJointMassHubble(i)
        
def fitJointMassHubble( nQuasars ):
    
    if not os.path.isfile( 'hubbleLawParameters.pkl'):
        hubbleParameters, powerLawIndex, powerLawIndexError = \
          getPowerLawForDifferentHubbleParameters()

        hubbleToPowerLawCorrelation, hubbleToPowerLawCorrelationError = \
          curve_fit( straightLine, hubbleParameters, powerLawIndex, \
                         p0=[1.,1.], sigma=powerLawIndexError)

        pkl.dump( hubbleToPowerLawCorrelation, open('hubbleLawParameters.pkl','wb'))
    
        
    #Draw a fake observation
    totalNumberQuasars = 1e3
    observedPDF = drawFakeObservation(nQuasars, nYears=5., iHubbleParameter=70.)

    nwalkers = 10
    ndim = 2
    burn_len=100
    chain_len=1000
    nthreads = 4
    pos0 = np.random.rand(nwalkers,ndim)*2. 
    pos0[:,0] +=70.
    pos0[:,1] += 12

    args = (observedPDF['x'], np.log10(observedPDF['y']), \
               observedPDF['yError'] / (observedPDF['y']*np.log(10.) ))
               
    #plt.errorbar( observedPDF['x'], np.log10(observedPDF['y']), \
    #            yerr=observedPDF['yError'] / (observedPDF['y']*np.log(10.) ), fmt=',')
    #plt.show()
    dmsampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                          lnprob, \
                                          args=args, \
                                          threads=nthreads )
                                          
    pos, prob, state  = dmsampler.run_mcmc(pos0, burn_len)

    pos, prob, state  = dmsampler.run_mcmc(pos, chain_len)
    samples = dmsampler.flatchain
    outputPickle = '../output/CDM/statReach/statisticalReach_%i.pkl' % nQuasars
    pkl.dump(samples, open(outputPickle, 'wb'))
    

def lnprob( theta, xTrue, yTrue, error ):

    yTheory = theModelToSample( xTrue, *theta)
    
    likelihood =  np.sum(norm.logpdf( yTheory, yTrue, error))
    peakTimeParameters = pkl.load(open('peakTimeMassParams.pkl','rb'))
    peakTime = straightLine(theta[1], *peakTimeParameters)
    priorMass = norm.pdf( theta[1], 13.0, 1.0)


    if (peakTime > 4) or (peakTime) < 0:
        return -np.inf

    return np.sum(likelihood*priorMass)

    
def drawFakeObservation(totalNumberQuasars, nYears=1., iHubbleParameter=70.):
    '''
    Get the true one from the simulations

    '''

    pklFileName = '../output/CDM/combinedPDF_'+str(iHubbleParameter)+'.pkl'
    if os.path.isfile(pklFileName):
        finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
    else:
        raise ValueError("No pickle file found (%s) "%pklFileName)
    
    finalMergedPDFdict['yError'] /= np.max(finalMergedPDFdict['y'])
    finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])


          



    lessThanPeak =  (finalMergedPDFdict['x'] < \
      finalMergedPDFdict['x'][np.argmax(finalMergedPDFdict['y'])]) &\
      (finalMergedPDFdict['y'] > 0.02)
    
    selectPDFx = finalMergedPDFdict['x'][lessThanPeak] + 2.56
    selectPDFy = finalMergedPDFdict['y'][ lessThanPeak]
   
    #interpolate selectX
    bins = np.linspace(  np.log10(10.),  np.max(selectPDFx), 21)


    hubbleLawParameters = pkl.load(open('hubbleLawParameters.pkl','rb'))
    beta = straightLine( iHubbleParameter, *hubbleLawParameters)
    peakTimeParameters = pkl.load(open('peakTimeMassParams.pkl','rb'))
    peakTime = straightLine(13., *peakTimeParameters)
    newSelectx = np.linspace(np.min(selectPDFx), np.max(selectPDFx), 1e8)
    newSelecty = 10**(newSelectx*beta + peakTime) 
#    plt.plot(newSelectx, newSelecty/np.sum(newSelecty))
#    plt.yscale('log')
#    plt.show()
    selectedTimeDelays = np.random.choice(newSelectx, \
           p=newSelecty /np.sum(newSelecty), \
           size= np.int(totalNumberQuasars))

    pdf, x = np.histogram(selectedTimeDelays, bins=bins)
    pdf = pdf.astype(float)
    error = np.sqrt(pdf)
    
    
    xc = (x[1:] + x[:-1])/2.
    error /= np.max(pdf)
         
    pdf /= np.max(pdf)
    inputPDF = {'x':xc[pdf>0], 'y':pdf[pdf>0], 'yError':error[pdf>0]}

    #plt.errorbar(inputPDF['x'], inputPDF['y'], yerr=inputPDF['yError'],fmt='o')
    #plt.show()
    return inputPDF


    
def theModelToSample( timeDelays, hubble, effectiveSampleMass ):
    '''
    This model will take in some parameters
    and output the probability time delays
    '''
    hubbleLawParameters = pkl.load(open('hubbleLawParameters.pkl','rb'))
    beta = straightLine(hubble, *hubbleLawParameters)
    peakTimeParameters = pkl.load(open('peakTimeMassParams.pkl','rb'))
    peakTime = straightLine(effectiveSampleMass, *peakTimeParameters)
    probabilityLogTimeDelay = peakTime + timeDelays*beta
    probabilityLogTimeDelay -= probabilityLogTimeDelay[np.argmin(np.abs(peakTime-timeDelays))]

    return probabilityLogTimeDelay
    


if __name__ == '__main__':
    main()
