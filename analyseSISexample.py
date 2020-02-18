


import cosmolopy.distance as dist
import os
from scipy.optimize import curve_fit
import emcee
from scipy.stats import norm
#Naughty, but i cant be botherd to put this in
from determineNumericalArtifacts import *
from scipy.ndimage import gaussian_filter as gauss
from matplotlib import gridspec 
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

def main():
    velocityDispersions =  np.linspace(200,300,3)
    color=['blue','green','purple','pink']
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 1)
    ax1 = plt.subplot(gs[:3,0])
    ax2 = plt.subplot(gs[3:, 0])
    plt.subplots_adjust(hspace=0)
    for iColor, iVelocityDispersion in enumerate(velocityDispersions):
        jsonFileName = "../output/SISexample/SIS_example_z0.2_%i_5.0_4.0.json" % iVelocityDispersion
        if not os.path.isfile(jsonFileName):
            print(jsonFileName)
            continue

        newFileName = jsonFileName+'.clean.pkl'
        if not os.path.isfile(newFileName):
            cleanMultipleImages(jsonFileName)

        data = pkl.load(open(newFileName,'rb'))[0]

        xc, y, yError = getHistogram( data, biasWeight=False, \
                                          weight=True, \
                                    bins=np.linspace(-1,3,150) )
        xc += np.log(1.74)
        y = np.cumsum(y)/np.sum(y)
        yError =  np.sqrt(np.cumsum(yError**2)/(np.arange(len(yError))+1))
        #y = 1. - np.cumsum(y*(xc[1] - xc[0]))
        #yError = np.sqrt(np.cumsum(yError**2))
        ax1.errorbar( xc , 1.-y, yerr=yError,fmt='.', \
                color=color[iColor], label=str(iVelocityDispersion)+' km/s')
        

        xAnalytical, yAnalytical = \
          getAnalyticExpression( xc, iVelocityDispersion, zSource=5., \
                                zLens=0.2 )
        #yAnalytical = 1. - np.cumsum(yAnalytical)*(xAnalytical[1]-xAnalytical[0])

        yAnalytical = np.cumsum(yAnalytical)/np.sum(yAnalytical)
        
        ax1.plot(xAnalytical, 1.-yAnalytical,':', \
                     color=color[iColor])
        ax2.errorbar( xc, y-yAnalytical,  \
                          fmt='-', color=color[iColor])

    ax1.plot(0,0,':',label='Analytical')
    ax1.legend()
    ax1.set_xticklabels([])
    box = dict(color='white')
    ax1.set_xlabel(r'$log(\Delta t)$', bbox=box)
    ax1.set_ylabel(r'P(>$log(\Delta t)$)', bbox=box)
    ax2.set_ylabel(r'$\Delta P(>log(\Delta t)$)', bbox=box)
    
    fig.align_ylabels([ax1,ax2])
    
    ax2.set_ylim(-0.05,0.05)
    #ax1.set_yscale('log')
    ax1.set_xlim(0.5,2.5)
    ax1.set_ylim(0.0,1.)

    ax2.set_xlim(0.5,2.5)
    ax2.plot([0., 2.2], [0.,0.], '--')
    ax2.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)

    
    plt.savefig('../plots/analyseSIS.pdf')
    plt.show()

def testSourcePlaneConvergence():

    resolutions = [0.25, 1.0, 2., 4.0, 6.0]
    
    color=['blue','green','purple','pink','cyan']
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 1)
    ax1 = plt.subplot(gs[:3,0])
    ax2 = plt.subplot(gs[3:, 0])
    plt.subplots_adjust(hspace=0)


    x= np.linspace(-1,3,150)
    xc = (x[1:] + x[:-1])/2.
    xAnalytical, yAnalytical = \
          getAnalyticExpression( xc, 250., zSource=5., \
                                zLens=0.2 )

    yAnalytical = np.cumsum(yAnalytical)/np.sum(yAnalytical)
        
    ax1.plot(xAnalytical, 1.-yAnalytical,':', \
                     color='black', label='Analytical')
                     
    for iColor, iResolution in enumerate(resolutions):
        jsonFileName = "../output/SISexample/SIS_example_z0.2_250_5.0_"+str(iResolution)+".json"
        if not os.path.isfile(jsonFileName):
            print(jsonFileName)
            continue

        newFileName = jsonFileName+'.clean.pkl'
        if not os.path.isfile(newFileName):
            cleanMultipleImages(jsonFileName)

        data = pkl.load(open(newFileName,'rb'))
        if type(data) == list:
            data = data[0]


        xc, y, yError = getHistogram( data, biasWeight=False, \
                                          weight=True, \
                                    bins=np.linspace(-1,3,150) )
        xc += np.log(1.74)

        xAnalytical, yAnalytical = \
          getAnalyticExpression( xc, 250., zSource=5., \
                                zLens=0.2 )

        yAnalytical = np.cumsum(yAnalytical)/np.sum(yAnalytical)
        
        y = np.cumsum(y)/np.sum(y)
        yError =  np.sqrt(np.cumsum(yError**2)/(np.arange(len(yError))+1))
        
        #yError = np.sqrt(np.cumsum(yError**2))
        ax1.errorbar( xc , 1.-y, yerr=yError,fmt='.', \
                color=color[iColor], label='%0.3f kpc' % (1./iResolution/10.))
        


        ax2.errorbar( xc, y-yAnalytical,  \
                          fmt='-', color=color[iColor])

    #ax1.plot(0,0,':',label='Analytical')
    ax1.legend(title='Source Plane Pixel Size')
    ax1.set_xticklabels([])
    box = dict(color='white')
    ax1.set_xlabel(r'$log(\Delta t)$', bbox=box)
    ax1.set_ylabel(r'P(>$log(\Delta t)$)', bbox=box)
    ax2.set_ylabel(r'$\Delta P(>log(\Delta t)$)', bbox=box)
    
    fig.align_ylabels([ax1,ax2])
    
    ax2.set_ylim(-0.05,0.05)
    #ax1.set_yscale('log')
    ax1.set_xlim(0.5,2.5)
    ax1.set_ylim(0.0,1.)

    ax2.set_xlim(0.5,2.5)
    ax2.plot([0., 2.2], [0.,0.], '--')
    ax2.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)

    
    plt.savefig('../plots/SISresolution.pdf')
    plt.show()

    
    
def getAnalyticExpression( logTimeDelay, velocityDispersion, zSource=1., zLens=0.2,kernelSize = 3. ):
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 1.}
    cosmo = dist.set_omega_k_0(cosmo)
    Dl = dist.angular_diameter_distance(zLens, **cosmo)
    Dls = dist.angular_diameter_distance(zSource, z0=zLens, **cosmo)
    Ds = dist.angular_diameter_distance(zSource, **cosmo)
    
    cInKmPerSecond = 3e5
    cInMpcPerSecond = 9.7156e-15
    seconds2days = 1./60./60./24
    timeDelayDistance =  getTimeDelayDistance( zLens, zSource, 100.)
    lensPlaneDistanceMpc = np.arange(500)/1000.*1e-4
    angle = lensPlaneDistanceMpc / Dl
    
    analytic = 8.*np.pi*(velocityDispersion/cInKmPerSecond)**2*timeDelayDistance*Dls/Ds*angle*seconds2days
    
    maxTimeDelay = np.log10(32.*np.pi**2*(velocityDispersion/cInKmPerSecond)**4*Dl*Dls/Ds*(1.+zLens)/cInMpcPerSecond*seconds2days)

    #logTimeDelay = np.linspace(-3,maxTimeDelay,100)
    probability =  (10**logTimeDelay)**2 
    probability =  probability / probability[ np.argmin(np.abs(logTimeDelay-maxTimeDelay))]
    probability[ logTimeDelay > maxTimeDelay] = 0
    dX = logTimeDelay[1]-logTimeDelay[0]
    #probability = gauss(probability, 1)
    #characterstic scale in kpc
    epsilon0 = 4.*np.pi*(velocityDispersion/cInKmPerSecond)**2\
      *Dl*Dls/Ds*1e3
    subsample = 4. 
    dy = 0.1/epsilon0*Dl/Ds/subsample
    

    timeDelayOnePixel = dy / 10**maxTimeDelay
     #timeDelayOnePixel/dX


    #pdb.set_trace()
    box_kernel = Box1DKernel(kernelSize)
    probability = convolve(probability, box_kernel)
    probability /= np.sum(probability*dX)
    print("convolution kernel size is ", dX*kernelSize)
    return logTimeDelay, probability




    
    

    
def getTimeDelayDistance(zLens, zSource, HubbleConstant, omegaLambda=1.0):
        '''
        Get the time delay distance for this particle lens
        '''

        #Wessels distance class
        
        omegaMatter = 1. - omegaLambda
        OmegaK = 1. - omegaMatter - omegaLambda
        

        cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : HubbleConstant/100.}
        cosmo = dist.set_omega_k_0(cosmo)    
    
        Dls =  dist.angular_diameter_distance(zSource, z0=zLens, **cosmo)
        Dl =  dist.angular_diameter_distance(zLens,**cosmo)
        Ds =  dist.angular_diameter_distance(zSource, **cosmo)
        
        cInMpcPerSecond = 9.7156e-15
        
        return  (1.+zLens)*Dl*Ds/Dls/cInMpcPerSecond
    

    
if __name__ == '__main__':
    testSourcePlaneConvergence()
    main()

