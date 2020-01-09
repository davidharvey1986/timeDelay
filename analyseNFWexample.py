from analyseSISexample import *
from nfwProperties import *
def main():
    masses =  np.arange(1,10)*1e14
        

    color=['blue','green','purple','pink','red','orange','purple','cyan','magenta','black']

    gs = gridspec.GridSpec(5, 1)
    ax1 = plt.subplot(gs[:3,0])
    ax2 = plt.subplot(gs[4:, 0])
    for iColor, iMass in enumerate(masses):

        jsonFileName = "../output/NFWexample/NFW_example_z0.2_%0.2f_4.json" % np.log10(iMass)

        if not os.path.isfile(jsonFileName):
            print(jsonFileName)
            continue

        newFileName = jsonFileName+'.clean.pkl'
        if not os.path.isfile(newFileName):
            cleanMultipleImages(jsonFileName)

        data = pkl.load(open(newFileName,'rb'))[0]

        xc, y, yError = getHistogram( data, biasWeight=False, \
                                          weight=True, \
                                    bins=np.linspace(0.,4,250) )
        xc -= np.log(0.94)

        pdf = {'x': xc, 'y':y, 'yError':yError}
        powerLawFitClass = powerLawFit( pdf, yMin=1e-5, curveFit=True, inputYvalue='y' )

I j        
        print(powerLawFitClass.params['params'])
        #y = 1. - np.cumsum(y*(xc[1] - xc[0]))
        #yError = np.sqrt(np.cumsum(yError**2))
        ax1.errorbar( xc , y, yerr=yError,fmt='.', \
                color=color[iColor], label='Numerical')
        

        xAnalytical, yAnalytical = getAnalyticExpression( xc, iMass, zSource=5., zLens=0.2 )
        #yAnalytical = 1. - np.cumsum(yAnalytical)*(xAnalytical[1]-xAnalytical[0])

        
        ax1.plot(xAnalytical, yAnalytical,':', \
                     color=color[iColor], label='Analytical')
        ax2.errorbar( xc, y-yAnalytical,  \
                          fmt='-', color=color[iColor])
        


    ax1.set_xlabel(r'$log(\Delta t)$')
    ax1.set_ylabel(r'p($log(\Delta t)$)')
    #ax1.set_xlim(0.,2.2)
    #ax1.set_yscale('log')


    plt.show()
def getAnalyticExpression( logTimeDelay, mass, zSource=1., zLens=0.2, concentration=10 ):
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

    G = grav_const()
    k = Dls / Ds
    einsteinRadius = np.sqrt( 4.*G*mass/cInKmPerSecond**2*k*Dl)
    scaleRadius = scale_radius( mass, concentration, zLens)
    maxTimeDelay = 2.*scaleRadius**2*einsteinRadius*1/k/Dl*(1.+zLens)
    pdb.set_trace()
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
    kernelSize = 3 #timeDelayOnePixel/dX
if __name__ == '__main__':
    main()
