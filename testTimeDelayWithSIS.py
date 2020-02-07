#!/usr/local/bin/python3

from runWesselCodeForGivenLensRedshift import *
from astropy.io import fits

def getParameterDefinitions( parameterFile ):
    with open(parameterFile, "r") as f:
        source = "global defs\n" + f.read();
        codeObj = compile(source, parameterFile, "exec")
    exec(codeObj)

    
def testTimeDelayWithMassSheetDegeneracy( massSheet=0., velocityDispersion=400):
    '''
    Make it a relatively massive galaxy to get good statistics
    '''
    
    
    exampleFitsFile = '../data/SIS_v_'+str(massSheet)+'.fits'
    createSurfaceDensityMap( velocityDispersion, exampleFitsFile, massSheet=massSheet)
    getExampleParameterDefinitions()
    defs["fits"]["file"] = exampleFitsFile
    defs["H0"] = 100
    defs["zSource"] = 5.

    dumpJSONFName = "../output/SISexample/SIS_example_z"+str(defs["zLens"])+"_"+str(massSheet)+".json"

    if not os.path.isfile(dumpJSONFName):
        allRaw = runTest( defs )
                
        #Put the results in to a json file
        SingleLensStatsIntegrator.SingleLensStatsIntegrator.DumpAllRawStats(allRaw, dumpJSONFName)

        
def testTimeDelayWithNFW(m200=1e14):
    getExampleParameterDefinitions()

    exampleFitsFile =  r'../data/NFW_%0.2f.fits' % np.log10(m200)
    dumpJSONFName = "../output/NFWexample/NFW_example_z%s_%0.2f_4.json" % \
      (defs["zLens"], np.log10(m200))
    print("Finding %s and dumping in %s" % (exampleFitsFile,dumpJSONFName))
    
    defs["fits"]["file"] = exampleFitsFile
    defs["H0"] = 100
    defs["zSource"] = 3.0
    defs['sourcePlaneSampling'] = 5.

    allRaw = runTest( defs )
                
    #Put the results in to a json file
    SingleLensStatsIntegrator.SingleLensStatsIntegrator.DumpAllRawStats(allRaw, dumpJSONFName)

    
def testTimeDelayWithSIS(velocityDisperson=400):
    exampleFitsFile = '../data/SIS_v_'+str(velocityDisperson)+'.fits'
    createSurfaceDensityMap( velocityDisperson, exampleFitsFile)
    getExampleParameterDefinitions()
    defs["fits"]["file"] = exampleFitsFile
    defs["H0"] = 70.
    defs["zSource"] = 5.0
    defs['sourcePlaneSampling'] = 6.

    dumpJSONFName = "../output/SISexample/SIS_example_z"+str(defs["zLens"])+"_"+str(velocityDisperson)+"_"+str(defs["zSource"])+"_"+str(defs["sourcePlaneSampling"])+".json"

    allRaw = runTest( defs )
                
    #Put the results in to a json file
    SingleLensStatsIntegrator.SingleLensStatsIntegrator.DumpAllRawStats(allRaw, dumpJSONFName)

def runTest( defs ):
    '''
    This is the same for each test and is just eh main part of Wessel's code
    '''
    #First initialise a single lens integrator
    oneStat = SingleLensStatsIntegrator.SingleLensStatsIntegrator(defs)
    lensName = oneStat.lens.toString() + " - "
    #CHeck to see if no lensing exists
    if oneStat.NoStrongLensing():
        print("No lensing in", lensName, ". Skipping.")
        return
        
    #Do the integration over all source planes
    plane = oneStat.SinglePlaneForKernel(-1 )
    allRaw = [{**{"weight" : 1, "z" : defs["zSource"]}, **plane.ForRawData()}]
    #allRaw = oneStat.IntegrateStats()
    return allRaw

def createSurfaceDensityMap( velocityDisperson, outputFile, massSheet=0. ):
    '''
    Create a surfaceDensityMap
    for an SIS give some velocity dispersion

    surface density of SIS
     = sigma**2 / (2Gr) 

     massSheet a uniform sheet of mass to be added to see what
     happens. Is in units of 1e13
    '''
    dPix = 1e-4
    xVector = (np.arange(1000) - 500.)*dPix #so that dpix = 1e-4 Mpc
    yVector = (np.arange(1000) - 500.)*dPix
    xGrid, yGrid = np.meshgrid( xVector, yVector )
    rGrid = np.sqrt(xGrid**2 + yGrid**2)

    G = 4.3e-9 #Mpc/M_sun (km/s)2

    density = velocityDisperson**2 / (2.*G*rGrid)
    density[ np.isfinite(density) == False ] = np.max(density[ np.isfinite(density)])

    density += massSheet*1e13

    fits.writeto(outputFile,density, clobber=True )

def getExampleParameterDefinitions():
    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/'+\
      'WesselsCodeOriginal/batch/CDM/z_0.20'

    exampleParameterFile = glob.glob(dataDir+'/*.py')[0]
    
    getParameterDefinitions( exampleParameterFile )

if __name__ == '__main__':
    #testTimeDelayWithSIS(velocityDisperson=100)
    #testTimeDelayWithSIS(velocityDisperson=200)
    testTimeDelayWithSIS(velocityDisperson=250)
    #testTimeDelayWithSIS(velocityDisperson=300)
    #testTimeDelayWithSIS(velocityDisperson=350)
    #testTimeDelayWithSIS(velocityDisperson=400)
    #testTimeDelayWithSIS(velocityDisperson=450)
    #testTimeDelayWithSIS(velocityDisperson=500)
    #for i in range(1,10):
    #    testTimeDelayWithNFW(m200=i*1e14)
