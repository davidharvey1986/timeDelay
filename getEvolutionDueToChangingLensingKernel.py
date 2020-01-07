#!/usr/local/bin/python3
'''
This script is designed to take the an model and determine what
the effect of the lens kernel is, and not the growth of a lens.
I.e. does the ratio change because the lenses are chaning, or because the
redshift of the lens is changing?
'''
from convolveDistributionWithLineOfSight import *

def main( iRedshift=0.25):
    '''
    NOT POITN DOING 0.20 SINCE THIS IS ALREADY DONE!

    First thing i need to do is recalcualte all the redshift 0
    json files with the same lens, but forced to be different redshifts (iRedshift)

    This adopts the sa,e code of runWesselCodeForGivenLensRedshift.py

    '''



    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/WesselsCodeOriginal/batch/CDM/z_0.20'
    redshiftDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM'
    outputDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/forcedLensRedshift'
   

    parameterFiles = glob.glob(dataDir+'/*.py')
    

    for iParameterFile in parameterFiles:
        print("# Loading definitions from", iParameterFile)
        getParameterDefinitions( iParameterFile )
        defs["H0"] = 100
        defs["zLens"] = iRedshift
        densityFileName = iParameterFile.split('/')[-1]
        
        dumpJSONFName = outputDir+"/%s.%0.2f.raw.json" % ( densityFileName, iRedshift)
            
        print("Looking for %s" % dumpJSONFName)
        if os.path.isfile(dumpJSONFName):
            continue
        #First initialise a single lens integrator
        oneStat = SingleLensStatsIntegrator.SingleLensStatsIntegrator(defs)
        lensName = oneStat.lens.toString() + " - "
        #CHeck to see if no lensing exists
        if oneStat.NoStrongLensing():
            print("No lensing in", lensName, ". Skipping.")
            continue

        #Do the integration over all source planes
        allRaw = oneStat.IntegrateStats()
                

        #Put the results in to a json file
        SingleLensStatsIntegrator.SingleLensStatsIntegrator.DumpAllRawStats(allRaw, dumpJSONFName)




def getTimeDelayDistance( zLens, zSource, HubbleConstant, omegaLambda=1.0):
        '''
        Get the time delay distance for this particle lens
        '''

        #Wessels distance class
        
        omegaMatter = 1. - omegaLambda
        OmegaK = 1. - omegaMatter - omegaLambda
        
        distanceClass = Distances.Distances(  zLens, zSource, omegaMatter, OmegaK, HubbleConstant)

        
        cInMpcPerSecond = 9.7156e-15
        
        return  np.log10((1.+zLens)*distanceClass.Dl*distanceClass.Ds/distanceClass.Dls/cInMpcPerSecond)



def getParameterDefinitions( parameterFile ):
    with open(parameterFile, "r") as f:
        source = "global defs\n" + f.read();
        codeObj = compile(source, parameterFile, "exec")
    exec(codeObj)


if __name__ == "__main__":
    if len(sys.argv) == 1 :
        getAnalticalEstimation()
    else:
        main(iRedshift=np.float(sys.argv[1]))
