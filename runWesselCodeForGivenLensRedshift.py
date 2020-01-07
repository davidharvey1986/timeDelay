#!/usr/local/bin/python3

import sys
import PlaneLenser.SingleLensStatsIntegrator as SingleLensStatsIntegrator
import matplotlib.pyplot as plt
import re
from scipy.stats import ks_2samp as ks_2samp
import numpy as np
import random
import os
import json
import glob

def main(zLens=0.20):

    getSimulationTimeDelaysForGivenLensRedshift( zLens=np.float(zLens) )
    
def getSimulationTimeDelaysForGivenLensRedshift( zLens=0.20):
    '''
    THis is mostly scraped from Wessel's code
    however i want to do it myself to make it clearer
    
    For now it is only a single plane at a redshift of zsource =8
    as defined in the parameter files in 
    /Users/DavidHarvey/Documents/Work/TimeDelay/WesselsCodeOriginal/batch/CDM/
    This and other parameters are loaded in to defs and can be changed

    In fact i dont need this, i can run all the timeDelays and then just change the time
    delay distance when i analyse them....
    '''
    
    print("Running for Lens Redshift=%0.2f" % zLens)


    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/'+\
      'WesselsCodeOriginal/batch/CDM/z_%0.2f' % zLens
    


    parameterFiles = glob.glob(dataDir+'/*.py')
    

    for iParameterFile in parameterFiles:
        print("# Loading definitions from", iParameterFile)
        getParameterDefinitions( iParameterFile )
        defs["H0"] = 100

        densityFileName = iParameterFile.split('/')[-1]
        
        dumpJSONFName = "../output/CDM/z_%0.2f/%s.raw.json" % (zLens, densityFileName)
        #the main four halos are  ['B002','B005','B008','B009']
 
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
        
def getParameterDefinitions( parameterFile ):
    with open(parameterFile, "r") as f:
        source = "global defs\n" + f.read();
        codeObj = compile(source, parameterFile, "exec")
    exec(codeObj)


if __name__ == "__main__":
    if len(sys.argv) == 1 :
        main(zLens=0.2)
    else:
        main(zLens=sys.argv[1])
