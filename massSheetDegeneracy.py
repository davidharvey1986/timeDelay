#!/usr/local/bin/python3

'''
mass sheet degeneracy
'''

from runWesselCodeForGivenLensRedshift import *
from astropy.io import fits

def getParameterDefinitions( parameterFile ):
    with open(parameterFile, "r") as f:
        source = "global defs\n" + f.read();
        codeObj = compile(source, parameterFile, "exec")
    exec(codeObj)


def testForGivenMassSheet( iMassSheet ):
    '''
    Generate an image with some added mass sheet degeneracy

    To do this i will need an example lens

    '''

    

    inputParameterFile = \
      '/Users/DavidHarvey/Documents/Work/TimeDelay/'+\
      'WesselsCodeOriginal/batch/CDM/z_0.37/B002_cluster_0_3_total_sph.fits.py'
      
    print("# Loading definitions from", inputParameterFile)
    getParameterDefinitions( inputParameterFile )

    
    imageFile = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections/B002_EAGLE_CDM/z_0.37/HIRES_MAPS/cluster_0_3_total_sph.fits'

    
    data = fits.open(imageFile)[0].data

    data += iMassSheet*1e13

    newFitsFile = '../data/B002_cluster_0_3_'+str(iMassSheet)+'.fits'
    fits.writeto( newFitsFile, data, overwrite=True)
    
    defs["fits"]["file"] = newFitsFile

    dumpJSONFName = '../output/CDM/massSheetTest/z_0.37/massSheet_'+str(iMassSheet)+'.json'

    oneStat = SingleLensStatsIntegrator.SingleLensStatsIntegrator(defs)

    #Do the integration over all source planes
    allRaw = oneStat.IntegrateStats()
                
    #Put the results in to a json file
    SingleLensStatsIntegrator.SingleLensStatsIntegrator.DumpAllRawStats(allRaw, dumpJSONFName)
    print(dumpJSONFName)

if __name__ == "__main__":
    if len(sys.argv) == 1 :
        raise ValueError("Input a mass sheet fraction")
    else:
        testForGivenMassSheet(np.float(sys.argv[1]))
