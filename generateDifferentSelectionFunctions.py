#!/usr/local/bin/python3

'''
Can i predict what would happen if you got the wrong selection function?

'''
from convolveDistributionWithLineOfSight import *
from scipy.stats import lognorm
import time
import lsstSelectionFunction as lsstSelect
from termcolor import colored

def selectionFunctionEnsembleHalos():
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    
    hubbleParameters = \
      np.array([50., 60., 70., 80., 90., 100.])
    #hubbleParameter = 70.
    hubbleParameters = [70.]
    halos = ['B002','B005','B008','B009']
    for halo in halos:
        allFiles = glob.glob(dirD+'/CDM/z*/%s_cluster_0_*.json' % halo )

        for hubbleParameter in hubbleParameters:
        
  
            
            pklFileName = \
              '../output/CDM/selectionFunction/SF_%s_%i_lsst.pkl' \
              % (halo,hubbleParameter )
            finalMergedPDFdict = \
              selectionFunction(allFiles, \
                                newHubbleParameter=hubbleParameter,\
                                useLsst = True)
                                
            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)
  
def selectionFunctionIndividualLenses(  ):
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
    
    nHubble = 11
    nOmegaM = 5
    nOmegaK = 5
    nOmegaL = 5
    hubbleParameters = np.linspace(65,75,nHubble)
    omegaMatter = np.linspace(0.25, 0.35, nOmegaM)
    omegaK = np.linspace(-0.02, 0.02, nOmegaK)
    omegaL = np.linspace(0.65,0.75,nOmegaL)
    totalTimes = nHubble*nOmegaM*nOmegaK*nOmegaL*len(allFiles)
    
    pklFileName = "../output/CDM/selectionFunction/sparselyPopulatedParamSpace.pkl"
    listOfSelectionFunctions = []
    i = 0
    for iHubbleParameter in hubbleParameters:
        for iOmegaM in omegaMatter:
            for iOmegaK in omegaK:
                for iOmegaL in omegaL:
                    timeStart = time.time()
                    for iFile in allFiles:
                        i += 1
                        
                        cosmology = {'H0':iHubbleParameter, \
                        'OmegaM':iOmegaM, 'OmegaL':iOmegaL, \
                                        'OmegaK':iOmegaK}
                        
                        finalMergedPDFdict = \
                          selectionFunction([iFile], \
                                cosmology=cosmology,\
                                useLsst = True)
                        listOfSelectionFunctions.append(finalMergedPDFdict)
                    timeEnd = time.time()
                    timediff = (timeEnd - timeStart)/len(allFiles)
                    print(colored("Time left finish time %0.2f hours" % \
                                    (timediff*(totalTimes-i)/60./60.),'red'))

    pkl.dump(listOfSelectionFunctions,open(pklFileName,'wb'), 2)
            
  

       
def selectionFunctionIndividualLensesForData( ):
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
    
    hubbleParameters = \
      np.array([50., 60., 70., 80., 90., 100.])
    #hubbleParameter = 70.

    for hubbleParameter in hubbleParameters:
        
        for iFile in allFiles:
  
            
            fileName = iFile.split('/')[-1]
            zLens =  iFile.split('/')[-2]
            pklFileName = \
              '../output/CDM/selectionFunction/SF_%s_%s_%i_data.pkl' \
              % (zLens,fileName,hubbleParameter )
            finalMergedPDFdict = \
              selectionFunction([iFile], \
                                newHubbleParameter=hubbleParameter,\
                                useLsst = 21)
                                
            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)
            
def selectFunctionForAllLenses():
    
    
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
    
   

    pklFileName = \
          '../output/CDM/selectionFunction/allHalosFiducialCosmology.pkl' \

    finalMergedPDFdict = \
         selectionFunction(allFiles, useLsst = True)
                                                                
    pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

    

def selectionFunction( listOfJsonFiles, cosmology=None, \
                           medianRedshift=1.0, useLsst=27 ):
    '''
    Combine the given list of Json Files into a single 
    histogram.

    It will generatea an array of time delays for each input json file
    then find the weight mean and variance of these where the weight is te 
    source and lens plane confihuration.
    
    Using the biased pdf

    Use lsst doesnt requre a median redshift as i use the integrated lum func
    (see lsstSelectionFunction.py)
    '''


    allFinalMergedPDF = None
    matchToThisXaxis = np.linspace(-3,4,100)
    zLenses = np.array([0.20, 0.25, 0.37, 0.50, 0.74, 1.0])
    if cosmology is None:
        cosmology={'H0':70., 'OmegaM':0.3, 'OmegaK':0., 'OmegaL':0.7}
    for iJsonFile in listOfJsonFiles:
         cluster = \
           timeDelayDistribution( iJsonFile, cosmology=cosmology, \
                                    outputPklFile='dontWrite')
        
         z0lens = cluster.zLens
         dzLens = zLenses[ np.arange(len(zLenses))[ z0lens == zLenses]+1] - z0lens
         z0source  = cluster.zLens
         for iSourcePlane in cluster.finalPDF['finalLoS']:

             if useLsst is not None:
                 selectionFunction = \
                   lsstSelect.getSelectionFunction(  iSourcePlane.data['z'], limitingObsMag=useLsst)
             else:
                selectionFunction = \
                  getSourceRedshiftWeight( iSourcePlane.data['z'], medianRedshift)

             
             dZ = iSourcePlane.data['z'] - z0source
             
             iWeight = \
               np.repeat(iSourcePlane.data['weight'],\
                             len(matchToThisXaxis))*\
                    selectionFunction*dZ*dzLens
             z0source = iSourcePlane.data['z']
             
             #they dont all have the same x, so match to that
             iMatchPdf = \
               iSourcePlane.interpolateGivenPDF( matchToThisXaxis, \
                            iSourcePlane.biasedTimeDelayWithLineOfSightPDF )
                            
             iMatchLensOnlyPdf = \
               iSourcePlane.interpolateGivenPDF( matchToThisXaxis, \
                            iSourcePlane.biasedTimeDelayPDF )

             
             if allFinalMergedPDF is None:
                allFinalMergedPDF = iMatchPdf
                allLensPlaneMergedPDF = iMatchLensOnlyPdf
                weightTable = iWeight
             else:
                allFinalMergedPDF = \
                  np.vstack( (allFinalMergedPDF, iMatchPdf) )
                allLensPlaneMergedPDF = \
                  np.vstack((allLensPlaneMergedPDF, iMatchLensOnlyPdf))
                weightTable = np.vstack(( weightTable , iWeight ))
                
                
    if allFinalMergedPDF is None:
        print("No redshifts found for this list of redshifts")
        return None

    nFluxRatios = np.nansum( allFinalMergedPDF/allFinalMergedPDF, axis=0)


    
    finalMergedPDF = np.sum( weightTable*allFinalMergedPDF, axis=0)/np.sum(weightTable)
    
    lensPlaneMergedPDF = np.sum( weightTable*allLensPlaneMergedPDF, axis=0)/np.sum(weightTable)

    diffTableFinal = np.zeros( weightTable.shape)
    diffTableLensOnly = np.zeros( weightTable.shape)
    for i in np.arange(allLensPlaneMergedPDF.shape[0]):
        diffTableFinal[i, :] = allLensPlaneMergedPDF[i, :] - finalMergedPDF
        diffTableLensOnly[i,:] = allLensPlaneMergedPDF[i,:] - lensPlaneMergedPDF


    #Weighted error.
    finalMergedPDFerror = np.sqrt(np.sum(weightTable*diffTableFinal**2, axis=0)/np.sum(weightTable)/nFluxRatios)
    lensPlanePDFerror = np.sqrt(np.sum(weightTable*diffTableLensOnly**2, axis=0)/np.sum(weightTable)/nFluxRatios)

    
    #normalise it as well
    dX = iSourcePlane.timeDelayWithLineOfSightPDF['dX']
    finalMergedPDFerror /= np.sum(finalMergedPDF*dX)
    lensPlanePDFerror /= np.sum(lensPlaneMergedPDF*dX)

    finalMergedPDF /= np.sum(finalMergedPDF*dX)
    lensPlaneMergedPDF /= np.sum(lensPlaneMergedPDF*dX)

    
    finalMergedPDFdict = \
      { 'x':matchToThisXaxis, 'y':finalMergedPDF, \
        'yError':finalMergedPDFerror,\
        'yLensPlane':lensPlaneMergedPDF, \
        'yLensPlaneError':lensPlanePDFerror,
      'nFluxRatios':nFluxRatios,
      'cosmology':cosmology, \
      'fileNames':listOfJsonFiles}

    return finalMergedPDFdict


def getSourceRedshiftWeight( z, zMed=1.0 ):

    zStar = zMed/1.412
    weight = z**2*np.exp(-(z/zStar)**(1.5))
    return weight

if __name__ == "__main__":
    #selectionFunctionEnsembleHalos()
    selectFunctionForAllLenses()
#    selectionFunctionIndividualLenses()
    #selectionFunctionIndividualLensesForData()
