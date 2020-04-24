#!/usr/local/bin/python3

'''
Can i predict what would happen if you got the wrong selection function?

'''
from convolveDistributionWithLineOfSight import *
from scipy.stats import lognorm

import lsstSelectionFunction as lsstSelect

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
            
def selectionFunctionIndividualLenses( ):
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
    
    hubbleParameters = \
      np.linspace(60,80,21)
    #hubbleParameter = 70.

    for hubbleParameter in hubbleParameters:
        
        for iFile in allFiles:
  
            
            fileName = iFile.split('/')[-1]
            zLens =  iFile.split('/')[-2]
            pklFileName = \
              '../output/CDM/selectionFunction/SF_%s_%s_%i_lsst.pkl' \
              % (zLens,fileName,hubbleParameter )
            finalMergedPDFdict = \
              selectionFunction([iFile], \
                                newHubbleParameter=hubbleParameter,\
                                useLsst = True)
                                
            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)
       
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
            
def selectFunctionForAllLenses(useLsst=None):
    
    
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_0_*.json')
    

    hubbleParameters = \
      np.linspace(60,80,21)
   

    for hubbleParameter in hubbleParameters:
        if useLsst is not None:
        
            pklFileName = \
              '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
              % (hubbleParameter )
            finalMergedPDFdict = \
              selectionFunction(allFiles, \
                            newHubbleParameter=hubbleParameter,\
                                useLsst = useLsst)
                                                                
            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)
        else:
            zMed = np.linspace(1.5, 2.5, 11)
            for izMed in zMed:
                pklFileName = \
                '../output/CDM/selectionFunction/SF_%i_%0.2flsst.pkl' \
                % (hubbleParameter, zMed )
        
                finalMergedPDFdict = \
                  selectionFunction(allFiles, \
                                    newHubbleParameter=hubbleParameter,\
                                    medianRedshift=izMed,\
                                    useLsst = False)
                                
                pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

    

def selectionFunction( listOfJsonFiles, newHubbleParameter=None, \
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

    for iJsonFile in listOfJsonFiles:
         cluster = \
           timeDelayDistribution( iJsonFile, \
                            newHubbleParameter=newHubbleParameter)
        
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
      'nFluxRatios':nFluxRatios}

    return finalMergedPDFdict


def getSourceRedshiftWeight( z, zMed=1.0 ):

    zStar = zMed/1.412
    weight = z**2*np.exp(-(z/zStar)**(1.5))
    return weight

if __name__ == "__main__":
    #selectionFunctionEnsembleHalos()
#    selectFunctionForAllLenses()
    selectionFunctionIndividualLenses()
    #selectionFunctionIndividualLensesForData()
