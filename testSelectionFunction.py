#!/usr/local/bin/python3

'''
Can i predict what would happen if you got the wrong selection function?

'''
from convolveDistributionWithLineOfSight import *
from scipy.stats import lognorm


def main():
    
    
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B*_cluster_*_*.json')
    

    hubbleParameter = 70.
    zMed = np.linspace(1.5, 2.5, 11)
    for izMed in zMed:
        pklFileName = '../output/CDM/selectionFunction/SF_%0.2f.pkl' \
          % (izMed )
      
        finalMergedPDFdict = \
          selectionFunction(allFiles, newHubbleParameter=hubbleParameter,\
                            medianRedshift=izMed )

        pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

    

def selectionFunction( listOfJsonFiles, newHubbleParameter=None, \
                           medianRedshift=1.0 ):
    '''
    Combine the given list of Json Files into a single 
    histogram.

    It will generatea an array of time delays for each input json file
    then find the weight mean and variance of these where the weight is te 
    source and lens plane confihuration.
    
    '''


    allFinalMergedPDF = None
    matchToThisXaxis = None
    for iJsonFile in listOfJsonFiles:
         cluster = timeDelayDistribution( iJsonFile, newHubbleParameter=newHubbleParameter)
         if matchToThisXaxis is None:
            matchToThisXaxis = cluster.finalPDF['finalLoS'][0].timeDelayWithLineOfSightPDF['x']

         for iSourcePlane in cluster.finalPDF['finalLoS']:
             print(iSourcePlane.data['z'])
             selectionFunction = \
               getSourceRedshiftWeight( iSourcePlane.data['z'], medianRedshift)
               
             iWeight = \
               np.repeat(iSourcePlane.data['weight'],\
                             len(matchToThisXaxis))*\
                    selectionFunction

             #they dont all have the same x, so match to that
             iMatchPdf = iSourcePlane.interpolateGivenPDF( matchToThisXaxis, iSourcePlane.timeDelayWithLineOfSightPDF )
             iMatchLensOnlyPdf = iSourcePlane.interpolateGivenPDF( matchToThisXaxis, iSourcePlane.timeDelayPDF )

             
             if allFinalMergedPDF is None:
                allFinalMergedPDF = iMatchPdf
                allLensPlaneMergedPDF = iMatchLensOnlyPdf
                weightTable = iWeight
             else:
                allFinalMergedPDF = np.vstack( (allFinalMergedPDF, iMatchPdf) )
                allLensPlaneMergedPDF = np.vstack((allLensPlaneMergedPDF, iMatchLensOnlyPdf))
                weightTable = np.vstack(( weightTable , iWeight ))
                
                
    if allFinalMergedPDF is None:
        print("No redshifts found for this list of redshifts")
        return None

    finalMergedPDF = np.sum( weightTable*allFinalMergedPDF, axis=0)/np.sum(weightTable)
    nFluxRatios = np.nansum( allFinalMergedPDF/allFinalMergedPDF, axis=0)
    
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

    
    finalMergedPDFdict = { 'x':matchToThisXaxis, 'y':finalMergedPDF, 'yError':finalMergedPDFerror,\
                               'yLensPlane':lensPlaneMergedPDF, 'yLensPlaneError':lensPlanePDFerror,
                               'nFluxRatios':nFluxRatios}

    return finalMergedPDFdict


def getSourceRedshiftWeight( z, zMed=1.0 ):

    zStar = zMed/1.412
    weight = z**2*np.exp(-(z/zStar)**(1.5))
    return weight

if __name__ == "__main__":
    main()
