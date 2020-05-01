#!/usr/local/bin/python3
from timeDelayDistributionClass import *
from plotAsFunctionOfDensityProfile import *



def saveAllLensesForMultipleCosmologies(rerun=False):
    
    
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B**cluster_0_*_total_*.json')


    hubbleParameters = np.linspace(60,80,11)
    omegaMatter = np.linspace(0.25, 0.35, 11)
    omegaK = np.linspace(-0.02, 0.02, 11)
    omegaL = np.linspace(0.65,0.75,11)
    cosmologyList = {'H0':hubbleParameters, 'OmegaM':omegaMatter, 'OmegaK':omegaK, 'OmegaL':omegaL}

    for iCosmoPar in cosmologyList.keys():
        cosmology = {'H0':70., 'OmegaM':0.3, 'OmegaK':0, 'OmegaL':0.7}
        
        for iParInCosmoList in cosmologyList[iCosmoPar]:
            
            cosmology[iCosmoPar] = iParInCosmoList

            pklFileName = \
              "../output/CDM/combinedPDF_h%0.2f_oM%0.4f_oK%0.4f_%0.4f.pkl" % \
                        (cosmology['H0'],cosmology['OmegaM'],cosmology['OmegaK'], \
                               cosmology['OmegaL'])
            print(cosmology)
                               
            if os.path.isfile(pklFileName):
                continue
            finalMergedPDFdict = combineJsonFiles(allFiles, cosmology=cosmology)
            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)
            
       
        
def getMagBiasedPDFs():
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B**cluster_0_*_total_*.json')


    hubbleParameters = [100.]
    for iHubbleParameter in hubbleParameters:
        pklFileName = '../output/CDM/combinedPDF_'+str(iHubbleParameter)+'Biased.pkl'

        finalMergedPDFdict = combineJsonFiles(allFiles, newHubbleParameter=iHubbleParameter)

        
        pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

def saveMassSheetsAsPickles():
    '''
    I dont need to combine here for now, just save as indieivual

    '''
    dataDir = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM/massSheetTest/z_0.37/'
    allJsonFiles = glob.glob(dataDir+'/*.json')
    for iJson in allJsonFiles:

        pklFileName = iJson+'.pkl'
        finalMergedPDFdict = combineJsonFiles([iJson])
        
        pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)


def combineMassBins( ):
    '''
    '''
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/CDM'
    catDir = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections'
    haloNames = ['B002','B005','B008','B009']

    allJsonFiles = np.array([])
    for i in haloNames:
        allJsonFiles  = np.append( allJsonFiles, glob.glob(dirD+'/z_*/'+i+'_cluster_0_*_total*.json'))
    
    nJsonFiles = len(allJsonFiles)
    allMasses = {'mass':np.zeros(nJsonFiles), 'json':allJsonFiles }
    rGrid = getRadGrid()
    for i, iJson in enumerate(allJsonFiles):
        nHalos = substructure(iJson)
        if nHalos > 1:
            continue
        allMasses['json'] = np.append(allMasses['json'], iJson)
        redshift = iJson.split('/')[-2]
        halo = iJson.split('/')[-1].split('_')[0]
        clusterInt = iJson.split('/')[-1].split('_')[2]
        project = iJson.split('/')[-1].split('_')[3]
        
        mass = getHaloMass( catDir+'/'+halo+'_EAGLE_CDM/'+redshift+'/HIRES_MAPS/'+\
                                'cluster_'+clusterInt+'_'+project+'_total_sph.fits',\
                                radialGrid=rGrid)

        allMasses['mass'][i] = mass

    #Five equal mass bins
    allMasses['json'] = allMasses['json'][np.argsort( allMasses['mass'])]
    allMasses['mass'] = np.sort( allMasses['mass'])
    
    nMassBins = 4
    jsonsPerMassBin = np.floor(nJsonFiles/nMassBins)
    
    for iMassBin in np.arange(nMassBins):

        samplesLo = np.int(jsonsPerMassBin*iMassBin)
        samplesHi = np.int(np.min([jsonsPerMassBin*(iMassBin+1), nJsonFiles]))
        halosInMassBin = allMasses['json'][samplesLo:samplesHi]
        meanMass = np.mean( allMasses['mass'][samplesLo:samplesHi] )

        finalMergedPDFdict = combineJsonFiles(halosInMassBin, newHubbleParameter=70.)
        pklFileName = dirD+'/massBin_%0.5f.pkl' % meanMass
        pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)



def saveIndividualHalos(rerun=False):
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
    haloNames = ['B002','B005','B008','B009']

    for iHalo in haloNames:
        individualHaloFiles = glob.glob(dirD+'/CDM/*/'+iHalo+'*cluster_0_*_total_*.json')

        
        pklFileName = '../output/CDM/combinedPDF_'+iHalo+'.pkl'

        
        if os.path.isfile(pklFileName)  & (not rerun) :
            continue
        else:
            finalMergedPDFdict = combineJsonFiles(individualHaloFiles, newHubbleParameter=70.)

            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

        
def saveIndividualHalosAndRedshifts(rerun=False):
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
    allRedshifts = [ i.split('/')[-1] for i in glob.glob(dirD+'/CDM/z*')]
    haloNames = ['B002','B005','B008','B009']
    for iHalo in haloNames:
        for i in allRedshifts:
            if i == 'z_0.74':
                continue
            individualHaloFiles = glob.glob(dirD+'/CDM/'+i+'/'+iHalo+'*cluster_0_*_total_*.json')

            pklFileName = '../output/CDM/combinedPDF_'+i+'_'+iHalo+'.pkl'
            if  os.path.isfile(pklFileName)  & (not rerun):
                continue
            else:
                finalMergedPDFdict = combineJsonFiles(individualHaloFiles, newHubbleParameter=70.)

            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)
        


def saveAllLensesForMultipleHubbleParameters(rerun=False):
    
    
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
             
    allFiles = glob.glob(dirD+'/CDM/z*/B**cluster_0_*_total_*.json')


    hubbleParameters = [100.] #[100., 50., 60., 70., 80., 90.]
    hubbleParameters = np.linspace(60,80,21)

    for iHubbleParameter in hubbleParameters:
        pklFileName = '../output/CDM/combinedPDF_'+str(iHubbleParameter)+'.pkl'
        if os.path.isfile(pklFileName)  & (not rerun):
            continue
        else:
            finalMergedPDFdict = combineJsonFiles(allFiles, cosmology=cosmology)

        pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

    
def forcedRedshiftHalos(rerun=False):
    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
    redshifts = ['0.20','0.25','0.37','0.50','0.74']
    
    for iRedshift in redshifts:
        if iRedshift == '0.20':
            individualHaloFiles = glob.glob(dirD+'/CDM/z_'+iRedshift+'/*cluster_0_*_total*json')
        else:
            individualHaloFiles = glob.glob(dirD+'/CDM/forcedLensRedshift/z_'+iRedshift+'/*'+iRedshift+'*json')
        pklFileName = '../output/CDM/forcedLensRedshift/combinedPDF_'+iRedshift+'.pkl'

        if os.path.isfile(pklFileName) & (not rerun) :
            continue
        else:
            finalMergedPDFdict = combineJsonFiles(individualHaloFiles, newHubbleParameter=70.)

        pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'), 2)

        
def saveAllRedshifts( rerun=False, integrate=False):
    '''
    If integrate then all lenses up to and including that redshift
    '''

    dirD = '/Users/DavidHarvey/Documents/Work/TimeDelay/output/'
    allRedshifts = [ i.split('/')[-1] for i in glob.glob(dirD+'/CDM/z*')]
    for jCount, iRedshift in enumerate(allRedshifts):

        if integrate:
            allFiles = np.array([])
            for kRedshift in allRedshifts[:jCount+1]:
                allFiles = np.append( allFiles, glob.glob(dirD+'/CDM/'+kRedshift+'/B*cluster_0_*_total*.json'))
                pklFileName = '../output/CDM/combinedIntegratedPDF_'+iRedshift+'.pkl'
                                                 
        else:
            allFiles = glob.glob(dirD+'/CDM/'+iRedshift+'/B**cluster_0_*_total*.json')
            pklFileName = '../output/CDM/'+iRedshift+'/combinedPDF.pkl'


            
            
        if len(allFiles) == 0:
            print("No files found for redshift %s" % iRedshift)
            continue

        
        if (os.path.isfile(pklFileName)) & (not rerun):
            continue
        else:

                
            finalMergedPDFdict = combineJsonFiles(allFiles, newHubbleParameter=70.)
            
            pkl.dump(finalMergedPDFdict,open(pklFileName,'wb'),2)
        
      
    
def combineJsonFiles( listOfJsonFiles, cosmology=None, zLens=None):
    '''
    Combine the given list of Json Files into a single 
    histogram.

    It will generatea an array of time delays for each input json file
    then find the weight mean and variance of these where the weight is te 
    source and lens plane confihuration.
    
    Choose a source plane if rquired.
    '''

    timeDelayBins = None #np.linspace(-2,3,100)
    allFinalMergedPDF = None
    matchToThisXaxis = np.linspace(-3,4,200)
    zLenses = np.array([0.20, 0.25, 0.37, 0.50, 0.74, 1.0])
    
    for iJsonFile in listOfJsonFiles:
         cluster = \
           timeDelayDistribution( iJsonFile, \
                cosmology=cosmology, \
                timeDelayBins=timeDelayBins,zLens=zLens)

         z0 = cluster.zLens

         dzLens = zLenses[ np.arange(len(zLenses))[ z0 == zLenses]+1] - z0


         
         for iSourcePlane in cluster.finalPDF['finalLoS']:

             dZsource = iSourcePlane.data['z'] - z0
             
             #This weight is due to the volume at a given redshift,
             iWeight = np.repeat(iSourcePlane.data['weight'],len(matchToThisXaxis))*dZsource*dzLens[0]
             z0 =  iSourcePlane.data['z']
             #they dont all have the same x, so match to that
             iMatchPdf = \
               iSourcePlane.interpolateGivenPDF( matchToThisXaxis, \
                                    iSourcePlane.timeDelayWithLineOfSightPDF )

             iMatchBiasedPdf = \
               iSourcePlane.interpolateGivenPDF( matchToThisXaxis, \
                                    iSourcePlane.biasedTimeDelayWithLineOfSightPDF )
                                    
             iMatchLensOnlyPdf = \
               iSourcePlane.interpolateGivenPDF( matchToThisXaxis, \
                                    iSourcePlane.timeDelayPDF )
             iMatchBiasOnlyPdf = \
               iSourcePlane.interpolateGivenPDF( matchToThisXaxis, \
                                    iSourcePlane.biasedTimeDelayPDF )

             if allFinalMergedPDF is None:
                allFinalMergedPDF = iMatchPdf
                allFinalMergedBiasedPDF = iMatchBiasedPdf
                allLensPlaneMergedPDF = iMatchLensOnlyPdf
                allBiasedLensPlanePDF = iMatchBiasOnlyPdf
                weightTable = iWeight
             else:
                allFinalMergedPDF = \
                  np.vstack( (allFinalMergedPDF, iMatchPdf) )
                allFinalMergedBiasedPDF = \
                  np.vstack( (allFinalMergedBiasedPDF, iMatchBiasedPdf) )
                allLensPlaneMergedPDF = \
                  np.vstack((allLensPlaneMergedPDF, iMatchLensOnlyPdf))
                allBiasedLensPlanePDF = \
                  np.vstack((allBiasedLensPlanePDF, iMatchBiasOnlyPdf))
               
                weightTable = np.vstack(( weightTable , iWeight ))

                 
                
    if allFinalMergedPDF is None:
        print("No redshifts found for this list of redshifts")
        return None

    nFluxRatios = np.nansum( allFinalMergedPDF/allFinalMergedPDF, axis=0)
    nBiasedFluxRatios = np.nansum( allFinalMergedBiasedPDF/allFinalMergedBiasedPDF, axis=0)

    
    finalMergedPDF = np.nansum( weightTable*allFinalMergedPDF, axis=0)/np.nansum(weightTable)
    finalMergedBiasedPDF = np.nansum( weightTable*allFinalMergedBiasedPDF, axis=0)/np.nansum(weightTable)
    lensPlaneMergedPDF = np.nansum( weightTable*allLensPlaneMergedPDF, axis=0)/np.nansum(weightTable)
    biasedLensPlanePDF = np.nansum( weightTable*allBiasedLensPlanePDF, axis=0)/np.nansum(weightTable)

    diffTableFinal = np.zeros( weightTable.shape)
    diffTableBiasedFinal = np.zeros( weightTable.shape)
    diffTableLensOnly = np.zeros( weightTable.shape)
    diffTableBiasLens = np.zeros( weightTable.shape)

    for i in np.arange(allLensPlaneMergedPDF.shape[0]):
        diffTableFinal[i, :] = allLensPlaneMergedPDF[i, :] - finalMergedPDF
        diffTableBiasedFinal[i, :] = allFinalMergedBiasedPDF[i, :] - finalMergedBiasedPDF
        diffTableLensOnly[i,:] = allLensPlaneMergedPDF[i,:] - lensPlaneMergedPDF
        diffTableBiasLens[i,:] = allBiasedLensPlanePDF[i,:] - biasedLensPlanePDF


    #Weighted error.
    finalMergedPDFerror = np.sqrt(np.nansum(weightTable*diffTableFinal**2, axis=0)/np.nansum(weightTable)/nFluxRatios**2)
    finalMergedBiasedPDFerror = np.sqrt(np.nansum(weightTable*diffTableBiasedFinal**2, axis=0)/np.nansum(weightTable)/nBiasedFluxRatios**2)
    
    lensPlanePDFerror = np.sqrt(np.nansum(weightTable*diffTableLensOnly**2, axis=0)/np.nansum(weightTable)/nFluxRatios**2)
    biasedLensPDFerror = np.sqrt(np.nansum(weightTable*diffTableBiasLens**2, axis=0)/np.nansum(weightTable)/nFluxRatios**2)

    
    #normalise it as well
    dX = iSourcePlane.timeDelayWithLineOfSightPDF['dX']
    
    finalMergedPDFerror /= np.nansum(finalMergedPDF*dX)
    finalMergedBiasedPDFerror /= np.nansum(finalMergedBiasedPDF*dX)
    lensPlanePDFerror /= np.nansum(lensPlaneMergedPDF*dX)
    biasedLensPDFerror /= np.nansum(lensPlaneMergedPDF*dX)

    finalMergedPDF /= np.nansum(finalMergedPDF*dX)
    finalMergedBiasedPDF /= np.nansum(finalMergedBiasedPDF*dX)
    lensPlaneMergedPDF /= np.nansum(lensPlaneMergedPDF*dX)
    biasedLensPlanePDF /= np.nansum(lensPlaneMergedPDF*dX)

    
    finalMergedPDFdict = { 'x':matchToThisXaxis, 'y':finalMergedPDF, 'yError':finalMergedPDFerror,\
                            'yBiased':finalMergedBiasedPDF, 'yBiasedError':finalMergedBiasedPDFerror,\
                            'yLensPlane':lensPlaneMergedPDF, 'yLensPlaneError':lensPlanePDFerror, \
                            'yBiasedLens':biasedLensPlanePDF, 'yBiasedLensError':biasedLensPDFerror,\
                               'nFluxRatios':nFluxRatios}


    return finalMergedPDFdict


    
def convolveTimeDelayDistributionWithLineOfSight( inputJsonFile ):
    '''
    I need to convolve the distribution of Time Delays with the line of sight
    effects.

    TimeDelay' = T*(1.+kappa_ext)

    '''

    cluster = timeDelayDistribution( inputJsonFile)

    for i in cluster.finalPDF['finalLoS']:

        plt.plot(i.timeDelayWithLineOfSightPDF['x'],i.timeDelayWithLineOfSightPDF['y'],'--')
    plt.plot(cluster.finalMergedPDF['x'],cluster.finalMergedPDF['y'])
    plt.show()
    
    #First i need to bin the data



if __name__ == "__main__":
    #getMagBiasedPDFs()
    #saveMassSheetsAsPickles()
    #combineMassBins( )
    #forcedRedshiftHalos( rerun=True)
    #saveAllRedshifts( rerun=True)
    #saveAllRedshifts( rerun=True, integrate=False)
    ##saveIndividualHalosAndRedshifts( rerun=True)
    #saveIndividualHalos( rerun=True)
    #saveAllLensesForMultipleHubbleParameters( rerun=True)
    saveAllLensesForMultipleCosmologies()
