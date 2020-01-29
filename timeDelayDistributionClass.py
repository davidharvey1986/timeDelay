#!/usr/local/bin/python3

from magnificationBias import *

from PlaneLenser import InterpolateLineOfSightTable as getLoS
import json
from PlaneLenser import ValueBinner
import PlaneLenser.SingleLensStatsIntegrator as SingleLensStatsIntegrator
import numpy as np
from matplotlib import pyplot as plt
from PlaneLenser import Distances
import pickle as pkl
import os
import glob
import sys
import ipdb as pdb
from getHaloMass import *
from determineNumericalArtifacts import *

class timeDelayDistribution:
    '''
    This class will take the time delay distribution for a given lens redshift
    and loop over each source redshift and convolve the distributions with the e
    expected kappa distribution
    '''

    def __init__( self, inputJsonFileName, \
                      timeDelayBins=None, \
                      outputPklFile=None, \
                     newHubbleParameter=100,\
                      zLens=None):
                      
        self.newHubbleParameter = newHubbleParameter
        self.timeDelayBins = timeDelayBins
        if outputPklFile is None:
            if newHubbleParameter is not None:
                outputPklFile = inputJsonFileName+'_'+str(newHubbleParameter)+'.pkl'
            else:
                outputPklFile = inputJsonFileName+'.pkl'

        self.inputJsonFileName  = inputJsonFileName
        self.cleanTimeDelays = inputJsonFileName+'.clean.pkl'
        
        if not os.path.isfile(self.cleanTimeDelays):
            cleanMultipleImages(inputJsonFileName)

        self.outputPklFile = outputPklFile
        if zLens is None:
            self.getLensRedshift()
        else:
            self.zLens = zLens
        try:
            self.loadFromPickle()
        except:
            self.convolveTimeDelayDistribution()


    def getLensRedshift( self ):
        '''
        Get the lens redshift directly from the 
        name of the file
        '''
        self.zLens = np.float(self.inputJsonFileName.split('/')[-2].split('_')[1])

    def getHaloMass( self ):
        '''
        Get the mass of the cluser in question
        '''
        dataDir = '/Users/DavidHarvey/Documents/Work/WDM/data/withProjections'
        halo = self.inputJsonFileName.split('/')[-1].split('_')[0]
        haloID = np.float(self.inputJsonFileName.split('/')[-1].split('_')[2])
        haloDir = dataDir+'/'+halo+'_EAGLE_CDM/z_'+str(self.zLens)
        catalog = np.loadtxt(haloDir+'/catalog.dat',\
                        dtype=[('id',float),('idfof',float),\
                               ('m200',float),('r200',float)])
                               
        self.mass = catalog['m200'][ haloID == catalog['id']][0]

        
    def convolveTimeDelayDistribution( self ):
        
        jsonData = json.load(open(self.inputJsonFileName, 'rb'))
        cleanTimeDelayData = pkl.load(open(self.cleanTimeDelays,'rb'))

        self.finalPDF = {'lensPlaneOnly':jsonData, 'finalLoS': []}
        
            
        for iSourcePlaneIndex in range(len(jsonData)):
            iSourcePlane = jsonData[iSourcePlaneIndex]
            try:
                iTimeDelayData = cleanTimeDelayData[iSourcePlaneIndex]
            except:
                pdb.set_trace()
            iSourcePlaneDist = \
                  singleSourcePlaneDistribution( self.zLens, \
                            iSourcePlane, iTimeDelayData, \
                            newHubbleParameter=self.newHubbleParameter,\
                            timeDelayBins=self.timeDelayBins)
            if iSourcePlaneDist.flag == -1:
                continue
            iSourcePlaneDist.__module__ == 'convolveDistributionWithLineOfSight'
            self.finalPDF['finalLoS'].append(iSourcePlaneDist)
        self.saveInPickle()
        
    def loadFromPickle( self ):
        if not os.path.isfile(self.outputPklFile):
            raise ValueError("Cannot find pickle file doing again")
        else:
            print("Found pickle file %s" % self.outputPklFile)
        tmpDict = pkl.load(open(self.outputPklFile, 'rb'))
        self.__dict__.update(tmpDict)
        
        
    def saveInPickle( self ):
        pkl.dump(self.__dict__,open(self.outputPklFile,'wb'))
                        
            
class singleSourcePlaneDistribution:
    '''
    This is a single source plane distribution
    '''

    def __init__( self, lensRedshift, jsonData, timeDelayData, \
            newHubbleParameter = None, timeDelayBins=None ):

        
        self.timeDelayBins = timeDelayBins
        self.data = jsonData
        self.timeDelayData = timeDelayData
        self.zLens = lensRedshift
        self.zSource = jsonData['z']


        self.getLineOfSightDistribution()
        self.getSourcePlaneWeighting()
        
        self.flag = self.binTimeDelays()

        if self.flag == -1:
            return
        
        
        self.convolveTimeDelayDistributionWithLineOfSight()

        if newHubbleParameter is not None:
            self.timeDelayWithLineOfSightPDF['x'] += self.rescaleToNewHubbleParameter(newHubbleParameter)
            self.timeDelayPDF['x'] += self.rescaleToNewHubbleParameter(newHubbleParameter)

    def rescaleToNewHubbleParameter( self, newHubbleParameter):
         timeDelayDistanceHubbleX =   self.getTimeDelayDistance( newHubbleParameter )
         timeDelayDistanceHubble100 = self.getTimeDelayDistance( 100. )
         
         return np.log10(timeDelayDistanceHubbleX/timeDelayDistanceHubble100)


    def getMagnificationBias( self ):
        
        self.magBias= magnificationBias( self.zSource,  self.minMagnification)
    
        
    def getTimeDelayDistance( self, HubbleConstant, omegaLambda=1.0):
        '''
        Get the time delay distance for this particle lens
        '''

        #Wessels distance class
        
        omegaMatter = 1. - omegaLambda
        OmegaK = 1. - omegaMatter - omegaLambda
        
        distanceClass = Distances.Distances(  self.zLens, self.data["z"], omegaMatter, OmegaK, HubbleConstant)

        
        cInMpcPerSecond = 9.7156e-15
        
        return  (1.+self.zLens)*distanceClass.Dl*distanceClass.Ds/distanceClass.Dls/cInMpcPerSecond
    


    def binTimeDelays( self, minMagRatio=0.1 ):
        '''
        Bin the time delays in log space
        '''
        secondsToDays = 1./60./60./24.


        minMagRatioIndexes = \
            (self.timeDelayData['magnificationRatio'] > minMagRatio) & \
            (self.timeDelayData['minCentralDistance'] > 1.)

        #Minus the dependecncy on h
        self.logDoubleTimeDelay = \
          np.log10(self.timeDelayData['timeDelay'][minMagRatioIndexes]*secondsToDays)


        
        if len(self.logDoubleTimeDelay) < 20:
            return -1
            #raise ValueError("Less than 20 time delays found (%i)" %   len(self.logDoubleTimeDelay)  )


        #Get the lower of the two magnifications
        self.minMagnification = \
            self.timeDelayData['minimumMagnification'][minMagRatioIndexes]
        
        #self.getMagnificationBias()
        self.magBias = self.timeDelayData['biasedTimeDelay'][minMagRatioIndexes]
       

        if self.timeDelayBins is None:
            timeDelayRange =  \
              [np.min(self.logDoubleTimeDelay), \
                np.max(self.logDoubleTimeDelay)]
            self.dDeltaTimeDelay = \
              self.lineOfSightPDF['dX'][0]/self.lineOfSightPDF['x'][-1]\
              *(timeDelayRange[1]-timeDelayRange[0])   
            self.timeDelayBins = \
              np.arange( timeDelayRange[0], timeDelayRange[1], \
                             self.dDeltaTimeDelay)
        else:
            timeDelayRange =  [np.min(self.timeDelayBins), \
                        np.max(self.timeDelayBins)]

            self.dDeltaTimeDelay = \
                self.lineOfSightPDF['dX'][0]/self.lineOfSightPDF['x'][-1]\
                *(timeDelayRange[1]-timeDelayRange[0])
          
        print("dDeltaTimeDelay is :", self.dDeltaTimeDelay)
        #Since the effective is polar and we are in cartesian
        #I need to weight the timedelays
        centralisationWeight = \
            1./self.timeDelayData['minCentralDistance'][minMagRatioIndexes]**2
            
        y, x = np.histogram( self.logDoubleTimeDelay, \
                          bins=self.timeDelayBins, density=True, \
                          weights=centralisationWeight)
                          
        xc = (x[1:] + x[:-1])/2.
        dX = x[1:] - x[:-1]

        #Removing calibartion
        xc -= np.log(0.94)

        yBiased, xBiased = \
            np.histogram( self.logDoubleTimeDelay, \
                          bins=self.timeDelayBins, density=True, \
                          weights=self.magBias*centralisationWeight )

        
        self.timeDelayPDF = {'x':xc, 'y':y, 'dX':self.dDeltaTimeDelay}
        
        self.biasedTimeDelayPDF = {'x':xc, 'y':yBiased, 'dX':self.dDeltaTimeDelay}
        
        return 1

    


    def getSourcePlaneWeighting( self ):
        '''
        Given that at different redshifts the dV/dZ is different
        I need the weighting so when I combine the distributions
        They are the same
        '''

        myTmpDistances = Distances.Distances(0, self.zLens, 0.272, 0, 100.)
        lensWeight, _, _, _ = myTmpDistances.Kernel_And_dVdZ( self.zLens)

        # correct for the error in the weighting
        myTmpDistances = Distances.Distances(self.zLens, self.data["z"], 0.272, 0, 100.)
        sourceWeight, _, _, wrongSourceWeight = myTmpDistances.Kernel_And_dVdZ(self.data["z"])

        #Becasue i weight a single redshift by volume i need to normalise to number
        #of halos in the sample
        lensWeight *= 1./len(glob.glob('../output/CDM/z_%0.2f/B*cluster_0_*_total*.json'% self.zLens))


        self.data["weight"] *= lensWeight / wrongSourceWeight * sourceWeight
        
    def getLineOfSightDistribution( self ):
        '''
        Use Wessels code to interpolate between the LoS pdfs for different
        source redshift planes
        '''
        #this is breaking down z<0.3:

        self.lineOfSightTable = getLoS.InterpolateLineOfSightTable('CDM',self.data['z'])
        #Given that this will be in mu not kappa, and since mu = 2kappa
        pdfInMagnitudes = np.array([ i[3] for i in self.lineOfSightTable.data])
        magnitudes = np.array([ i[0] for i in self.lineOfSightTable.data])
        dX = np.array([ i[2] - i[1] for i in self.lineOfSightTable.data])
        kappa = (magnitudes - 1)/2.

       
        dKappa = dX/2. 
        pdfInKappa = pdfInMagnitudes*2.
        
        self.lineOfSightPDF = {'x':kappa, 'y':pdfInKappa, 'dX':dKappa}
        

    def interpolateGivenPDF( self, givenTimeDelay, PDF ):
        '''
        Interpolate the closest pdf value for time delay
        '''
        
        
        interpolation =  np.interp( givenTimeDelay, PDF['x'], PDF['y'])

        interpolation[ (givenTimeDelay < PDF['x'][0]) | (givenTimeDelay > PDF['x'][-1])] = 0
        #renormalise
        dX = givenTimeDelay[1] - givenTimeDelay[0]

        #interpolation = interpolation/(np.sum(interpolation)*dX)
        
        return interpolation


    def convolveTimeDelayDistributionWithLineOfSight( self ):
        '''
        The convolution goes as 
        
        P(log(T_obs)) = int dT'/ T' T/T' P(log(T'))P(1. - T'/T_obs)
        
        '''

        #First set up the T observed vector
        pdfDeltaTimeDelayObserved = np.zeros(len(self.timeDelayPDF['x'])+len(self.lineOfSightPDF['x']) -1 )
        
        pdfDeltaTimeDelayObservedBiased = \
          np.zeros(len(self.timeDelayPDF['x'])+len(self.lineOfSightPDF['x']) -1 )
                
        startValue = \
          self.timeDelayPDF['x'][0] - len(self.lineOfSightPDF['x'])*self.dDeltaTimeDelay

        #THe values at which i will work out each element of the PDF
        deltaTimeDelayObserved = \
          np.arange( len(pdfDeltaTimeDelayObserved))*self.dDeltaTimeDelay \
          + startValue
        #The list of values for each element of the PDF i will integrate over
        deltaTimeDelayTrue = \
          np.arange( len(pdfDeltaTimeDelayObserved))*self.dDeltaTimeDelay \
          + startValue
        
        probabilityTimeDelayTrue = \
          self.interpolateGivenPDF( deltaTimeDelayObserved, \
                                    self.timeDelayPDF )

        probabilityBiasedTimeDelayTrue = \
              self.interpolateGivenPDF( deltaTimeDelayObserved , \
                                            self.biasedTimeDelayPDF)
        
        for iPdf, iTimeDelayObserved in enumerate(deltaTimeDelayObserved):

            kappa = 1. - 10**iTimeDelayObserved/10**deltaTimeDelayTrue
            
            probabilityLineOfSightKappa = \
              self.interpolateGivenPDF( kappa , self.lineOfSightPDF)
      
            probabilityObservedTimeDelay = np.log(10)*self.dDeltaTimeDelay*\
              10**iTimeDelayObserved/10**deltaTimeDelayTrue*\
              probabilityLineOfSightKappa

            pdfDeltaTimeDelayObserved[iPdf] = \
              np.sum(probabilityObservedTimeDelay*probabilityTimeDelayTrue)
              
            pdfDeltaTimeDelayObservedBiased[iPdf] = \
              np.sum(probabilityObservedTimeDelay*probabilityBiasedTimeDelayTrue)
                   
            
            
            
        pdfDeltaTimeDelayObserved /=  np.sum(pdfDeltaTimeDelayObserved*self.dDeltaTimeDelay)
          
        pdfDeltaTimeDelayObservedBiased /= \
          np.sum(pdfDeltaTimeDelayObservedBiased*self.dDeltaTimeDelay)
          
        
        self.timeDelayWithLineOfSightPDF = \
          {'x':deltaTimeDelayObserved, 'y':pdfDeltaTimeDelayObserved,'dX':self.dDeltaTimeDelay}

        self.biasedTimeDelayWithLineOfSightPDF = \
           {'x':deltaTimeDelayObserved, 'y':pdfDeltaTimeDelayObservedBiased, 'dX':self.dDeltaTimeDelay}
        
