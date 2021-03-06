

'''
I want to determine the numerical artifacts in the histogram, so i want to histogram the time delays as a function
of distance from the centre of the halo

'''
import numpy as np
import json
import ipdb as pdb
import pickle as pkl
from matplotlib import pyplot as plt
#from analyseSISexample import *
from powerLawFit import *
from magnificationBias import *
import cosmolopy.distance as dist
import glob 
def main(fname, bias=False, **kwargs):
    '''
    Plot the images as a function of distance from the centre
    '''

    data = pkl.load(open(fname,'rb'))

    xc, y, yError = getHistogram( data, biasWeight=bias )

    yError /= np.max(y[powerLawFitIndex])

    y /= np.max(y[powerLawFitIndex])

    params, cov = curve_fit( straightLine, xc[powerLawFitIndex], \
                                 np.log10(y[powerLawFitIndex]), p0=[1,1])
    error = np.sqrt(np.diag(cov))

    inputPDF = {'x':xc[powerLawFitIndex], 'y': y[powerLawFitIndex], \
                    'yError':yError[powerLawFitIndex] }
    
    mcmcFit = powerLawFit( inputPDF )
    params =  mcmcFit.params['params']
    error= mcmcFit.params['error']
    
    yError[ yError > y ] = y[ yError > y ]*0.99
    plt.errorbar(xc-np.log10(0.5*0.7**2), y, yerr=yError,fmt='o', color=kwargs['color'])

    kwargs['label'] +=  "%0.2f+/- %0.2f)" % (params[1], error[1])
    plt.plot(xc-np.log10(0.5*0.7**2), 10**straightLine( xc, *params),'--',\
                 **kwargs )
    print("The power law model follows %0.2f +/- %0.2f" % (params[1], error[1]))
    plt.yscale('log')
    plt.xlim(2,3.3)
    plt.ylim(1e-2, 2.)
    plt.ylabel(r'P(log[$\Delta t$]) / P(log[$\Delta t_{\rm peak}$])')

    plt.xlabel(r'log($\Delta t$ / days)')
    
    plt.legend()


def plotAsFunctionMagRatio(fname, **kwargs):
    '''
    plot the same pdf but as function of magnification ratio
    '''
    nBins=2
    
    data = pkl.load(open(fname,'rb'))
    nBins =10
    magnificationBins = [0., 0.1, 1.]
    seconds2days = 1./60./60./24

    for iBin in range(nBins):

        inBin = (data['magnificationRatio'] > magnificationBins[iBin]) &\
          (data['magnificationRatio'] < magnificationBins[iBin+1]) & \
          (np.log10(data['timeDelay']*seconds2days) > 0.7) 

        magBinnedData = {}
        for iKey in data.keys():
            magBinnedData[ iKey] = data[iKey][inBin]
           
        xc, y, yError = getHistogram( magBinnedData )

        plt.errorbar( xc, y, yerr=yError, label=str(iBin))
    xc, y, yError = getHistogram( data )

    plt.errorbar( xc, y, yerr=yError, label=str(iBin))
    plt.yscale('log')
    plt.xlim(1.2,2.4)
    plt.ylim(1e-3, 1.0)
    plt.ylabel(r'p(log($\Delta t$ / days)) < log$(Delta t_{\rm peak}$)')
    plt.xlabel(r'log($\Delta t$ / days)')
    plt.legend()
   


def getHistogram( data, weight=True, biasWeight=False, bins=None):
    seconds2days = 1./60./60./24
    if bins is None:
        bins=100
    if weight:
        weightedTime = (1./data['minCentralDistance'])**2
    else:
        weightedTime = np.ones(len(data['minCentralDistance']))

    if biasWeight:
        zSource = 5.
        print(data['minimumMagnification'])
        magBiasWeight = data['biasedTimeDelay'] / data['timeDelay']

        weightedTime *= magBiasWeight
        
    index = (data['minCentralDistance'] > 1.) & \
      (data['magnificationRatio'] > 0.1)

    rescale = getTimeDelayDistance( 0.2, 3.0, 70.)/ getTimeDelayDistance( 0.2, 3.0, 100.)

    y, x = np.histogram(np.log10(data['timeDelay'][index]*seconds2days*rescale), \
                bins=bins, density=False, weights=weightedTime[index])
    y = y.astype(float)
    yErrorFrac = 1./np.sqrt(y)
    y /= np.max(y)
    dX = x[1:] - x[:-1]
    y /= (np.sum(y)*dX)
    xc = (x[1:] + x[:-1])/2.
    yError = y*yErrorFrac
    yError[ y < yError ] = y[ y < yError]*0.99

    return xc, y, yError

def cleanMultipleImages(jsonFile, zSource=5.):
    '''
    Loop through the high resolution json file and
    get the position of each time delay

    PLus clean out those that are not on opposite sides of the lens
    '''

    jsonDataAllSourcePlanes = json.load(open(jsonFile,'rb'))

    #since  len(jsonData['doubleTimeDelay']) =  len(jsonData['positionX'])
    #Each list in positionX corresponds to 1 time delay

    
    cleanData = []
    print("Cleaning %s" % jsonFile)
    noTimes = np.array([])
    for jsonData in jsonDataAllSourcePlanes:
        

        times = np.array(jsonData['doublesTime'])
        if len(times) == 0:
            
            data = {"timeDelay":noTimes, "minCentralDistance":noTimes, \
                    "imageSeparation":noTimes, "magnificationRatio":noTimes, \
                    "minimumMagnification":noTimes, \
                    "biasedTimeDelay":noTimes}
                    
            cleanData.append(data) 
            continue
        xDistance = np.array(jsonData['positionX']) - 500.
        yDistance = np.array(jsonData['positionY']) - 500.
        centralDistance = np.sqrt( xDistance**2+yDistance**2)
        centralDistance[ times == 0 ] = 999999
        parity = xDistance / np.abs(xDistance)
        newTimeDelays = np.array( jsonData['doubleTimeDelay'])
        minCentralDistance = np.min(centralDistance, axis=1)
        
        absoluteMags = np.abs( np.array(jsonData['doubles'] ))
        absoluteMags[ absoluteMags == 0 ] = 999999
        minimumMagnification = np.min(absoluteMags, axis=1)
        magnificationRatio = np.array(jsonData['doubleRatios'])
        print("Getting magnification bias for source Redshift %s" % jsonData['z'])
        magBiasWeight = \
            magnificationBias( jsonData['z'], minimumMagnification)
        print("Done")
        tNoZerosIndex = np.argsort(times, axis=1)
        xNoZeros = np.take_along_axis(xDistance,tNoZerosIndex,axis=1)
        yNoZeros = np.take_along_axis(yDistance,tNoZerosIndex,axis=1)
        tNoZeros = np.take_along_axis(times,tNoZerosIndex,axis=1)
        tDelays = tNoZeros[:,-1] - tNoZeros[:,-2]
        
        imageSeparation = \
            np.sqrt((xNoZeros[:,-2] - xNoZeros[:,-1])**2+\
                    (yNoZeros[:,-2] - yNoZeros[:,-1])**2)

            
        data = {"timeDelay":newTimeDelays, "minCentralDistance":minCentralDistance, \
            "imageSeparation":imageSeparation, "magnificationRatio":magnificationRatio, \
            "minimumMagnification":minimumMagnification, \
            "biasedTimeDelay":magBiasWeight}
        
        cleanData.append(data) 
    pkl.dump(cleanData, open(jsonFile+'.clean.pkl','wb'))


def cleanAllFiles():
    '''
    loop through the main files and clean
    like here
    '''
    allFiles = glob.glob('../output/CDM/z_0.74/*.json')

    for iFile in allFiles:
        cleanMultipleImages(iFile)
        print(iFile)

        
    
    
def getTimeDelayDistance(zLens, zSource, HubbleConstant, omegaLambda=1.0):
        '''
        Get the time delay distance for this particle lens
        '''

        #Wessels distance class
        
        omegaMatter = 1. - omegaLambda
        OmegaK = 1. - omegaMatter - omegaLambda
        

        cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : HubbleConstant/100.}
        cosmo = dist.set_omega_k_0(cosmo)    
    
        Dls =  dist.angular_diameter_distance(zSource, z0=zLens, **cosmo)
        Dl =  dist.angular_diameter_distance(zLens,**cosmo)
        Ds =  dist.angular_diameter_distance(zSource, **cosmo)
        
        cInMpcPerSecond = 9.7156e-15
        
        return  (1.+zLens)*Dl*Ds/Dls/cInMpcPerSecond



def sisExample():
    #NEEDS TO BE DONE
    #fname = '../output/SISexample/SIS_example_z0.2_400_4.SISexample.json'
    #cleanMultipleImages(fname)
    #main(fname+".clean.pkl",
    #    label=r"Source Plane Resolution: 0.2kpc  ($\beta $ = ",\
    #    color='green')
        
    #NEEDS TO BE DONE
    #fname = '../output/NFWexample/NFW_example_z0.2_14.00_4.json'
    #cleanMultipleImages(fname)
    #main(fname+".clean.pkl",
    #    label=r"NFW  ($\beta $ = ",\
    #    color='green')




    #NEEDS TO BE DONE
    fname = '../output/SISexample/SIS_example_z0.2_405.SISexample.json'
    #cleanMultipleImages(fname)
    main(fname+".clean.pkl",\
         label=r"Source Plane Resolution: 1kpc  ($\beta $ = ",\
          color='red')

        
    fname = '../output/SISexample/SIS_example_z0.2_400_0.25.SISexample.json'
    #cleanMultipleImages(fname)
    main(fname+".clean.pkl", \
        label=r"Source Plane Resolution: 4kpc  ($\beta $ = ", \
        color='orange')
        
    plt.savefig('../plots/SISexample.pdf')
    
    plt.show()

    
if __name__ == '__main__':
    sisExample()
