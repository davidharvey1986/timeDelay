



import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb as pdb
import pickle as pkl
from cosmolopy import distance as dist
import os

class lens:
    def __init__( self, position, resolution = 1000 ):
        '''
        The lens set up, epslion 0 is for an SIS
        epsilon0 = 4pi*(sigma/c)^2*dlds/dls
        
        But for now everything is unitless
        '''
        
        self.position = position
        
        xVector = np.linspace( -1, 1, resolution ) 
        yVector = np.linspace( -1, 1, resolution )  
          
        xGrid, yGrid = \
          np.meshgrid( xVector, yVector)
          
        xPotentialGrid = xGrid-position[0]
        yPotentialGrid = yGrid-position[1]
        
        self.x = xGrid
        self.y = yGrid
        
        self.potential = np.sqrt( xPotentialGrid**2 + yPotentialGrid**2)

class source:
    def __init__( self, position, resolution = 1000 ):
        self.position = position
        
class lensSourceConfiguration:
    
    def __init__(self, lens, source ):
        self.lens = lens
        self.source = source

    def getTimeArrivalSurface( self ):
        
        self.timeArrivalSurface = \
          0.5*(self.lens.x - self.source.position[0])**2 + \
          0.5*(self.lens.y - self.source.position[1])**2 - \
          self.lens.potential
        
    def getImagePositions( self, threshold=1e-11 ):
        '''
        i.e. where the time arrival surface gradient is
        0
        '''
        gradient = np.gradient(self.timeArrivalSurface)
        
        self.derivativeTimeArrivalSurface = \
          gradient[0]*gradient[1]

        self.imageIndex =  \
          np.abs(self.derivativeTimeArrivalSurface) < threshold
          
        imageTimeArrivals = \
          self.timeArrivalSurface[  self.imageIndex ]
        
        self.timeArrivals =  \
          np.unique(imageTimeArrivals)
            
          
        self.numberMultipleImages = \
          len( self.timeArrivals )
        
    def getTimeDelays( self ):
        self.timeDelays = np.array([])
        self.timeArrivals = np.sort(self.timeArrivals)[::-1]
        for iImage in xrange(self.numberMultipleImages-1):
            iDelays = self.timeArrivals[iImage] - \
              np.delete(self.timeArrivals, np.arange(iImage+1))

            
            self.timeDelays = np.append(self.timeDelays,iDelays)
           
    def plotTimeArrivalSurface( self ):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface( self.lens.x, \
                         self.lens.y,\
                         self.timeArrivalSurface )
        plt.show()

        
    def plotImagePositions( self ):
        images = np.zeros( self.timeArrivalSurface.shape)
        images[self.imageIndex] = 1.
        plt.imshow( images )
        plt.show()
        
def getTimeDelays(sourceResolution):
    if  os.path.isfile('../output/SIS_analytical_example.pkl'):
        return pkl.load(open('../output/SIS_analytical_example.pkl','rb'))
                          

    allSourcePositions = np.linspace(0, 1,sourceResolution)
    
    #If the lens is cicually symmetric i dont need to integrate over
    #all positions as they are the same, so just take one position
    #and when i histogram it weight it by its radial distace

    lensPotentialPosition = [0., 0.]
    galaxyLens = lens(lensPotentialPosition)

    allConfigurations = []
    allTimeDelays = np.array([])
    allWeights = np.array([])
    
    for iSource in range(np.int(sourceResolution/10), sourceResolution):
        print("%i/%i" % (iSource+1, sourceResolution))

        for jSource in range(np.int(sourceResolution/10), sourceResolution):
            iSourcePosition =\
              [allSourcePositions[iSource], allSourcePositions[jSource]]
        
            galaxySource = source(iSourcePosition)
    
            galaxyImagePlane = \
              lensSourceConfiguration( galaxyLens, galaxySource)
            galaxyImagePlane.getTimeArrivalSurface()
            galaxyImagePlane.getImagePositions()
            galaxyImagePlane.getTimeDelays()

            allConfigurations.append( galaxyImagePlane ) 
            
            allTimeDelays = \
              np.append(allTimeDelays, galaxyImagePlane.timeDelays )

            weights = np.zeros(len(galaxyImagePlane.timeDelays)) +\
                        allSourcePositions[iSource]
            allWeights = \
              np.append(allWeights, weights)

    pkl.dump([allTimeDelays,allWeights], \
        open('../output/SIS_analytical_example.pkl','wb'))
        
    
    return allTimeDelays, allWeights

def timeDelayDistanceForSIS( velocityDispersion, zLens, zSource ):
    '''
    At the moment, i have saved the fermat potential in unitless
    SIS
    i.e deltaPHI

    so i need to times by 
    eta0^2 Ds / (Dls*Dl)*(1+z)
    
    eta0 = 4 pi (v/c)^2 * Dl Dls / Dls

    '''
    
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 1.}
    cosmo = dist.set_omega_k_0(cosmo)
    Dl = dist.angular_diameter_distance(zLens, **cosmo)
    Dls = dist.angular_diameter_distance(zSource, z0=zLens, **cosmo)
    Ds = dist.angular_diameter_distance(zSource, **cosmo)
    
    cInKmPerSecond = 3e5
    cInMpcPerDay = 9.7156e-15*60.*60.*24

    Eta0 = 4.*np.pi*(velocityDispersion/cInKmPerSecond)*Dl*Dls / Ds

    TimeDelayDistance  = (1+zLens)/cInMpcPerDay*Ds/(Dl*Dls)*Eta0**2

    return TimeDelayDistance
    
def main():
    unitLessTimeDelays, weights = getTimeDelays(50)

    zLens = 0.2
    zSource = 1.0
    velocityDispersion = 100.
    timeDelayDistance = \
      timeDelayDistanceForSIS( velocityDispersion, zLens, zSource )
    timeDelays = unitLessTimeDelays*timeDelayDistance/206265.

    #Have an excess of timedelays at large T, for now i dont
    #care but i should see where these are coming from
    y, x =np.histogram( timeDelays,\
            bins = 10**np.linspace(0,3,30))

    dx = x[1:] - x[:-1]
    y = y.astype(float)
    y /= np.sum(y*dx)
    xc = (x[1:] + x[:-1])/2.
    plt.plot(xc,y)

    plt.plot(xc, 0.0001*xc**2)
    plt.xscale('log')
    plt.yscale('log')

    plt.show()
    
if __name__ == '__main__':
    main()
