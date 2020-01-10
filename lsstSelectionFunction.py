'''
Using the luminosity functions of quasars
i want the redshift distributions

'''

from magnificationBias import *
from generateDifferentSelectionFunctions import *

def main():

    redshifts = np.linspace(0.1, 5., 100)
    nQuasars = []
    Volume = []
    for iRedshift in redshifts:
        luminosityFunctionClass = \
          luminosityFunction( iRedshift )
        comovingVolume = \
          distance.diff_comoving_volume( iRedshift, \
                        **luminosityFunctionClass.cosmo)
       
        nQuasars.append(np.sum(luminosityFunctionClass.luminosityFunction['y'])*np.abs(luminosityFunctionClass.dMag))
        Volume.append(comovingVolume)


    
    Volume = np.array(Volume)
    nQuasars = np.array(nQuasars)

    plt.plot(redshifts, nQuasars/np.max(nQuasars))
    plt.plot(redshifts, Volume/np.max(Volume))
    total = (Volume*nQuasars)
    plt.plot(redshifts, total/np.max(total))
    

    
    plt.show()


if __name__ == '__main__':
    main()
        
