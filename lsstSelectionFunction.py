'''
Using the luminosity functions of quasars
i want the redshift distributions

'''

from generateDifferentSelectionFunctions import *

def main():

    redshifts = np.linspace(0.1, 10., 100)
    total = []
    for iRedshift in redshifts:

        ana = getSourceRedshiftWeight( iRedshift, zMed=2.)
        #plt.plot(iRedshift, ana/np.max(ana), ':')
        total.append(getSelectionFunction( iRedshift))

    plt.plot(redshifts, total)
    

    
    plt.show()
    
def getSelectionFunction( iRedshift, limitingObsMag=27):

    luminosityFunctionClass = \
          luminosityFunction( iRedshift, limitingObsMag=limitingObsMag )

    nQuasars = \
      np.sum(luminosityFunctionClass.luminosityFunction['y'])*\
      np.abs(luminosityFunctionClass.dMag)
    comovingVolume = \
          distance.diff_comoving_volume( iRedshift, \
                        **luminosityFunctionClass.cosmo)
       
    return nQuasars*comovingVolume/1e9

if __name__ == '__main__':
    main()
        
