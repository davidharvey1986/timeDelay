
from matplotlib import gridspec
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
from scipy.stats import lognorm
from lsstSelectionFunction import *

def main(fiducialHubble = 70. ):
    '''
    Analyse the different selection functions and the impact 
    on the power law index
    '''
    fig = plt.figure(figsize=(6,7))
    gs = gridspec.GridSpec(13,1)
    axisC = plt.subplot( gs[0:4,0] )
    axisA = plt.subplot( gs[5:12,0])

    zMed = np.linspace(1.5, 2.5, 3)
    zMax = 8.
    axisC.set_xlabel(r'Source Redshift, $z_s$')
    axisC.set_ylabel(r'P($z_s$)')

    plotLSST( [axisA,axisC], fiducialHubble)
    plotPDFwithNoSelectionFunction(  [axisA,axisC], fiducialHubble)


    #axisA.set_yscale('log')

    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
    axisA.set_ylabel(r'P(log[$\Delta t$])')

    axisA.set_xlim(-2.,3.)
    #axisA.set_ylim(1e-2,2.)
    #axisB.set_xlabel(r'Median Source Redshift, $z_{\rm med}$')
    #axisB.set_ylabel(r'$\beta$')
    plt.savefig('../plots/selectionFunctionEffect.pdf')

    plt.show()


def plotLSST( axisArr, fiducialHubble ):

    redshifts = np.linspace( 0.1, 8.0)
    selectFunc = []
    for iRedshift in redshifts:
        selectFunc.append( getSelectionFunction( iRedshift))
    selectFunc = np.array(selectFunc)

    dZ = redshifts[1] - redshifts[0]
    axisArr[1].plot( redshifts, selectFunc/(np.sum(selectFunc)*dZ))

    
    color = 'black'
        
    pklFileName = '../output/CDM/selectionFunction/SF_%i_lsst.pkl' \
          % (fiducialHubble )
          
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

    
    finalMergedPDFdict['y'] /= np.sum( finalMergedPDFdict['y'])*(finalMergedPDFdict['x'][1]- finalMergedPDFdict['x'][0])

    axisArr[0].plot( finalMergedPDFdict['x'], finalMergedPDFdict['y'], \
                    color=color, label= "LSST")
            
                    
def plotPDFwithNoSelectionFunction( axArr, fiducialHubble , color='blue' ):

    pklFileName = '../output/CDM/combinedPDF_%0.1f.pkl' \
          % (fiducialHubble )
          
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
    finalMergedPDFdict['y'] /= np.sum( finalMergedPDFdict['y'])*(finalMergedPDFdict['x'][1]- finalMergedPDFdict['x'][0])
    axArr[0].plot( finalMergedPDFdict['x'], finalMergedPDFdict['y'], \
                    color=color, label= "No Selection Function")

                    
def getSourceRedshiftWeight( z, zMed=1.0 ):

    zStar = zMed/1.412
    weight = z**2*np.exp(-(z/zStar)**(1.5))
    return weight/np.sum(weight)


if __name__ == '__main__':
    main()

