from plotHubbleDistributions import *
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc
from scipy.stats import lognorm
def main( ):
    '''
    Analyse the different selection functions and the impact 
    on the power law index
    '''
    fig = plt.figure(figsize=(6,7))
    gs = gridspec.GridSpec(15,1)
    axisC = plt.subplot( gs[0:4,0] )
    axisA = plt.subplot( gs[5:12,0])
    axisB = plt.subplot( gs[13:,0])

    zMed = np.linspace(1.5, 2.5, 11)
    zMax = 5.
    axisC.set_xlabel(r'Source Redshift, $z_s$')
    axisC.set_ylabel(r'P($z_s$)')

    #For aesthetics                                                         
    jet = cm = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=np.min(zMed)*0.98, \
                            vmax=np.max(zMed))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #####
    zs = np.linspace(0., zMax, 100)
    for i in zMed:
        axisC.plot( zs, getSourceRedshiftWeight( zs, zMed=i),
                        color=scalarMap.to_rgba(i))
    fiducialHubble = 70.
  
    #plot origninal pdf
    pklFileName = '../output/CDM/combinedPDF_%0.1f.pkl' % fiducialHubble
          
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

    plotPDF( finalMergedPDFdict, 'blue', \
                 'Original', axisA, yType='yBiased', nofill=True )
        
    for iColor, izMed in enumerate(zMed):
        
        color = scalarMap.to_rgba(izMed)
        
        pklFileName = '../output/CDM/selectionFunction/SF_%i_%0.2f.pkl' \
          % (fiducialHubble, izMed )
          
        finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))

        #finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])

        plotPDF( finalMergedPDFdict, color, \
                r"log($M/M_\odot$)=%0.2f" % np.float(izMed),
                     axisA, yType='y', nofill=True )
            
        #####FIT POWER LAW TO THE DISTRIBUTION##############
        powerLawFitClass = powerLawFit( finalMergedPDFdict )
                                             
     
        ######################################################
        axisB.errorbar( np.float( izMed), \
                    powerLawFitClass.params['params'][1], \
                    yerr=powerLawFitClass.params['error'][1], \
                    fmt='o', color=color)

        

    #axisA.set_yscale('log')

    axisA.set_xlabel(r'log($\Delta t$/ days)', labelpad=-1)
    
    axisA.set_ylabel(r'P(log[$\Delta t$])')

    axisA.set_xlim(-2.,3.)
    #axisA.set_ylim(1e-2,2.)
    axisB.set_xlabel(r'Median Source Redshift, $z_{\rm med}$')
    axisB.set_ylabel(r'$\beta$')
    plt.savefig('../plots/selectionFunctionEffect.pdf')

    plt.show()




    

    
def getSourceRedshiftWeight( z, zMed=1.0 ):

    zStar = zMed/1.412
    weight = z**2*np.exp(-(z/zStar)**(1.5))
    return weight/np.sum(weight)


if __name__ == '__main__':
    main()
