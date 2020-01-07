'''
COnvolve the distribution with a Gaussian of 3 days then take 
I think this should not change it in the slightest

'''
from scipy.ndimage import gaussian_filter as gauss
from plotHubbleDistributions import *

def main():
    
    fig = plt.figure( figsize = (10, 10))

    gs = gridspec.GridSpec(10,1)

    axisA = plt.subplot( gs[0:7,0])
    axisB = plt.subplot( gs[7:,0])
    
    

    colors = ['r','b','g','c','orange','k']

    ratioYearToMonth = []
    pklFileName = '../output/CDM/combinedPDF_100.0.pkl'
    finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
        
    finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])
    convolvedWithObservationalNoise = gauss( finalMergedPDFdict['y'], np.log10(100.))
    
    axisA.plot(finalMergedPDFdict['x']+2.56, finalMergedPDFdict['y'], \
                   label=r"Without Obs", color='green')

                   
    axisA.plot(finalMergedPDFdict['x']+2.56, convolvedWithObservationalNoise,\
                   label=r"With Obs", color='red')
    axisB.plot(finalMergedPDFdict['x']+2.56,finalMergedPDFdict['y']/convolvedWithObservationalNoise)
    axisB.plot([0,4],[1,1],'k--')
    axisA.legend()
    axisA.set_yscale('log')
 
    axisB.set_xlabel(r'log($\Delta T$/ days)')
    axisA.set_ylabel(r'P(log($\Delta T$/ days))')
    axisA.set_xlim(0.6,3.)
    axisB.set_xlim(0.6,3.)
    axisA.set_ylim(2e-3,1.2)
    axisB.set_ylim(0.5,2)
    axisB.set_yscale('log')
    axisA.set_xticklabels([])

    plt.show()


if __name__ == '__main__':
    main()
