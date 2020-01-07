from plotHubbleDistributions import *

def main():
    '''
    Create a plot of the integrated probability distribution over all redshift and lenses
    for different hubble parameters
    '''
    fig = plt.figure( figsize = (10, 10))

    gs = gridspec.GridSpec(10,1)

    axisA = plt.subplot( gs[0:7,0])
    axisB = plt.subplot( gs[8:,0])
    
    
    hubbleParameters = [50., 60., 70., 80., 90., 100.]
    colors = ['r','b','g','c','orange','k']

    dmModels = ['CDM','WDM']
    dmModelsLs = ['--',':']
    for iLs, iDM in enumerate(dmModels):
        ratioYearToMonth = []
        for iColor, iHubbleParameter in enumerate(hubbleParameters):
            pklFileName = '../output/'+iDM+'/combinedPDF_'+str(iHubbleParameter)+'.pkl'
            if os.path.isfile(pklFileName):
                finalMergedPDFdict = pkl.load(open(pklFileName,'rb'))
            else:
                raise ValueError("No pickle file found (%s) "%pklFileName)
            finalMergedPDFdict['y'] /= np.max(finalMergedPDFdict['y'])

            axisA.plot(finalMergedPDFdict['x']+2.56,finalMergedPDFdict['y'],label=r"H0=%i kms/Mpc" % iHubbleParameter, color=colors[iColor])


            #####FIT POWER LAW TO THE DISTRIBUTION##############
            peakTime = np.max( finalMergedPDFdict['y'] )
            fittedProbability =  \
              (finalMergedPDFdict['y'] <peakTime) & (finalMergedPDFdict['y']>1e-2)
            params, error = sis.fitPowerLaw( finalMergedPDFdict['x'][fittedProbability],finalMergedPDFdict['y'][fittedProbability])
            axisA.plot( finalMergedPDFdict['x']+2.56, 10**sis.straightLine( finalMergedPDFdict['x'], *params), ls=dmModelsLs[iLs], color=colors[iColor])
            lsstError = 3.
            yearLogProb = (365.+lsstError)**(params[1]+1) - (365.-lsstError)**(params[1]+1)
            monthLogProb = (30.+lsstError)**(params[1]+1) - (30.-lsstError)**(params[1]+1)
            ratio = yearLogProb / monthLogProb
            ratioYearToMonth.append(ratio)

            print(iHubbleParameter, params[1], ratio)
        
        ######################################################
        axisB.errorbar( hubbleParameters, ratioYearToMonth,fmt='k*')
        params, cov = sis.curve_fit( sis.straightLine, hubbleParameters, np.log10(ratioYearToMonth), p0=[1.,1.])
    
        axisB.plot( hubbleParameters, 10**sis.straightLine( np.array(hubbleParameters), *params), 'k',ls=dmModelsLs[iLs])
    
    axisB.set_xlabel(r'H0 (km/s/Mpc)')
    axisB.set_ylabel(r'P($\Delta T_{365}$) / P($\Delta T_{30}$)')
    axisA.legend()
    axisA.set_yscale('log')
 
    axisA.set_xlabel(r'log($\Delta T$/ days)')
    axisA.set_ylabel(r'P(log($\Delta T$/ days))')
    axisA.set_xlim(-1.4,3.5)
    axisA.set_ylim(1e-2,1.2)
    plt.savefig('../plots/compareWDMwithCDM.pdf')
    plt.show()
    
    
if __name__ == '__main__':
    compareCDMtoWDMhubbleDistributions()
