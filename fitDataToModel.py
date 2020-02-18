import plotAsFunctionOfDensityProfile as getDensity
from powerLawFit import *
import fitHubbleParameter as fitHubble
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis
import corner as corner
from hubbleInterpolatorClass import *
from scipy.stats import norm
import matplotlib.lines as mlines
from matplotlib import rcParams
rcParams["font.size"] = 16
from scipy.ndimage import gaussian_filter as gauss


def main():

    selectedTimeDelays = getObservations()
    
    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData('exactPDFpickles/trainingData.pkl')
    hubbleInterpolaterClass.extractPrincipalComponents()
    hubbleInterpolaterClass.learnPrincipalComponents()

    fitHubbleClass = \
      fitHubble.fitHubbleParameterClass( selectedTimeDelays, \
                                    hubbleInterpolaterClass)


    pdb.set_trace()
def getObservations():
    data = np.loadtxt( '../data/oguriTimeDelays.txt',\
                           dtype=[('name',object), ('zs', float), ('zl', float),\
                                      ('timeDelays', float)])

    timeDelays = np.log10(np.sort(data['timeDelays']))

    probability=np.ones(len(timeDelays))

    cumsum = np.cumsum(probability) / np.sum(probability)

    error = np.sqrt(np.cumsum(probability))/ np.sum(probability)
    plt.plot(timeDelays,cumsum)
    plt.show()
    return {'x':timeDelays, 'y':cumsum, 'error':error}


if __name__ == '__main__':
    main()
    
    
    
