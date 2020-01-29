

from hubbleInterpolatorClass import *

def main():

    interpolateToTheseTimes=  \
      np.linspace(-3, 4, 1000)
    powerLawIndex = np.linspace(0,1.,10)
    hubbleInterpolaterClass = hubbleInterpolator()
    hubbleInterpolaterClass.getTrainingData()
    hubbleInterpolaterClass.extractPrincipalComponents()

    #for iWeight in 10**np.linspace(-4,2.,10):
    hubbleInterpolaterClass.learnPrincipalComponents(weight=1)
    print(hubbleInterpolaterClass.getGaussProcessLogLike())
    for i in powerLawIndex:
        interpolatedProb = \
          hubbleInterpolaterClass.predictPDF( interpolateToTheseTimes, \
                                            np.array([i,0.7, -1.5] ))

        plt.plot(interpolateToTheseTimes, interpolatedProb/np.max(interpolatedProb))
    plt.show()
if __name__ == '__main__':
    main()
