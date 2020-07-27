
from predictConstraintsOnHubble import selectRandomSampleOfTimeDelays
import os
import pickle as pkl
import ipdb as pdb
import progressbar
import numpy as np
from matplotlib import pyplot as plt

def main(fiducialNobs= 100):
    '''
    It is taking too long to generate this variance,
    it should be analytical but im an idiot
    So just smash it here and divide the error bars 
    by sqrt(n) each time
    '''

    strappePDF = \
      selectRandomSampleOfTimeDelays( fiducialNobs)
    
    newPDF = \
      selectRandomSampleOfTimeDelays( fiducialNobs*10)

    plt.plot(strappePDF['x'], strappePDF['error'][0] / newPDF['error'][0])
    plt.plot(strappePDF['x'], strappePDF['error'][1] / newPDF['error'][1])

    plt.show()
  




if __name__ == '__main__':
    main()
