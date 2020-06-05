from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np

def main():


    dmModels = ['CDM','L8','L11p2']

    for iDMmodel in dmModels:

        pklFile = \
          '../output/%s/selectionFunction/SF_ensemble_fiducialCosmo.pkl' \
          % iDMmodel


        finalMergedPDFdict = pkl.load(open(pklFile, 'rb'))
        finalMergedPDFdict['y'] = \
              1. - np.cumsum(finalMergedPDFdict['y'])/\
              np.sum(finalMergedPDFdict['y'])


        plt.plot( finalMergedPDFdict['x'], finalMergedPDFdict['y'], \
                      label=iDMmodel)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
