
import pickle as pkl
import numpy as np
import glob

def main():

    for i in glob.glob('pickles/multiFitSamples_*.pkl'):
        samplesA = pkl.load(open(i, 'rb'))
        samplesB = pkl.load(open('picklesCumSumGaussError/%s' \
                                     % i.split('/')[1], 'rb'))

        total = np.vstack((samplesA,samplesB))

        pkl.dump(total, open('exactPDFpickles/%s' % i.split('/')[1], 'wb'))
        print(samplesA.shape, samplesB.shape, total.shape)

def mainTrue():

    for i in glob.glob('picklesTrueHubbleOld/multiFitSamples_*.pkl'):
        samplesA = pkl.load(open(i, 'rb'))
        samplesB = pkl.load(open('picklesTrueHubbleNew/%s' \
                                     % i.split('/')[1], 'rb'))

        total = np.vstack((samplesA,samplesB))

        pkl.dump(total, open('picklesTrueHubble/%s' % i.split('/')[1], 'wb'))
        print(samplesA.shape, samplesB.shape, total.shape)
        
if __name__ == '__main__':
    main()
        
