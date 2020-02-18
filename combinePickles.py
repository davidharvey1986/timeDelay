
import pickle as pkl
import numpy as np
import glob

def main():

    for i in glob.glob('picklesMinimumDelay30/multiFitSamples_*.pkl'):
        samplesA = pkl.load(open(i, 'rb'))
        samplesB = pkl.load(open('picklesMinimumDelay70/%s' \
                                     % i.split('/')[1], 'rb'))

        total = np.vstack((samplesA,samplesB))

        pkl.dump(total, open('picklesMinimumDelay/%s' % i.split('/')[1], 'wb'))
        print(samplesA.shape, samplesB.shape, total.shape)

def mainTrue():

    for i in glob.glob('exactPDFpickles30/multiFitSamples_*.pkl'):
        samplesA = pkl.load(open(i, 'rb'))
        samplesB = pkl.load(open('exactPDFpickles70/%s' \
                                     % i.split('/')[1], 'rb'))

        total = np.vstack((samplesA,samplesB))

        pkl.dump(total, open('exactPDFpickles/%s' % i.split('/')[1], 'wb'))
        print(samplesA.shape, samplesB.shape, total.shape)
        
if __name__ == '__main__':
    main()
    mainTrue()
