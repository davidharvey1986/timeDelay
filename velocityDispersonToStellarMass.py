'''
using https://arxiv.org/pdf/1607.04275.pdf
convert stellar mass to velocity disperson
'''
import sys
import numpy as np
import ipdb as pdb
def main( stellarMass ):

    Mb = 10**(10.26)
    alpha1 = 0.403
    alpha2 = 0.293
    sigmaB = 10**2.073
    if stellarMass <= Mb:
        velocity = sigmaB*(np.float(stellarMass)/Mb)**alpha1
    else:
        velocity = sigmaB*(np.float(stellarMass)/Mb)**alpha2
    
    return velocity


if __name__ == '__main__':
    main(sys.argv[1])
