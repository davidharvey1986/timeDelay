import math
import numpy as np

def GaussianSmoothing(nGrid, defs):
    """
    Returns a function that computes the gaussian smoothing filter
    as a function of wavenumber. Needs nGrid and sigma.
    """

    # sigma in integer units, waveNumberMod also.
    # kernel of gaussian convolution: rho_k -> rho_k g_k with g_k == Exp[-sigma^2 k^2 / 2], sigma = n * dx = n * L / N, k = m * 2 pi / L -> sigma^2 k^2 / 2 == 2 pi^2 n m / N^2
    sigma = defs["sigma"]
    twoPi2Sigma2OverN2 = math.pi**2 / (nGrid**2) * sigma**2
    def result(waveNumber, waveNumberMod2):
        return np.exp(-twoPi2Sigma2OverN2 * waveNumberMod2)

    return result, "Gaussian with deviation " + str(sigma)

def TophatFilter(nGrid, defs):
    """
    Returns a function that computes the amplitude of the
    tophat filter as given wavenumber. nGrid is unused.
    """
    cutoff = defs["cutoff"]
    def result(waveNumber, waveNumberMod2):
        amp = 1
        if any(abs(it) > cutoff for it in waveNumber):
            amp = 0
        return amp

    return result, "Tophat filter with cutoff " + str(cutoff)

def CloudInCellConvolution(nGrid, defs):
    """
    Returns a function that computes Product_i [sin(pi k_i / 2 k_N)/(pi k_i / 2 k_N) ]^order, i.e. the fourier transform of cloud-in-cell, triangular shaped cloud, and whatever. This can be your deconvolution for bilinear interpolation == CIC,
        simply provide order = -2. Simulate bilenear interpolation by setting order = 2.
        
        See eq. 13 in https://arxiv.org/pdf/0804.0070.pdf
    """
    n = defs["order"]

    cloudSize = defs["cloudSize"] if "cloudSize" in defs else 1

    # kN = π/H = 2π/(2H), H == grid spacing = 1 / N. Larger clouds? Effectively larger grid spacing.
    kN = math.pi * nGrid / cloudSize

    piOver2kN = math.pi / (2 * kN)

    def result(waveNumber, waveNumberMod2):
        amp = 1
        for ki in waveNumber:
            k = ki * piOver2kN
            amp *= np.sinc(k)
#            print(ki, k, np.sinc(k), amp)
        return math.pow( amp, n)

    return result, "CloudInCellConvolution of order " + str(n) + ", cloud size " + str(cloudSize)

class PolyFilters:
    filterDict = {
        "gaussian" : GaussianSmoothing,
        "tophat" : TophatFilter,
        "CIC" : CloudInCellConvolution
    }
    def __init__(self, nGrid, defsList):
        self.defs = defsList
        self.filters = []
        asStrings = [];
        for it in defsList:
            if it["kind"] in self.filterDict:
                nextFilter, asString = self.filterDict[it["kind"]](nGrid, it)
                asStrings.append(asString)
                self.filters.append(nextFilter)
        filterWord = "filters:"
        if len(self.filters) == 1:
            filterWord = "filter:"
        elif len(self.filters) == 0:
            filterWord = "filters."
        print("###########################################")
        print("# PolyFilters compiled", len(self.filters), filterWord)
        for it in asStrings:
            print("#", it)
        print("###########################################")

    def __call__(self, waveNumber, waveNumberMod2):
        result = 1
        for it in self.filters:
            result *= it(waveNumber, waveNumberMod2)
        return result

    def size(self):
        return len(self.filters)
