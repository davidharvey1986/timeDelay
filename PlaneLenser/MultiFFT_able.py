import pyfftw
import numpy as np
import random


import PlaneLenser.FFT_able



class FFTZeroFields(Exception):
    pass

class FFTWrongState(Exception):
    pass

class FFTWrongSize(Exception):
    pass

class FFTNegativeFrequenciesInLastComplexDimension(Exception):
    pass

class MultiFFT_able_TestFailed(Exception):
    pass

def ListStr(li):
    return "[%s]" % ", ".join(map(str, li)) # https://stackoverflow.com/a/5445983

class MultiFFT_able:
    """A class that holds m fields of n-dimensional real-valued data in configuration space or complex valued data in Fourier space.

       You can transform between these two states with R2C() and C2R() (complex-to-real and vice versa.)
       As this project is expected to always start from real valued data, the normalization
       1/(number of points) is applied during the C2R() call. This is arbitrary,
       it could have been applied in the R2C() call.
       If you want manual control, call C2R_unnormalized().

       This class uses FFTW under the hood, so it uses exactly the FFTW data layout. Read their docs if you need:
       size of the last dimension when in fourier state, will be (n/2 + 1) complex values, with n the size of the
       last dimension in configuration space.

       It will have iterators some day.

       Data members with names that start with an underscore, are considered 'private': can't touch this!
    """
    pass

    __configState = 0
    __fourierState = 1
    __defaultDTypeReal = 'float64'
    __defaultDTypeComplex = 'complex128'

    def Copy(self):
        """Returns a deep copy of the object, hence doubling the memory use."""
        result = MultiFFT_able(self._nFields, self._layout, self.StateToString( self._state ) )
        result._data[:] = self._data[:]
        return result

    def __init__ (self, nFields, layout, state = "configuration"):
        """Pass an iterable that contains the size of each dimension, e.g. [1024, 768], and the state of either \"configurarion\" (default) or \"fourier\"."""


        self._state = self.StringToState(state)

        self._nFields = int(nFields);
        if ( nFields < 1 ):
            raise FFTWrongSize("Need to have at least one field. Your nFields: " + str(nFields));


        self._layout = [int(x) for x in layout]
        self._layoutHalf = [int(x) // 2 for x in layout]

        self.nPoints = 1
        for x in self._layout:
            self.nPoints *= x

        self.normalization = 1. / self.nPoints

        self.nD = len(self._layout)

        # had to follow this: https://github.com/pyFFTW/pyFFTW/issues/29

        # // integer division
        lastDimSizeComplex = (self._layout[-1] // 2 + 1)
        # [n0, n1, n2]
        self._layoutMany = self._layout[:]
        self._layoutMany[-1] = lastDimSizeComplex;

        self._layoutManyRealPadded = self._layoutMany[:]
        self._layoutManyRealPadded[-1] *= 2 # symmetry in last dim: z* = z

        self._data = pyfftw.zeros_aligned(self._layoutMany + [nFields], dtype = self.__defaultDTypeComplex )
        # https://stackoverflow.com/a/12116854  ellipsis

        self._realViewPadded = self._data.view(self.__defaultDTypeReal).reshape(*(self._layoutManyRealPadded + [nFields]))
        self._realView = self._realViewPadded[..., :self._layout[-1], :nFields]
        self._complexView = self._data.view(self.__defaultDTypeComplex)

        fftAxes = [x for x in range(len(self._layout))]

        self._planR2C = pyfftw.FFTW(self._realView, self._complexView, axes = fftAxes, direction = "FFTW_FORWARD");
        self._planC2R = pyfftw.FFTW(self._complexView, self._realView, axes = fftAxes, direction = "FFTW_BACKWARD");


    def StateToString(self, instate):
        return str("configuration" if instate == self.__configState else "fourier")

    def StringToState(self, instate):
        result = -1;
        if instate == "configuration":
            result = self.__configState
        elif instate == "fourier":
            result = self.__fourierState
        else:
            raise FFTWrongState("Wrong starting state for FFT_able: " + repr(instate));
        return result


    def __str__(self):
        return ("MultiFFT_able(" +
            ListStr(self._layout)
            + ", " +
            str("configuration" if self._state == 1 else "fourier") + ")\n"
            "  Actual layout: " + ListStr(self._layoutMany) + "\n" +
             ("  data shape: " + ListStr( self._data.shape )) + "\n" +
             ("  realView shape: " + ListStr( self._realView.shape )) + "\n" +
             ("  complexView shape: " + ListStr( self._complexView.shape ))
            )

    def BluntDataLoop(self, inputFunction, currentDimension = 0, hyperPlane = None):
        """Call the function that you supply as first argument, passing a single
           rod in the n-dimensional data to your function. You loop over a single rod.
           No indices / wave numbers passed. Good for global re-scaling and
           randomization. """

        if ( currentDimension == 0 ) and ( hyperPlane == None ) :
            if self._state == self.__configState:
                hyperPlane = self._realView
            else:
                hyperPlane = self._complexView

        if currentDimension == self.nD - 2:
            # the rods in this 2d plane
            for i in hyperPlane:
                inputFunction(i)
        else:
            for i in hyperPlane:
                self.BluntDataLoop(inputFunction, currentDimension + 1, i)

    def RodLoopWithWaveNumbers(self, inputFunction, currentDimension = 0, hyperPlane = None, indexList = []):
        """Call the function that you supply as first argument, passing to your function
           a vector reference to a gridpoint (complex or real, depending on state) and
           wavenumber k at which it lives.
           Your function need not return a value, as it is passed a vector
           reference. You can do the modifications in place in your function.
           Not sure this is the most efficient way of doing things, but it might be.
           """

        if ( len(indexList) < self.nD ):
            indexList = [0] * self.nD

        if ( currentDimension == 0 ) and ( hyperPlane == None ) :
            if self._state == self.__configState:
                hyperPlane = self._realView
            else:
                hyperPlane = self._complexView

        if currentDimension == self.nD - 1:
            # the rods in this 2d plane
            waveNumbers = self.IndicesToWaveNumbers(indexList)
            for index, entry in enumerate(hyperPlane):
                indexList[currentDimension] = index;
                waveNumbers[currentDimension] = index - self._layout[currentDimension] if index > self._layoutHalf[currentDimension] else index

                inputFunction(entry, waveNumbers)

        else:
            for index, rod in enumerate( hyperPlane ):
                indexList[currentDimension] = index;
                self.RodLoopWithWaveNumbers(inputFunction, currentDimension + 1, rod, indexList)


    def RandomizeRod(self, rod):
#        for x in rod:
#            x = random.random()
        for d in rod:
            for x in range(len(d)):
                d[x] = random.random()

    def Randomize(self):
        random.seed(1)
        self.BluntDataLoop( self.RandomizeRod )

    def Normalize(self):
        self.Rescale(self.normalization)

    def R2C(self):
        if ( self._state != self.__configState ):
            raise FFTWrongState("Calling R2C while we are not in configuration space.")
        self._planR2C()
        self._state = self.__fourierState

    def C2R(self):
        self.C2R_unnormalized()
        self.Normalize()

    def C2R_unnormalized(self):
        if ( self._state != self.__fourierState ):
            raise FFTWrongState("Calling C2R while we are not in fourier space.")
        self._planC2R(normalise_idft = False) # We want no surprises: remove the built-in renormalization of pyFFTW, and do it ourselves.)
        self._state = self.__configState

    def __getitem__(self, args):
        """Getting arbitrary dimensional data, but using the signed frequency / position values."""
        shifted = tuple(self.WaveNumbersToIndices(args))
        if ( self._state == self.__configState ):
            return self._realView[shifted]
        else:
            return self._complexView[shifted]

    def __setitem__(self, args):
        """Getting arbitrary dimensional data, but using the signed frequency / position values."""
        shifted = tuple(self.WaveNumbersToIndices(args))
        if ( self._state == self.__configState ):
            return self._realView[shifted]
        else:
            return self._complexView[shifted]

    def WaveNumbersToIndices(self, args, result = []):
        """Convert wavenumbers (-n/2 + 1, ... -1, 0, 1 ... n /2 ) to indices (0, ... n - 1). Optionally provide a pre-allocated result list."""
        if not (len(args) >= self.nD):
            raise FFTWrongSize("Called operator[] with wrong number of indices, dimensions: " + str(self.nD))

        if (len(result) < len(args) ):
            result = [0] * len(args)

        for i in range( min(len(args), self.nD) ):
            result[i] = (self._layout[i] + args[i] if args[i] < 0 else args[i])
            if i == self.nD - 1 and args[i] < 0 and self._state == self.__fourierState:
                raise FFTNegativeFrequenciesInLastComplexDimension("Fourier transform of real field has hermitian symmetry: only positive frequencies are stored.")

        # asking for a single field entry, or the vector of nFields?
        if len(args) > self.nD:
            result[self.nD] = args[self.nD]
#        print("Mapping " + ListStr(args) + " to " + ListStr(result));
        return result

    def IndicesToWaveNumbers(self, args, result = []):
        """Convert indices (0, ... n - 1) to wavenumbers (-n/2 + 1, ... -1, 0, 1 ... n /2 ). Optionally provide a pre-allocated result list."""
        if not (len(args) == self.nD):
            raise FFTWrongSize("Called operator[] with wrong number of indices, dimensions: " + str(self.nD))

        if (len(result) < len(args) ):
            result = [0] * len(args)

        for i in range( min(len(args), self.nD) ):
            result[i] = (args[i] - self._layout[i] if args[i] > self._layoutHalf[i] else args[i])
#        print("Mapping " + ListStr(args) + " to " + ListStr(result));

        # asking for a single field entry, or the vector of nFields?
        if len(args) > self.nD:
            result[self.nD] = args[self.nD]

        return result

    def Rescale(self, scale):
        def Helper(rod):
            for d in rod:
                for i in range(len(d)):
                    d[i] *= scale

        self.BluntDataLoop(Helper)

def MultiFFT_able_OneToMany(one, howmany):
    result = MultiFFT_able(howmany, one._layout, one.StateToString(one._state))

    def CopyEntry(entry, wavenumbers):
        thisvec = result[wavenumbers]
        for i in range(howmany):
            thisvec[i] = entry
#        thisvec[:] = [entry for i in range(howmany)]
        return 10041982

    one.RodLoopWithWaveNumbers(CopyEntry, readonly = True)
    return result

def PrintXY(x, y):
    print(x, y)
    return x

if __name__ == "__main__":
    """For now, this is just a long linear list of individual test for the MultiFFT_able class."""

    try:
        x = MultiFFT_able(1, [2., 2, 3], "configuration xyz")
    except FFTWrongState:
        pass
    else:
        raise MultiFFT_able_TestFailed("FFTWrongState should have been raised.")

    try:
        x = MultiFFT_able(2, ["a", 2, 3])
    except ValueError:
        pass
    else:
        raise MultiFFT_able_TestFailed("ValueError should have been raised.")

    y = MultiFFT_able(2, [1024.5, 1022.1])
    print(y, "\n")

    y.Randomize()

    try:
        y.C2R()
    except FFTWrongState:
        pass
    else:
        raise MultiFFT_able_TestFailed("FFTWrongState should have been raised.")

    yCopy = y.Copy()
    print(y[0, 0], "==", yCopy[0, 0])
    print(y[0, 1], "==", yCopy[0, 1])

    print("First few entries, of type:", type(y[0,0]))
    print("0, 0:", y[0,0])
    print("-3, -4:", y[-3,-4])
    print("3, 4:", y[3,4])
    print("-3, 4:", y[-3,4])
    print()

    y00AtStart = np.copy(y._realView[0,0])
    y.R2C()
    y00afterR2C = np.copy(y._realView[0,0])

    if y00AtStart[0] == y00afterR2C[0] or y00AtStart[1] == y00afterR2C[1]:
        raise MultiFFT_able_TestFailed("[0, 0] component was unchanged during R2C: is it really in-place?")

    try:
        y.R2C()
    except FFTWrongState:
        pass
    else:
        raise MultiFFT_able_TestFailed("FFTWrongState should have been raised.")

    print("First few entries, of type:", type(y[0,0]))
    print("0, 0:", y[0,0])
    print("3, 4:", y[3,4])
    print("-3, 4:", y[-3,4], "\n")

    try:
        print("-3, -4:", y[-3,-4])
    except FFTNegativeFrequenciesInLastComplexDimension:
        pass
    else:
        raise MultiFFT_able_TestFailed("FFTNegativeFrequenciesInLastComplexDimension should have been raised.")

    y.C2R()
    y00afterR2CC2R = np.copy(y._realView[0, 0])

    if not np.isclose(y00AtStart[0], y00afterR2CC2R[0]) or not np.isclose(y00AtStart[1], y00afterR2CC2R[1]):
        raise MultiFFT_able_TestFailed("[0, 0] component was changed during R2C - C2R. Normalization wrong?")


    print(y[0, 0], "=?", yCopy[0, 0])
    print(y[0, 1], "=?", yCopy[0, 1])

    # next up: verify that back and forth actually preserves the values
    try:
        np.testing.assert_allclose(y._realView, yCopy._realView, rtol = 1.e-7);
    except AssertionError:
        print("They should have been all close!")
        raise
    else:
        print("Seems to work: numpy does not complain about the values of our arrays")

    print(y[0, 0], "==", yCopy[0, 0])
    print(y[0, 1], "==", yCopy[0, 1])

#    visually inspect the conversion between indices and frequencies, don't know how to test that other than copying the
#    conversion functions and redundantly testing that they are equal...
#    y2 = MultiFFT_able([8, 8, 8]);
#    y2.Randomize()
#    y2.RodLoopWithWaveNumbers(PrintXY)
#    y2.R2C()
#    y2.RodLoopWithWaveNumbers(PrintXY)

#    y.Rescale(2.5)
#    allEqual = True
#    # this is very very slow, so lets skip most of it..
#    for ix in range(y._layout[0] // 5):
#        for iy in range(y._layout[1] // 5):
#            wn = tuple(y.IndicesToWaveNumbers([ix, iy]));
#            for fi in range(2):
#                wnfn = (wn[0], wn[1], fi)
##                print(y[wnfn], yCopy[wnfn] * 2.5)
#                allEqual = allEqual and np.isclose(y[wnfn], yCopy[wnfn] * 2.5)
#
#    if not allEqual:
#        raise MultiFFT_able_TestFailed("MultiFFT_able.Rescale fails test.")
#


    # next up: set all fields in a multifft_able from a single fft_able
    y0 = FFT_able.FFT_able([1024, 1024])
    y0.Randomize()

    for i in range(2):
        if i == 1:
            y0.R2C()

        yn = MultiFFT_able_OneToMany(y0, 7)


        allEqual = True
        # this is very very slow, so lets skip most of it..
        for ix in range(y._layout[0] // 5):
            for iy in range(y._layout[1] // 5):
                wn = tuple(y.IndicesToWaveNumbers([ix, iy]));
                for fi in range(yn._nFields):
                    wnfn = (wn[0], wn[1], fi)
                    allEqual = allEqual and np.isclose(y0[wn], yn[wnfn])

        if not allEqual:
            raise MultiFFT_able_TestFailed("MultiFFT_able_OneToMany fails test.")



    print("Made it to the end: all tests passed. Congratulations!")

