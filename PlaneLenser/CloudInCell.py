import numpy as np
import math
#import kernprof

globalFeedBackHack = False

class CloudInCell:
    """The simplest implementation of an interpolator: Cloud-in-Cell (CIC), also known as bilinear interpolation. Study this if you want to implement higher order schemes.
    
    Assumes unit steps in the indices.
    
    Defines several constants, used by ProcessSingleLens:
    
    - range2d: [[xbegin, xend], [ybegin, yend]], where the end is exclusive (as in array ranges). For CIC [[0, 2], [0, 2]]
    
    
    Implements the methods:
    - Interpolate( fList, pos2d ): interpolates on the equi-spaced sampling of fList with ranges that correspond to range2d, on the relative position pos = [delta_x, delta_y], with 0 <= delta < 1 for both delta_x and delta_y. Returns the interpolated f(x).
    - Invert ( fList, value ): your interpolation scheme must be invertible, for the time being. Return 2d position at which this value is reached, if any.
    - [no, unused] Extremize( fList ), returns the 2d position and the value at the single extremum of the interpolated f(x).
    """
    
    range2d = [[0, 2], [0, 2]]

    def Interpolate( fList, pos2d ):
        """Returns the interpolated value of f on pos2d in {{0,1},{0,1}}."""
        def wf(x):
            return abs(1 - x)
        return fList[0, 0] * wf(pos2d[0]) * wf(pos2d[1]) + \
            fList[0, 1] * wf(pos2d[0]) * wf(pos2d[1] + 1) + \
            fList[1, 0] * wf(pos2d[0] + 1) * wf(pos2d[1]) + \
            fList[1, 1] * wf(pos2d[0] + 1) * wf(pos2d[1] + 1)


    def Invert( fList1, value1, fList2, value2, canSwap = True ):
        """Since this interpolation scheme is so trivial, it can be inverted if the input has no saddle point. The last input
           parameter canSwap should be left untouched: of the inversion fails when trying to find x1 and then x0(x1), it
           will call itself once more with the x's swapped (x1 <-> x0) and canSwap = false (to avoid infinite recursion).
           
           We can invert the 2D interpolation, if we have two constraints. That is, value1 must be a value that
           can be obtained by interpolation of fList1, and idem for value2.
           value1 = (1 - x0) (1 - x1) fList[0, 0] + ....
           value2 = idem
           We know fList1, fList2, value1, value2, and solve for x0 and x1.
           The solution first gets x1( fList[12], value[12] ), and then x0(x1).
           This means that there is an arbitrary choice in the order of solving: first x0 or first x1.
           It turns out that in some cases, one of the two solutions is numerically unstable: singularities etc.
           In that case, we simply transpose the system and call the same function once more with canSwap = false.
           
           The analytical solution turns out to be high-school maths, faij = fList1[i, j], same for fb:
           x0 = (value1 - fa00 + fa00*x1 - fa01*x1)/(-fa00 + fa10 + fa00*x1 - fa01*x1 - fa10*x1 + fa11*x1)
           gives a curve on which bilinear interpolation of of fList1 gives the result f1([x0,x1]) = value1.

           x0 = (value2 - fb00 + fb00*x1 - fb01*x1)/(-fb00 + fb10 + fb00*x1 - fb01*x1 - fb10*x1 + fb11*x1)
           gives a curve on which bilinear interpolation of of fList2 gives the result f2([x0,x1]) = value2.

           If both curves intersect, we have a solution.
           (value1 - fa00 + fa00*x1 - fa01*x1)/(-fa00 + fa10 + fa00*x1 - fa01*x1 - fa10*x1 + fa11*x1) ==
           (value2 - fb00 + fb00*x1 - fb01*x1)/(-fb00 + fb10 + fb00*x1 - fb01*x1 - fb10*x1 + fb11*x1)
           
           ->
           
           (value1 - fa00 + fa00*x1 - fa01*x1) * (-fb00 + fb10 + fb00*x1 - fb01*x1 - fb10*x1 + fb11*x1) -
           (value2 - fb00 + fb00*x1 - fb01*x1) * (-fa00 + fa10 + fa00*x1 - fa01*x1 - fa10*x1 + fa11*x1) == 0

            That's a simply quadratic equation. High school.
           """
            # test if the input is sane
        myDomain = np.array([[0, 2], [0, 2]])
#        f1Min, f1Max = CloudInCell.FunctionRangeOnDomain(fList1, myDomain)
#        f2Min, f2Max = CloudInCell.FunctionRangeOnDomain(fList2, myDomain)
#
#        if (value1 > f1Max or value1 < f1Min or value2 > f2Max or value2 < f2Min):
#            return []

        guess = [0, 0];
        
        a = value1
        b = value2
#         for i in 0 1; do
#         for j in 0 1; do
#         echo "fa${i}${j} = fList1[${i}, ${j}]"
#         echo "fb${i}${j} = fList2[${i}, ${j}]"
#         done
#         done
        if ( globalFeedBackHack ): print("fList1:", fList1, "\nfList2:", fList2)
        fa00 = fList1[0, 0]
        fb00 = fList2[0, 0]
        fa01 = fList1[0, 1]
        fb01 = fList2[0, 1]
        fa10 = fList1[1, 0]
        fb10 = fList2[1, 0]
        fa11 = fList1[1, 1]
        fb11 = fList2[1, 1]

        def x0(x1):
            return (a - fa00 + fa00*x1 - fa01*x1)/(-fa00 + fa10 + fa00*x1 - fa01*x1 - fa10*x1 + fa11*x1)

        def Power(a1, a2):
            return a1**a2
        
        def Sqrt(a1):
            if ( a1 < 0): return math.inf
            return math.sqrt(a1)

        if (-((fa10 - fa11)*(fb00 - fb01)) + (fa00 - fa01)*(fb10 - fb11)) == 0.:
            if ( globalFeedBackHack ):
                np.set_printoptions(precision=16)
                print("abcCenter contains 1/", (-((fa10 - fa11)*(fb00 - fb01)) + (fa00 - fa01)*(fb10 - fb11)), "=", "(-((", "%24.15e" % fa10, " - ",  "%24.15e" % fa11, ")*(",  "%24.15e" % fb00, " - ",  "%24.15e" % fb01, ")) + (",  "%24.15e" % fa00, " - ",  "%24.15e" % fa01, ")*(",  "%24.15e" % fb10," - ",  "%24.15e" % fb11, "))")
            abcCenter = math.inf
        else:
            abcCenter = 0.5 * ((b*(-fa00 + fa01 + fa10 - fa11) - 2*fa10*fb00 + fa11*fb00 + fa10*fb01 +\
                    2*fa00*fb10 - fa01*fb10 - fa00*fb11 + a*(fb00 - fb01 - fb10 + fb11))/\
                    (-((fa10 - fa11)*(fb00 - fb01)) + (fa00 - fa01)*(fb10 - fb11)))
        if Power((fa10 - fa11)*(fb00 - fb01) - (fa00 - fa01)*(fb10 - fb11),2) == 0.:
            abcDiscriminant = math.inf
        else:
            abcDiscriminant = 0.5 * Sqrt((Power(b,2)*Power(fa00 - fa01 - fa10 + fa11,2) +\
                    Power(fa11,2)*Power(fb00,2) - 2*fa10*fa11*fb00*fb01 +\
                    Power(fa10,2)*Power(fb01,2) - 2*fa01*fa11*fb00*fb10 - 2*fa01*fa10*fb01*fb10 +\
                    4*fa00*fa11*fb01*fb10 + Power(fa01,2)*Power(fb10,2) + 4*fa01*fa10*fb00*fb11 -\
                    2*fa00*fa11*fb00*fb11 - 2*fa00*fa10*fb01*fb11 - 2*fa00*fa01*fb10*fb11 +\
                    Power(fa00,2)*Power(fb11,2) + Power(a,2)*Power(fb00 - fb01 - fb10 + fb11,2) -\
                    2*b*(-(fa00*fa11*fb00) - fa10*fa11*fb00 + Power(fa11,2)*fb00 -\
                    fa00*fa10*fb01 + Power(fa10,2)*fb01 + 2*fa00*fa11*fb01 - fa10*fa11*fb01 +\
                    Power(fa01,2)*fb10 + 2*fa00*fa11*fb10 + fa00*(fa00 - fa10 - fa11)*fb11 +\
                    a*(fa00 - fa01 - fa10 + fa11)*(fb00 - fb01 - fb10 + fb11) -\
                    fa01*(fa11*(fb00 + fb10) + fa10*(-2*fb00 + fb01 + fb10 - 2*fb11) +\
                    fa00*(fb10 + fb11))) +\
                    2*a*(fa01*fb00*fb10 - 2*fa00*fb01*fb10 + fa01*fb01*fb10 - fa01*Power(fb10,2) +\
                    fa10*fb00*(fb01 - 2*fb11) + fa00*fb00*fb11 - 2*fa01*fb00*fb11 +\
                    fa00*fb01*fb11 + fa00*fb10*fb11 + fa01*fb10*fb11 - fa00*Power(fb11,2) +\
                    fa10*fb01*(-fb01 + fb10 + fb11) +\
                    fa11*(-Power(fb00,2) - 2*fb01*fb10 + fb00*(fb01 + fb10 + fb11))))/\
                    Power((fa10 - fa11)*(fb00 - fb01) - (fa00 - fa01)*(fb10 - fb11),2))
    
        soln1 = abcCenter - abcDiscriminant
        soln2 = abcCenter + abcDiscriminant
    
        if ( globalFeedBackHack ):
            print("soln:", abcCenter, "+/-", abcDiscriminant, "=", [soln1, soln2])

        def valid(candidatex1):
            if (not np.isnan(candidatex1)) and candidatex1 >= 0 and candidatex1 < 1:
                mtx0 = x0(candidatex1)
                return (not np.isnan(mtx0)) and mtx0 >= 0 and mtx0 < 1, mtx0
            return False, None

        c1, tx0_1 = valid(soln1);
        c2, tx0_2 = valid(soln2);
        # are the two solutions the same one?
#        if abcDiscriminant == 0.:
        if abs(abcDiscriminant) < 1e-10: # relative to cell size 1.
            c2 = False
            tx0_2 = None

        result = []
        if c1:
            result.append([tx0_1, soln1])
#            print("error 1 1:", CloudInCell.Interpolate(fList1, [tx0_1 , soln1]) - value1)
#            print("error 1 2:", CloudInCell.Interpolate(fList2, [tx0_1 , soln1]) - value2)

        if c2:
            result.append([tx0_2, soln2])
#            print("error 2 1:", CloudInCell.Interpolate(fList1, [tx0_2 , soln2]) - value1)
#            print("error 2 2:", CloudInCell.Interpolate(fList2, [tx0_2 , soln2]) - value2)

#        if c1 and c2:
#            print("CONSIDER INCREASING THE RESOLUTION:\nFound double solution for interpolation inversion: [", soln1, tx0_1, "] and [", soln2, tx0_2, "]")

        if (len(result) < 1) and canSwap:
            # transposing lists: https://stackoverflow.com/questions/6473679/transpose-list-of-lists
            fList1T = np.asarray(list(map(list, zip(*fList1))))
            fList2T = np.asarray(list(map(list, zip(*fList2))))
#            print("From", fList1, "to", fList1T)
            resultT = CloudInCell.Invert(fList2T, value2, fList1T, value1, False)

            for pair in resultT:
                pair[0], pair[1] = pair[1], pair[0]
                
            result = resultT

#        if globalFeedBackHack and tx1 is None and tx0 is None:
#            print("Time wasted on", fList1, value1, fList2, value2)

        return result

    
    
    

#    @profile
    def InvertNumerically( fList1, value1, fList2, value2, canSwap = True ):
        """Since this interpolation scheme is so trivial, it can be inverted if the input has no saddle point. The last input
           parameter canSwap should be left untouched: of the inversion fails when trying to find x1 and then x0(x1), it
           will call itself once more with the x's swapped (x1 <-> x0) and canSwap = false (to avoid infinite recursion).
           
           We can invert the 2D interpolation, if we have two constraints. That's way, value1 must be a value that
           can be obtained by interpolation of fList1, and vice versa for value2.
           value1 = (1 - x0) (1 - x1) fList[0, 0] + ....
           value2 = idem
           We know fList1, fList2, value1, value2, and solve for x0 and x1.
           The solution first gets x1( fList[12], value[12] ), and then x0(x1).
           This means that there is an arbitrary choice in the order of solving: first x0 or first x1.
           It turns out that in some cases, one of the two solutions is numerical unstable: singularities etc.
           In that case, we simply transpose the system and call the same function once more with canSwap = false.
           """

        # test if the input is sane
        myDomain = np.array([[0, 2], [0, 2]])
        f1Min, f1Max = CloudInCell.FunctionRangeOnDomain(fList1, myDomain)
        f2Min, f2Max = CloudInCell.FunctionRangeOnDomain(fList2, myDomain)

        if (value1 > f1Max or value1 < f1Min or value2 > f2Max or value2 < f2Min):
            return []

        guess = [0, 0];
        
        a = value1
        b = value2
#         for i in 0 1; do
#         for j in 0 1; do
#         echo "fa${i}${j} = fList1[${i}, ${j}]"
#         echo "fb${i}${j} = fList2[${i}, ${j}]"
#         done
#         done
        if ( globalFeedBackHack ): print("fList1:", fList1, "\nfList2:", fList2)
        fa00 = fList1[0, 0]
        fb00 = fList2[0, 0]
        fa01 = fList1[0, 1]
        fb01 = fList2[0, 1]
        fa10 = fList1[1, 0]
        fb10 = fList2[1, 0]
        fa11 = fList1[1, 1]
        fb11 = fList2[1, 1]

        def JacCompute(x0, x1):
            JacdEdX_00 = fa10*(1 - x1) + fa00*(-1 + x1) + (-fa01 + fa11)*x1
            JacdEdX_01 = fa01 + fa00*(-1 + x0) - fa01*x0 - fa10*x0 + fa11*x0
            JacdEdX_10 = fb10*(1 - x1) + fb00*(-1 + x1) + (-fb01 + fb11)*x1
            JacdEdX_11 = fb01 + fb00*(-1 + x0) - fb01*x0 - fb10*x0 + fb11*x0

            JacDet = JacdEdX_00 * JacdEdX_11 - JacdEdX_01 * JacdEdX_10
            invJacDet = 1 / JacDet
            invJac00 = JacdEdX_11 * invJacDet
            invJac01 = JacdEdX_10 * invJacDet
            invJac10 = JacdEdX_01 * invJacDet
            invJac11 = JacdEdX_00 * invJacDet
            
#            eigenVector_00 = invJac00 - invJac11 - math.sqrt(math.pow(invJac00,2) + 4*invJac01*invJac10 - 2*invJac00*invJac11 + math.pow(invJac11,2))
#            eigenVector_01 = 2*invJac10
#
#            eigenVector_10 = invJac00 - invJac11 + math.sqrt(math.pow(invJac00,2) + 4*invJac01*invJac10 - 2*invJac00*invJac11 + math.pow(invJac11,2))
#            eigenVector_11 = 2*invJac10

            xs = [x0, x1]
            epsilon0 = CloudInCell.Interpolate(fList1, xs) - a
            epsilon1 = CloudInCell.Interpolate(fList2, xs) - b

            dx0 = -(epsilon0 * invJac00 + epsilon1 * invJac01)
            dx1 = -(epsilon0 * invJac10 + epsilon1 * invJac11)
            
            return dx0, dx1, epsilon0, epsilon1


        # try from 4 corners.
        limits = [[0, 1], [0, 1]]
        bestGuess = [[0, 0], 1e30]
        lastGuess = [[0, 0], 1e30]
        failed = False
        for i in range(3):
          for j in range(2):
            tx0 = i
            tx1 = j
            
            if i == 2:
                tx0 = 0.5
                tx1 = 0.5
            
            maxit = 40
            limit = 1.e-7

            if globalFeedBackHack: print("start tx:", tx0, tx1)

            mf = limit + 1
            k = 0
            while abs(mf) > limit and k < maxit:
                k += 1
                dx0, dx1, e0, e1 = JacCompute(tx0, tx1)
                errorSum = e0*e0 + e1*e1
                if errorSum < bestGuess[1]:
                    bestGuess[0] = [tx0, tx1]
                    bestGuess[1] = errorSum
                lastGuess[0] = [tx0, tx1]
                lastGuess[1] = errorSum
                if globalFeedBackHack: print("tx:", tx0, tx1, "\ndx:", dx0, dx1, "\ne:", e0, e1)
                norm = math.sqrt(dx0*dx0 + dx1*dx1)
                if ( norm > 0.1):
                    dx0 /= k * norm # use k: increasing (changing) fraction, to avoid ping-pong in the loop.
                    dx1 /= k * norm

                if (tx0 + dx0 < 0):
                    renorm = - tx0 / dx0
                    dx0 *= renorm
                    dx1 *= renorm
                if (tx1 + dx1 < 0):
                    renorm = - tx1 / dx1
                    dx0 *= renorm
                    dx1 *= renorm
                if (tx0 + dx0 > 1):
                    renorm = (1 - tx0) / dx0
                    dx0 *= renorm
                    dx1 *= renorm
                if (tx1 + dx1 > 1):
                    renorm = (1 - tx1) / dx1
                    dx0 *= renorm
                    dx1 *= renorm

                if dx0 > 0 and tx0 > limits[0][0]:
                    limits[0][0] = tx0
                if dx1 > 0 and tx1 > limits[1][0]:
                    limits[1][0] = tx1

                if dx0 < 0 and tx0 < limits[0][1]:
                    limits[0][1] = tx0
                if dx1 < 0 and tx1 < limits[1][1]:
                    limits[1][1] = tx1

                if abs(dx0) == 0 and abs(dx1) == 0:
                    dx0 = 1e-8
                    dx1 = 1e-8

                tx0 += dx0
                tx1 += dx1
                mf = abs(e0) + abs(e1)

#                d = 0.001 if np.isnan(d) else d
#                mf = f(tx1)
#                if globalFeedBackHack: print("f:", mf)
#                mf = 0.001 + limit if np.isnan(mf) else mf
#                dtx1 = - mf / d
#                if not np.isfinite(mf) or not np.isfinite(d):
#                    dtx1 = 0.1
#                    tx1 += dtx1
#                    continue
##                print("fixed dtx1:", dtx1)
#                dtx1 = abs(dtx1) if tx1 == 0 else -abs(dtx1) if tx1 == 1 else 0.1 * np.sign(dtx1) if ( abs(dtx1) > 0.1 ) else dtx1
#                # let's help the newton-raphson scheme a little bit, by making sure that the jump does not overshoot:
#                limitCounter = 0
#                lastdtx1 = dtx1
#                while abs(f(tx1 + dtx1)) > abs(mf) and limitCounter < 30 and not (tx1 + dtx1 is tx1):
#                    # two checks: limitCounter limits the decrease to 0.8^30 = 1.e-3,
#                    # and simply checking for numerical overflow: if
#                    # tx1 + dtx1 == tx1, dtx1 has become so small that it does not add in the
#                    # significant digits. Keep the last value that does change tx1.
#                    lastdtx1 = dtx1
#                    dtx1 *= 0.8
#                    limitCounter += 1

#                tx1 += dtx1
#                print (tx1, f(tx1), dtx1)
            if ( abs(mf) < limit ):
                # we converged
                break
            if ( abs(mf) < limit ):
              # we converged
              break

        if ( abs(mf) > limit ) and (limits[0][0] == 0 or limits[0][1] == 1 or limits[1][0] == 0 or limits[1][1] == 1):
#            print("Failed numerical inversion of args:", fList1, value1, fList2, value2, "will still return", tx0, tx1, lastGuess, "Although had best guess:", bestGuess)
#            exit(0)
            failed = True
        else:
            tx0 = (limits[0][0] + limits[0][1] ) * 0.5
            tx1 = (limits[1][0] + limits[1][1] ) * 0.5
#        tx0 = x0(tx1) # tx0 >= or tx0 < 0 ? Extrapolations! Real value should be found by neighbouring interpolation.
        coordinatesAreValid = (tx1 is not None) and (tx0 is not None)
#        coordinatesAreValid = coordinatesAreValid and not np.isnan(f(tx1)) and not (abs(f(tx1)) > limit or tx0 >= 1 or tx0 < 0 or tx1 >= 1 or tx1 < 0)
        if globalFeedBackHack: print("tx:", tx0, tx1)
#        if (not coordinatesAreValid) and canSwap:
#            # transposing lists: https://stackoverflow.com/questions/6473679/transpose-list-of-lists
#            fList1T = np.asarray(list(map(list, zip(*fList1))))
#            fList2T = np.asarray(list(map(list, zip(*fList2))))
##            print("From", fList1, "to", fList1T)
#            resultT = CloudInCell.InvertNumerically(fList2T, value2, fList1T, value1, False)
#
#            for pair in resultT:
#                pair[0], pair[1] = pair[1], pair[0]
#
#            if len(resultT) > 0:
#                tx0 = resultT[0]
#                tx1 = resultT[1]
#
##            if tx1 is None and tx0 is None:
##                print("Time wasted on", fList1, value1, fList2, value2)
#            # important to return from here, and no longer apply the test below:
#            # test already applied in the inner CloudInCell.Invert,
#            # and we know that here, in the outer call, we are in an unstable
#            # situation with zero denominators.
#        elif (not coordinatesAreValid):
#            if abs(f(tx1)) > limit or tx0 >= 1 or tx0 < 0 or tx1 >= 1 or tx1 < 0 :
##            print("HELP!", fList1, value1, fList2, value2, "FAILED.\n\n")
#                return []
        return [[tx0, tx1]] if not failed else []


    def Extremize( f ):
        """Returns the position of the solution to grad f == 0, if it exists.
           Note that for CIC, the value only exists if it is a saddle point."""
        denominator = f[0, 0] + f[1, 1] - f[0, 1] - f[1, 0]
        yNum = f[0, 0] - f[1, 0]
        xNum = f[0, 0] - f[0, 1]

        noValue = denominator == 0

        if not noValue:
            xPos = xNum / denominator
            yPox = yNum / denominator

            noValue = xPos < 0 or xPos >= 1 or yPos < 0 or yPos >= 1

        if not noValue:
            return xPos, yPox

        return None, None


    def FunctionRangeOnDomain( fList, domain ):
        """Provides two extreme values of the interpolated function in the domain. In the CIC case, simply the largest and smallest values in the list. """

        minVal = fList[domain[0, 0], domain[1, 0]]
        maxVal = minVal
        
        for ix in range(domain[0, 0], domain[0, 1]):
            for iy in range(domain[1, 0], domain[1, 1]):
                if fList[ix, iy] < minVal:
                    minVal = fList[ix, iy]
                if fList[ix, iy] > maxVal:
                    maxVal = fList[ix, iy]

        return minVal, maxVal

if __name__ == "__main__":
    tester = CloudInCell.Interpolate(np.asarray([[ 535.41940381,  535.41940381], [ 535.41940381,  535.41940381]]), np.asarray([0.7438185170487468, 0.5]))
    print(tester)
