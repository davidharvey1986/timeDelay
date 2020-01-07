from scipy.integrate import quad
import numpy as np
import math

class Distances:
    def __init__(self, zLens, zSource, OmegaM, OmegaK, H0):
        #DH Has changed this so i cna add H0
        
        self.zl = zLens         # redshift lens plane
        self.zs = zSource       # redshift source plane

        self.cOverH0 = 2.99792458e5/H0                  # units of c/H0: Mpc/h
        self.OmegaV = 1 - OmegaM - OmegaK          # density parameter of vacuum
        self.OmegaM = OmegaM                       # density parameter of matter/dust
        self.OmegaK = OmegaK
        self.K = - OmegaK / self.cOverH0**2        # curvature parameter: (H0/c)^2*(OmegaV+OmegaM-1)
    
        self.Dls = self.RedshiftToComovAngDistance(self.zl, self.zs) # angular diameter distance from lens to source
        self.Ds = self.RedshiftToComovAngDistance(0.0, self.zs)      # angular diameter distance of source
        self.Dl = self.RedshiftToComovAngDistance(0.0, self.zl)      # angular diameter distance of lens

    def __str__(self):
        return "Distances - Dls: " + str(self.Dls) + " Mpc/h, Ds: " + str(self.Ds) + " Mpc/h, Dl: " + str(self.Dl) + " Mpc/h, Kernel: " + str(self.DistanceKernelSurfaceToPotential()) + " Mpc/h"

    def integrand(self, z):
        zp1 = z + 1
        return 1./np.sqrt(zp1**3 * self.OmegaM + zp1**2. * self.OmegaK + self.OmegaV)

    def RedshiftToComovAngDistance(self, z1, z2):
        def integrand(z):
            return self.integrand(z)
        integrated = quad(integrand, z1, z2) # integrated = (interpolation integrale value, int error)
        integral = integrated[0] * self.cOverH0
        if (self.K == 0.0):
            fk = integral
        else:
            if (self.K < 0.):
                fk = np.abs(self.K)**(-0.5) * np.sinh(np.abs(self.K)**0.5 * integral)
            else:
                fk = self.K**(-0.5) * np.sin(self.K**0.5 * integral)
        return fk

    def RedshiftToAngDistance(self, z1, z2):
        # And here A. correctly added 1/(1+z) to go from comoving to angular diameter distance.
        return 1./(1. + z2) * self.RedshiftToComovAngDistance(z1, z2)

    def RedshiftToComovingDistance(self, z1, z2):
        def integrand(z):
            zp1 = z + 1
            return 1./np.sqrt(zp1**3 * self.OmegaM + zp1**2. * self.OmegaK + self.OmegaV)
        integrated = quad(integrand, z1, z2) # integrated = (interpolation integrale value, int error)
        integral = integrated[0] * self.cOverH0
        return integral
    
    def ComovingToAngDistance(self, wz1z2, z2):
        if (self.K == 0.0): fk = wz1z2
        else:
            if (self.K < 0.): fk = np.abs(self.K)**(-0.5) * np.sinh(np.abs(self.K)**0.5 * wz1z2)
            else: fk = self.K**(-0.5) * np.sin(self.K**0.5 * wz1z2)
        return 1./(1. + z2) * fk
    
    def ComovingDistanceToRedshift(self, w, zguess=0.5): # w is the comving distance from redsift 0 to z
        def func(z):
            return w - self.RedshiftToComovingDistance(0, z)
        vfunc = np.vectorize(func)
        z_guess = zguess
        z_exit, = fsolve(vfunc, z_guess)
        return z_exit

    def DistanceKernelSurfaceToPotential(self):
        return self.Dls * self.Dl / self.Ds

    def Kernel_And_dVdZ(self, atZ):
        """
        Returns kernel value and dVolumedZ.
        Helper for constructing a list of kernels and integration
        weights from zLens to maximum zSource.
        """
        Dls = self.RedshiftToComovAngDistance(self.zl, atZ)
        Ds = self.RedshiftToComovAngDistance(0.0, atZ)
        thisKernel = Dls * self.Dl / Ds
        # dV/dz = r^2 dr/dz = a(z) r^2 / H(z)
        oneOverHz = self.cOverH0 * self.integrand(atZ)
        thisdVdz = Ds**2 / (1 + atZ) * oneOverHz
        thisdKerneldz = self.Dl * self.Dl / (Ds*Ds) * oneOverHz
        print("Kernel_And_dVdZ", thisdVdz, "Dls:", Dls, "Ds", Ds, "self z:", self.zl, "target z:", atZ)
        return thisdVdz, thisKernel, thisdKerneldz, self.Dl**2 / (1 + atZ) * oneOverHz

    def FindKernelValue(self, kv):
        """
        Finds z at which the lensing kernel has the desired value.
        No checking: dont put in stupid values.
        Helper for constructing a list of kernels and integration
        weights from zLens to maximum zSource.
        """
        thisfindz = self.zl
        _, thisKernel, thisdKerneldz, _ = self.Kernel_And_dVdZ(thisfindz)
        if thisdKerneldz == 0:
            raise Exception("What the hell? d (Dl Dls / Ds) /dz == 0.")
        while 0.5 * abs((thisKernel - kv)/(thisKernel + kv)) > 1e-14:
#                print("finding kernel", thisKernel, kv, thisfindz, 0.5 * (kv - thisKernel) / thisdKerneldz)
            thisfindzPre = thisfindz
            thisfindz += 0.5 * (kv - thisKernel) / thisdKerneldz
            if thisfindz == thisfindzPre:
                # underflow. Probably we're there.
                break
            if not math.isfinite(thisfindz):
                raise Exception("Oh no, could not find kernel value " + str(kv))
            dVdz, thisKernel, thisdKerneldz, wrongdVdz = self.Kernel_And_dVdZ(thisfindz)
        return thisfindz, thisdKerneldz, wrongdVdz #dVdz yes, return wrong value, for consistency with correction to right value later, in BinListOfIterablesAndWeights

    def ComputeIntegrationDVolumeAndKernelForLensAndMinimalKernel(self, minimalKernel, nSteps, minimalKernelDelta):
        """
        Returns a list of kernel values and volume weights.
        That is,
        kernel(zs) = [...] the lens strength,
        and
        dV/dz dV = [...] the integral volume element,
        so you can sum your results for all those lens strengths.
        First and last value of kernel should match input value of minimalKernel.
        """

        zMax = self.zs
        zSource = self.zl

        # first things first: what is the best we get?
        # Always at highest z.
        _, maxKernel, dMaxKerneldz, _ = self.Kernel_And_dVdZ(zMax)
        if maxKernel < minimalKernel:
            print("No lensing for this image: minimal value for kernel never reached.", maxKernel, "<", minimalKernel)
            return []
        
        # Second things second: what is the minimal zSource for lensing?


        zSource, dStartKerneldz, _ = self.FindKernelValue(minimalKernel)
       
        # now, cut that in nSteps equal-weight pieces.
        nSteps = max(1, nSteps)
        dKernel = (maxKernel - minimalKernel) / (nSteps)
        if dKernel < minimalKernelDelta:
            dKernel = minimalKernelDelta
            smallerNSteps = max(1, int((maxKernel - minimalKernel) / minimalKernelDelta + 1))
            minimalKernel = maxKernel - (smallerNSteps) * dKernel
            nSteps = smallerNSteps

        print("# Getting integration kernel and weight from", minimalKernel, "@z", zSource, "to", maxKernel, "@z", zMax, "with steps dKernel:", dKernel)
        
        # and construct the values.
        result = []
        thisz = zSource
        weightCounter = 0
        for i in range(nSteps):
            # Hm, yeah, we're taking the kernel
            # value at the upper limit of the integration bin.
            # Because larger values give more signal.
            nextKernel = minimalKernel + (1 + i) * dKernel
            lastz = thisz
            thisz, _, dVdz = self.FindKernelValue(nextKernel)
            thisWeight = dVdz * (thisz - lastz)
            weightCounter += thisWeight
            result.append({"kernel" : nextKernel, "weight" : thisWeight, "z" : thisz})

        if weightCounter > 0:
            for it in result:
                it["weight"] /= weightCounter
                print(it)
    
        return result
