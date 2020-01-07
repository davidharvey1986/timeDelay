import numpy as np
import random
import PlaneLenser.CloudInCell
import matplotlib.pyplot as plt
import math

def PointMass(pos, PosObj, params = {'shift':0.5}):
    # Point Mass function
    PosObj = [PosObj[0]+params['shift'], PosObj[1]+params['shift']]
    if (pos == PosObj):
        return 1
    else:
        return 0

def SIS(pos, PosObj, params = {'e': 0}):
    # Singular Isothermal Sphere
    xi2 = (pos[0]-PosObj[0])**2 + (pos[1]-PosObj[1])**2
    return 1./(np.sqrt(xi2) + params['e'])

def NFW(pos, PosObj, params = {'rs': 1., 'e': 0}):
    # Navarro-Frenk-White profile, scale radius rs
    rs = params['rs']
    xi = np.sqrt((pos[0]-PosObj[0])**2 + (pos[1]-PosObj[1])**2)
    Xi = 0
    if (xi<rs):
        Xi = 2.*rs/np.sqrt(rs*rs-xi*xi)*np.arctanh(np.sqrt((rs-xi)/(rs+xi)))
    elif (xi>rs):
        Xi = 2.*rs/np.sqrt(-rs*rs+xi*xi)*np.arctan(np.sqrt((-rs+xi)/(rs+xi)))
    else:
        Xi = 2.*rs/np.sqrt(-rs*rs+xi*xi+params['e'])*np.arctan(np.sqrt((-rs+xi+params['e'])/(rs+xi+params['e'])))
        return rs**3/(-rs*rs+xi*xi+params['e'])*(1-Xi)
    return rs**3/(-rs*rs+xi*xi)*(1-Xi)

def Sersic(pos, PosObj, params = {'n': 1., 'xie': 1.}):
    # Sérsic profile, effective radius xie, Sérsic shape parameter n
    n, xie = params['n'], params['xie']
    xi = np.sqrt((pos[0]-PosObj[0])**2 + (pos[1]-PosObj[1])**2)
    if ((0.5<n) and (n<10)):
        bn = 2*n - 1/3 + 4/(405*n)
    return np.exp(bn*(1-(xi/xie)**n))

def ShowMockLens(dims, shape = PointMass, params = {'e':0,'rs':1.,'n':1.,'xie':1., 'shift':0.5}):
    lens = LensDensity(dims, shape, params)
    print('Max and min of field: ',np.max(lens),np.min(lens))
    plt.pcolor(lens)
    plt.colorbar()
    plt.show()
    return lens

def SetSingleKeyInLensParams(params, key, value):
    if key not in params:
        params[key] = value

def SetMissingKeysInLensParams(params):
    SetSingleKeyInLensParams(params, 'sideLength', 1)
    SetSingleKeyInLensParams(params, 'e', 0)
    SetSingleKeyInLensParams(params, 'rs', 1)
    SetSingleKeyInLensParams(params, 'n', 1)
    SetSingleKeyInLensParams(params, 'xie', 1)
    SetSingleKeyInLensParams(params, 'shift', 0.5)

def LensDensity(dims, shape = PointMass, params = {}):
    SetMissingKeysInLensParams(params)
    print(params)
    surface = np.zeros(dims)
    # position of the object creating the lensing
    #ObjectX = random.gauss(dims[0]/2, dims[0]/4) # or .uniform
    #ObjectY = random.gauss(dims[1]/2, dims[1]/4)
    # to avoid singularity: -0.5
    dimScale = params['sideLength'] / dims[0]
    shift = params['shift']
    PosObj = [(dims[0]/2-shift) * dimScale, (dims[1]/2-shift) * dimScale]
    print('Center Main Lens: ',PosObj)
    for x in range(dims[0]):
        for y in range(dims[1]):
            surface[x, y] = shape([x * dimScale, y * dimScale], PosObj, params)
    return surface

def CutoffAtRelativeRadius(density, relrad):
    dims = density.shape
    dimScale = 1. / dims[0]
    xCenter = 0.5 * (1 - dimScale)
    yCenter = (dims[1] / 2 - 0.5) / dims[1]
    for x in range(dims[0]):
        rx = x / (1. * dims[0]) - xCenter
        for y in range(dims[1]):
            ry = y / (1. * dims[1]) - yCenter
            r = math.sqrt(rx*rx + ry*ry)
            if ( r > relrad):
                density[x, y] = 0.


def FactorSIS(sigma = (2/3)*1e-3, zl = 0.3, zls = 0.2, zs = 0.5):
    cOverH0 = 3e3
    Dls = 2*cOverH0/((1+zls))*(1-1/(1+zls)**0.5)
    Ds = 2*cOverH0/((1+zs))*(1-1/(1+zs)**0.5)
    Dl = 2*cOverH0/((1+zl))*(1-1/(1+zl)**0.5)
    print("Dls:", Dls, "Ds:", Ds, "Dl:", Dl)
    thetaE = 4*np.pi*sigma**2*Dls/Ds
    xiE = Dl*thetaE
    f = Dl*thetaE/2
    print('FactorSIS: ',f, 'xiE: ',xiE)
    return f, xiE

class LensModel:
    def __init__(self):
        pass


if __name__ == "__main__":
    lens = ShowMockLens([30, 30], Sersic)




























