import numpy as np
def grav_const():
    '''                                                                         
    Return grav constant in Pc / M_sun (km/s)^2                                 
    '''

    return 0.0042992854
def virial_radius( mass, redshift, overdensity=200 ):
    '''                                                                         
    Given the virial mass, work out the radius at the overdensity               
    given                                                                       
                                                                                
    return the virial radius in kpc                                             
    '''
    return (mass/(4./3.*np.pi* \
            critical_density( z_lens=redshift)\
            *overdensity))**(1./3.)/1e3

def scale_radius( mass, concentration, redshift, overdensity=200. ):
    '''                                                                         
    Determine the scale radius from the mass of a halo                          
                                                                                
    inpits: mass in units M_SUN                                                 
    overdensity : the radius at which the radius is determine,                  
    default is 200 (~virial)                                                    
                                                                                
    return in kpc                                                               
    '''


    virial_rad = virial_radius( mass, redshift)


    scale_radius = virial_rad / concentration

    return scale_radius


def critical_density( z_lens=0.3):
    '''
    PURPOSE : Function that calculates the critical
            density at a given lens                  
            Assuming a cosmology of
            omega_matter = 0.3, omega_lambda = 0.7, 
            hubble=0.7
            and negligble radiation and curvature                                          

    Arguments :                                                                    
            z_lens : redshift of the lens 

    RETURNS : Critical density in units of M_sun / pc^3

    '''
    G = grav_const() # in (km/s)^2 pc M^-1
    hubble = 70.0/1e6 # In km/s/pc
  

    hubble_at_z = hubble*np.sqrt(0.3*(1.0+z_lens)**3+0.7)
    
    criticalDensity = 3.0/(8.0*np.pi*G) * hubble_at_z**2
    
    return criticalDensity
