import numpy as np

def Boltzmann_Inversion(kbT,dist):
  U_BI=-kbT*np.log(dist)
  return U_BI

def get_target_dict(kbT, dist_dict):
    U_BI=Boltzmann_Inversion(kbT,dist_dict['dist'])-Boltzmann_Inversion(kbT,dist_dict['dist'])[-1]
    bin_centers=dist_dict['bin_centers']
    return bin_centers, U_BI