#ASTR5400 term project
#author: Lingjian Chen

import numpy as np
import matplotlib.pyplot as plt

n=1

def f(z):
    return 1 - (1/6.)*z**2 + (n/1200.)*z**4

def df(z):
    return  - (1 / 3.) * z  + (n*4 / 1200.) * z ** 3

#taylor expansion region
step = 1e-4 # step size of z
z_center = 1e-3 # size of central region
z = np.linspace(0, z_center, int(z_center/step)) # z_i's in central region


# w and dw/dz in central region
w = f(z)
dw_dz = df(z)

# starting point for numerical integral
z_now = z[-1]
w_now = w[-1]
dw_dz_now = dw_dz[-1]

############# NUMERICAL INTEGRAL #################

while w_now>0: #truncate at w==0
    z_prev = z_now
    w_prev = w_now
    dw_dz_prev = dw_dz_now

    # UPDTAE by single step Euler method
    z_now = z_prev + step
    dw_dz_now = dw_dz_prev + step*(-w_prev**n - 2./z_prev*dw_dz_prev)
    w_now = w_prev + dw_dz_prev*step

print(z[-1])

########## PLOTS #################

plt.plot(z,w)
plt.show()
#
# # density profile
# rho_c = 1
# rho = rho_c*w**n
# plt.plot(z,rho)
# plt.show()
#
# # pressure profile
# K = 1
# P = K*rho**(1+1./n)
# plt.plot(z,P)
# plt.show()