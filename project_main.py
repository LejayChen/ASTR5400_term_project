#ASTR5400 term project
#author: Lingjian Chen

import numpy as np
import matplotlib.pyplot as plt

n=3

def f(z):
    return 1 - (1/6.)*z**2 + (n/1200.)*z**4

def df(z):
    return  - (1 / 3.) * z  + (n*4 / 1200.) * z ** 3

#taylor expansion region
z = np.linspace(0,1,1000)
w = f(z)
dw_dz = df(z)

z_now = z[-1]
w_now = w[-1]
dw_dz_now = dw_dz[-1]

while w_now>0:
    z_prev = z_now
    w_prev = w_now
    dw_dz_prev = dw_dz_now

    z_now = z_prev + 0.0001
    w_now = w_prev + dw_dz_prev*0.0001
    #dw_dz_now = (w_now - w_prev)/(z_now - z_prev)

    z = np.append(z, z_now)
    w = np.append(w, w_now)

plt.plot(z,w)
plt.show()
