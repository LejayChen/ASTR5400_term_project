# ASTR5400 term project
# Fall 2017
# author: Lingjian Chen

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from astropy.io import ascii

def f(zed):
    ''' w(z) in central region'''

    return 1 - (1/6.)*zed**2 + (n/1200.)*zed**4


def df(zed):
    '''dw/dz in central region'''

    return - (1 / 3.) * zed + (n*4 / 1200.) * zed ** 3

############ SETUP ##########################
# set up for central region

n = 3.  # polytropic index
step = 1e-4  # step size of z --> about 60000 z's
z_center = 1e-3  # size of central region

# z, w and dw/dz in central region
z = np.linspace(0, z_center, int(z_center/step))  # z_i's in central region
w = f(z)
dw_dz = df(z)

# starting point for numerical integral
# for two-step Adams-Bashforth I need two point to start with,
# and I picked last two points in the central approximation
z_now = (z[-2], z[-1])
w_now = (w[-2], w[-1])
dw_dz_now = (dw_dz[-2], dw_dz[-1])

# import yrec model
yrec = ascii.read('yrec.csv')

############# NUMERICAL INTEGRAL #################

while w_now[1] > 0 and z_now[1] < 20:  # truncate at w==0
    z_prev = z_now
    w_prev = w_now
    dw_dz_prev = dw_dz_now

    # UPDTAE by two-step method
    # update: calculate a set of new values for next step, and then move second value in each tuple to first one,
    # and then insert new value into second value in each tuple
    z_now = (z_prev[0]+step, z_prev[1]+step)
    dw_dz_now = (dw_dz_prev[1], dw_dz_prev[1] + 3./2*step*(-w_prev[1]**n - 2./z_prev[1]*dw_dz_prev[1])-1./2*step*(-w_prev[0]**n - 2./z_prev[0]*dw_dz_prev[0]))
    w_now = (w_prev[1], w_prev[1] + 3./2*dw_dz_prev[1]*step-1./2*dw_dz_prev[0]*step)

    # append the z,w,dw_dz list
    z = np.append(z, z_now[1])
    w = np.append(w, w_now[1])
    dw_dz = np.append(dw_dz, dw_dz_now[1])

# cut off last value which due to the truncation point is w<0
z = z[0:-1]
w = w[0:-1]
dw_dz = dw_dz[0:-1]

# print out z_n to check
z_n = z[-1]
print('z_n:', z_n)

########## w(z) PLOT #################
plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=16)
plt.subplots(figsize=(7.5, 6))

plt.plot(z, w)
plt.xlim([0, z[-1]+0.5])
plt.ylim([0, 1.05])
plt.xlabel('z', fontsize=16)
plt.ylabel('w', fontsize=16)
plt.savefig('w_z.png')
plt.show()

############### PHYSICAL PROPERTIES #############

mu = 0.62 # mean molecular weight [dimensionless]
beta1 = 0.25 # a guess of gas pressure fraction [dimensionless]
beta2 = 0.982 # a guess of gas pressure fraction [dimensionless]
rho_c1 = 2.*1e3 # central density (kg/m^3)
rho_c2 = 150.*1e3 # central density (kg/m^3) (same as in solar core)
Re = 8315.   # gas constant (J K^-1 kg^-1)
a = 7.5657e-16  # radiation constant (J m^-3 K^-4)
G = 6.67e-11  # gravitational constant (m^3 kg^-1 s^-2)

K1 = (3*Re**4/a*mu**4)**(1./3)*((1.-beta1)/beta1**4.)**(1./3) # mks system unit
A1 = np.sqrt(4*np.pi*G/((n+1)*K1)*rho_c1**((n-1.)/float(n))) # mks system unit

K2 = (3*Re**4/a*mu**4)**(1./3)*((1.-beta2)/beta2**4.)**(1./3) # mks system unit
A2 = np.sqrt(4*np.pi*G/((n+1)*K2)*rho_c2**((n-1.)/float(n))) # mks system unit

print('central density for early/late type star: 2 and 150 g/cm^3')

#----------------------BOTH MODELS PLOTS------------------------------

############## DENSITY PROFILE ######################################

rho1 = rho_c1*w**n
rho2 = rho_c2*w**n
plt.plot(z/A1/6.95e8, rho1/1000, label='early type', color='b')  # rho in g/cm^3
plt.plot(z/A2/6.95e8, rho2/1000, label='late type', color='r')  # rho in g/cm^3
plt.plot(yrec['r'], yrec['rho'], label='yrec model', color='g')
plt.xlabel(r'r/r$_\odot$', fontsize=15)
plt.ylabel(r'$\rho(r)$ [g/cm^3]', fontsize=15)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=15)
plt.show()

################ PRESSURE PROFILE ##################

# early type
P1 = K1*rho_c1**(1+1./n)*w**(n+1)
plt.plot(z/A1/6.95e8, P1/0.1, label='early type', color='b')
plt.plot(z/A1/6.95e8, P1/0.1*beta1, label='P_gas,early', color='b',linestyle='-.')
plt.plot(z/A1/6.95e8, P1/0.1*(1-beta1), label='P_rad,early', color='b',linestyle='--')

# late type
P2 = K2*rho_c2**(1+1./n)*w**(n+1)
plt.plot(z/A2/6.95e8, P2/0.1, label='late type', color='r')
plt.plot(z/A2/6.95e8, P2/0.1*beta2, label='P_gas,late', color='r',linestyle='-.')
plt.plot(z/A2/6.95e8, P2/0.1*(1-beta2), label='P_rad,late', color='r',linestyle='--')

plt.plot(yrec['r'], yrec['P'], label='yrec model', color='g')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'r/r$_\odot$', fontsize=15)
plt.ylabel(r'P(r) [Ba]', fontsize=15)
plt.legend(fontsize=15)
plt.show()

############### MASS PROFILE ##########################

# early type
m1 = 4*np.pi*rho_c1*(z/A1)**3*(-1/z*dw_dz)/1.99e30
plt.plot(z/A1/6.95e8, m1, label='early type',color='b')

# late type
m2 = 4*np.pi*rho_c2*(z/A2)**3*(-1/z*dw_dz)/1.99e30
plt.plot(z/A2/6.95e8, m2, label='late type',color='r')

plt.plot(yrec['r'], yrec['m'], label='yrec model', color='g')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'r/r$_\odot$', fontsize=15)
plt.ylabel(r'cumulative m(r)/M$_\odot$', fontsize=15)
plt.legend(fontsize=15)
plt.show()

#############TOTAL MASS, RADIUS #######################

#early type
rho_avg1 = rho_c1*(-3/z_n*dw_dz[-1])
R1 = z_n/A1
M1 = 4./3*np.pi*R1**3*rho_avg1/1.99e30
R1 = R1/6.95e8
delta_R1 = step/A1/6.95e8

#late type
rho_avg2 = rho_c2*(-3/z_n*dw_dz[-1])
R2 = z_n/A2
M2 = 4./3*np.pi*R2**3*rho_avg2/1.99e30
R2 = R2/6.95e8
delta_R2 = step/A1/6.95e8

print('Early type,  M:', round(M1,2),'M_sun,', 'R:', round(R1), 'R_sun')
print('Late type,  M:', round(M2,2),'M_sun,', 'R:', round(R2, 2), 'R_sun')

########### TEMPERATURE ###################
# find root in P = R*rho*T/mu + aT^4/3


def fun(T, rho, P):
    return a/3.*T**4 + Re*rho/mu*T - P

T1 = []
T2 = []
for i in range(len(z)):
    # early type
    roots1 = optimize.root(fun, 1e7, args=(rho1[i], P1[i]))
    T1.append(max(roots1.x))

    # late type
    roots2 = optimize.root(fun, 1e6, args=(rho2[i], P2[i]))
    T2.append(max(roots2.x))

T1 = np.array(T1)
T2 = np.array(T2)

plt.plot(z/A1/6.95e8, T1, label='early type',color='b')
plt.plot(z/A2/6.95e8, T2, label='late type',color='r')
plt.plot(yrec['r'], yrec['T'], label='yrec model', color='g')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'r/R$_\odot$', fontsize=14)
plt.ylabel('Temperature [K]', fontsize=14)
plt.legend(fontsize=15)
plt.show()

########## ENERGY GENERATION ##############
x_1 = 0.7 # as in the sun
x_cno = 0.0138  # as in the sun, source: http://www.ucolick.org/~woosley/ay220-15/handouts/xsol_ag89.pdf
e_pp0 = 1.08e-5  # from 'introduction to modern astrophysics'
e_cno0 = 8.24e-24  # from 'introduction to modern astrophysics'
nu_pp = 4  # from 'introduction to modern astrophysics'
nu_cno = 19.9  # from introduction to modern astrophysics'

# pp-chain and CNO cycle (1==early, 2==late)
e_pp1 = e_pp0*(rho1/1000.)*x_1**2*(T1/1e6)**nu_pp
e_cno1 = e_cno0*(rho1/1000.)*x_1*x_cno*(T1/1e6)**nu_cno

e_pp2 = e_pp0*(rho2/1000.)*x_1**2*(T2/1e6)**nu_pp
e_cno2 = e_cno0*(rho2/1000.)*x_1*x_cno*(T2/1e6)**nu_cno

# total energy generation
e_1 = e_pp1 + e_cno1
e_2 = e_pp2 + e_cno2

plt.plot(z/A1/6.95e8, e_1, linestyle='-', color='b', label='early-type total')
plt.plot(z/A1/6.95e8, e_pp1, linestyle='-.', color='b', label='early-type pp')
plt.plot(z/A1/6.95e8, e_cno1, linestyle='--', color='b', label='early-type CNO')

plt.plot(z/A2/6.95e8, e_2, linestyle='-', color='r', label='late-type total')
plt.plot(z/A2/6.95e8, e_pp2, linestyle='-.', color='r', label='late-type pp')
plt.plot(z/A2/6.95e8, e_cno2, linestyle='--', color='r', label='late-type CNO')

plt.plot(yrec['r'], yrec['epsilon'], label='yrec model', color='g')

plt.xlabel(r'r/r$_\odot$', fontsize=14)
plt.ylabel('energy generation [erg/s/g]', fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-50, 1e11])
plt.legend(fontsize=14)

# indicate r_core (define r_core when e_pp drops to 0.1 percent to central value)
r_core_1 = z[len(e_pp1[e_pp1>0.001*e_pp1[0]])]
r_core_2 = z[len(e_pp2[e_pp2>0.001*e_pp2[0]])]
plt.axvline(x=r_core_1/A1/6.95e8, color='b', linestyle='-.')
plt.axvline(x=r_core_2/A2/6.95e8, color='r', linestyle='-.')
print('r_core/R for early/late type: '+str(round(r_core_1/A1/R1/6.95e8, 2))+' '+str(round(r_core_2/A2/R2/6.95e8, 2)))
plt.show()

############# LUMINOSITY ############################

dl1 = (e_pp1 + e_cno1)*np.pi*(z/A1)**2*step/A1*1e6  # luminosity in one shell (erg/s)
dl2 = (e_pp2 + e_cno2)*np.pi*(z/A2)**2*step/A2*1e6  # luminosity in one shell (erg/s)

plt.plot(z/A1/6.95e8, dl1, color='b', label='early type')
plt.plot(z/A2/6.95e8, dl2, color='r', label='late type')
plt.plot(yrec['r'], yrec['luminosity']/(step/A2), label='yrec model', color='g')

plt.xlabel('r/r$_\odot$', fontsize=14)
plt.ylabel('luminosity*dr [erg/s]', fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=15)
plt.show()

L1 = sum(dl1) # total luminosity (early type)
L2 = sum(dl2) # total luminosity (late type)
L_sun = 3.828e33 # from wikipedia
print('luminosity for early/late type (L_sun):', round(L1/L_sun, 2), ',', round(L2/L_sun,4)) # solar unit

########## OPACITY ########################
kappa_0 = 1e16  # unit?
kappa1 = kappa_0*P1*T1**(-4.5)
kappa2 = kappa_0*P2*T2**(-4.5)

plt.plot(z/A1/6.95e8, kappa1, label='early type',color='b')
plt.plot(z/A2/6.95e8, kappa2, label='late type',color='r')
plt.plot(yrec['r'], yrec['kappa'], label='yrec model', color='g')

plt.xlabel(r'r/R$_\odot$', fontsize=14)
plt.ylabel(r'$\kappa$ [cm^2/g]', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.show()

########## TEMPERATURE GRADIENT ##################
nabla1 = P1[0:-1]/T1[0:-1]*np.diff(T1)/np.diff(P1)  # P/T*dT/dP (dT, dP by discrete differentials)
nabla2 = P2[0:-1]/T2[0:-1]*np.diff(T2)/np.diff(P2)  # P/T*dT/dP
nabla_ad1 = 0.25  # equation see report section 3.2.4
nabla_ad2 = 0.38  # equation see report section 3.2.4

plt.plot(z[0:-1]/A1/6.95e8, nabla1, label='early type', color='b')
plt.plot(z[0:-1]/A2/6.95e8, nabla2, label='late type', color='r')

# indicate adiabatic gradients
plt.axhline(y=nabla_ad1, label=r'$\nabla_{ad,early}$', linestyle='--', color='b')
plt.axhline(y=nabla_ad2, label=r'$\nabla_{ad,late}$', linestyle='--', color='r')

plt.xlabel(r'r/R$_\odot$', fontsize=14)
plt.ylabel(r'Temperature Gradient ($\nabla$)', fontsize=14)
plt.xscale('log')
plt.legend(fontsize=14)
plt.show()