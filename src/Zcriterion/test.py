import numpy as np
import scipy as sp
import matplotlib as mpl
import Zcriterion.plasma as plasma
from scipy import special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import Zcriterion.runaway as runaway



# =========================
# Parameters
# =========================
m = plasma.m_e
e = plasma.e
c = plasma.c
Z_eff = 1.55
T_e = 5 * 1.602e-19   # J (5 eV in Joules)
n_terms = 40
E_min = 1.0  # Lower integration limit 
E_points = np.linspace(1, 1000, 100000)  # Energy points from 1 to 1000
n_e = 2.44 * 10**20 # m^-3
u = np.sqrt(T_e * e/(m*c**2)) # Normalized electron thermal velocity
xi = -3*(1+Z_eff)/16
E_points = np.linspace(1,100, 1000)

def dreicer_gamma(E_points):
    result = np.zeros_like(E_points)
    for j,  E_init in enumerate(E_points):
         
        x = 1/(4*u**2*E_points)
        x0 = x[0]
        x1 = x[-1]
        beta = -xi


        gamma_upper_x1 = special.gammaincc(beta + 1, x1) * special.gamma(beta + 1)
        gamma_upper_x0 = special.gammaincc(beta + 1, x0) * special.gamma(beta + 1)
                
        # Apply recurrence relation to get gamma function for beta
        term_x1 = (gamma_upper_x1 - x1**beta*np.exp(-x1))/beta
        term_x0 = (gamma_upper_x0 - x0**beta*np.exp(-x0))/beta
                
        gamma_diff = term_x1 - term_x0
        result[j] = (4*u**2)**(-xi)*gamma_diff*np.exp(-np.sqrt((1+Z_eff)/(u**2*E_init)))
        return result
            


series = (1 + (-4*u**2*E_init)*(xi + 1))
dreicer_helander = 4*u**2 * E_init * u**(2*xi) * E_init**(xi) * np.exp(-1/(4*u**2*E_init) - np.sqrt((1+Z_eff)/(u**2*E_init))) * series

dndt = E_init**xi*np.exp(-1/(4*u**2*E_init) - np.sqrt((1+Z_eff)/(u**2*E_init)))
#dreicer_num = sp.integrate.cumulative_simpson(dndt/E_init,E_init, initial=0)

plt.plot(dreicer_gamma, E_init, label='Gamma function')
plt.plot(dreicer_helander, E_init, label='Helander approximation')
#plt.plot(E_init, dreicer_num, label='Numerical integration', ls='--')
plt.xlabel('E_init')
plt.ylabel('Dreicer growth rate')
plt.legend()
plt.show()