import numpy as np
import scipy as sp
import Zcriterion.ADAS.rates as rates
import Zcriterion.atomicPhysics as atomic

#ARC-parametrar
a  = 1.18  # minor radius (m)
Ip = 12e6   # Target plasma current (A)
j0 = np.array([Ip / (np.pi * a**2)])  # current density (A/m^2) 

#simuleringsparametrar
n_Ne = np.geomspace(1e18, 1e23, 100)
n_D = np.geomspace(1e20, 1e25, 100)
n_D0 = 1.5e20 * np.ones(len(n_D))
n_T0 = 1.5e20 * np.ones(len(n_D))
n = [n_D0, n_T0, n_D, n_Ne]
T_e = 5.0 * np.ones(len(n_D))
Z_eff = np.linspace(1, 3 ,2)
#print(n)
rates_scd = []
rates_acd = []
rates_plt = []
rates_prb = []

species = ['D', 'T', 'D', 'Ne']

for i in species:
    ion = rates.get_rate(i, 'scd')
    rec = rates.get_rate(i, 'acd')
    plt = rates.get_rate(i, 'plt')
    prb = rates.get_rate(i, 'prb')
    rates_scd.append(ion)
    rates_acd.append(rec)
    rates_plt.append(plt)
    rates_prb.append(prb)

tempmatris = []
#ADAS.rates.get_rate(species, rateName), där ’species’ är t.ex. ’D’, ’T’, ’Ne’. etc, 
#och rateName är ’scd’, ’acd’, ’plt’ eller ’prb’, vilket står för jonisering, rekombinering, linjestrålning och bromsstrålning

for i in range(len(n_D)):
    row = []
    n_Ne_konstant = n_Ne[i] * np.ones(len(n_Ne))
    n = [n_D0, n_T0, n_D, n_Ne_konstant]
    temp = atomic.equilibriumTemperature(n, j0, T_e,  rates_scd, rates_acd, rates_plt, rates_prb)
    row.append(temp)
    tempmatris.append(row)

#print(tempmatris)


heatmap_matrix = np.array([row[0] for row in tempmatris])


import matplotlib.pyplot as plt

plt.imshow(
    heatmap_matrix,
    aspect='auto',
    origin='lower',
    extent=[n_D[0], n_D[-1], n_Ne[0], n_Ne[-1]],
cmap="hot", vmin=0, vmax=30)


plt.xscale("log")
plt.yscale("log")

plt.xlabel("n_D [/m^3]")
plt.ylabel("n_Ne [/m^3]")
plt.colorbar(label="T_e [ev]")

plt.show()




#for i in range(len(n_Ne)):
 #   row = []
  #  for j in range(len(n_D)):
   #   n_loop = [np.array([n[0][0]]), np.array([n[1][0]]), np.array([n[2][j]]), np.array([n[3][i]])]
     # [arr[0], arr[1][1], arr[2][1]]
    #  temp = atomic.equilibriumTemperature(n_loop, j0, T_e,  rates_scd, rates_acd, rates_plt, rates_prb)
       #def equilibriumTemperature(n_Z,j0,T_guess,ionRate,recombRate,lineRadRate,bremsRate = None,solveInLogScale = True,solver = 'brentq')
      #row.append(temp)
    ##tempmatris.append(row)
#print(tempmatris)
#for i in range(len(a)):
 #   row = []  # ny rad i matrisen
  #  for j in range(len(b)):
   ##    row.append(value)  # lägg till i raden
   # result.append(row)  # lägg till raden i matrisen