import numpy as np
import scipy as sp
import Zcriterion.ADAS.rates as rates

import Zcriterion.atomicPhysics as atomic
import Zcriterion.GeriMap as gerimap

#ARC-parametrar
a  = 1.18  # minor radius (m)
Ip = 12e6   # Target plasma current (A)
j0 = np.array([Ip / (np.pi * a**2)])  # current density (A/m^2) 

#simuleringsparametrar
n_Ne = np.geomspace(1e16, 1e21, 30)
n_D = np.geomspace(1e20, 5e22, 30)
n_D0 = 1.5e20 * np.ones(len(n_D))
n_T0 = 1.5e20 * np.ones(len(n_D))
n = [n_D0, n_T0, n_D, n_Ne]
T_e = 5.0 * np.ones(len(n_D))
Z_eff = np.linspace(1, 3 ,2)

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
    n_Ne_konstant = n_Ne[i] * np.ones(len(n_D))
    n = [n_D0, n_T0, n_D, n_Ne_konstant]
    temp = atomic.equilibriumTemperature(n, j0, T_e,  rates_scd, rates_acd, rates_plt, rates_prb)
    row.append(temp)
    tempmatris.append(row)

#print(tempmatris)
#print('Neon:', n_Ne)
#print('Deuterium', n_D)

#försöker lösa det som matris direkt

heatmap_matrix = np.array([row[0] for row in tempmatris]) #föratt göra matrisen plot-bar
#print(heatmap_matrix)
import matplotlib.pyplot as plt

gerimap.register #för att få rätt colormap som i deras artikel
GeriMap = gerimap.get() 

fig, ax = plt.subplots()
levels = np.linspace(0, 30, 300)

cp = plt.contourf(n_D, n_Ne, heatmap_matrix, levels=levels, cmap = GeriMap, extend="max")
cbar = plt.colorbar(cp,label="Temperatur (eV)")
important_levels = np.linspace(0, 30, 7)
cbar.set_ticks(important_levels)
iso_levels = [10, 100, 1000, 1500]
cs = ax.contour(n_D, n_Ne, heatmap_matrix, levels=iso_levels, colors='gray', linewidths=1, data=iso_levels)


plt.ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
plt.xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()
