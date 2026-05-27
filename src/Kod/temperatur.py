import numpy as np
import scipy as sp
#import pandas as pd
import sys
sys.path.append('/home/loe/Desktop/Zcriterion/src')
import Zcriterion.ADAS.rates as rates

import Zcriterion.atomicPhysics as atomic
import Zcriterion.GeriMap as gerimap

#ARC-parametrar
a  = 1.18  # minor radius (m)
Ip = 12e6   # Target plasma current (A)
j0 = np.array([Ip / (np.pi * a**2)])  # current density (A/m^2) 

#simuleringsparametrar
n_Ne =np.geomspace(1e16, 1e21, 3)
n_D = np.geomspace(1e20, 5e22, 3)
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
print(rates_scd)

for i in range(len(n_D)):
    row = []
    n_Ne_konstant = n_Ne[i] * np.ones(len(n_D))
    n = [n_D0, n_T0, n_D, n_Ne_konstant]
    temp = atomic.equilibriumTemperature(n, j0, T_e,  rates_scd, rates_acd, rates_plt, rates_prb)
    row.append(temp)
    tempmatris.append(row)

heatmap_matrix = np.array([row[0] for row in tempmatris]) #föratt göra matrisen plot-bar
print(heatmap_matrix)
import matplotlib.pyplot as plt

gerimap.register #för att få rätt colormap som i deras artikel
GeriMap = gerimap.get() 

fig, ax = plt.subplots()
levels = np.linspace(0, 30, 300)

cp = plt.contourf(n_D, n_Ne, heatmap_matrix, levels=levels, cmap = GeriMap, extend="max")
cbar = plt.colorbar(cp,label="Temperatur (eV)")
important_levels = np.linspace(0, 30, 7)
cbar.set_ticks(important_levels)
iso_levels = [10, 100, 200, 1000]
cs = ax.contour(n_D, n_Ne, heatmap_matrix, levels=iso_levels, colors='gray', linewidths=1, data=iso_levels)
#ax.clabel(cs, iso_levels)

plt.ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
plt.xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
#plt.show()

densitymatris = []
Z_matris = []
for i in range(len(n_D)):
    density_row = []
    Z_row = []
    n_Ne_konstant = n_Ne[i] * np.ones(len(n_D))
    T = heatmap_matrix[i,:]
    n = [n_D0, n_T0, n_D, n_Ne_konstant]
    n_j, Z = atomic.coronalEquilibrium(n, T,  rates_scd, rates_acd)
    #density_row.append(n_j)
    #Z_row.append(Z)
    densitymatris.append(n_j)
    Z_matris.append(Z)
#densitymatris=np.array(densitymatris).T
#Z_matris=np.array(Z_matris).T
z = Z_matris[0]
print("shape", heatmap_matrix.shape, len(n_Ne))

Z_eff = densitymatris @ (z ** 2) / (densitymatris @ z)

fig, ax = plt.subplots()
levels = np.linspace(1, 3, 300)

c1 = plt.contourf(n_D, n_Ne, Z_eff, levels=levels, cmap = GeriMap, extend="max")
cbar = plt.colorbar(c1,label=r'$Z_\mathrm{eff}$')
important_levels = np.linspace(0, 3, 7)
cbar.set_ticks(important_levels)
Z_levels = [1.01, 1.1, 1.5]
c2 = ax.contour(n_D, n_Ne, Z_eff, levels=Z_levels, colors='gray', linewidths=1, data=Z_levels)


plt.ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
plt.xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
plt.xscale("log")
plt.yscale("log")
plt.rcParams.update({
        "font.family": "DejaVu Serif",  # lik Computer Modern
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "mathtext.fontset": "dejavuserif",  # matchar math text
    })
plt.tight_layout()
#plt.show()

densitymatris = np.array(densitymatris)
# Namn på laddningstillstånden
names = ['D0', 'D0_jon', 'T', 'T_jon', 'D', 'D_jon', 'Ne', 'Ne_jon1', 'Ne_jon2','Ne_jon3', 'Ne_jon4', 'Ne_jon5', 'Ne_jon6', 'Ne_jon7', 'Ne_jon8',
'Ne_jon9', 'Ne_jon10']


k = 0 #neonaxeln
j = 0 #deuteriumaxeln
# Skapa dictionary med 3x3-lager för varje laddningstillstånd
densities = {name: densitymatris[:, :, i] for i, name in enumerate(names)}
globals().update(densities)
#print(D0_0, D_0)
n_D_joner = np.array([D0 + D0_jon, D + D_jon])
n_T_joner = np.array([T, T_jon])
n_Ne_joner= np.array([Ne, Ne_jon1, Ne_jon2, Ne_jon3, Ne_jon4, Ne_jon5, Ne_jon6, Ne_jon7, Ne_jon8, Ne_jon9, Ne_jon10])

total_D = np.sum(n_D_joner[:, k, j]) #vänster neon, höger deuterium
total_T = np.sum(n_T_joner[:, k, j])
total_Ne = np.sum(n_Ne_joner[:, k, j])


np.savez(r'/home/loe/Desktop/DREAM/examples/ARC/Temp och Z/Temp och Z_3.npz', Temp = heatmap_matrix, D0 = D0, 
         D0_jon = D0_jon, 
         T = T, T_jon = T_jon, D = D, D_jon = D_jon, Ne = Ne, Ne_jon1 = Ne_jon1, Ne_jon2 = Ne_jon2,
   
      Ne_jon3 = Ne_jon3, Ne_jon4 = Ne_jon4, Ne_jon5 = Ne_jon5, Ne_jon6 = Ne_jon6, Ne_jon7 = Ne_jon7, 
         Ne_jon8 = Ne_jon8, Ne_jon9 = Ne_jon9, Ne_jon10 = Ne_jon10, n_D = n_D, n_Ne = n_Ne, z = z)
np.savez(r'/home/loe/Desktop/DREAM/examples/ARC/Data/Z_eff3.npz', Z_eff = Z_eff)

print("Klar")
