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
n_Ne = np.geomspace(1e16, 1e21, 3)
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
    n = [n_D0, n_T0, n_D, n_Ne_konstant]
    n_j, Z = atomic.coronalEquilibrium(n, T_e,  rates_scd, rates_acd)
    #density_row.append(n_j)
    #Z_row.append(Z)
    densitymatris.append(n_j)
    Z_matris.append(Z)
#print(densitymatris)
#density_matrix = np.array([row[0] for row in densitymatris]) #föratt göra matrisen plot-bar
#Z_matrix = np.array([row[0] for row in Z_matris]) #föratt göra matrisen plot-bar

#print(density_matrix)
#d = densitymatris[0]
#print(d.shape)
#print(d)
z = Z_matris[0]
#print(d)
#print(z)
#dot =d@(z**2)/(d @ z)
Z_eff = densitymatris @ (z ** 2) / (densitymatris @ z)
#print(dot)
#print(Z_eff)

fig, ax = plt.subplots()
levels = np.linspace(1, 3, 300)

c1 = plt.contourf(n_D, n_Ne, Z_eff, levels=levels, cmap = GeriMap, extend="max")
cbar = plt.colorbar(c1,label="Z (eV)")
important_levels = np.linspace(0, 3, 7)
cbar.set_ticks(important_levels)
Z_levels = [1.01, 1.1, 1.4, 1.8]
c2 = ax.contour(n_D, n_Ne, Z_eff, levels=Z_levels, colors='gray', linewidths=1, data=Z_levels)


plt.ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
plt.xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
#plt.show()

#np.savez(r'C:\Users\loeir\Kod ARC\heatmap_matrix.npz', Temp = heatmap_matrix, n_D = n_D, n_Ne = n_Ne)
#np.savez(r'C:\Users\loeir\Kod ARC\Z_eff.npz', Z_eff = Z_eff, z = z, n_D = n_D, n_Ne = n_Ne)
#['Temp','D', 'D_jon', 'T', 'T_jon', 'Ne', 'Ne_jon1', 'Ne_jon2','Ne_jon3', 'Ne_jon4', 'Ne_jon5', 'Ne_jon6', 'Ne_jon7', 'Ne_jon8',
#'Ne_jon9', 'Ne_jon10']
#np.savez(r'C:\Users\loeir\Kod ARC\Temp och Z.npz')
#print(densitymatris)
#for i in len(z):
#    i = np.column_stack([A[:,i] for A in densitymatris])
#print(z)
def d(typ):
    C = np.column_stack([A[:,typ] for A in densitymatris])
    return C

#print(d(z[0]))
#['Temp','D', 'D_jon', 'T', 'T_jon', 'Ne', 'Ne_jon1', 'Ne_jon2','Ne_jon3', 'Ne_jon4', 'Ne_jon5', 'Ne_jon6', 'Ne_jon7', 'Ne_jon8',
#'Ne_jon9', 'Ne_jon10']
np.savez(r'C:\Users\loeir\Kod ARC\Temp och Z.npz', Temp = heatmap_matrix, D = (d(z[0]) + d(z[4])) , D_jon = (d(z[1]) + d(z[5])), T = d(z[2]), T_jon = d(z[3]), 
         Ne = d(z[6]), Ne_jon1 = d(z[7]), Ne_jon2 = d(z[8]), Ne_jon3 = d(z[9]), Ne_jon4 = d(z[10]), Ne_jon5 = d(z[11]), Ne_jon6 = d(z[12]), 
         Ne_jon7 = d(z[13]), Ne_jon8 = d(z[14]), Ne_jon9 = d(z[15]), Ne_jon10=d(z[16]), n_D = n_D, n_Ne = n_Ne, z = z)
print(d(z[1]), d(z[16]))
Neon = np.array([d(z[6]),d(z[7]),d(z[8]),d(z[9])])
print(Neon)
print(Neon[:, 2, 0])

