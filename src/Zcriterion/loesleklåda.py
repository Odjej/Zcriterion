import numpy as np
import matplotlib.pyplot as plt
data_temp = np.load(r'C:\Users\loeir\Kod ARC\heatmap_matrix.npz')
data_Z = np.load(r'C:\Users\loeir\Kod ARC\Z_eff.npz')

#Temp = heatmap_matrix, n_D = n_D, n_Ne = n_Ne
T = data_temp["Temp"]
n_D = data_temp["n_D"]
n_Ne = data_temp["n_Ne"]



fig, ax = plt.subplots()
levels = np.linspace(0, 30, 300)
cp = plt.contourf(n_D, n_Ne, T, levels=levels, cmap = "inferno", extend="max")
cbar = plt.colorbar(cp,label="Temperatur (eV)")
important_levels = np.linspace(0, 30, 7)
cbar.set_ticks(important_levels)
iso_levels = [10, 100, 1000, 1500]
cs = ax.contour(n_D, n_Ne, T, levels=iso_levels, colors='gray', linewidths=1, data=iso_levels)
#ax.clabel(cs, iso_levels)

plt.ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
plt.xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()


#Z
Z_eff = data_Z["Z_eff"] 
z = data_Z["z"] 
n_D = data_Z["n_D"] 
n_Ne = data_Z["n_Ne"]
levels = np.linspace(1, 3, 300)

c1 = plt.contourf(n_D, n_Ne, Z_eff, levels=levels, cmap = "inferno", extend="max")
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
plt.show()