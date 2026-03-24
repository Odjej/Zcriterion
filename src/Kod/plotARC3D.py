import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.constants import e, c
import numpy as np
import sys
sys.path.append('/home/loe/Desktop/Zcriterion/src')
import Zcriterion.GeriMap as gerimap

#laddar data
data_r = np.load(r'/home/loe/Desktop/DREAM/examples/ARC/Data/runaways100.npz')
data_z = np.load(r'/home/loe/Desktop/DREAM/examples/ARC/Data/Z_eff100.npz')
data_n = np.load(r'/home/loe/Desktop/DREAM/examples/ARC/Temp och Z/Temp och Z_100.npz')
globals().update(data_r)
globals().update(data_z)
globals().update(data_n)

#definierar kvoterna och undersöker gränser
a  = 1.18  # minor radius (m)
A = np.pi * a**2
norm_kvot = Ip0 / (A * e * c)
kvot_Ips = np.divide(runaways, Ips)
kvot_Ip0 = np.divide(runaways, Ip0)
kvot_Dreicer = np.divide(Dreicer_RE, norm_kvot)
kvot_Compton = np.divide(Compton_RE, norm_kvot)
kvot_Tritium = np.divide(Tritium_RE, norm_kvot)
#print(np.max(runaways), np.argmax(runaways))
#print(np.max(kvot_Ip0))

#skaffar colormapen
gerimap.register 
GeriMap = gerimap.get() 
cmap = gerimap

plt.rcParams.update({
        "font.family": "DejaVu Serif",  # lik Computer Modern
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "mathtext.fontset": "dejavuserif",  # matchar math text
    })

#startar figurerna
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

#fixar fontent i figurerna


#nivåer till colorbars och relevanta iso_z
levels_kvot_Ip = np.linspace(0, 1, 11)
levels_kvot_Ip0 = np.linspace(0, 0.7, 8)
levels_RE = np.linspace(0, 8*10**6, 9)
levels_Z_eff = np.linspace(1, 1.5, 11)
levels_temp = np.linspace(0,5, 11)
iso_z = [1.0001, 1.001, 1.01 ]
iso_z_label = {level: f'{level:.0e}' for level in iso_z}


#figuren för runaways
RE = ax1.contourf(n_D, n_Ne, runaways, levels=levels_RE, cmap = GeriMap)
cbar = plt.colorbar(RE,label="Runawayström (A)")
cbar.set_ticks(levels_RE)
#ax1.clabel(cs, levels=iso_levels_RE)
ax1.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax1.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax1.set_xscale("log")
ax1.set_yscale("log")
fig1.tight_layout()


#figuren för RE/I_p för samma punkt som RE hade maximum
#K = ax2.contourf(n_D, n_Ne, kvot_Ips, levels=levels_kvot_Ip, cmap = GeriMap)
#kbar = plt.colorbar(K,label=r'$I_\mathrm{Re}/I_\mathrm{p}$')
#kbar.set_ticks(levels_kvot_Ip)
#ks = ax2.contour(n_D, n_Ne, Z_eff, levels=iso_z, colors='red', linewidths=2)
#ax2.clabel(ks, fmt=iso_z_label)
#ax2.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
#ax2.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
#ax2.set_xscale("log")
#ax2.set_yscale("log")
#fig2.tight_layout()

#figur där RE/I_p där I_p är strömmen vid t = 0
P = ax3.contourf(n_D, n_Ne, kvot_Ip0, levels=levels_kvot_Ip0, cmap = GeriMap)
pbar = plt.colorbar(P,label=r'$I_\mathrm{Re}/I_\mathrm{p0}$')
pbar.set_ticks(levels_kvot_Ip0)
iso_level = [0.01]
ps = ax3.contour(n_D, n_Ne, kvot_Ip0, levels=iso_level, colors='gray', linewidths=1, data=iso_level)
ax3.clabel(ps, iso_level)
#manual_positions = [(1e20,1e22), (1e19,1e22), (1e18,1e21), (1e17,1e20)]
ax3.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax3.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax3.set_xscale("log")
ax3.set_yscale("log")
fig3.tight_layout()

#Z_effplot
fig4, ax4 = plt.subplots()
levels = np.linspace(1, 3, 300)
c1 = ax4.contourf(n_D, n_Ne, Z_eff, levels=levels_Z_eff, cmap = GeriMap, extend="max")
cbar = plt.colorbar(c1,label=r'$Z_\mathrm{eff}$')
cbar.set_ticks(levels_Z_eff)
Z_levels = [1.01, 1.1, 1.5]
Z_label = {level: f'{level:.0e}' for level in Z_levels}
c2 = ax4.contour(n_D, n_Ne, Z_eff, levels=Z_levels, colors='gray', linewidths=1, data=Z_levels)
ax4.clabel(c2, Z_levels) #manual_positions = [(1e20,1e22), (1e19,1e22), (1e18,1e21), (1e17,1e20)]
ax4.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax4.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax4.set_xscale("log")
ax4.set_yscale("log")
fig4.tight_layout()

#temperatur
fig5, ax5 = plt.subplots()
levels = np.linspace(0, 5, 11)

cp = ax5.contourf(n_D, n_Ne, Temp, levels=levels_temp, cmap = GeriMap, extend="max")
cbar = plt.colorbar(cp,label="Temperatur (eV)")
cbar.set_ticks(levels_temp)
iso_levels = [10, 100, 1000]
cs = ax5.contour(n_D, n_Ne, Temp, levels=iso_levels, colors='gray', linewidths=1, data=iso_levels)
ax5.clabel(cs, iso_levels)
ax5.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax5.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax5.set_xscale("log")
ax5.set_yscale("log")
fig5.tight_layout()

#logaritmerad Dreicer
fig6, ax6 = plt.subplots()
levels = np.linspace(-350, -28, 13) #om man vill plotta från 1e-12, annars böraj vid -350 om du vill fånga allt
P = ax6.contourf(n_D, n_Ne, np.log10(kvot_Dreicer), levels=levels, cmap = GeriMap)
pbar = plt.colorbar(P,label=r'$I_\mathrm{Dr,Re}/(I_\mathrm{p0}/Aec)$')
pbar.set_ticks(levels)
#iso_level = [0.01]
#ps = ax6.contour(n_D, n_Ne, kvot_Dreicer, colors='gray', linewidths=1, data=iso_level)#levels=iso_level, 
#ax6.clabel(ps, iso_level)
#manual_positions = [(1e20,1e22), (1e19,1e22), (1e18,1e21), (1e17,1e20)]
ax6.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax6.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax6.set_xscale("log")
ax6.set_yscale("log")
fig6.tight_layout()

#logaritmerad Compton
fig7, ax7 = plt.subplots()
levels = np.linspace(-11, -9, 9)
P = ax7.contourf(n_D, n_Ne, np.log10(kvot_Compton), levels=levels, cmap = GeriMap)
pbar = plt.colorbar(P,label=r'$I_\mathrm{C,Re}/(I_\mathrm{p0}/Aec)$')
pbar.set_ticks(levels)
#iso_level = [0.01]
#ps = ax6.contour(n_D, n_Ne, kvot_Dreicer, colors='gray', linewidths=1, data=iso_level)#levels=iso_level, 
#ax6.clabel(ps, iso_level)
#manual_positions = [(1e20,1e22), (1e19,1e22), (1e18,1e21), (1e17,1e20)]
ax7.set_ylabel(r'$n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax7.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax7.set_xscale("log")
ax7.set_yscale("log")
fig7.tight_layout()

#logaritmerad Tritium
fig8, ax8 = plt.subplots()
levels = np.linspace(-12, -6, 13)
P = ax8.contourf(n_D, n_Ne, np.log10(kvot_Tritium), levels=levels, cmap = GeriMap)
pbar = plt.colorbar(P,label=r'$I_\mathrm{T,Re}/(I_\mathrm{p0}/Aec)$')
pbar.set_ticks(levels)
#iso_level = [0.01]
#ps = ax6.contour(n_D, n_Ne, kvot_Dreicer, colors='gray', linewidths=1, data=iso_level)#levels=iso_level, 
#ax6.clabel(ps, iso_level)
#manual_positions = [(1e20,1e22), (1e19,1e22), (1e18,1e21), (1e17,1e20)]
ax8.set_ylabel(r'$ log n_\mathrm{Ne}(\mathrm{m}^{-3})$')
ax8.set_xlabel(r'$n_\mathrm{D}(\mathrm{m}^{-3})$')
ax8.set_xscale("log")
ax8.set_yscale("log")
fig8.tight_layout()

print(np.log10(kvot_Dreicer), np.max(np.log10(kvot_Dreicer)), np.max(kvot_Dreicer))
#, np.log10(kvot_Compton), np.log10(kvot_Tritium))
print(np.max(runaways), np.argmax(runaways))

plt.show()


