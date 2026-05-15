import numpy as np
import Zcriterion.plasma as plasma
from scipy import special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import locale
locale.setlocale(locale.LC_NUMERIC, "de_DE")

LATEX_FONT_SIZE = 11
TEXTWIDTH_PT = 426.7913

def pt_to_inches(pt):
    return pt / 72.27

fig_width = pt_to_inches(TEXTWIDTH_PT)
fig_height = fig_width / 1.618

figsize = ((fig_width, fig_height))

plt.rcParams.update({
    "font.family": "sans serif",
    "mathtext.fontset": "cm",
    "font.size": LATEX_FONT_SIZE,
    "axes.titlesize": LATEX_FONT_SIZE,
    "axes.labelsize": LATEX_FONT_SIZE,
    "xtick.labelsize": LATEX_FONT_SIZE * 0.8,
    "ytick.labelsize": LATEX_FONT_SIZE * 0.8,
    "legend.fontsize": LATEX_FONT_SIZE,
    "axes.formatter.use_locale": True,
})


m_e  = plasma.m_e
e    = plasma.e
c    = plasma.c
Z_eff = 1.5
T_e   = 5 * 1.602e-19  # J 
E_min = 1.0
E_points = np.linspace(600, 1000, 10000)

u  = np.sqrt(T_e / (m_e * c**2))
xi = -3 * (1 + Z_eff) / 16



def dreicer_gamma(E_points, E_min):
    result = np.zeros_like(E_points)
    beta = -xi  

    for j, E_init in enumerate(E_points):
        x1 = 1 / (4 * u**2 * E_init)
        x0 = 1 / (4 * u**2 * E_min)

        
        gamma_x1 = special.gammaincc(beta, x1) * special.gamma(beta)
        gamma_x0 = special.gammaincc(beta, x0) * special.gamma(beta)

        gamma_diff = gamma_x1 - gamma_x0  

        result[j] = (4 * u**2)**(-xi) * gamma_diff * np.exp(-np.sqrt((1 + Z_eff) / (u**2 * E_init)))

    return result



def dreicer_helander(E_points):
    xi = -3 * (1 + 1) / 16
    result = np.zeros_like(E_points)
    for j, E_init in enumerate(E_points):
        series = 1 + (-4 * u**2 * E_init) * (xi + 1)
        result[j] = (4 * u**2 * E_init * E_init**xi
                     * np.exp(-1 / (4 * u**2 * E_init)
                              - np.sqrt((1 + 1) / (u**2 * E_init)))
                     * series)
    return result

def dreicer_serie(E_points):
    xi = -3 * (1 + Z_eff) / 16
    result = np.zeros_like(E_points)
    for j, E_init in enumerate(E_points):
        series = 1 + (-4 * u**2 * E_init) * (xi + 1)
        result[j] = (4 * u**2 * E_init * E_init**xi
                     * np.exp(-1 / (4 * u**2 * E_init)
                              - np.sqrt((1 + Z_eff) / (u**2 * E_init)))
                     * series)
    return result


def dreicer_numerical(E_points, E_min, num_points=100000):
    E_max  = np.max(E_points)
    E_fine = np.linspace(E_min, E_max, num_points)

    integrand = E_fine**(xi - 1) * np.exp(-1 / (4 * u**2 * E_fine)
                                           - np.sqrt((1 + Z_eff) / (u**2 * E_fine)))
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

    F_cumulative = integrate.cumulative_simpson(integrand, x=E_fine, initial=0)
    return np.interp(E_points, E_fine, F_cumulative)



n_gamma    = dreicer_gamma(E_points, E_min)
n_helander = dreicer_helander(E_points)
n_num      = dreicer_numerical(E_points, E_min)
n_serie    = dreicer_serie(E_points)

fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=300)
 
ax.plot(E_points, n_num,
        color='green', lw=1.4, ls="solid",
        label=r"Numerisk")
 
ax.plot(E_points, n_gamma,
        color='blue', lw=1.2, ls=(0, (4, 2, 1, 2)),   # dash-dot-dash
        label=r"Analytisk ($\Gamma$)")
 
ax.plot(E_points, n_helander,
        color='orange', lw=1.2, ls=(0, (4, 2)),           # dotted
        label=r"Helander")

ax.plot(E_points, n_serie,
        color='red', lw=1.2, ls=(0, (1, 2)),           # dashed
        label=r"Analytisk (serie)")
 
ax.set_xlim(600, 950)
ax.set_yscale("log")
ax.set_xlabel(r"$E_1$")
ax.set_ylabel(r"$\log_{10} \int \,  \gamma_{\mathrm{seed}}^{\mathrm{D}}/E\,\mathrm{d}E$")
ax.legend(frameon=False, handlelength=2, labelspacing=0.3)

plt.show()