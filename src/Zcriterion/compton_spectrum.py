"""
Reproduces Fig. 2b of Zaar et al. (arXiv:2603.20485v1) for custom parameters:

    C1 = 1.922,  C2 = 0.954,  C3 = 0.041,  Gamma_flux = 1.5e16 m^-2 s^-1

Plots:
    - Gamma_gamma(W_gamma) / Gamma_0   (normalised gamma energy spectrum)
    - sigma_bar_eff(Wc)                (Compton cross-section averaged over
                                        spectrum, normalised to sigma_T)
      for critical energies Wc = 0, 1, 10, 100 keV  (solid/dashed/dot-dash/dotted)

Equations implemented
---------------------
Gamma_gamma  : Eq. (4.14)-(4.15)
sigma(Wg,Wc) : Eq. (4.19)   [Klein-Nishina integrated above theta_c]
cos_theta_c  : Eq. (4.18)
sigma_bar_eff: Eq. (4.20) normalised by sigma_T, then divided by Gamma_flux
               to give the spectrum-averaged reduced cross-section
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
from scipy.constants import m_e, c, e, physical_constants
import locale
# ── GeriMap (inline fallback — swap for real import if available) ─────────────
from matplotlib.colors import LinearSegmentedColormap
locale.setlocale(locale.LC_NUMERIC, "de_DE")

_stops = [
    (0.000, (0.00, 0.00, 0.50)),
    (0.100, (0.00, 0.00, 1.00)),
    (0.250, (0.00, 1.00, 1.00)),
    (0.400, (0.00, 1.00, 0.00)),
    (0.550, (1.00, 1.00, 0.00)),
    (0.700, (1.00, 0.50, 0.00)),
    (0.850, (1.00, 0.00, 0.00)),
    (1.000, (0.50, 0.00, 0.00)),
]
_cdict = {"red": [], "green": [], "blue": []}
for pos, (r, g, b) in _stops:
    _cdict["red"].append((pos, r, r))
    _cdict["green"].append((pos, g, g))
    _cdict["blue"].append((pos, b, b))
GeriMap = LinearSegmentedColormap("GeriMap", _cdict, N=512)
GeriMap.set_under("black")
mpl.colormaps.register(GeriMap, force=True)

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

# ── Physical constants ────────────────────────────────────────────────────────
m_e_eV   = m_e * c**2 / e          # electron rest energy in eV  (~511 keV)
r_e      = physical_constants["classical electron radius"][0]   # m
sigma_T  = (8 * np.pi / 3) * r_e**2                            # m^2

# ── Parameters (your ARC / SPARC-DD values) ───────────────────────────────────
C1          = 1.922
C2          = 0.954
C3          = 0.041
Gamma_flux  = 1.5e16   # m^-2 s^-1  (total photon flux)

# ── Gamma energy spectrum  Eq. (4.14)-(4.15) ─────────────────────────────────
def gamma_spectrum_shape(Wgamma_eV):
    """Returns Gamma_gamma / Gamma_0  (unit-normalised shape, range [0,1])."""
    Wgamma_MeV = np.asarray(Wgamma_eV, dtype=float) / 1e6
    # guard against log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(np.where(Wgamma_MeV > 0, Wgamma_MeV, np.nan)) + C1) / C2 \
            + C3 * Wgamma_MeV**2
    return np.exp(-np.exp(-z) - z + 1)

def gamma_spectrum(Wgamma_eV):
    """Gamma_gamma(Wgamma)  (shape x Gamma_0, where Gamma_0 normalises flux)."""
    shape = gamma_spectrum_shape(Wgamma_eV)
    # Gamma_0 chosen so integral of Gamma_gamma dWgamma = Gamma_flux
    # We compute the normalisation numerically once.
    return shape  # will be scaled below

# Numerical Gamma_0 so that int Gamma_gamma dWg = Gamma_flux
_Wg_norm   = np.logspace(3, 9, 4000)   # 1 keV – 1 GeV
_shape_norm = gamma_spectrum_shape(_Wg_norm)
_integral   = np.trapz(_shape_norm, _Wg_norm)
Gamma_0     = Gamma_flux / _integral    # [m^-2 s^-1 eV^-1]

def Gamma_gamma(Wgamma_eV):
    return Gamma_0 * gamma_spectrum_shape(Wgamma_eV)

# ── Critical angle  Eq. (4.18) ────────────────────────────────────────────────
def cos_theta_c(wgamma, wc):
    """
    wgamma = Wgamma / (me c^2),  wc = Wc / (me c^2)
    Returns cos(theta_c); NaN where scattering into runaway regime impossible.
    Special case: wc = 0  ->  cos_theta_c = 1 (all deflections contribute).
    """
    if wc == 0.0:
        return 1.0
    denom = wgamma - wc
    if denom <= 0:
        return np.nan
    cos_tc = 1.0 - (wc / wgamma) / denom
    return cos_tc if abs(cos_tc) <= 1.0 else np.nan

# ── Klein-Nishina cross-section  Eq. (4.19) ───────────────────────────────────
def sigma_KN(Wgamma_eV, Wc_eV):
    """
    Integrated Klein-Nishina cross-section for electrons scattered into the
    runaway region (theta > theta_c).  Returns sigma in m^2.
    """
    wg = Wgamma_eV / m_e_eV
    wc = Wc_eV    / m_e_eV

    ctc = cos_theta_c(wg, wc)
    # No contribution when scattering is kinematically impossible
    if np.isnan(ctc) or wg <= wc:
        return 0.0

    A = (wg**2 - 2*wg - 2) / wg**3 \
        * np.log((1 + 2*wg) / (1 + wg*(1 - ctc)))

    B = (1.0 / (2*wg)) * (
          1.0 / (1 + wg*(1 - ctc))**2
        - 1.0 / (1 + 2*wg)**2
    )

    D = -(1.0 / wg**3) * (
          1 - wg
        - (1 + 2*wg) / (1 + wg*(1 - ctc))
        - wg * ctc
    )

    return (3 * sigma_T / 8) * (A + B + D)

# Vectorised wrapper
sigma_KN_vec = np.vectorize(sigma_KN)

# ── sigma_bar_eff(Wc)  Eq. (4.20) normalised ─────────────────────────────────
def sigma_bar_eff(Wc_eV, Wg_grid, Gg_grid):
    """
    sigma_eff / sigma_T  averaged over the gamma spectrum.
    Numerically: int Gamma_gamma(Wg) * sigma(Wg,Wc) dWg  / (Gamma_flux * sigma_T)
    """
    sig = sigma_KN_vec(Wg_grid, Wc_eV)
    integrand = Gg_grid * sig
    return np.trapz(integrand, Wg_grid) / (Gamma_flux * sigma_T)

# ── Grids ─────────────────────────────────────────────────────────────────────
# x-axis: photon energy Wgamma (used for both sigma/sigma_T and Gamma/Gamma_0)
Wg_plot   = np.logspace(3, 9, 800)        # 1 keV -> 1 GeV [eV]

# Dense integration grid for spectrum-averaged sigma_bar_eff
Wg_int    = np.logspace(3, 9, 3000)
Gg_int    = Gamma_gamma(Wg_int)

# Four fixed Wc values; for each compute sigma(Wg, Wc)/sigma_T vs Wg (Fig 2b)
Wc_lines  = [0, 1e3, 18.6e3, 100e3]         # 0, 1, 10, 100 keV
styles    = ["solid", "dotted", "dashed", "dashdot"]

print("Computing sigma(Wgamma, Wc)/sigma_T curves ...")
sig_curves = []
for Wc_eV in Wc_lines:
    sig = sigma_KN_vec(Wg_plot, Wc_eV) / sigma_T
    sig = np.nan_to_num(sig, nan=0.0)
    sig_curves.append(sig)
    print(f"  Wc = {Wc_eV/1e3:.0f} keV  ->  peak sigma/sigma_T = {np.max(sig):.4f}")

# Spectrum-averaged sigma_bar_eff(Wc) plotted vs Wc for comparison
Wc_vals   = np.logspace(3, 8, 300)
print("\nComputing spectrum-averaged sigma_bar_eff(Wc) ...")
sbar_vals = np.array([sigma_bar_eff(wc, Wg_int, Gg_int) for wc in Wc_vals])
print(f"  sigma_bar_eff(Wc->0) = {sbar_vals[0]:.4f}  (paper Table1 SPARC DD = 0.3921)")



fig, ax1 = plt.subplots(figsize=figsize, layout="constrained", dpi=300)
ax1.set_rasterization_zorder(0)


# ── Left axis: sigma(Wgamma, Wc)/sigma_T for four fixed Wc ──────────────────
labels_Wc = [r"$W_\mathrm{c}=0$",
             r"$W_\mathrm{c}=1\ \mathrm{keV}$",
             r"$W_\mathrm{c}=18.6\ \mathrm{keV}$", 
             r"$W_\mathrm{c}=100\ \mathrm{keV}$"]

for sig, ls, lbl in zip(sig_curves, styles, labels_Wc):
    ax1.plot(Wg_plot, sig, color="black", ls=ls, lw=1.2, label=lbl)


ax1.set_xlabel(r"$W_\gamma \ [\mathrm{eV}]$")
ax1.set_ylabel(r"$\sigma/\sigma_\mathrm{T},\ \Gamma_\gamma/\Gamma_0$")
ax1.set_xscale("log")
ax1.set_xlim(1e4, 1e8)
ax1.set_ylim(0, 1.0)
ax1.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8])

# ── Right axis: Gamma_gamma / Gamma_0 ─────────────────────────────────────────
Gg_norm = gamma_spectrum_shape(Wg_plot)   # = Gamma_gamma / Gamma_0
ax1.plot(Wg_plot, Gg_norm,
         color="tab:blue", lw=1.4, ls="solid",
         label=r"$\Gamma_\gamma/\Gamma_0$")


# ── Combined legend ───────────────────────────────────────────────────────────
lines1, labs1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labs1,
           frameon=False, handlelength=1.5,
           labelspacing=0.3, loc="center right")

# ── Save ──────────────────────────────────────────────────────────────────────

plt.show()
