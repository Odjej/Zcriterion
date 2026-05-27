import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")

W_MAX_EV = 18.6e3          #


def F_beta(Wc_eV):
    """
    Fraction of tritium beta-decay electrons born above the critical energy Wc.

    Parameters
    ----------
    Wc_eV : array-like
        Critical energy in electronvolts.

    Returns
    -------
    F : ndarray
        F_beta in [0, 1].  Zero whenever Wc >= Wmax.
    """
    Wc_eV = np.asarray(Wc_eV, dtype=float)
    w = Wc_eV / W_MAX_EV                         # normalised critical energy

    # Polynomial piece (valid for w < 1, i.e. Wc < Wmax)
    F = 1.0 - (35.0 / 8.0) * w**1.5 \
            + (21.0 / 4.0) * w**2.5 \
            - (15.0 / 8.0) * w**3.5

    # Heaviside: F_beta = 0 when Wc >= Wmax
    F = np.where(w < 1.0, F, 0.0)
    F = np.clip(F, 0.0, 1.0)
    return F



Wc = np.logspace(2, 7, 2000)   # 100 eV → 10 MeV

F = F_beta(Wc)



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

fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=300)

ax.plot(Wc, F, color="#1f77b4", lw=2.0, label=r"$F_\beta$")

ax.axvline(W_MAX_EV, color="gray", ls="--", lw=1.0, alpha=0.7)
ax.text(W_MAX_EV * 1.12, 0.55,
        r"$W_\mathrm{max} = 18{,}6\,\mathrm{keV}$",
        color="gray", va="center")


ax.set_xscale("log")
ax.set_xlim(1e2, 1e7)
ax.set_ylim(-0.02, 1.05)

ax.set_xlabel(r"$W_{\mathrm{c}} \, [\mathrm{eV}]$")
ax.set_ylabel(r"$F_{\beta}(W_{\mathrm{c}})$")


ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(True, which="both", ls=":", alpha=0.4)

plt.tight_layout()
plt.show()
plt.savefig("tritium_beta_spectrum.png", dpi=300, bbox_inches="tight")
