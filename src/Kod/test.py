from matplotlib import animation
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import locale
from scipy.special import erf
from matplotlib.lines import Line2D

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
    "axes.labelsize": LATEX_FONT_SIZE * 1.5,
    "xtick.labelsize": LATEX_FONT_SIZE * 1.4,
    "ytick.labelsize": LATEX_FONT_SIZE * 1.5,
    "legend.fontsize": LATEX_FONT_SIZE * 1.5,
    "axes.formatter.use_locale": True,
})
# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

v_th = 0.7
v_c = 4.0
N = 400

diffusion = 0.2
E_acc = 0.06
steps = 400

# ------------------------------------------------------------
# Friction (Coulomb drag)
# ------------------------------------------------------------

def friction(v, v_th=1.0):
    v = np.clip(np.abs(v), 1e-3, None)
    x = v / v_th
    return erf(x)/v**2 - (2/(v_th*np.sqrt(np.pi))) * np.exp(-x**2) / v

# ------------------------------------------------------------
# Velocity grid (for visualization)
# ------------------------------------------------------------

v = np.linspace(-3, 3, 2400)

f = (1 / (np.sqrt(np.pi) * v_th**3)) * np.exp(-(v**2) / v_th**2)
f /= np.max(f)



fig, ax = plt.subplots(figsize=figsize)
ax.fill_between(v, 0, f, where=(v >= 0),
                facecolor='none', edgecolor='black', hatch='///', zorder=2, label=r'$v > v_\mathrm{th}$')
ax.fill_between(v, 0, f, where=(v < 0),
                facecolor='none', edgecolor='black', hatch='O', zorder=2, label=r'$v < v_\mathrm{th}$')
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.15)

ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$f_M(v)$")

tick_positions = [ 0]
tick_labels    = [ r'$v_\mathrm{th}$']
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.get_xticklabels()[0].set_color('black')




ax.axvline(0, color='black', linestyle=':', lw=1)
ax.plot(v, f, lw=2, color="black", linestyle='-', zorder=1)



plt.legend(loc='upper right', frameon=False)
fig.set_size_inches(12.8, 7.2)
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.85)
ax.set_yticks([])
plt.show()

