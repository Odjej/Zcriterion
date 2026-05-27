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
    "xtick.labelsize": LATEX_FONT_SIZE * 1.5,
    "ytick.labelsize": LATEX_FONT_SIZE * 1.5,
    "legend.fontsize": LATEX_FONT_SIZE * 1.5,
    "axes.formatter.use_locale": True,
})
# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

v_th = 1.8
v_c = 4.0
N = 400
 
diffusion = 0.2
E_acc = 0.15
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

v = np.linspace(-2, 6, 2400)

f = (1 / (np.sqrt(np.pi) * v_th**3)) * np.exp(-(v**2) / v_th**2)
f /= np.max(f)

# ------------------------------------------------------------
# Particles
# ------------------------------------------------------------

velocities = np.random.normal(0, v_th/np.sqrt(2), N)

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=figsize)

ax.set_xlim(-2, 6)
ax.set_ylim(-0.1, 1.15)

ax.set_xlabel(r"$v_\parallel$")
ax.set_ylabel(r"$f_M(v_\parallel)$")


ax.axvline(0, color='black', linestyle=':', lw=1)
ax.axvline(v_c, color='red', linestyle='--', lw=1)

tick_positions = [4, 0]
tick_labels    = [ r'$v_c$', r'$v_\mathrm{th}$']
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.get_xticklabels()[0].set_color('red')
# initial scatter
y_particles = np.interp(velocities, v, f)

colors = np.where(velocities > v_c, "red", "green")
ax.plot(v, f, lw=2, color="black", linestyle='-', zorder=1)
scatter = ax.scatter(
    velocities,
    y_particles,
    s=18,
    c=colors,
    alpha=0.75
)

ax.legend(handles=[
    plt.Line2D([], [], color='black', lw=2, linestyle='-',
               label='Maxwell-Boltzmannfördelning'),

    plt.Line2D([], [], linestyle='None', marker='o',
               markerfacecolor='green',
               label='Termiska elektroner'),

    plt.Line2D([], [], linestyle='None', marker='o',
               markerfacecolor='red',
               label='Skenande elektroner')
],
frameon=False,
handlelength=2,
labelspacing=0.3, loc="upper center")

ax.text(5, 0.5, r'Skenande Region',
        ha='center', color='black')

# ------------------------------------------------------------
# Dynamics
# ------------------------------------------------------------

span = None

def init():
    global span
    span = ax.axvspan(v_c, 6, color='salmon', alpha=0.1, zorder=0)
    scatter.set_offsets(np.column_stack((velocities, y_particles)))
    return scatter, span

def update(frame):
    global velocities

    bulk = velocities <= v_c
    runaway = ~bulk

    velocities[bulk] += np.random.normal(0, diffusion, bulk.sum())
    drag = friction(velocities[bulk], v_th)
    velocities[bulk] -= 0.03 * np.sign(velocities[bulk]) * drag
    velocities[bulk] += 0.03 * E_acc * np.sign(velocities[bulk])

    velocities[bulk] = np.clip(velocities[bulk], -2, 6)

    velocities[runaway] += 0.15 * np.sign(velocities[runaway])
    velocities = np.clip(velocities, -2, 6)

    escaped = velocities >= 5.9
    velocities[escaped] = np.random.normal(0, v_th / np.sqrt(2), escaped.sum())

    y_particles = np.interp(velocities, v, f)

    scatter.set_offsets(np.column_stack((velocities, y_particles)))
    scatter.set_color(np.where(velocities > v_c, "red", "green"))

    return scatter, span

# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------


fig.set_size_inches(12.8, 7.2)
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.85)
ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=10,
    blit=False
)
ax.set_yticks([])
plt.show()

ani.save("dreicer_animation.gif", writer="pillow", fps=30, dpi=150)