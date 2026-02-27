import numpy as np
import scipy as sp
import matplotlib as mpl
import Zcriterion.plasma as plasma
from scipy import special
import scipy.integrate as integrate
mpl.use("pgf", force=True)
import matplotlib.pyplot as plt
import math

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage[swedish]{babel}
        \usepackage{lmodern}
        \usepackage{amsmath,amssymb,mathtools}
        \usepackage{physics}
        \usepackage{siunitx}
    """,
})


mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
})


def constants(Zeff, Te, n_e):
    mu0 = plasma.mu0
    eps0 = plasma.eps0
    e = plasma.e
    m_e = plasma.m_e
    c = plasma.c
    
    u = np.sqrt(Te/(m_e*c**2))
    lnL = plasma.lnLc(Te, n_e)
    I_p = 12e6 # A (12 MA) 
    tau_c = plasma.calc_tau_c(Te, n_e) # s
    k = 1.0 # dimensionless constant
    R_0 = 4.62 # m
    a = 1.18 # m
    L = mu0 * R_0 # H
    I_A = (4*np.pi*m_e*c)/(mu0*e) # A
    sigma = plasma.calc_spitzerCond(Te, n_e, Zeff) # S/m
    v_Te = np.sqrt(2 * Te / m_e) # m/s
    j_par = I_p /(np.pi * a**2) # A/m^2
    E_c = m_e * c /(e*tau_c) # V/m
    j_0 = j_par
    n_hat = j_par/(e*c)
    s = sigma * E_c / j_0 # dimensionless electric field
    alpha = (2*L*I_p)/(R_0*mu0*I_A*lnL)*np.sqrt(1/((Zeff+5)))
    
    def C_Z(Zeff):
       return (np.sqrt((Zeff + 5)))/(2**(3/2))*k*lnL*n_e/n_hat*u**(-(27+3*Zeff)/8)
   
    return s*alpha* C_Z(Zeff)
# =========================
# Analytical Dreicer series
# =========================
def dreicer_analytical(E_points, E_min, Zeff, Te, n_terms, gamma_func = False):

    c = plasma.c
    m_e = plasma.m_e

    u = np.sqrt(Te/(m_e*c**2))
    a = -3*(1+Zeff)/16

    result = np.zeros_like(E_points)

    if gamma_func == False:
        for j, E1 in enumerate(E_points):

            prefactor = np.exp(-(1/(4*u**2*E1)) - np.sqrt((1+Zeff)/(u**2*E1)))

            first = 4*u**2 * E1**(a+1)

            series = 0
            for n in range(1, n_terms+1):
                term = (4*u**2)**(n+1)
                term *= E1**(a+n+1)
                term *= special.gamma(a+n+1)/special.gamma(a+1)
                term *= (-1)**n
                series += term

            result[j] = prefactor * (first + series)
    
    else:
    # =========================
    # Analytical using incomplete gamma function
    # =========================
        for j, E1 in enumerate(E_points):
            prefactor = (4*u**2)**(-a)
            x1 = 1/(4*u**2*E1)
            x0 = 1/(4*u**2*E_min)  
            if -a > 0:
                gamma_diff = special.gammaincc(-a, x1) - special.gammaincc(-a, x0)
                gamma_diff *= special.gamma(-a)
            else:
                s = -a
                s_plus1 = s + 1
                
                # Compute Γ(s+1, x) for both bounds
                gamma_upper_x1 = special.gammaincc(s_plus1, x1) * special.gamma(s_plus1)
                gamma_upper_x0 = special.gammaincc(s_plus1, x0) * special.gamma(s_plus1)
                
                # Apply recurrence
                term_x1 = (gamma_upper_x1 - x1**s * np.exp(-x1)) / s
                term_x0 = (gamma_upper_x0 - x0**s * np.exp(-x0)) / s
                
                gamma_diff = term_x1 - term_x0
            
            result[j] = prefactor * gamma_diff * np.exp(-np.sqrt((1+Zeff)/(u**2*E1)))
            
    return result


# =========================
# Numerical integral (Simpson cumulative)
# =========================
def dreicer_numerical(E_points, E_min, Zeff, Te, num_points=100000):
    """
    Compute F(E)/E de from E0 to E1 of the Dreicer integrand
    """
    c = plasma.c
    m_e = plasma.m_e

    u = np.sqrt(Te/(m_e*c**2))
    a = -3*(1+Zeff)/16
    
    def integrand(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Handle x=0 or negative values
            x_safe = np.maximum(x, 1e-10)
            
            # Calculate the two exponential terms
            term1 = -1/(4*u**2*x_safe)
            term2 = -np.sqrt((1+Zeff)/(u**2*x_safe))
            
            # Integrand = x^(a-1) * exp(term1 + term2)
            result = x_safe**(a-1) * np.exp(term1 + term2)
            
        return result
    
    # Create fine grid from E_min to max(E_points)
    E_max = np.max(E_points)
    x_fine = np.linspace(E_min, E_max, num_points)
    y_fine = integrand(x_fine)
    
    # Handle any numerical issues (NaN, Inf)
    y_fine = np.nan_to_num(y_fine, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate cumulative integral using Simpson's rule
    F_cumulative = integrate.cumulative_simpson(y_fine, x=x_fine, initial=0)
    
    # Interpolate to get values at original E_points
    result = np.interp(E_points, x_fine, F_cumulative)
    
    return result

def dreicer_helander(E_points, Te, n_e, Zeff = 1.0):
    """
    Dreicer seed integral from Helander.
    """
    
    mu0 = plasma.mu0
    eps0 = plasma.eps0
    e = plasma.e
    m_e = plasma.m_e
    c = plasma.c
    
    u = np.sqrt(Te/(m_e*c**2))
    lnL = plasma.lnLc(Te, n_e)
    I_p = 12e6 # A (12 MA) 
    tau_c = plasma.calc_tau_c(Te, n_e) # s
    k = 1.0 # dimensionless constant
    R_0 = 4.62 # m
    a = 1.18 # m
    L = mu0 * R_0 # H
    I_A = (4*np.pi*m_e*c)/(mu0*e) # A
    sigma = plasma.calc_spitzerCond(Te, n_e, Zeff) # S/m
    v_Te = np.sqrt(2 * Te / m_e) # m/s
    j_par = I_p /(np.pi * a**2) # A/m^2
    E_c = m_e * c /(e*tau_c) # V/m
    j_0 = j_par
    n_hat = j_par/(e*c)
    s = sigma * m_e/ (n_hat * e**2 * tau_c) # dimensionless electric field
    alpha = (np.sqrt(2*np.pi))/3 * (L*I_p)/(mu0*R_0*I_A*lnL)
    
    result = np.zeros_like(E_points)
    
    
    
    for j, E1 in enumerate(E_points):
        F_E = (3*k*lnL)/(2*np.pi**(1/2)*u**(15/4))*n_e/n_hat / (E1**(3/8)) * np.exp(-1/(4*u**2 * E1) - np.sqrt(2/(u**2 * E1)))
        result[j] = s * alpha * F_E * E1 * 4 * u**2
    
    return result
    
    

def plot_dreicer_comparison(E_points, n_seed_num, n_seed_series, n_seed_gamma, n_seed_helander, 
                           save_path="dreicer_comparison.pdf", figsize=(15, 8), cm_units=True):
    """
    Plot comparison of different Dreicer seed calculations.
    """
    
    # Convert cm to inches if needed
    if cm_units:
        cm_to_inch = 1/2.54
        figsize_inch = (figsize[0] * cm_to_inch, figsize[1] * cm_to_inch)
    else:
        figsize_inch = figsize
    
    # Create figure
    plt.figure(figsize=figsize_inch)
    
    # Plot all four methods
    plt.plot(E_points, n_seed_num, label=r"Numerisk integral", linewidth=1.5)
    plt.plot(E_points, n_seed_series, "--", label=r"Analytisk serie", linewidth=1.5)
    plt.plot(E_points, n_seed_gamma, ":", label=r"Analytisk (Inkomplett $\Gamma$)", linewidth=1.5)
    plt.plot(E_points, n_seed_helander, "-.", label=r"Helander", linewidth=1.5)
    
    # Formatting
    plt.yscale("log")  
    plt.xlabel(r"$E_1$", fontsize=12)
    plt.ylabel(r"$n^*$", fontsize=12)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    
def plot_ratios(E_points, n_seed_num, n_seed_series, n_seed_gamma, n_seed_helander):
    """
    Plot ratios compared to numerical result
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(E_points, n_seed_series / n_seed_num, 
                label=r'Serie / Numerisk', linewidth=1.5)
    ax.semilogx(E_points, n_seed_gamma / n_seed_num, 
                '--', label=r'Gamma / Numerisk', linewidth=1.5)
    ax.semilogx(E_points, n_seed_helander / n_seed_num, 
                ':', label=r'Helander / Numerisk', linewidth=1.5)
    
    ax.set_xlabel(r'$E_1$')
    ax.set_ylabel(r'Kvot till numerisk')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ratios.pdf", bbox_inches="tight")
    plt.show()


# =========================
# Parameters
# =========================
Zeff = 10
Te = 5 * 1.602e-19   # J (5 eV in Joules)
n_terms = 40
E_min = 1.0  # Lower integration limit 
E_points = np.linspace(1, 1000, 100000)  # Energy points from 1 to 1000
n_e = 2.44 * 10**20 # m^-3
num = dreicer_numerical(E_points, E_min, Zeff, Te)
ana_series = dreicer_analytical(E_points, E_min, Zeff, Te, n_terms, gamma_func=False)
ana_gamma = dreicer_analytical(E_points, E_min, Zeff, Te, n_terms, gamma_func=True)


n_seed_series = constants(Zeff, Te, n_e) * ana_series 
n_seed_gamma = constants(Zeff, Te, n_e) * ana_gamma
n_seed_num = constants(Zeff, Te, n_e) * num
n_seed_helander = dreicer_helander(E_points, Te, n_e, Zeff=1.0)

plot_dreicer_comparison(E_points, n_seed_num, n_seed_series, n_seed_gamma, n_seed_helander)
plot_ratios(E_points, n_seed_num, n_seed_series, n_seed_gamma, n_seed_helander)