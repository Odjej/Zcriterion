import numpy as np
import scipy as sp
import Zcriterion.plasma as plasma
from scipy import special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math

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
    Compute ∫ from E0 to E1 of the Dreicer integrand
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

# =========================
# Parameters
# =========================
Zeff = 1.51
Te = 5 * 1.602e-19   
n_terms = 40
E_min = 1.0  # Lower integration limit 
E_points = np.linspace(1, 1000, 10000)  # Energy points from 1 to 1000

num = dreicer_numerical(E_points, E_min, Zeff, Te)
ana_series = dreicer_analytical(E_points, E_min, Zeff, Te, n_terms, gamma_func=False)
ana_gamma = dreicer_analytical(E_points, E_min, Zeff, Te, n_terms, gamma_func=True)

difference_ana = ana_series/np.where(ana_gamma != 0, ana_gamma, 1)  # Avoid division by zero for relative difference
difference_ser_num = num/np.where(ana_series != 0, ana_series, 1)  
difference_gam_num = num/np.where(ana_gamma != 0, ana_gamma, 1)  


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dreicer Function Comparisons', fontsize=16)

# Plot 1: All three methods together (top left)
ax1 = axes[0, 0]
ax1.plot(E_points, np.log(num + 1e-100), 'b-', label='Numerical (Simpson)', linewidth=2)
ax1.plot(E_points, np.log(ana_series + 1e-100), 'r--', label='Analytical (series)', linewidth=2)
ax1.plot(E_points, np.log(ana_gamma + 1e-100), 'g:', label='Analytical (gamma)', linewidth=2)
ax1.set_xlabel("E")
ax1.set_ylabel("log(F(E))")
ax1.set_title("All methods comparison")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Gamma vs Series (top right)
ax2 = axes[0, 1]
ax2.plot(E_points, difference_ana, 'r-', linewidth=2)
ax2.set_xlabel("E")
ax2.set_ylabel("Relative difference")
ax2.set_title(f"Series / Gamma \nMax diff: {np.nanmax(np.abs(difference_ana)):.2e}")
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Series vs Numerical (bottom left)
ax3 = axes[1, 0]
ax3.plot(E_points, difference_ser_num, 'b-', linewidth=2)
ax3.set_xlabel("E")
ax3.set_ylabel("Relative difference")
ax3.set_title(f"Numerical / Series \nMax diff: {np.nanmax(np.abs(difference_ser_num)):.2e}")
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Gamma vs Numerical (bottom right)
ax4 = axes[1, 1]
ax4.plot(E_points, difference_gam_num, 'g-', linewidth=2)
ax4.set_xlabel("E")
ax4.set_ylabel("Relative difference")
ax4.set_title(f"Numerical / Gamma \nMax diff: {np.nanmax(np.abs(difference_gam_num)):.2e}")
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.show()

# Print all differences
print("Max difference (gamma vs series):", np.nanmax(np.abs(difference_ana)))
print("Max difference (series vs numerical):", np.nanmax(np.abs(difference_ser_num)))
print("Max difference (gamma vs numerical):", np.nanmax(np.abs(difference_gam_num)))

