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
def dreicer_analytical(E, Zeff, Te, n_terms, gamma_func = False):

    c = plasma.c
    m_e = plasma.m_e

    u = np.sqrt(Te/(m_e*c**2))
    a = -3*(1+Zeff)/16

    result = np.zeros_like(E)

    if gamma_func == False:
        for j, E1 in enumerate(E):

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
        A = 1/(4*u**2)
        C = 2*np.sqrt(1+Zeff)
        prefactor = A**(a+1)

        for j, E1 in enumerate(E):

            x1 = A/E1
            x0 = A/E[0]

            series = 0.0

            for n in range(n_terms):

                s = n/2 - a - 1

                if s <= 0:
                    continue

                gamma1 = special.gamma(s) * special.gammaincc(s, x1)
                gamma0 = special.gamma(s) * special.gammaincc(s, x0)

                term = ((-C)**n / math.factorial(n)) * (gamma1 - gamma0)
                series += term

            result[j] = prefactor * series



    return result


# =========================
# Numerical integral (Simpson cumulative)
# =========================
def dreicer_numerical(E, Zeff, Te):
    """
    Compute F(E) = ∫₀ᴱ x^a * exp(-1/(4u²x) - √((1+Zeff)/(u²x))) dx
    """
    c = plasma.c
    m_e = plasma.m_e

    u = np.sqrt(Te/(m_e*c**2))
    a = -3*(1+Zeff)/16
    
    # Define the integrand function
    def integrand(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            exp1 = -1/(4*u**2*x)
            exp2 = -np.sqrt((1+Zeff)/(u**2*x))
            result = x**a * np.exp(exp1 + exp2)
            # Set x=0 to 0 (the integrand diverges but area under curve is 0 at x=0)
            result[x == 0] = 0
        return result
    
    # Create a fine grid for integration
    x_fine = np.linspace(E.min(), E.max(), 1000)
    y_fine = integrand(x_fine)
    
    # Use cumulative_simpson to get integral from E_min to each point
    F_cumulative = integrate.cumulative_simpson(y_fine, x=x_fine)
    
    # Interpolate to match original E points
    result = np.interp(E, x_fine[1:], F_cumulative)
    
    return result


# =========================
# Parameters
# =========================
Zeff = 1
Te = 5 * 1.602e-19   
n_terms = 80
E = np.linspace(1, 1000, 500)  


# =========================
# Compute
# =========================
num = dreicer_numerical(E, Zeff, Te)
ana_series = dreicer_analytical(E, Zeff, Te, n_terms, gamma_func=False)
ana_gamma = dreicer_analytical(E, Zeff, Te, n_terms, gamma_func=True)

difference = (num - ana_series)/np.where(num != 0, num, 1)  # Avoid division by zero for relative difference


# =========================
# Plot
# =========================
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(E, np.log(num + 1e-100), label='Numerical (Simpson)')
plt.plot(E, np.log(ana_series + 1e-100), linestyle='dashed', label='Analytical (series)')
plt.plot(E, np.log(ana_gamma + 1e-100), linestyle='dotted', label='Analytical (gamma)')
plt.xlabel("E")
plt.ylabel("log(F(E))")
plt.title("Dreicer integral comparison")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(E, difference)
plt.xlabel("E")
plt.ylabel("Relative difference")
plt.title(f"Max difference: {np.nanmax(np.abs(difference)):.2e}")
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.show()

print("Max difference:", np.nanmax(np.abs(difference)))