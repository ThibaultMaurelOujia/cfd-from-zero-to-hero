import os
import time
import scipy
import platform
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import fftpack
import matplotlib.cm as cm
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import LogFormatterMathtext
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

plt.rcParams['text.usetex'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.style.use('classic')

import numpy as np
import matplotlib.pyplot as plt

def init_step(N, pos):
    u = np.zeros(N)
    u[:pos] = 1.0
    return u

def init_gaussian(N, _=None):
    i = np.arange(N)
    mu = (N - 1) / 2
    sigma2 = N / 4.0
    return np.exp(- (i - mu)**2 / (2 * sigma2))

def scheme_euler_explicit(u, nu):
    return u + nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1))

def scheme_euler_implicit(u, nu):
    N = u.size
    main_diag = (1 + 2*nu) * np.ones(N)
    off_diag  = -nu * np.ones(N-1)
    A = np.diag(main_diag) \
      + np.diag(off_diag, 1) \
      + np.diag(off_diag, -1)
    A[0, -1] = -nu
    A[-1, 0] = -nu
    return np.linalg.solve(A, u)

def scheme_crank_nicolson(u, nu):
    N = u.size
    main_diag = (1 + nu) * np.ones(N)
    off_diag  = - (nu/2) * np.ones(N-1)
    A = np.diag(main_diag) \
      + np.diag(off_diag, 1) \
      + np.diag(off_diag, -1)
    A[0, -1] = - (nu/2)
    A[-1, 0] = - (nu/2)
    b = u + (nu/2) * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
    return np.linalg.solve(A, b)

def scheme_spectral(u, dt, dx, nu):
    D = nu * dx**2 / dt
    u_hat = np.fft.fft(u)
    k     = 2*np.pi * np.fft.fftfreq(len(u), d=dx)
    du_dx_2 = np.fft.ifft(-k**2*u_hat).real
    return u + dt * D * du_dx_2

def scheme_spectral_rk4(u, dt, dx, nu):
    D = nu * dx**2 / dt
    def rhs(v):
        v_hat = np.fft.fft(v)
        k     = 2*np.pi * np.fft.fftfreq(len(v), d=dx)
        v_xx  = np.fft.ifft(-k**2 * v_hat).real
        return D * v_xx
    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u +     dt*k3)
    return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def exact_solution(u0, x, c, t):
    x_shift = (x - c*t) % 1.0
    return np.interp(x_shift, x, u0)

def run_scheme(name, init_func, scheme_func, params):
    N   = params['N_x']
    c   = params['c']
    CFL = params['CFL']
    T   = params['T_final']
    step_plot = params['step_plot']
    dx = 1.0 / N
    dt = CFL / (N * c)
    nu = params.get('nu', c*dt/dx)
    nsteps = int(T / dt)
    x  = np.linspace(0, 1, N, endpoint=False)
    u0 = init_func(N, params.get('pos', None))
    u  = u0.copy()
    fig, ax = plt.subplots()
    line, = ax.plot(x, u, lw=1.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    plt.ion()
    plt.show()
    errors = {'L2': [], 'Linf': []}
    times  = []
    for i in range(1, nsteps+1):
        if name == 'scheme_spectral':
            u = scheme_func(u, dt, dx, nu)
        elif name == 'scheme_spectral_rk4':
            u = scheme_func(u, dt, dx, nu)
        else:
            u = scheme_func(u, nu)
        if i % step_plot == 0 or i == nsteps:
            t = i * dt
            u_exact = exact_solution(u0, x, c, t)
            err_L2  = np.linalg.norm(u - u_exact) / np.sqrt(N)
            err_inf = np.max(np.abs(u - u_exact))
            errors['L2'].append(err_L2)
            errors['Linf'].append(err_inf)
            times.append(t)
            line.set_ydata(u)
            ax.set_title(f"{name} — t={t:.3f}, L2={err_L2:.2e}, Linf={err_inf:.2e}")
            plt.pause(0.01)
    plt.ioff()
    plt.show()
    return times, errors

params = {
    'N_x': 1280,
    'c': 1.0,
    'CFL': 0.01,
    'T_final': 0.1,
    'step_plot': 50,
    'pos': 1280//2,
    'nu': 0.05
}

# Euler explicite
# run_scheme('euler_explicit', init_gaussian, scheme_euler_explicit, params)

# Euler implicite
# run_scheme('euler_implicit', init_gaussian, scheme_euler_implicit, params)

# Crank–Nicolson
# run_scheme('crank_nicolson', init_gaussian, scheme_crank_nicolson, params)

# Pseudo-spectral
# run_scheme('scheme_spectral', init_gaussian, scheme_spectral, params)

# Pseudo-spectral
run_scheme('scheme_spectral_rk4', init_gaussian, scheme_spectral_rk4, params)






































