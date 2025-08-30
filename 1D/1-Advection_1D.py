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

def init_step(N, pos):
    u = np.zeros(N)
    u[:pos] = 1.0
    return u

def init_gaussian(N, _=None):
    i     = np.arange(N)
    mu    = (N - 1)/2
    sigma2= N/4.0
    return np.exp(- (i - mu)**2 / (2*sigma2))

def scheme_ftcs(u, nu):
    return u - 0.5*nu*(np.roll(u, -1) - np.roll(u, 1))

def scheme_upwind(u, nu, c):
    if c > 0:
        return u - nu*(u - np.roll(u, 1))
    else:
        return u - nu*(np.roll(u, -1) - u)

def scheme_lax_friedrichs(u, nu):
    return 0.5*(np.roll(u, -1) + np.roll(u, 1)) - 0.5*nu*(np.roll(u, -1) - np.roll(u, 1))

def scheme_spectral(u, dt, dx):
    u_hat = np.fft.fft(u)
    k     = 2*np.pi * np.fft.fftfreq(len(u), d=dx)
    du_dx = np.fft.ifft(1j*k*u_hat).real
    return u - dt * du_dx

def scheme_spectral_rk4(u, dt, dx, c):
    def rhs(v):
        v_hat = np.fft.fft(v)
        k     = 2*np.pi * np.fft.fftfreq(len(v), d=dx)
        dv_dx = np.fft.ifft(1j*k*v_hat).real
        return -c * dv_dx

    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u +     dt*k3)
    return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def scheme_spectral_implicit(u, dt, dx, c):
    u_hat = np.fft.fft(u)
    k = 2*np.pi * np.fft.fftfreq(len(u), d=dx)
    denom = 1 + 1j * c * k * dt
    u_hat_new = u_hat / denom
    return np.fft.ifft(u_hat_new).real

def exact_solution(u0, x, c, t):
    x_shift = (x - c*t) % 1.0
    return np.interp(x_shift, x, u0)

def run_scheme(name, init_func, scheme_func, params):
    N   = params['N_x'];    c = params['c']
    CFL = params['CFL'];    T = params['T_final']
    step_plot = params['step_plot']
    dx = 1.0/N
    dt = CFL/(N*c)
    nu = c*dt/dx
    nsteps = int(T/dt)
    x  = np.linspace(0, 1, N, endpoint=False)
    u0 = init_func(N, params.get('pos', None))
    u  = u0.copy()
    fig, ax = plt.subplots()
    line, = ax.plot(x, u, lw=1.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('x');  ax.set_ylabel('u')
    plt.ion(); plt.show()
    errors = {'L2': [], 'Linf': []}
    times  = []
    for i in range(1, nsteps+1):
        if name == 'spectral':
            u = scheme_func(u, dt, dx)
        elif name == 'spectral_rk4':
            u = scheme_func(u, dt, dx, c)
        elif name == 'spectral_implicit':
            u = scheme_func(u, dt, dx, c)
        elif name == 'upwind':
            u = scheme_func(u, nu, c)
        else:
            u = scheme_func(u, nu)
        if i % step_plot == 0 or i == nsteps:
            t = i*dt
            u_exact = exact_solution(u0, x, c, t)
            err_L2  = np.linalg.norm(u - u_exact)/np.sqrt(N)
            err_inf = np.max(np.abs(u - u_exact))
            errors['L2'].append(err_L2)
            errors['Linf'].append(err_inf)
            times.append(t)
            line.set_ydata(u)
            ax.set_title(f"{name} — t={t:.3f}, L2={err_L2:.2e}, Linf={err_inf:.2e}")
            plt.pause(0.01)
    plt.ioff(); plt.show()
    return times, errors

params = {
    'N_x': 1280,
    'c': 1.0,
    'CFL': 0.01,
    'T_final': 0.5,
    'step_plot': 50,
    'pos': 1280//2
}

run_scheme('FTCS', init_step, scheme_ftcs, params)
run_scheme('upwind', init_gaussian, scheme_upwind, params)
run_scheme('lax_friedrichs', init_gaussian, scheme_lax_friedrichs, params)
run_scheme('spectral', init_gaussian, scheme_spectral, params)
run_scheme('spectral_rk4', init_step, scheme_spectral_rk4, params)
run_scheme('spectral_implicit', init_step, scheme_spectral_implicit, params)

def init_1(x, pos=0):
    x = x * 0
    x[:pos] = 1
    return x

def init_2(x):
    N = x.shape[0]
    mu = (N - 1) / 2.0
    sigma2 = N / 4.0
    i = np.arange(N)
    g = np.exp(- (i - mu) ** 2 / (2 * sigma2))
    return g

def FTCS():
    T_final = 0.5
    step_plot = 100
    c = 1.0
    CFL = 0.01
    N_x = 1280
    Delta_x = 1 / N_x
    Delta_t = CFL / (N_x * c)
    print(c * Delta_t / Delta_x)
    x = np.linspace(0, 1, N_x, endpoint=False)
    u_x = init_1(x, N_x // 2)
    plt.plot(x, u_x)
    plt.show(block=False)
    for i in range(int(T_final / Delta_t)):
        u_x = (
            u_x
            - 0.5 * c * (Delta_t / Delta_x) * (np.roll(u_x, -1) - np.roll(u_x, 1))
        )
        if i % step_plot == 0:
            plt.clf()
            plt.plot(x, u_x)
            plt.pause(0.01)
    plt.show()

def upwind():
    T_final = 0.5
    step_plot = 100
    c = 1.0
    CFL = 0.01
    N_x = 1280
    Delta_x = 1 / N_x
    Delta_t = CFL / (N_x * c)
    nu = c * Delta_t / Delta_x
    print(nu)
    x = np.linspace(0, 1, N_x, endpoint=False)
    u_x = init_2(x)
    plt.plot(x, u_x)
    plt.show(block=False)
    for i in range(int(T_final / Delta_t)):
        if c > 0:
            u_x = u_x - nu * (u_x - np.roll(u_x, 1))
        else:
            u_x = u_x - nu * (np.roll(u_x, -1) - u_x)
        if i % step_plot == 0:
            plt.clf()
            plt.plot(x, u_x)
            plt.pause(0.01)
    plt.show()

def Lax_Friedrichs():
    T_final = 0.1
    step_plot = 100
    c = 1.0
    CFL = 0.01
    N_x = 1280
    Delta_x = 1 / N_x
    Delta_t = CFL / (N_x * c)
    print(c * Delta_t / Delta_x)
    x = np.linspace(0, 1, N_x, endpoint=False)
    u_x = init_2(x)
    plt.plot(x, u_x)
    plt.show(block=False)
    for i in range(int(T_final / Delta_t)):
        u_x = (
            0.5 * (np.roll(u_x, -1) + np.roll(u_x, 1))
            - 0.5 * c * (Delta_t / Delta_x) * (np.roll(u_x, -1) - np.roll(u_x, 1))
        )
        if i % step_plot == 0:
            plt.clf()
            plt.plot(x, u_x)
            plt.pause(0.01)
    plt.show()

def spectral():
    T_final = 0.1
    step_plot = 100
    c = 1.0
    CFL = 0.01
    N_x = 1280
    Delta_x = 1 / N_x
    Delta_t = CFL / (N_x * c)
    print(c * Delta_t / Delta_x)
    x = np.linspace(0, 1, N_x, endpoint=False)
    u_x = init_2(x)
    plt.plot(x, u_x)
    plt.show(block=False)
    for i in range(int(T_final / Delta_t)):
        u_hat = np.fft.fft(u_x)
        k = 2 * np.pi * np.fft.fftfreq(N_x, d=Delta_x)
        du_hat = 1j * k * u_hat
        du_dx = np.fft.ifft(du_hat).real
        u_x = u_x - Delta_t * du_dx
        if i % step_plot == 0:
            plt.clf()
            plt.plot(x, u_x)
            plt.pause(0.01)
    plt.show()

















































