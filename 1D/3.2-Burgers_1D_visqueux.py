import time
import numpy as np
import matplotlib.pyplot as plt








plt.rcParams['text.usetex'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.style.use('classic')






def init_step(N, pos):
    u = np.zeros(N)
    u[:pos] = 1.0
    return u


def init_sine(N, _=None):
    x = np.linspace(0, 1, N, endpoint=False)
    return np.sin(2 * np.pi * x)


def init_turbulence(N, _=None):
    exponent=2.0
    k = np.fft.fftfreq(N, d=1.0/N)
    phi = 2*np.pi * np.random.rand(N)
    A = np.where(k!=0, np.abs(k)**(-exponent/2), 0.0)
    u_hat = A * np.exp(1j*phi)
    u = np.fft.ifft(u_hat).real
    u -= u.mean()
    u /= u.std()
    return u


def forcing_base(x, phi, k_f=2):
    return np.sin(2*np.pi * k_f * x + phi)


def forcing_term(u, dx, A, k_f, phi):
    x = np.linspace(0,1,u.size,endpoint=False)
    return A * np.sin(2*np.pi*k_f*x + phi)


def weno3_upwind_flux(u):
    u_0 = 1.5 * u - 0.5 * np.roll(u, 1)
    u_1 = 0.5 * u + 0.5 * np.roll(u, -1)
    beta_0 = (u - np.roll(u, 1))**2
    beta_1 = (np.roll(u, -1) - u)**2
    d_0, d_1 = 2/3, 1/3
    eps = 1e-6
    alpha_0 = d_0 / (eps + beta_0)**2
    alpha_1 = d_1 / (eps + beta_1)**2
    omega_0 = alpha_0 / (alpha_0 + alpha_1)
    omega_1 = alpha_1 / (alpha_0 + alpha_1)
    u_new = omega_0 * u_0 + omega_1 * u_1
    f_new = 0.5 * u_new**2
    F_half = np.where(u_new > 0, f_new, np.roll(f_new, -1))
    return F_half


def spatial_derivative(u, nu, dx):
    F_half = weno3_upwind_flux(u)
    F_conv = -(F_half - np.roll(F_half, 1)) / dx
    F_diff = nu*(np.roll(u,-1) - 2*u + np.roll(u,1)) / dx**2
    return F_conv + F_diff 


def ssp_rk3_step(u, nu, dt, dx):
    L = lambda v: spatial_derivative(v, nu, dx)
    u1 = u + dt * L(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1))
    u3 = (u + 2 * (u2 + dt * L(u2))) / 3.0
    return u3


def scheme_weno3_upwind_ssprk3(u, nu, dt, dx):
    return ssp_rk3_step(u, nu, dt, dx)


def run_scheme_with_spectrum_and_energy(name, init_func, scheme_func, params):
    N      = params['N_x']
    T      = params['T_final']
    dt     = params['dt']
    dx     = params['dx']
    nu     = params['nu']
    
    nsteps = int(T / dt)
    x      = np.linspace(0, 1, N, endpoint=False)
    u0     = init_func(N, params['pos'])
    E0     = 0.5 * np.sum(u0**2) * dx
    F_base = forcing_base(x, phi=1.0, k_f=2)
    k_vals = np.fft.rfftfreq(N, d=dx)
    energy_list = []
    time_list = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    line_u, = ax1.plot(x, u0, lw=1.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-2.1, 2.1)
    ax1.set_ylabel('u')
    ax1.set_title('Profil de vitesse')
    u_hat = np.fft.rfft(u0)
    E = np.abs(u_hat)**2
    line_E, = ax2.loglog(k_vals, E, lw=1.5)

    ax2.set_xlabel('k')
    ax2.set_ylabel('E(k)')
    ax2.set_title("Spectre d'énergie")
    ax2.set_xlim(k_vals[1], k_vals[-1])
    ax2.set_ylim(10**int(np.log10(E[E>1e-10].min())), 10**int(np.log10(E[E<1e50].max()))) 
    E_tot = 0.5 * np.sum(u0**2) * dx
    energy_list.append(E_tot)
    time_list.append(0.0)
    line_energy, = ax3.plot(time_list, energy_list, lw=1.5)

    ax3.set_xlabel('t')
    ax3.set_ylabel('Énergie totale')
    ax3.set_title("Énergie totale vs temps")
    ax3.set_xlim(0, T)
    plt.tight_layout()
    plt.ion()
    plt.show()
    phi = 0
    u = u0.copy()


    for n in range(1, nsteps+1):
        u = scheme_func(u, nu, dt, dx)
        line_u.set_ydata(u)
        u_hat = np.fft.rfft(u)
        E = np.abs(u_hat)**2
        line_E.set_data(k_vals, E)
        ax2.set_ylim(10**int(np.log10(E[E>1e-10].min())), 10**int(np.log10(E[E<1e50].max()))) 
        E_curr = 0.5 * np.sum(u**2) * dx
        dE      = E0 - E_curr
        phi     += (np.random.rand() - 0.5) * dt
        F_base  = forcing_base(x, phi=phi, k_f=2)
        I_F     = np.sum(u * F_base) * dx
        max_inj_frac = dt**(1/2)
        max_inj      = max_inj_frac * E0
        dE_step = np.clip(dE, -max_inj, max_inj)
        if abs(I_F) < 1e-14:
            A_inj = 0.0
        else:
            A_inj = dE_step / (dt * I_F)
        u += dt * A_inj * F_base
        t = n * dt
        E_tot = 0.5 * np.sum(u**2) * dx
        energy_list.append(E_tot)
        time_list.append(t)
        line_energy.set_data(time_list, energy_list)
        ax3.set_ylim(min(energy_list), max(energy_list))
        if n % params['plot_every'] == 0 or n == nsteps:
            ax1.set_title(f"{name} — t={n*dt:.3f}")
            plt.pause(0.01)
    plt.ioff()
    plt.show()
    return u


params = {
    'N_x': 1024,
    'dx'  : 1/1024,
    # 'dx'  : 1/2048,
    # 'nu'  : 1e-2,
    # 'nu'  : 1e-3,
    # 'nu'  : 1e-4,
    'nu'  : 1e-5,
    'dt'  : 0.0001,
    'T_final': 1,
    'plot_every': 5,
    'pos': 256
}

run_scheme_with_spectrum_and_energy('scheme_weno3_upwind_ssprk3', init_turbulence, scheme_weno3_upwind_ssprk3, params)





















