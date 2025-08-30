import time
import numpy as np
import matplotlib.pyplot as plt

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

def scheme_upwind(u, lam):
    U = u**2/2
    F = U * (u>0) + np.roll(U, -1) * (u<0)
    return u - lam * (F - np.roll(F, +1))

def scheme_lax_friedrichs(u, lam):
    return 1 / 2 * (np.roll(u, -1) + np.roll(u, +1)) - 1 / 2 * lam * (np.roll(u**2/2, -1) - np.roll(u**2/2, +1))

def scheme_lax_wendroff(u, lam):
    a = 0.5*(u + np.roll(u, -1))
    term1 = -0.5*lam*(np.roll(u**2/2,-1) - np.roll(u**2/2,1))
    term2 = 0.5*lam**2*(a*(np.roll(u,-1)-u) - np.roll(a,1)*(u-np.roll(u,1)))
    return u + term1 + term2

def scheme_mac_cormack(u, lam):
    U = u**2/2
    u_star = u - lam * (np.roll(U, -1) - U)
    U_star = u_star**2/2
    return 1 / 2 * (u + u_star) - lam / 2 * (U_star - np.roll(U_star, +1))

def minmod(a, b):
    return np.where(a*b>0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)

def godunov_burgers(uL, uR):
    fL, fR = 0.5*uL**2, 0.5*uR**2
    if uL < uR:
        if uL > 0:
            return fL
        elif uR < 0:
            return fR
        else:
            return 0.0
    else:
        s = 0.5*(uL+uR)
        return fL if s>0 else fR

def scheme_mac_cormack_tvd(u, lam):
    N = u.size
    du_minus = u - np.roll(u, 1)
    du_plus  = np.roll(u, -1) - u
    sigma    = minmod(du_minus, du_plus)
    uL = u + 0.5*sigma
    uR = np.roll(u, -1) - 0.5*np.roll(sigma, -1)
    F_half = np.empty(N)
    for i in range(N):
        F_half[i] = godunov_burgers(uL[i], uR[i])
    u_star = u - lam*(F_half - np.roll(F_half, 1))
    du_minus_s = u_star - np.roll(u_star, 1)
    du_plus_s  = np.roll(u_star, -1) - u_star
    sigma_s    = minmod(du_minus_s, du_plus_s)
    uL_s = u_star + 0.5*sigma_s
    uR_s = np.roll(u_star, -1) - 0.5*np.roll(sigma_s, -1)
    F_half_s = np.empty(N)
    for i in range(N):
        F_half_s[i] = godunov_burgers(uL_s[i], uR_s[i])
    u_new = 0.5*(u + u_star) - 0.5*lam*(F_half_s - np.roll(F_half_s, 1))
    return u_new

def weno3(u, lam):
    u_0  = 3 / 2 * u - 1 / 2 * np.roll(u, 1)
    u_1  = 1 / 2 * u + 1 / 2 * np.roll(u, -1)
    beta_0 = (u - np.roll(u, 1))**2
    beta_1 = (np.roll(u, -1) - u)**2
    d_0 = 2 / 3
    d_1 = 1 / 3
    eps = 1e-6
    alpha_0 = d_0 / (eps + beta_0)**2
    alpha_1 = d_1 / (eps + beta_1)**2
    omega_0 = alpha_0 / (alpha_0 + alpha_1)
    omega_1 = alpha_1 / (alpha_0 + alpha_1)
    u_new = omega_0 * u_0 + omega_1 * u_1
    f_new = 1 / 2 * u_new ** 2 
    return u - lam * (f_new - np.roll(f_new, 1))

def weno3_upwind(u, lam):
    u_0 = 1.5*u - 0.5*np.roll(u, 1)
    u_1 = 0.5*u + 0.5*np.roll(u, -1)
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
    return u - lam * (F_half - np.roll(F_half, 1))

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

def spatial_derivative(u, dx):
    F_half = weno3_upwind_flux(u)
    return -(F_half - np.roll(F_half, 1)) / dx

def ssp_rk3_step(u, dt, dx):
    L = lambda v: spatial_derivative(v, dx)
    u1 = u + dt * L(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1))
    u3 = (u + 2 * (u2 + dt * L(u2))) / 3.0
    return u3

def scheme_weno3_upwind_ssprk3(u, dt, dx):
    return ssp_rk3_step(u, dt, dx)

def run_scheme(name, init_func, scheme_func, params):
    N      = params['N_x']
    lam    = params['dt'] / params['dx']
    T      = params['T_final']
    dt     = params['dt']
    dx     = params['dx']
    nsteps = int(T / dt)
    x = np.linspace(0, 1, N, endpoint=False)
    u = init_func(N, params.get('pos', None)).copy()
    fig, ax = plt.subplots()
    line, = ax.plot(x, u, lw=1.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x'); ax.set_ylabel('u')
    plt.ion(); plt.show()
    for n in range(1, nsteps+1):
        if name == 'spectral':
            u = scheme_func(u, dt, dx)
        elif name == 'scheme_weno3_rk4':
            u = scheme_func(u, dt, dx)
        elif name == 'scheme_weno3_upwind_ssprk3':
            u = scheme_func(u, dt, dx)
        else:
            u = scheme_func(u, lam)
        if n % params['plot_every'] == 0 or n==nsteps:
            line.set_ydata(u)
            ax.set_title(f"{name} t={n*dt:.3f}")
            # STOP
        try:
            plt.pause(0.01)
        except OSError:
            time.sleep(0.01)
    plt.ioff(); plt.show()
    return u

params = {
    'N_x': 512,
    'dx'  : 1/512,
    'dt'  : 0.0005,
    'T_final': 0.3,
    'plot_every': 1,
    'pos': 256
}

# upwind
# run_scheme('upwind', init_step, scheme_upwind, params)

# Lax–Friedrichs
# run_scheme('lax_friedrichs', init_step, scheme_lax_friedrichs, params)

# Lax–Wendroff
# run_scheme('lax_wendroff', init_sine, scheme_lax_wendroff, params)

# MacCormack
# run_scheme('mac_cormack', init_step, scheme_mac_cormack, params)
# run_scheme('scheme_mac_cormack_tvd', init_step, scheme_mac_cormack_tvd, params)

# weno3
# run_scheme('weno3', init_step, weno3, params)

# run_scheme('scheme_weno3_rk4', init_step, scheme_weno3_rk4, params)


# run_scheme('weno3', init_step, weno3_upwind, params)
# run_scheme('weno3', init_sine, weno3_upwind, params)
# run_scheme('weno3', init_turbulence, weno3_upwind, params)


# run_scheme('scheme_weno3_upwind_ssprk3', init_step, scheme_weno3_upwind_ssprk3, params)
run_scheme('scheme_weno3_upwind_ssprk3', init_sine, scheme_weno3_upwind_ssprk3, params)
# run_scheme('scheme_weno3_upwind_ssprk3', init_turbulence, scheme_weno3_upwind_ssprk3, params)





# params = {
#     'N_x': 512,
#     'dx'  : 1/512,
#     'dt'  : 0.0001,
#     'T_final': 0.02,
#     'plot_every': 5,
#     'pos': 256
# }


# run_scheme('scheme_weno3_rk4', init_sine, scheme_weno3_rk4, params)
# run_scheme('scheme_weno3_rk4', init_turbulence, scheme_weno3_rk4, params)



















