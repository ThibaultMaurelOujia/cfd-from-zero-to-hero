import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.style.use('classic')

def init_gaussian_2d(Nx, Ny, sigma=0.1):
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2*sigma**2))

def scheme_upwind_2d(u, vx, vy, dx, dy):
    u_x = ((u - np.roll(u, 1, axis=0)) * (vx>=0) + (np.roll(u, -1, axis=0) - u) * (vx<0)) / dx
    u_y = ((u - np.roll(u, 1, axis=1)) * (vy>=0) + (np.roll(u, -1, axis=1) - u) * (vy<0)) / dy
    return vx * u_x + u_y * vy

def rk4_step(u, vx, vy, dx, dy, dt):
    k1 = -scheme_upwind_2d(u,              vx, vy, dx, dy)
    k2 = -scheme_upwind_2d(u + 0.5*dt*k1,  vx, vy, dx, dy)
    k3 = -scheme_upwind_2d(u + 0.5*dt*k2,  vx, vy, dx, dy)
    k4 = -scheme_upwind_2d(u +     dt*k3,  vx, vy, dx, dy)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def run_scheme_2d(init_func, scheme_func, params):
    Nx, Ny     = params['Nx'], params['Ny']
    dx, dy     = 1.0/Nx, 1.0/Ny
    vx, vy     = params['vx'], params['vy']
    cfl        = params['CFL']
    plot_every = params['plot_every']
    T          = params['T_final']
    dt_candidates = []
    if vx != 0:
        dt_candidates.append(dx/abs(vx))
    if vy != 0:
        dt_candidates.append(dy/abs(vy))
    if not dt_candidates:
        raise ValueError("vx et vy sont nuls : rien à advecter.")
    dt = cfl * min(dt_candidates)
    u = init_func(Nx, Ny)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(u.T, cmap='viridis', vmin=u.min(), vmax=u.max())
    fig.colorbar(pcm, ax=ax)
    ax.set_title("t = 0.00")
    plt.ion(); plt.show()
    nsteps = int(T/dt)
    for n in range(1, nsteps+1):
        u = scheme_func(u, vx, vy, dx, dy, dt)
        if n % plot_every == 0 or n == nsteps:
            pcm.set_array(u.T.ravel())
            pcm.set_clim(vmin=u.min(), vmax=u.max())
            ax.set_title(f"t = {n*dt:.2f}")
            ax.set_xlim(0, Nx)
            ax.set_ylim(0, Ny)
            plt.pause(0.01)
    plt.ioff(); plt.show()
    return u

params = {
    'Nx': 256,
    'Ny': 256,
    'CFL': 0.5,
    'T_final': 1.0,
    'plot_every': 1,
    'vx': 0.0,
    'vy': -1.0
}

run_scheme_2d(init_gaussian_2d, rk4_step, params)











