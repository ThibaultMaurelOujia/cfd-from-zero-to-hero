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

def scheme_diffusion_2d(u, nu, dx, dy):
    lap_x = np.roll(u, -1, axis=0) - 2*u + np.roll(u, +1, axis=0)
    lap_y = np.roll(u, -1, axis=1) - 2*u + np.roll(u, +1, axis=1)
    return u + nu*(lap_x + lap_y)

def run_diffusion_2d(init_func, scheme_func, params):
    Nx, Ny    = params['Nx'], params['Ny']
    D         = params['D']
    cfl       = params['CFL']
    T         = params['T_final']
    plot_every= params['plot_every']

    dx = dy = 1.0 / Nx
    dt = cfl * dx*dx / (4*D)
    nu = D * dt / (dx*dx)
    u = init_func(Nx, Ny)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(u.T, cmap='inferno', vmin=u.min(), vmax=u.max())
    fig.colorbar(pcm, ax=ax, label='u')
    ax.set_title("t = 0.00")
    plt.ion(); plt.show()
    nsteps = int(T / dt)
    for n in range(1, nsteps+1):
        u = scheme_func(u, nu, dx, dy)
        if n % plot_every == 0 or n == nsteps:
            pcm.set_array(u.T.ravel())
            pcm.set_clim(vmin=u.min(), vmax=u.max())
            ax.set_title(f"t = {n*dt:.3f}")
            ax.set_xlim(0, Nx)
            ax.set_ylim(0, Ny)
            plt.pause(0.01)
    plt.ioff(); plt.show()
    return u

params = {
    'Nx': 256,
    'Ny': 256,
    'D':   0.1,
    'CFL': 0.5,
    'T_final': 1.0,
    'plot_every': 10
}

run_diffusion_2d(init_gaussian_2d, scheme_diffusion_2d, params)



























