import time
import numpy as np
import matplotlib.pyplot as plt







plt.rcParams['text.usetex'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.style.use('classic')



def compute_figsize(Lx, Ly, min_w=7.0, min_h=8.0, max_w=15.0, max_h=10.0):
    ratio = Lx / Ly
    if ratio >= 1.0:
        h = min_h
        w = ratio * h
    else:
        w = min_w
        h = w / ratio
    if w > max_w:
        w = max_w
        h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    return w, h

NL = 9
cxs = np.array([0, 1, 0, -1,  0, 1, -1, -1, 1])
cys = np.array([0, 0, 1,  0, -1, 1,  1, -1,-1])
ws  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

SCALE = 0.1

def init_velocity_pressure_Taylor_Green(Lx, Ly, Nx, Ny):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    ux =  np.sin(2*np.pi * X) * np.cos(2*np.pi * Y)
    uy = -np.cos(2*np.pi * X) * np.sin(2*np.pi * Y)
    return ux, uy

def init_kelvin_helmholtz(Lx, Ly, Nx, Ny, U0=1.0, delta=0.005, amp=0.1, kx=4):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x,y,indexing='ij')
    ux = U0 * np.tanh((Y - 0.5)/delta)
    uy = amp * np.sin(2*np.pi * kx * X) * np.exp(- (Y-0.5)**2 / (2*delta**2))
    return ux, uy 

def init_kelvin_helmholtz_periodic(Lx, Ly, Nx, Ny, U0=1.0, delta=0.005, amp=0.1, kx=4):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x,y,indexing='ij')
    ux = U0 * np.tanh((Y - 0.5)/delta)
    uy = amp * np.sin(2*np.pi * kx * X) * np.exp(- (Y-0.5)**2 / (2*delta**2))
    quarter = Ny // 4
    half    = Ny // 2
    start   = quarter
    end     = quarter + half
    ux_center = ux[:, start:end].copy()
    uy_center = uy[:, start:end].copy()
    ux[:, :start] = ux_center[:, :start][:, ::-1]
    uy[:, :start] = uy_center[:, :start][:, ::-1]
    ux[:, end:] = ux_center[:, -start:][:, ::-1]
    uy[:, end:] = uy_center[:, -start:][:, ::-1]
    rand_coef = 1e-6
    ux += np.random.random(ux.shape)*rand_coef
    uy += np.random.random(uy.shape)*rand_coef
    return ux, uy 

def init_one(Lx, Ly, Nx, Ny):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x,y,indexing='ij')
    ux = X * 0 + 1
    uy = np.zeros_like(ux)
    rand_coef = 1e-6
    ux += np.random.random(ux.shape)*rand_coef
    uy += np.random.random(uy.shape)*rand_coef
    return ux, uy 

def init_velocity_lbm(Lx, Ly, Nx, Ny, init_type='kelvin_helmholtz_periodic'):
    if init_type == 'taylor_green':
        ux, uy = init_velocity_pressure_Taylor_Green(Lx, Ly, Nx, Ny)
    elif init_type == 'kelvin_helmholtz':
        ux, uy = init_kelvin_helmholtz(Lx, Ly, Nx, Ny)
    elif init_type == 'kelvin_helmholtz_periodic':
        ux, uy = init_kelvin_helmholtz_periodic(Lx, Ly, Nx, Ny)
    elif init_type == 'one':
        ux, uy = init_one(Lx, Ly, Nx, Ny)
    else:
        raise ValueError(f"init_type inconnu '{init_type}'")
    ux, uy = ux * SCALE, uy * SCALE
    rho = np.ones((Nx, Ny)) 
    F = np.zeros((Nx, Ny, NL))
    usqr = ux**2 + uy**2
    for i in range(NL):
        cu = cxs[i]*ux + cys[i]*uy
        F[:,:,i] = ws[i] * rho * (1 + 3*cu + 9/2*cu**2 - 3/2*usqr)
    return F, ux, uy

def create_circular_mask(Lx, Ly, Nx, Ny, xc, yc, r):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return ((X - xc)**2 + (Y - yc)**2) > r*r

def create_square_mask(Lx, Ly, Nx, Ny, xc, yc, half_side):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    mask = (np.abs(X - xc) > half_side) | (np.abs(Y - yc) > half_side)
    return mask

def create_mask(Lx, Ly, Nx, Ny, mask_type='circular', xc=None, yc=None, r=None):
    if mask_type == 'circular':
        if xc is None or yc is None or r is None:
            raise ValueError("Pour un masque circulaire, xc, yc et r doivent être fournis")
        return create_circular_mask(Lx, Ly, Nx, Ny, xc, yc, r)
    elif mask_type == 'square':
        if xc is None or yc is None or r is None:
            raise ValueError("Pour 'square', xc, yc et r (demi‑côté) doivent être fournis")
        return create_square_mask(Lx, Ly, Nx, Ny, xc, yc, r)
    elif mask_type == 'none':
        return np.ones((Nx, Ny), dtype=bool)
    else:
        raise ValueError(f"Type de masque inconnu : {mask_type!r}")

def apply_bcs_periodic(F):
    return F

def apply_bcs_inflow_outflow_x(F):
    F[:,-1,[3,6,8]] = F[:,-2,[3,6,8]]
    F[:,0,[1,5,8]] = F[:,1,[1,5,8]]
    return F

def apply_bcs(F, bc_type='periodic'):
    if bc_type == 'periodic':
        F = apply_bcs_periodic(F)
    elif bc_type == 'inflow_outflow_x':
        F = apply_bcs_inflow_outflow_x(F)
    else:
        raise ValueError(f"bc_type inconnu '{bc_type}'")
    return F

def Stream(F_post):
    F_new = np.empty_like(F_post)
    for i, cx, cy in zip(range(NL), cxs, cys):
        F_new[:,:,i] = np.roll(np.roll(F_post[:,:,i], cx, axis=0), cy, axis=1)
    return F_new

def Bounce_back(F, mask):
    inv = np.array([0,3,4,1,2,7,8,5,6])
    F_bb = F.copy()
    F_bb[~mask] = F[~mask][:,inv]
    return F_bb

def Collide(F, tau, mask):
    rho = np.sum(F, axis = 2)
    ux  = np.sum(F * cxs, axis = 2) / rho
    uy  = np.sum(F * cys, axis = 2) / rho
    usqr = ux**2 + uy**2
    ux[~mask], uy[~mask] = 0, 0
    F_eq = np.empty_like(F)
    for i in range(NL):
        cu = cxs[i]*ux + cys[i]*uy
        F_eq[:,:,i] = ws[i] * rho * (1 + 3*cu + 9/2*cu**2 - 3/2*usqr)
    F += -(1./tau)*(F-F_eq)
    return F, ux, uy

def step_lbm(F, ux, uy, dx, dy, tau, mask, bc_type='periodic'):
    F = Stream(F)
    F = Bounce_back(F, mask)
    F, ux, uy = Collide(F, tau, mask)
    F = apply_bcs(F, bc_type=bc_type)
    return F, ux, uy

def run_Lattice_Boltzmann_2d(params):
    Lx, Ly = params.get('Lx', 1.0), params.get('Ly', 1.0)
    Nx, Ny   = params['Nx'], params['Ny']
    tau       = params['tau']
    T_final  = params['T_final']
    plot_every = params['plot_every']
    init_type = params.get('init_type', 'kelvin_helmholtz_periodic')
    bc_type = params['bc_type']
    mask_info = params.get('mask', {'type':'none'})
    mask = create_mask(Lx, Ly, Nx, Ny,
                       mask_type = mask_info['type'],
                       xc        = mask_info.get('xc'),
                       yc        = mask_info.get('yc'),
                       r         = mask_info.get('r'))
    dx, dy = Lx / Nx, Ly / Ny
    F, ux, uy = init_velocity_lbm(Lx, Ly, Nx, Ny, init_type)
    cmap_c = plt.get_cmap('RdBu').copy()
    cmap_c.set_bad('black')
    curl = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2*dx) \
         - (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2*dy)
    curl_plot = np.ma.masked_where(~mask, curl)
    max_abs = np.max(np.abs(curl))
    clim = 2/3 * max_abs
    w, h = compute_figsize(Lx, Ly)
    print(w, h)
    fig2, ax2 = plt.subplots(figsize=(w, h))
    pcm_c = ax2.pcolormesh(curl_plot.T, cmap=cmap_c, shading='auto', vmin=-clim, vmax=clim)
    ax2.set_title('Vorticité + Champ de vitesse normalisé')
    fig2.colorbar(pcm_c, ax=ax2)
    ax2.set_xlim(0, Nx)
    ax2.set_ylim(0, Ny)
    plt.ion()
    plt.show()
    it = 0
    while it < T_final:
        print(f"it={it:4d}")
        F, ux, uy = step_lbm(F, ux, uy, dx, dy, tau, mask=mask, bc_type=bc_type)
        it += 1
        if (it % plot_every == 0 and it > 0):
            curl      = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2*dx) \
                      - (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2*dy)
            curl_plot = np.ma.masked_where(~mask, curl)
            pcm_c.set_array(curl_plot.T.ravel())
            max_abs = np.max(np.abs(curl))
            clim = 2/3 * max_abs
            pcm_c.set_clim(-clim, clim)
            plt.suptitle(f"it = {it}")
            plt.pause(0.01)
    plt.ioff()
    plt.show()
    return F, ux, uy

params = {
    'Lx':       3,
    'Ly':       1,
    'Nx':       256,
    'Ny':       196,    
    'tau':       0.7,
    'T_final': 4000,
    'plot_every': 10,
    'init_type': 'one',
    'bc_type': 'inflow_outflow_x',
    'mask': {
    'type':     'circular',
    'xc':       0.7,
    'yc':       0.5,
    'r':        0.10
    }
}

F, ux, uy = run_Lattice_Boltzmann_2d(params)

















