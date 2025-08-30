import time
import numpy as np
import matplotlib.pyplot as plt



# conda create -n cfd_m2 python=3.10
# conda activate cfd_m2

# CONDA_SUBDIR=osx-64 conda install -c apple numpy scipy

# export OMP_NUM_THREADS=8
# export VECLIB_MAXIMUM_THREADS=8
# conda activate cfd_m2
# python 6-Navier_Stokes_2D.py






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

def init_velocity_pressure_Taylor_Green(Lx, Ly, Nx, Ny):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u =  np.sin(2*np.pi * X) * np.cos(2*np.pi * Y)
    v = -np.cos(2*np.pi * X) * np.sin(2*np.pi * Y)
    p =  0.25 * (np.cos(4*np.pi * X) + np.cos(4*np.pi * Y))
    return u, v, p

def init_kelvin_helmholtz(Lx, Ly, Nx, Ny, U0=1.0, delta=0.005, amp=0.1, kx=4):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x,y,indexing='ij')
    u = U0 * np.tanh((Y - 0.5)/delta)
    v = amp * np.sin(2*np.pi * kx * X) * np.exp(- (Y-0.5)**2 / (2*delta**2))
    p = np.zeros_like(u)
    return u, v, p

def init_kelvin_helmholtz_periodic(Lx, Ly, Nx, Ny, U0=1.0, delta=0.005, amp=0.1, kx=4):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x,y,indexing='ij')
    u = U0 * np.tanh((Y - 0.5)/delta)
    v = amp * np.sin(2*np.pi * kx * X) * np.exp(- (Y-0.5)**2 / (2*delta**2))
    p = np.zeros_like(u)
    quarter = Ny // 4
    half    = Ny // 2
    start   = quarter
    end     = quarter + half
    u_center = u[:, start:end].copy()
    v_center = v[:, start:end].copy()
    p_center = p[:, start:end].copy()
    u[:, :start] = u_center[:, :start][:, ::-1]
    v[:, :start] = v_center[:, :start][:, ::-1]
    p[:, :start] = p_center[:, :start][:, ::-1]
    u[:, end:] = u_center[:, -start:][:, ::-1]
    v[:, end:] = v_center[:, -start:][:, ::-1]
    p[:, end:] = p_center[:, -start:][:, ::-1]
    rand_coef = 1e-6
    u += np.random.random(u.shape)*rand_coef
    v += np.random.random(v.shape)*rand_coef
    p += np.random.random(p.shape)*rand_coef
    return u, v, p

def init_one(Lx, Ly, Nx, Ny):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x,y,indexing='ij')
    u = X * 0 + 1
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    rand_coef = 1e-6
    u += np.random.random(u.shape)*rand_coef
    v += np.random.random(v.shape)*rand_coef
    p += np.random.random(p.shape)*rand_coef
    return u, v, p

def init_velocity_pressure(Lx, Ly, Nx, Ny, init_type='kelvin_helmholtz_periodic'):
    if init_type == 'taylor_green':
        return init_velocity_pressure_Taylor_Green(Lx, Ly, Nx, Ny)
    elif init_type == 'kelvin_helmholtz':
        return init_kelvin_helmholtz(Lx, Ly, Nx, Ny)
    elif init_type == 'kelvin_helmholtz_periodic':
        return init_kelvin_helmholtz_periodic(Lx, Ly, Nx, Ny)
    elif init_type == 'one':
        return init_one(Lx, Ly, Nx, Ny)
    else:
        raise ValueError(f"init_type inconnu '{init_type}'")





# def apply_source_term(u, v, dt):
#     forcing_u = np.zeros_like(u)
#     forcing_v = np.zeros_like(v)
#     forcing_u[0, :] = 3.0
#     forcing_u[1, :] = 3.0
#     forcing_u[2, :] = 3.0
#     forcing_u[-1, :] = 3.0
#     forcing_u[-2, :] = 3.0
#     forcing_u[-0, :] = 3.0
#     u[0, :] += forcing_u[0, :] #* dt
#     return u, v





 
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

def apply_mask_div(q, mask, layers=1):
    q[~mask] = 0.0
    solid = ~mask
    dilated = np.zeros_like(solid, dtype=bool)
    for r in range(1, layers+1):
        dilated |= np.roll(solid,  ( r,  0), axis=(0,1))
        dilated |= np.roll(solid,  (-r, 0), axis=(0,1))
        dilated |= np.roll(solid,  ( 0,  r), axis=(0,1))
        dilated |= np.roll(solid,  ( 0, -r), axis=(0,1))
    fluid_border = mask & dilated
    q[fluid_border] = 0.0
    return q

def apply_immersed_boundary(u, v, mask, R=1, factor = -1):
    if mask is None:
        return u, v
    u_orig = u.copy()
    v_orig = v.copy()
    u_bc   = u.copy()
    v_bc   = v.copy()
    u_bc[~mask] = 0.0
    v_bc[~mask] = 0.0
    fluid_left  = np.roll(mask,  1, axis=0) & (~mask)
    fluid_right = np.roll(mask, -1, axis=0) & (~mask)
    fluid_down  = np.roll(mask,  1, axis=1) & (~mask)
    fluid_up    = np.roll(mask, -1, axis=1) & (~mask)
    for r in range(1, R+1):
        ghost = np.roll(fluid_left,  r-1, axis=0)
        u_bc[ghost] = factor * np.roll(u_orig,  r, axis=0)[ghost]
        ghost = np.roll(fluid_right, -(r-1), axis=0)
        u_bc[ghost] = factor * np.roll(u_orig, -r, axis=0)[ghost]
        ghost = np.roll(fluid_down,  r-1, axis=1)
        v_bc[ghost] = factor * np.roll(v_orig,  r, axis=1)[ghost]
        ghost = np.roll(fluid_up,   -(r-1), axis=1)
        v_bc[ghost] = factor * np.roll(v_orig, -r, axis=1)[ghost]
    return u_bc, v_bc

def ensure_min_thickness(mask, R):
    mask2 = mask.copy()
    Nx, Ny = mask2.shape
    T = 2 * R
    for i in range(Nx):
        j = 0
        while j < Ny:
            if not mask2[i, j]:
                j0 = j
                while j < Ny and not mask2[i, j]:
                    j += 1
                j1 = j
                L = j1 - j0
                if L < T:
                    need = T - L
                    left  = need//2 + (need % 2)
                    right = need//2
                    new_start = max(0, j0 - left)
                    new_end   = min(Ny, j1 + right)
                    mask2[i, new_start:new_end] = False
                    j = new_end
            else:
                j += 1
    for j in range(Ny):
        i = 0
        while i < Nx:
            if not mask2[i, j]:
                i0 = i
                while i < Nx and not mask2[i, j]:
                    i += 1
                i1 = i
                L = i1 - i0
                if L < T:
                    need = T - L
                    top    = need//2 + (need % 2)
                    bottom = need//2
                    new_start = max(0, i0 - top)
                    new_end   = min(Nx, i1 + bottom)
                    mask2[new_start:new_end, j] = False
                    i = new_end
            else:
                i += 1
    return mask2

def apply_bcs_periodic(u, v, p):
    return u, v, p

def apply_bcs_inflow_outflow_x(u, v, p, R=1):
    R = 2*R
    u_in = 4
    u[:R] = u_in
    v[:R] = 0
    p[:R] = 0
    u[-R:] = u[-R-1]
    v[-R:] = v[-R-1]
    p[-R:] = p[-R-1]
    
    # u[-1:] = 2
    # v[-1:] = 0
    # p[-1:] = 0
    
    return u, v, p

def apply_bcs(u, v, p, bc_type='periodic', R=1):
    if bc_type == 'periodic':
        u, v, p = apply_bcs_periodic(u, v, p)
    elif bc_type == 'inflow_outflow_x':
        return apply_bcs_inflow_outflow_x(u, v, p, R=R)
    else:
        raise ValueError(f"bc_type inconnu '{bc_type}'")
    return u, v, p

def weno3_reconstruct_axis(q, axis, eps=1e-6):
    q_hat0 = 3/2 * q - 1/2 * np.roll(q,  1, axis=axis)
    q_hat1 = 1/2 * q + 1/2 * np.roll(q, -1, axis=axis)
    q_hat2 = 3/2 * np.roll(q, -1, axis=axis) - 1/2 * np.roll(q, -2, axis=axis)
    beta0 = (q - np.roll(q,  1, axis=axis))**2
    beta1 = (np.roll(q, -1, axis=axis) - q)**2
    beta2 = (np.roll(q, -2, axis=axis) - np.roll(q, -1, axis=axis))**2
    d0, d1 = 1/3, 2/3
    eps = 1e-6
    alpha0 = d0 / (eps + beta0)**2
    alpha1 = d1 / (eps + beta1)**2
    alpha2 = d0  / (eps + beta2)**2
    sum_minus = alpha0 + alpha1
    omega0_minus = alpha0 / sum_minus
    omega1_minus = alpha1 / sum_minus
    sum_plus = alpha1 + alpha2
    omega1_plus  = alpha1 / sum_plus
    omega2_plus  = alpha2 / sum_plus
    q_half_minus = omega0_minus * q_hat0 + omega1_minus * q_hat1
    q_half_plus  = omega1_plus  * q_hat1 + omega2_plus  * q_hat2
    return q_half_minus, q_half_plus

def compute_advection_weno3_simple(u, v, dx, dy):
    uLx, uRx = weno3_reconstruct_axis(u, axis=0)
    u_half_x = np.where(u > 0, uLx, uRx)
    ux       = (u_half_x - np.roll(u_half_x,  1, axis=0)) / dx
    vLx, vRx = weno3_reconstruct_axis(v, axis=0)
    v_half_x = np.where(u > 0, vLx, vRx)
    vx       = (v_half_x - np.roll(v_half_x,  1, axis=0)) / dx
    uLy, uRy = weno3_reconstruct_axis(u, axis=1)
    u_half_y = np.where(v > 0, uLy, uRy)
    uy       = (u_half_y - np.roll(u_half_y,  1, axis=1)) / dy
    vLy, vRy = weno3_reconstruct_axis(v, axis=1)
    v_half_y = np.where(v > 0, vLy, vRy)
    vy       = (v_half_y - np.roll(v_half_y,  1, axis=1)) / dy
    A_u = u * ux + v * uy
    A_v = u * vx + v * vy
    return A_u, A_v

def compute_advection_weno3_riemann(u, v, dx, dy):
    uLx, uRx = weno3_reconstruct_axis(u, axis=0)
    vLx, vRx = weno3_reconstruct_axis(v, axis=0)
    Fu_L, Fu_R = uLx**2, uRx**2
    alpha_x_u = np.maximum(np.abs(uLx), np.abs(uRx))
    Fu = 0.5*(Fu_L + Fu_R) - 0.5*alpha_x_u*(uRx - uLx)
    Fv_L, Fv_R = uLx*vLx, uRx*vRx
    Fv_x = 0.5*(Fv_L + Fv_R) - 0.5*alpha_x_u*(vRx - vLx)
    uLy, uRy = weno3_reconstruct_axis(u, axis=1)
    vLy, vRy = weno3_reconstruct_axis(v, axis=1)
    Gu_L, Gu_R = uLy*vLy, uRy*vRy
    alpha_y_u = np.maximum(np.abs(vLy), np.abs(vRy))
    Gu = 0.5*(Gu_L + Gu_R) - 0.5*alpha_y_u*(uRy - uLy)
    Gv_L, Gv_R = vLy**2, vRy**2
    Gv = 0.5*(Gv_L + Gv_R) - 0.5*alpha_y_u*(vRy - vLy)
    A_u = (Fu - np.roll(Fu, 1, axis=0)) / dx + (Gu - np.roll(Gu, 1, axis=1)) / dy
    A_v = (Fv_x - np.roll(Fv_x, 1, axis=0)) / dx + (Gv - np.roll(Gv, 1, axis=1)) / dy
    return A_u, A_v

def upwind2_axis(q, vel, d, axis):
    q_im1 = np.roll(q,  1, axis=axis)
    q_im2 = np.roll(q,  2, axis=axis)
    q_ip1 = np.roll(q, -1, axis=axis)
    q_ip2 = np.roll(q, -2, axis=axis)
    return np.where(
        vel >= 0,
        (3*q - 4*q_im1 + q_im2) / (2*d),
        (-3*q + 4*q_ip1 - q_ip2) / (2*d)
    )

def compute_advection(u, v, dx, dy, method_scheme='weno3_simple'):
    if method_scheme == 'upwind2':
        ux = upwind2_axis(u, u, dx, axis=0)
        uy = upwind2_axis(u, v, dy, axis=1)
        vx = upwind2_axis(v, u, dx, axis=0)
        vy = upwind2_axis(v, v, dy, axis=1)
        A_u = u * ux + v * uy
        A_v = u * vx + v * vy
    elif method_scheme == 'weno3_simple':
        return compute_advection_weno3_simple(u, v, dx, dy)
    elif method_scheme == 'weno3_riemann':
        return compute_advection_weno3_riemann(u, v, dx, dy)
    else:
        raise ValueError(f"Unknown advection method '{method_scheme}'")
    return A_u, A_v

def compute_diffusion(u, v, dx, dy, nu):
    D_u = nu * (
        (np.roll(u, -1, axis=0) - 2*u + np.roll(u, +1, axis=0)) / dx**2 +
        (np.roll(u, -1, axis=1) - 2*u + np.roll(u, +1, axis=1)) / dy**2
    )
    D_v = nu * (
        (np.roll(v, -1, axis=0) - 2*v + np.roll(v, +1, axis=0)) / dx**2 +
        (np.roll(v, -1, axis=1) - 2*v + np.roll(v, +1, axis=1)) / dy**2
    )
    return D_u, D_v

import pyamg
from scipy.sparse import diags, eye, kron
def _build_amg_solver(Nx, Ny, dx, dy, strength='symmetric', aggregate=False):
    from scipy.sparse import diags, eye, kron
    e_x = np.ones(Nx)
    Tx = diags([e_x, -2*e_x, e_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
    Tx[0, -1] = 1; Tx[-1, 0] = 1
    e_y = np.ones(Ny)
    Ty = diags([e_y, -2*e_y, e_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
    Ty[0, -1] = 1; Ty[-1, 0] = 1
    Ix = eye(Nx, format='csr')
    Iy = eye(Ny, format='csr')
    A = kron(Tx, Iy) / dx**2 + kron(Ix, Ty) / dy**2
    if aggregate:
        ml = pyamg.smoothed_aggregation_solver(A,
                                               strength=strength)
    else:
        ml = pyamg.ruge_stuben_solver(A,
                                      strength=strength)
    return ml

def solve_pressure_poisson_amg(
    p, b, dx, dy,
    tol=1e-8, maxiter=100,
    full_solve=True 
):
    Nx, Ny = p.shape
    key = (Nx, Ny, round(dx,8), round(dy,8))
    cache = getattr(solve_pressure_poisson_amg, "_ml_cache", {})
    if key not in cache:
        cache[key] = _build_amg_solver(Nx, Ny, dx, dy,
                                       strength='symmetric',
                                       aggregate=True)
        solve_pressure_poisson_amg._ml_cache = cache
    ml = cache[key]
    b_vec = b.ravel()
    if not full_solve:
        P = ml.aspreconditioner()
        x = P(b_vec)
    else:
        x, info = ml.solve(b_vec,
                            tol=tol,
                            maxiter=maxiter,
                            x0=p.ravel(),
                            return_info=True)
        if info > 0:
            res = np.linalg.norm(ml.levels[0].A.dot(x) - b_vec, np.inf)
            print(f"[AMG] pas convergé en {info} iters, résidu ≃ {res:.2e}")
    p_new = x.reshape(Nx, Ny)
    p_new -= p_new.mean()
    return p_new

def solve_pressure_poisson_fft(p, b, dx, dy):
    Nx, Ny = p.shape
    b_zero_mean = b - np.mean(b)
    b_hat = np.fft.rfft2(b_zero_mean)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)[:, None]
    ky = 2 * np.pi * np.fft.rfftfreq(Ny, d=dy)[None, :]
    denom = -(kx**2 + ky**2)
    denom[0, 0] = 1.0
    p_hat = b_hat / denom
    p_hat[0, 0] = 0.0
    p_new = np.fft.irfft2(p_hat, s=(Nx, Ny))
    return p_new - p_new.mean()

from scipy.sparse.linalg import splu
def solve_pressure_poisson_direct(p, b, dx, dy):
    Nx, Ny = p.shape
    b_vec = b.ravel()
    key = (Nx, Ny, round(dx,8), round(dy,8))
    cache = getattr(solve_pressure_poisson_direct, "_direct_cache", {})
    if key not in cache:
        e_x = np.ones(Nx)
        Tx = diags([e_x,-2*e_x,e_x],[-1,0,1],(Nx,Nx),format='csr')
        Tx[0,-1]=1; Tx[-1,0]=1
        e_y = np.ones(Ny)
        Ty = diags([e_y,-2*e_y,e_y],[-1,0,1],(Ny,Ny),format='csr')
        Ty[0,-1]=1; Ty[-1,0]=1
        Ix = eye(Nx,format='csr'); Iy = eye(Ny,format='csr')
        A  = kron(Tx,Iy)/dx**2 + kron(Ix,Ty)/dy**2
        cache[key] = splu(A)
        solve_pressure_poisson_direct._direct_cache = cache
    lu = cache[key]
    x = lu.solve(b_vec)
    p_new = x.reshape(Nx, Ny)
    p_new -= p_new.mean()
    return p_new

_chol_cache = {}
def build_laplacian_matrix(Nx, Ny, dx, dy):
    e_x = np.ones(Nx)
    Tx = diags([e_x, -2*e_x, e_x], [-1, 0, 1], shape=(Nx, Nx), format='csr')
    Tx[0, -1] = Tx[-1, 0] = 1
    e_y = np.ones(Ny)
    Ty = diags([e_y, -2*e_y, e_y], [-1, 0, 1], shape=(Ny, Ny), format='csr')
    Ty[0, -1] = Ty[-1, 0] = 1
    Ix = eye(Nx, format='csr')
    Iy = eye(Ny, format='csr')
    A = kron(Tx, Iy) / dx**2 + kron(Ix, Ty) / dy**2
    return A.tocsc()

def solve_pressure_poisson_cholesky(p, b, dx, dy, eps=1e-8):
    Nx, Ny = p.shape
    key = (Nx, Ny, round(dx,8), round(dy,8))
    if key not in _chol_cache:
        A = build_laplacian_matrix(Nx, Ny, dx, dy)
        A = A + eps * eye(Nx*Ny, format='csc')
        _chol_cache[key] = splu(A)
    lu = _chol_cache[key]
    b_vec = b.ravel()
    b_vec = b_vec - b_vec.mean()
    x = lu.solve(b_vec)
    p_new = x.reshape(Nx, Ny)
    p_new -= p_new.mean()
    return p_new

def solve_pressure_poisson(p, b, dx, dy, method_poisson="direct", full_solve=True): 
    if method_poisson == "direct":
        return solve_pressure_poisson_direct(p, b, dx, dy)
    elif method_poisson == "cholesky":
        return solve_pressure_poisson_cholesky(p, b, dx, dy)
    elif method_poisson == "amg":
        return solve_pressure_poisson_amg(p, b, dx, dy, full_solve=full_solve)
        # return p
    elif method_poisson == "fft":
        return solve_pressure_poisson_fft(p, b, dx, dy)
    else:
        raise ValueError(f"solve_pressure_poisson : méthode inconnue «{method_poisson}»")

def project(u_star, v_star, p, dx, dy, dt):
    dpdx = (np.roll(p, -1,0) - np.roll(p, +1,0))/(2*dx)
    dpdy = (np.roll(p, -1,1) - np.roll(p, +1,1))/(2*dy)
    u_new = u_star - dt * dpdx
    v_new = v_star - dt * dpdy
    return u_new, v_new

def projection_step(u_star, v_star, p_old, dx, dy, dt, method_poisson='direct', full_solve=True, mask=None):
    div_q = (
        (np.roll(u_star,-1,0) - np.roll(u_star,+1,0))/(2*dx)
      + (np.roll(v_star,-1,1) - np.roll(v_star,+1,1))/(2*dy)
    )
    b = div_q / dt
    p_new = solve_pressure_poisson(p_old, b, dx, dy, method_poisson=method_poisson, full_solve=full_solve) 
    u_corr, v_corr = project(u_star, v_star, p_new, dx, dy, dt)
    return u_corr, v_corr, p_new

def rk3_substep(u_old, v_old, p_old, c0_rk3, c1_rk3, dx, dy, dt, nu, 
                full_solve, method_scheme='upwind2', method_poisson='direct', 
                mask=None, bc_type='periodic', factor_immersed_boundary = -1):
    if method_scheme == 'upwind2':
        R = 1
    elif method_scheme == 'weno3_simple' or method_scheme == 'weno3_riemann':
        R = 2
    u_old, v_old, p_old = apply_bcs(u_old, v_old, p_old, bc_type=bc_type, R=R)
    u_old, v_old = apply_immersed_boundary(u_old, v_old, mask=mask, R=R, factor = factor_immersed_boundary)
    
    
    # advection
    A_u, A_v = compute_advection(u_old, v_old, dx, dy, method_scheme=method_scheme)
    u_old, v_old = apply_immersed_boundary(u_old, v_old, mask, R=R, factor = factor_immersed_boundary)
    
    # diffusion
    D_u, D_v = compute_diffusion(u_old, v_old, dx, dy, nu)
    
    # projection
    u_star = u_old + dt * (-A_u + D_u)
    v_star = v_old + dt * (-A_v + D_v)
    u_star, v_star, p_old = apply_bcs(u_star, v_star, p_old, bc_type=bc_type, R=R)
    u_star, v_star = apply_immersed_boundary(u_star, v_star, mask, R=R, factor = factor_immersed_boundary)
    
    # résolution Poisson + correction
    u_corr, v_corr, p_corr = projection_step(
        u_star, v_star, p_old, dx, dy, dt,
        full_solve=full_solve, method_poisson=method_poisson, mask=mask
    )
    u_corr, v_corr, p_corr = apply_bcs(u_corr, v_corr, p_corr, bc_type=bc_type, R=R)
    u_corr, v_corr = apply_immersed_boundary(u_corr, v_corr, mask, R=R, factor = factor_immersed_boundary)
    u_new = c0_rk3*u_old + c1_rk3*u_corr
    v_new = c0_rk3*v_old + c1_rk3*v_corr
    p_new = c0_rk3*p_old + c1_rk3*p_corr
    return u_new, v_new, p_new

def step_rk3(u, v, p, dx, dy, dt, nu, method_scheme='upwind2', method_poisson='direct', mask=None, bc_type='periodic', factor_immersed_boundary = -1):
    u1, v1, p1 = rk3_substep(u,  v,  p, 0.0, 1.0, dx, dy, dt, nu,
                             method_scheme=method_scheme, method_poisson=method_poisson, full_solve=False, 
                             mask=mask, bc_type=bc_type, factor_immersed_boundary=factor_immersed_boundary)
    u2, v2, p2 = rk3_substep(u1, v1, p1, 3/4, 1/4, dx, dy, dt, nu,
                             method_scheme=method_scheme, method_poisson=method_poisson, full_solve=False, 
                             mask=mask, bc_type=bc_type, factor_immersed_boundary=factor_immersed_boundary)
    u3, v3, p3 = rk3_substep(u2, v2, p2, 1/3, 2/3, dx, dy, dt, nu,
                             method_scheme=method_scheme, method_poisson=method_poisson, full_solve=True, 
                             mask=mask, bc_type=bc_type, factor_immersed_boundary=factor_immersed_boundary)
    return u3, v3, p3

def run_navier_stokes_2d(params):
    Lx, Ly = params.get('Lx', 1.0), params.get('Ly', 1.0)
    Nx, Ny   = params['Nx'], params['Ny']
    nu       = params['nu']
    cfl      = params['CFL']
    T_final  = params['T_final']
    plot_every = params['plot_every']
    method_poisson   = params['method_poisson']
    method_scheme = params['method_scheme']
    
    init_type = params.get('init_type', 'kelvin_helmholtz_periodic')
    bc_type = params['bc_type']
    
    mask_info = params.get('mask', {'type':'none'})
    mask = create_mask(Lx, Ly, Nx, Ny,
                       mask_type = mask_info['type'],
                       xc        = mask_info.get('xc'),
                       yc        = mask_info.get('yc'),
                       r         = mask_info.get('r'))
    if method_scheme == 'upwind2':
        R = 1
    elif method_scheme == 'weno3_simple' or method_scheme == 'weno3_riemann':
        R = 2
    mask = ensure_min_thickness(mask, R)
    
    
    dx, dy = Lx / Nx, Ly / Ny
    u, v, p = init_velocity_pressure(Lx, Ly, Nx, Ny, init_type)
    u, v, p    = apply_bcs(u, v, p)
    
    # CFL advective
    umax = max(np.max(np.abs(u)), np.max(np.abs(v)), 1e-8)
    dt_adv = cfl * min(dx/umax, dy/umax)
    # CFL diffusif
    dt_diff = 0.5 * min(dx*dx, dy*dy) / nu
    dt = min(dt_adv, dt_diff)
    
    
    _ = solve_pressure_poisson_direct(np.zeros((Nx,Ny)),
                                  np.zeros((Nx,Ny)),
                                  dx, dy)
    cmap_c = plt.get_cmap('RdBu').copy()
    cmap_c.set_bad('black')
    curl = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dx) \
         - (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
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
    t = 0.0
    it = 0
    while t < T_final:
        print(f"it={it:4d}  dt={dt:4e}  max|u|={np.max(np.abs(u)):.3e}")
        umax   = max(np.max(np.abs(u)), np.max(np.abs(v)), 1e-8)
        dt_adv = cfl * min(dx/umax, dy/umax)
        dt_diff= 0.5 * min(dx*dx, dy*dy) / nu
        dt     = min(dt_adv, dt_diff)
        u, v, p = step_rk3(u, v, p, dx, dy, dt,
                           nu, method_scheme=method_scheme, method_poisson=method_poisson, 
                           mask=mask, bc_type=bc_type)
        t  += dt
        it += 1
        if it % plot_every == 0 or t >= T_final:
            curl      = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dx) \
                      - (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
            curl_plot = np.ma.masked_where(~mask, curl)
            pcm_c.set_array(curl_plot.T.ravel())
            max_abs = np.max(np.abs(curl))
            clim = 2/3 * max_abs
            pcm_c.set_clim(-clim, clim)
            ax2.set_title(f"Vorticité t = {t:.3f}")
            plt.pause(0.01)
    plt.ioff()
    plt.show()
    return u, v, p

params = {
    'Lx':       3,
    'Ly':       1,
    'Nx':       128,
    'Ny':       64,
    'CFL':      0.1,       
    'nu':       1e-6,
    'T_final': 20.0,
    'plot_every': 5,
    'method_poisson':  'direct',
    'method_scheme': 'weno3_riemann',
    'init_type': 'one',
    'bc_type': 'inflow_outflow_x',
    'mask': {
    'type':     'square',
    'xc':       0.4,
    'yc':       0.5,
    'r':        0.20
    }
}

run_navier_stokes_2d(params)





















































