import time
import numpy as np
import matplotlib.pyplot as plt








plt.rcParams['text.usetex'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.style.use('classic')






# =============================================================================
# 
# =============================================================================

def init_euler_turbulence(N, _=None):
    
    exponent=2.0
    gamma=1.4
    p0=1.0
     
    k = np.fft.fftfreq(N, d=1.0/N) 
    phi = 2*np.pi * np.random.rand(N) 
    A = np.where(k!=0, np.abs(k)**(-exponent/2), 0.0) 
    u_hat = A * np.exp(1j*phi)
    u = np.fft.ifft(u_hat).real 
    u -= u.mean()
    u /= u.std()
 
    rho = np.ones(N)                
    mom = rho * u               
    
    E   = p0/(gamma - 1) + 0.5 * rho * u**2

    return np.vstack([rho, mom, E])


def init_sod(N, x0=0.5): 
    x = np.linspace(0,1,N,endpoint=False)
    U = np.zeros((3,N))
    left  = x < x0
    right = ~left
    
    rho = left*1.0      + right*0.125
    u   = np.zeros(N)
    p   = left*1.0      + right*0.1
    E   = p/(1.4-1) + 0.5*rho*u**2
    U[0] = rho
    U[1] = rho*u
    U[2] = E
    return U




# def init_tuyere(N, A, rho0=1, u0=0.0, p0=1, gamma=1.4):
def init_tuyere(N, A, rho0=1.1768292682926829, u0=0.0, p0=101325, gamma=1.4):
    """
    Initialise U=(ρ A, ρu A, E A) uniforme sur toute la tuyère.
    """
    rho = rho0 * np.ones(N)
    u   = u0  * np.ones(N)
    E   = p0/(gamma-1) + 0.5*rho*u**2

    U = np.zeros((3, N))
    # print(A)
    U[0] = rho * A
    U[1] = rho * u * A
    U[2] = E   * A
    return U

def apply_bc(U, rho_in, u_in, p_in, A, rho_out=1, u_out=0, p_out=1, gamma=1.4, R=2): 
    
    E_in = p_in/(gamma-1) + 0.5 * rho_in * u_in**2 
    U[0, :R] = rho_in * A[:R]
    U[1, :R] = rho_in * u_in * A[:R]
    U[2, :R] = E_in * A[:R]

 
     
    E_out = p_out/(gamma-1) + 0.5 * rho_out * u_out**2
    U[0, -R:] = rho_out * A[-R:]
    U[1, -R:] = rho_out * u_out * A[-R:]
    U[2, -R:] = E_out * A[-R:]
    
    
    # U[1, -R:] = U[1, -R-30]# * A[-R:]
    
    # U[0, -R:] = U[0, -R-1] * A[-R:]
    # U[1, -R:] = U[1, -R-1] * A[-R:]
    # U[2, -R:] = U[2, -R-1] * A[-R:]
    
    
    # E_out = p_out/(gamma-1) + 0.5 * rho_out * u_out**2
    # U[0, -R:] = rho_out * A[-R:]
    # U[1, -R:] = U[1, -R-1] * A[-R:]
    # U[2, -R:] = E_out * A[-R:]
    return U
    
# Mach_in = 0.5
# u_in = Mach_in * np.sqrt(gamma * p_in / rho_in)



# =============================================================================
# 
# =============================================================================



def nozzle_area(N, A_th=3.0, A_out=0.0): # 3, 0.2 # 3, 0 
    
    x = np.linspace(0,1,N,endpoint=False)
    
    # A0 = np.sin(3*np.pi * x) / A_th - 1 / 2 - x * A_out
    A0 = np.sin(np.pi * x) / A_th - 1 / 2 - x * A_out
    A1 = - A0 + x * A_out
    
    A = A1 - A0
    
    dx = 1.0/N
    dA_dx = (np.roll(A, -1) - np.roll(A, +1)) / (2*dx)
    dA_dx[0] = dA_dx[1]
    dA_dx[-1] = dA_dx[-2]
    
    return A, dA_dx
    # return A/A * 1, dA_dx * 0




def weno3_reconstruct(U):  
    U_hat0 =  3/2 * U - 1/2 * np.roll(U,  1, axis=1) 
    U_hat1 =  1/2 * U + 1/2 * np.roll(U, -1, axis=1) 
    U_hat2 =  3/2 * np.roll(U, -1, axis=1) - 1/2 * np.roll(U, -2, axis=1)
 
    beta0 = (U - np.roll(U,  1, axis=1))**2
    beta1 = (np.roll(U, -1, axis=1) - U)**2
    beta2 = (np.roll(U, -2, axis=1) - np.roll(U, -1, axis=1))**2
 
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
 
    U_half_minus = omega0_minus * U_hat0 + omega1_minus * U_hat1
    U_half_plus  = omega1_plus  * U_hat1 + omega2_plus  * U_hat2

    return U_half_minus, U_half_plus


def hllc_flux(U_L, U_R, gamma=1.4): 
    
    rho_L = U_L[0]
    u_L   = U_L[1] / rho_L
    E_L   = U_L[2]
    p_L   = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    a_L   = (gamma * p_L / rho_L)**0.5

    rho_R = U_R[0]
    u_R   = U_R[1] / rho_R
    E_R   = U_R[2]
    p_R   = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
    a_R   = (gamma * p_R / rho_R)**0.5
     
    S_L = np.minimum(u_L - a_L, u_R - a_R)
    S_R = np.maximum(u_L + a_L, u_R + a_R)
     
    numer = p_R - p_L + rho_L*u_L*(S_L - u_L) - rho_R*u_R*(S_R - u_R)
    denom = rho_L*(S_L - u_L) - rho_R*(S_R - u_R)
    S_star = numer / denom
 
    rho_L_star = rho_L * (S_L - u_L) / (S_L - S_star)
    rho_R_star = rho_R * (S_R - u_R) / (S_R - S_star) 
    p_star = p_L + rho_L*(S_L - u_L)*(S_star - u_L)
 
    E_L_star = ((S_L - u_L)*E_L - p_L*u_L + p_star*S_star) / (S_L - S_star)
    E_R_star = ((S_R - u_R)*E_R - p_R*u_R + p_star*S_star) / (S_R - S_star)
 
    U_L_star = np.array([rho_L_star,
                         rho_L_star * S_star,
                         E_L_star])
    U_R_star = np.array([rho_R_star,
                         rho_R_star * S_star,
                         E_R_star]) 
    
    def flux(U, p):
        rho, m, E = U
        u = m/rho
        return np.array([m,
                         m*u + p,
                         u*(E + p)])
    
    F_L = flux(U_L, p_L)
    F_R = flux(U_R, p_R)
    # F_L_star = flux(U_L_star, p_star)
    # F_R_star = flux(U_R_star, p_star)

    # 
    # if S_L >= 0:
    #     return F_L
    # elif S_L <= 0 <= S_star:
    #     return F_L + S_L*(U_L_star - U_L) # flux(U_L_star, p_star)
    # elif S_star <= 0 <= S_R:
    #     return F_R + S_R*(U_R_star - U_R) # flux(U_R_star, p_star)
    # else:  # S_R <= 0
    #     return F_R
     
    mask_L         = (S_L >= 0)
    mask_star_L    = (S_L < 0) & (S_star >= 0)
    mask_star_R    = (S_star < 0) & (S_R >= 0)
    mask_R         = (S_R < 0)
 
    F_hllc                 = np.empty_like(U_L)  # shape (3, N_interfaces)
    F_hllc[:, mask_L]      = F_L[:, mask_L]
    F_hllc[:, mask_star_L] = (F_L[:, mask_star_L] + S_L[mask_star_L] * (U_L_star[:, mask_star_L] - U_L[:, mask_star_L]))
    F_hllc[:, mask_star_R] = (F_R[:, mask_star_R]+ S_R[mask_star_R] * (U_R_star[:, mask_star_R] - U_R[:, mask_star_R]))
    F_hllc[:, mask_R]      = F_R[:, mask_R]

    return F_hllc



def spatial_derivative(U, dx, A, dA_dx, gamma=1.4):
    
    U_L, U_R = weno3_reconstruct(U)
    
    F_half = hllc_flux(U_L, U_R)
    
    div_flux = - (F_half - np.roll(F_half, 1, axis=1)) / dx 
    
    
    rho = U[0] / A
    mom = U[1] / A
    E   = U[2] / A
    u   = mom / rho
    p   = (gamma - 1) * (E - 0.5 * rho * u**2)

    S = np.zeros_like(U)
    S[1] = p * dA_dx

    return div_flux + S

def ssp_rk3_step(u, dt, dx, A, dA_dx, gamma=1.4):
    """
    SSP RK3 pour u_t + ∂_x f(u) = 0
    """
    
    
    # # u_in = 0.5
    # # rho_in, p_in = 1, 1
    # Mach_in = 0 # 0.001
    # p_in    = 3 # 101325
    # rho_in  = p_in/800# 0.00375 # 0.001 # 1 # 1.225
    # u_in = Mach_in * np.sqrt(gamma * p_in / rho_in)
    # # U = apply_bc(U, rho_in, u_in, p_in, A, gamma=1.4, R=2)
    # # # U = apply_bc(U, rho_in, u_in, p_in, gamma=1.4, R=2) # !!!  
    
    
    # Mach_in = 0 # 0.001
    # p_out   = 10
    # p_in    = 3*p_out # 101325
    # rho_in  = p_in/800# 0.00375 # 0.001 # 1 # 1.225
    # rho_out  = p_out/300# 0.00375 # 0.001 # 1 # 1.225
    # u_in = Mach_in * np.sqrt(gamma * p_in / rho_in)
    
    
    R     = 287.0    
    
    # 3.69      3.7     13      14      20 
    p_atm = 101325        # Pa
    # p_in  = 3.0 * p_atm   # 3 atm
    p_in  = 4 * p_atm   # 3 atm       
    # p_out = 1.0 * p_atm   # 3 atm  
    p_out = 0.1 * p_atm   # 3 atm
    # p_out = 0.1 * p_atm   # 3 atm
    # T_in  = 800.0         # K
    T_in  = 800.0         # K
    T_out = 300.0         # K
     
    rho_in = p_in / (R * T_in)
    rho_out= p_out / (R * T_out)
     
    # Mach_in = 0.0
    # Mach_in = 0.001
    Mach_in = 0.1
    c_in    = np.sqrt(gamma * R * T_in)
    u_in    = Mach_in * c_in
    
    
    
    
    
    L = lambda v: spatial_derivative(v, dx, A, dA_dx, gamma=1.4)
    u1 = u + dt * L(u)
    # u1 = apply_bc(u1, rho_in, u_in, p_in, A, gamma=1.4, R=2)
    u1 = apply_bc(u1, rho_in, u_in, p_in, A, gamma=1.4, R=100, p_out=p_out, rho_out=rho_out)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1))
    # u2 = apply_bc(u2, rho_in, u_in, p_in, A, gamma=1.4, R=2)
    u2 = apply_bc(u2, rho_in, u_in, p_in, A, gamma=1.4, R=100, p_out=p_out, rho_out=rho_out)
    u3 = (u + 2 * (u2 + dt * L(u2))) / 3.0
    # u3 = apply_bc(u3, rho_in, u_in, p_in, A, gamma=1.4, R=2)
    u3 = apply_bc(u3, rho_in, u_in, p_in, A, gamma=1.4, R=100, p_out=p_out, rho_out=rho_out)
    return u3

def scheme_weno3_hllc_ssprk3(u, dt, dx, A, dA_dx, gamma=1.4):
    return ssp_rk3_step(u, dt, dx, A, dA_dx, gamma=1.4)








# =============================================================================
# 
# =============================================================================


def run_euler(name, scheme_func, params):
    
    N        = params['N_x']
    CFL      = params['CFL']
    T_final  = params['T_final']
    plot_every = params.get('plot_every', 10)

    
    # !!!
    # A, dA_dx = nozzle_area(N, A_th=100000.0, A_out=0.0)
    A, dA_dx = nozzle_area(N, A_th=10.0, A_out=0.0)
    A, dA_dx = nozzle_area(N, A_th=2.3, A_out=0.0)
    # A, dA_dx = nozzle_area(N, A_th=4.0, A_out=0.0)
    # A, dA_dx = nozzle_area(N, A_th=3.0, A_out=0.2)
    # A, dA_dx = nozzle_area(N, A_th=3.0, A_out=-0.1)
    # A, dA_dx = nozzle_area(N, A_th=5.0, A_out=-0.3)
    
    
    R     = 287.0
     
    dx = 1.0 / N
    U = init_tuyere(N, A)        # U.shape = (3,N)
    rho, m, E = U
    u = m / rho
    gamma = params.get('gamma', 1.4)
    p = (gamma-1)*(E - 0.5*rho*u**2)
    a = np.sqrt(gamma*p/rho)
    max_speed = np.max(np.abs(u) + a)
    dt = CFL * dx / max_speed
    print(dt)

 
    x = np.linspace(0, 1, N, endpoint=False)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
     
    axes[0].plot(x, A, label='A(x)', c='b')
    axes[0].plot(x, -A, c='b')
    Amax = np.max(np.abs(A))
    axes[0].set_ylim(-Amax, Amax)
    # axes[0].set_ylim(-np.abs(A), np.abs(A))
    axes[0].set_ylabel('A')
    axes[0].legend(loc='best')
     
    p = (gamma - 1)*(U[2] - 0.5*(U[1]**2/U[0]))
    # print(U[2])
    line_p, = axes[1].plot(x, p, label='p')
    axes[1].set_ylabel('p')
    axes[1].legend(loc='best')
    
    
    rho = U[0] / 101325
    T = p / (rho * R)
    line_T, = axes[2].plot(x, T, label='T')
    axes[2].set_ylabel('T')
    axes[2].legend(loc='best')
     
    u = U[1]/rho
    c = np.sqrt(gamma*p/rho)
    M = np.abs(u)/c
    line_M, = axes[3].plot(x, M, label='M')
    axes[3].set_ylabel('M')
    axes[3].set_xlabel('x')
    axes[3].legend(loc='best')
    
    plt.tight_layout()
    plt.ion()
    plt.show()
    
    
    t = 0.0
    it = 0
    while t < T_final:
        
        rho = U[0]
        mom = U[1]
        E   = U[2]
        u   = mom / rho
        p   = (gamma - 1)*(E - 0.5*rho*u**2)
        c   = np.sqrt(gamma*p/rho)
        dt  = CFL * dx / np.max(np.abs(u) + c)
        
        
        U = scheme_weno3_hllc_ssprk3(U, dt, dx, A, dA_dx, gamma)
    
    
        t  += dt
        it += 1
    
        if it % plot_every == 0 or t >= T_final:
            print(it, dt)
            
            rho = U[0]; mom = U[1]; E = U[2]
            u   = mom / rho
            p   = (gamma - 1)*(E - 0.5*rho*u**2)
            T   = p / (rho * R)
            c   = np.sqrt(gamma*p/rho)
            M   = np.abs(u)/c
            # print('----------------------------------')
            # print(p)
            # print(E)
            # print(rho)
            # print(u)
    
            line_p.set_ydata(p / 101325)
            line_T.set_ydata(T)
            line_M.set_ydata(M)
    
            for ax in axes[1:]:
                ax.relim()
                ax.autoscale_view()
    
            fig.suptitle(f"t = {t:.3f}")
            plt.pause(1e-3)

    plt.ioff(); plt.show()
    return U




# =============================================================================
# 
# =============================================================================


params = {
    'N_x'      : 1024,    
    'CFL'      : 0.5,   
    'T_final'  : 100,   
    'plot_every': 50,    
    'gamma'    : 1.4,  
}



run_euler('Euler 1D WENO3+HLLC+RK3', scheme_weno3_hllc_ssprk3, params)

# run_euler('Euler 1D WENO3+HLLC+RK3', init_euler_turbulence, scheme_weno3_hllc_ssprk3, params)




















# =============================================================================
# 
# =============================================================================





def theoretical():
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq
    
    def nozzle_area(N, A_th=3.0, A_out=0.0):
        x = np.linspace(0, 1, N, endpoint=False)
        A0 = np.sin(np.pi * x) / A_th - 1/2 - x * A_out
        A1 = -A0 + x * A_out
        A = A1 - A0
        return x, A
    
    def theoretical_mach_from_area(A, gamma=1.4):
        A = np.asarray(A)
        i_star = np.argmin(A)
        A_star = A[i_star]
        
        def area_mach_function(M, R):
            term = (2/(gamma+1))*(1 + 0.5*(gamma-1)*M*M)
            return (1.0/M) * term**((gamma+1)/(2*(gamma-1))) - R
        
        M = np.zeros_like(A)
        eps = 1e-6
        for i, Ai in enumerate(A):
            R = Ai / A_star
            # Si on est au throat
            if abs(R - 1.0) < 1e-6:
                M[i] = 1.0
            elif i < i_star:
                # branche subsonique
                M[i] = brentq(area_mach_function, eps, 1 - eps, args=(R,))
            else:
                # branche supersonique
                M[i] = brentq(area_mach_function, 1 + eps, 10.0, args=(R,))
        return M
    
    # Configuration et tracé
    N = 200
    configs = [
        {"A_th": 2.3, "A_out": 0.0, "label": "A_th=2.3, A_out=0.0"},
        {"A_th": 3.0, "A_out": 0.2, "label": "A_th=3.0, A_out=0.2"}
    ]
    
    plt.figure(figsize=(8, 5))
    for cfg in configs:
        x, A = nozzle_area(N, cfg["A_th"], cfg["A_out"])
        M = theoretical_mach_from_area(A)
        plt.plot(x, M, label=cfg["label"])
    
    plt.xlabel("Position x (non-dimensionnel)")
    plt.ylabel("Mach M(x)")
    plt.title("Distribution du nombre de Mach isentropique\npour différentes tuyères")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    import numpy as np
    from scipy.optimize import brentq
    
    # 1) Relations pour un choc normal
    def mach_after_shock(M1, gamma=1.4):
        """Mach après choc normal."""
        num = 1 + 0.5*(gamma-1)*M1**2
        den = gamma*M1**2 - 0.5*(gamma-1)
        return np.sqrt(num/den)
    
    def pressure_ratio_normal_shock(M1, gamma=1.4):
        """p2/p1 à travers un choc normal."""
        return 1 + 2*gamma*(M1**2 - 1)/(gamma+1)
    
    # 2) Distribution de Mach isentropique pour un profil de section A(x)
    def theoretical_mach_from_area(A, gamma=1.4):
        A = np.asarray(A)
        i_star = np.argmin(A)
        A_star = A[i_star]
    
        def area_mach_function(M, R):
            term = (2/(gamma+1))*(1 + 0.5*(gamma-1)*M*M)
            return (1.0/M) * term**((gamma+1)/(2*(gamma-1))) - R
    
        M = np.zeros_like(A)
        eps = 1e-6
        for i, Ai in enumerate(A):
            R = Ai / A_star
            if abs(R - 1.0) < eps:
                M[i] = 1.0
            elif i < i_star:
                M[i] = brentq(area_mach_function, eps, 1 - eps, args=(R,))
            else:
                M[i] = brentq(area_mach_function, 1 + eps, 10.0, args=(R,))
        return M
    
    # 3) Fonctions pour le profil de tuyère
    def nozzle_area(N, A_th=2.3, A_out=0.0):
        x = np.linspace(0, 1, N, endpoint=False)
        A0 = np.sin(np.pi * x) / A_th - 1/2 - x * A_out
        A1 = -A0 + x * A_out
        A = A1 - A0
        return x, A
    
    # 4) Pour chaque position x candidate, calcul de p_exit après choc
    def exit_pressure_for_shock(x_idx, A, A_star, A_exit, M_profile, p0, p_out, gamma=1.4):
        M1 = M_profile[x_idx]
        M2 = mach_after_shock(M1, gamma)
        # pression statique avant choc (isentropique)
        p1  = p0 * (1 + 0.5*(gamma-1)*M1**2)**(-gamma/(gamma-1))
        # pression après choc
        p2  = p1 * pressure_ratio_normal_shock(M1, gamma)
        # pression totale après choc
        p02 = p2 * (1 + 0.5*(gamma-1)*M2**2)**(gamma/(gamma-1))
        # Mach théorique en sortie
        Mach_exit = M_profile[-1]
        # pression à la sortie si on reste isentropique depuis M2
        p_exit = p02 * (1 + 0.5*(gamma-1)*Mach_exit**2)**(-gamma/(gamma-1))
        return p_exit - p_out
    
    # 5) Recherche de la position de choc
    N = 200
    x, A     = nozzle_area(N, A_th=2.3, A_out=0.0)
    M_profile = theoretical_mach_from_area(A)
    A_star   = A.min()
    A_exit   = A[-1]
    
    # rapports de pression totales (normalisés sur p0* = pression tot. au throat)
    p0    = 3.0   # p0_in / p0_throat
    p_out = 1.0   # p_exit / p0_throat
    
    def f_pos(pos):
        idx = int(pos * N)
        return exit_pressure_for_shock(idx, A, A_star, A_exit, M_profile, p0, p_out)
    
    # on cherche x/L tel que p_exit(p0,p_out) = 0
    x_shock = brentq(f_pos, 0.01, 0.99)
    print("Choc attendu autour de x/L =", x_shock)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









