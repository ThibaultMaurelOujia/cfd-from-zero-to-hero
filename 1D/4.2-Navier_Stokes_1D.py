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



 
def normal_shock_relations(M1, gamma=1.4): 
    p_ratio   = (2*gamma*M1**2 - (gamma-1)) / (gamma+1)
    rho_ratio = ((gamma+1)*M1**2) / ((gamma-1)*M1**2 + 2)
    T_ratio   = p_ratio / rho_ratio
    M2 = np.sqrt((M1**2*(gamma-1)+2) / (2*gamma*M1**2 - (gamma-1)))
    return M2, p_ratio, rho_ratio, T_ratio

# ----------------------------------------------------------------------
#  Rankine–Hugoniot
# ----------------------------------------------------------------------
def init_rankine_hugoniot(N,
                          M=2.03,
                          p0_atm=1.0,
                          T0=500.0,
                          x0=0.5,
                          domain_length=1.0,
                          gamma=1.4,
                          molecular_weight=2e-3):
     
    R_univ = 8.314462618  
    R_sp   = R_univ / molecular_weight 
    p1     = p0_atm * 101325.0  
    rho1   = p1 / (R_sp * T0)
    a1     = np.sqrt(gamma * p1 / rho1)
    U1     = M * a1       

     
    M2, p_ratio, rho_ratio, T_ratio = normal_shock_relations(M, gamma)
    p2   = p_ratio * p1
    rho2 = rho_ratio * rho1
    T2   = T_ratio * T0

     
    U2 = rho1 / rho2 * U1

     
    E1 = p1/(gamma-1) + 0.5 * rho1 * U1**2
    E2 = p2/(gamma-1) + 0.5 * rho2 * U2**2

     
    U = np.zeros((3, N))
    x = np.linspace(0, domain_length, N, endpoint=False)
    idx = x < x0*domain_length

     
    U[0, idx] = rho1
    U[1, idx] = rho1 * U1
    U[2, idx] = E1

     
    U[0, ~idx] = rho2
    U[1, ~idx] = rho2 * U2
    U[2, ~idx] = E2

    return U




# =============================================================================
# 
# =============================================================================


# https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf
# https://www.researchgate.net/figure/Linear-weights-for-the-WENO3-WENO5-and-WENO7-schemes_tbl4_283344505
# https://www.mathworks.com/matlabcentral/fileexchange/40956-example-of-weno3-lf-and-weno5-lf-scheme-for-1d-buckey-leverett-problem
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

     
    mask_L         = (S_L >= 0)
    mask_star_L    = (S_L < 0) & (S_star >= 0)
    mask_star_R    = (S_star < 0) & (S_R >= 0)
    mask_R         = (S_R < 0)
 
    F_hllc                 = np.empty_like(U_L)  # shape (3, N_interfaces)
    F_hllc[:, mask_L]      = F_L[:, mask_L]
    F_hllc[:, mask_star_L] = (F_L[:, mask_star_L] + S_L[mask_star_L] * (U_L_star[:, mask_star_L] - U_L[:, mask_star_L]))
    F_hllc[:, mask_star_R] = (F_R[:, mask_star_R] + S_R[mask_star_R] * (U_R_star[:, mask_star_R] - U_R[:, mask_star_R]))
    F_hllc[:, mask_R]      = F_R[:, mask_R]

    return F_hllc



# def spatial_derivative(U, dx):
    
#     U_L, U_R = weno3_reconstruct(U)
    
#     F_half = hllc_flux(U_L, U_R)
    
#     div_flux = - (F_half - np.roll(F_half, 1, axis=1)) / dx 
    
#     return div_flux
 
def spatial_derivative(U, dx, mu, Pr, gamma):
    # convectif
    U_L, U_R = weno3_reconstruct(U)
    F_conv   = hllc_flux(U_L, U_R, gamma)                # (3, N_interfaces)
    div_conv = - (F_conv - np.roll(F_conv,1,axis=1)) / dx


    rho = U[0]
    u   = U[1] / rho
    E   = U[2]
    
    p   = (gamma-1)*(E - 0.5*rho*u**2)
    T   = p / rho   
    
    cp    = gamma/(gamma-1)
    kappa = mu * cp / Pr


    du_dx = (np.roll(u,-1) - np.roll(u,1)) / (2*dx)
    dT_dx = (np.roll(T,-1) - np.roll(T,1)) / (2*dx)


    tau = mu * du_dx       
    q   = - kappa * dT_dx       


    tau_i = 0.5*(tau + np.roll(tau, -1))
    q_i   = 0.5*(q   + np.roll(q,   -1))


    u_i = 0.5*(u + np.roll(u,-1))
    F_visc = np.vstack([
        np.zeros_like(u_i),
        tau_i,
        u_i*tau_i + q_i
    ])
 
    div_visc = (np.roll(F_visc, -1, axis=1) - F_visc) / dx

    return div_conv + div_visc


def ssp_rk3_step(u, dt, dx, mu, Pr, gamma): 
    L = lambda v: spatial_derivative(v, dx, mu, Pr, gamma)
    u1 = u + dt * L(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1))
    u3 = (u + 2 * (u2 + dt * L(u2))) / 3.0
    return u3

def scheme_weno3_hllc_ssprk3(u, dt, dx, mu, Pr, gamma):
    return ssp_rk3_step(u, dt, dx, mu, Pr, gamma)








# =============================================================================
# 
# =============================================================================


def run_euler(name, init_func, scheme_func, params): 
    N        = params['N_x']
    CFL      = params['CFL']
    mu       = params['mu']
    # kappa    = params['kappa']
    Pr       = params['Pr']
    gamma    = params.get('gamma', 1.4)
    T_final  = params['T_final']
    plot_every = params.get('plot_every', 10)

 
    dx = 1.0 / N
    print(dx, mu)
     
    
    U = init_func(N)        # U.shape = (3,N)
    rho, m, E = U
    u = m / rho
    p = (gamma-1)*(E - 0.5*rho*u**2)
    a = np.sqrt(gamma*p/rho)
    max_speed = np.max(np.abs(u) + a)
    dt = CFL * dx / max_speed
    
    print(λ, dx, dt, mu)

    nsteps = int(T_final / dt)
 
    x = np.linspace(0,1,N,endpoint=False)
    fig, axes = plt.subplots(3,1,figsize=(10,8))
    lines = []
    
    for ax, var in zip(axes, ['ρ','u','E']):
        if var == 'u':
            y0 = (U[1]/U[0])   # vitesse
        else:
            idx = {'ρ':0,'E':2}[var]
            y0 = U[idx]
        line, = ax.plot(x, y0)
        lines.append((var, line))
        ax.set_ylabel(var)
        ax.legend([var])
    axes[-1].set_xlabel('x')
    plt.ion(); plt.show()


    for n in range(1, nsteps+1):
        U = scheme_func(U, dt, dx, mu, Pr, gamma)

        if n%plot_every==0 or n==nsteps:
            rho, m, E = U
            u = m/rho
            for var, line in lines:
                if var == 'u':
                    line.set_ydata(u)
                else:
                    idx = {'ρ':0,'E':2}[var]
                    line.set_ydata(U[idx])
                axes[['ρ','u','E'].index(var)].relim()
                axes[['ρ','u','E'].index(var)].autoscale_view()
            fig.suptitle(f"t = {n*dt:.3f}")
            plt.pause(1e-3)

    plt.ioff(); plt.show()
    return U




# =============================================================================
# 
# =============================================================================


params = {
    'N_x'      : 2**13,    
    'CFL'      : 0.001,     
    #
    'gamma'    : 1.4,    # rapport des chaleurs  
    # 'mu'       : 1e-5,   # viscosité dynamique
    'mu'       : 11e-6,   # viscosité dynamique
    # 'kappa'    : 1e-3,   # conductivité thermique
    'Pr'       : 0.68,   # nombre de Prandtl    0.71
    #
    'T_final'  : 0.1,    # temps final
    'plot_every': 1,    # intervalle d’affichage
}
 

# run_euler('Euler 1D WENO3+HLLC+RK3', init_sod, scheme_weno3_hllc_ssprk3, params)

# run_euler('Euler 1D WENO3+HLLC+RK3', init_euler_turbulence, scheme_weno3_hllc_ssprk3, params)


run_euler('Euler 1D WENO3+HLLC+RK3', init_rankine_hugoniot, scheme_weno3_hllc_ssprk3, params)












def test_weno3():
    
    import sympy as sp
 
    c_m1, c_0, c_p1 = sp.symbols('c_m1 c_0 c_p1')
    
    i = sp.symbols('i')
     
    eq1 = sp.Eq(c_m1 + c_0 + c_p1, 1)
     
    eq2 = sp.Eq(c_m1*(i-1) + c_0*i + c_p1*(i+1), i + sp.Rational(1,2))
     
    avg_m1 = ((i-1+sp.Rational(1,2))**3 - (i-1-sp.Rational(1,2))**3)/3
    avg_0 = ((i+sp.Rational(1,2))**3 - (i-sp.Rational(1,2))**3)/3
    avg_p1 = ((i+1+sp.Rational(1,2))**3 - (i+1-sp.Rational(1,2))**3)/3
    
    interface_val = (i+sp.Rational(1,2))**2
    
    eq3 = sp.Eq(c_m1*avg_m1 + c_0*avg_0 + c_p1*avg_p1, interface_val)
    
    sol = sp.solve([eq1,eq2,eq3],[c_m1,c_0,c_p1])
    print(sol)
    
    
    
    
    
    a,b,c= sp.symbols('a b c') 
    eq1=sp.Eq(a+b+c,1) 
    eq2=sp.Eq(a*i + b*(i+1) + c*(i+2), i+sp.Rational(1,2)) 
    avg_i = ((i+sp.Rational(1,2))**3 - (i-sp.Rational(1,2))**3)/3
    avg_ip1 = ((i+1+sp.Rational(1,2))**3 - (i+1-sp.Rational(1,2))**3)/3
    avg_ip2 = ((i+2+sp.Rational(1,2))**3 - (i+2-sp.Rational(1,2))**3)/3
    eq3=sp.Eq(a*avg_i + b*avg_ip1 + c*avg_ip2, interface_val)
    sol=sp.solve([eq1,eq2,eq3],[a,b,c])
    print(sol)
    
    
    
    
    
    a,b,c=sp.symbols('a b c') 
    eq1=sp.Eq(a+b+c,1)
    eq2=sp.Eq(a*(i-2) + b*(i-1) + c*i, i+sp.Rational(1,2))
    avg_im2= ((i-2+sp.Rational(1,2))**3 - (i-2-sp.Rational(1,2))**3)/3
    avg_im1= ((i-1+sp.Rational(1,2))**3 - (i-1-sp.Rational(1,2))**3)/3
    avg_i = ((i+sp.Rational(1,2))**3 - (i-sp.Rational(1,2))**3)/3
    eq3=sp.Eq(a*avg_im2 + b*avg_im1 + c*avg_i, interface_val)
    sol=sp.solve([eq1,eq2,eq3],[a,b,c])
    print(sol)

    
    
    
    
    












