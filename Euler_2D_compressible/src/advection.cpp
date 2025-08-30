#include "advection.hpp"

namespace navier_stokes {


inline double upwind2ndOrder_axis_x(const Field& q, const Field& vel, double d, int i, int j){
    if (vel[i][j] >= 0) {
        return (+3 * q[i][j] - 4 * q[i-1][j] + q[i-2][j]) / (2 * d);
    }
    else {
        return (-3 * q[i][j] + 4 * q[i+1][j] - q[i+2][j]) / (2 * d);
    }
}

inline double upwind2ndOrder_axis_y(const Field& q, const Field& vel, double d, int i, int j){
    if (vel[i][j] >= 0) {
        return (3 * q[i][j] - 4 * q[i][j-1] + q[i][j-2]) / (2 * d);
    }
    else {
        return (-3 * q[i][j] + 4 * q[i][j+1] - q[i][j+2]) / (2 * d);
    }
}


inline std::pair<double, double> weno3_axis_x(const Field& q, int i, int j){

    double eps = 1e-6;
    double d0 = 1.0/3.0, d1 = 2.0/3.0;

    double q_hat0 = 3.0/2.0 * q[i][j]   - 1.0/2.0 * q[i-1][j];
    double q_hat1 = 1.0/2.0 * q[i][j]   + 1.0/2.0 * q[i+1][j];
    double q_hat2 = 3.0/2.0 * q[i+1][j] - 1.0/2.0 * q[i+2][j];

    double beta0 = (q[i][j] - q[i-1][j])   * (q[i][j] - q[i-1][j]);
    double beta1 = (q[i+1][j] - q[i][j])   * (q[i+1][j] - q[i][j]);
    double beta2 = (q[i+2][j] - q[i+1][j]) * (q[i+2][j] - q[i+1][j]);

    double alpha0 = d0 / ((eps + beta0) * (eps + beta0));
    double alpha1 = d1 / ((eps + beta1) * (eps + beta1));
    double alpha2 = d0 / ((eps + beta2) * (eps + beta2));

    double sum_minus = alpha0 + alpha1;
    double omega0_minus = alpha0 / sum_minus;
    double omega1_minus = alpha1 / sum_minus;

    double sum_plus = alpha1 + alpha2;
    double omega1_plus  = alpha1 / sum_plus;
    double omega2_plus  = alpha2 / sum_plus;

    double q_half_minus = omega0_minus * q_hat0 + omega1_minus * q_hat1;
    double q_half_plus  = omega1_plus  * q_hat1 + omega2_plus  * q_hat2;

    return { q_half_minus, q_half_plus };
}

inline std::pair<double, double> weno3_axis_y(const Field& q, int i, int j){

    double eps = 1e-6;
    double d0 = 1.0/3.0, d1 = 2.0/3.0;

    double q_hat0 = 3.0/2.0 * q[i][j]   - 1.0/2.0 * q[i][j-1];
    double q_hat1 = 1.0/2.0 * q[i][j]   + 1.0/2.0 * q[i][j+1];
    double q_hat2 = 3.0/2.0 * q[i][j+1] - 1.0/2.0 * q[i][j+2];

    double beta0 = (q[i][j] - q[i][j-1])   * (q[i][j] - q[i][j-1]);
    double beta1 = (q[i][j+1] - q[i][j])   * (q[i][j+1] - q[i][j]);
    double beta2 = (q[i][j+2] - q[i][j+1]) * (q[i][j+2] - q[i][j+1]);

    double alpha0 = d0 / ((eps + beta0) * (eps + beta0));
    double alpha1 = d1 / ((eps + beta1) * (eps + beta1));
    double alpha2 = d0 / ((eps + beta2) * (eps + beta2));

    double sum_minus = alpha0 + alpha1;
    double omega0_minus = alpha0 / sum_minus;
    double omega1_minus = alpha1 / sum_minus;

    double sum_plus = alpha1 + alpha2;
    double omega1_plus  = alpha1 / sum_plus;
    double omega2_plus  = alpha2 / sum_plus;

    double q_half_minus = omega0_minus * q_hat0 + omega1_minus * q_hat1;
    double q_half_plus  = omega1_plus  * q_hat1 + omega2_plus  * q_hat2;

    return { q_half_minus, q_half_plus };
}

void reconstruct_weno3_states(const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E, 
    Field& rho_Lx, Field& rho_Rx, Field& rho_u_Lx, Field& rho_u_Rx, Field& rho_v_Lx, Field& rho_v_Rx, Field& E_Lx, Field& E_Rx, 
    Field& rho_Ly, Field& rho_Ry, Field& rho_u_Ly, Field& rho_u_Ry, Field& rho_v_Ly, Field& rho_v_Ry, Field& E_Ly, Field& E_Ry, 
    int Nx, int Ny, int N_ghost) {
    #pragma omp parallel for collapse(2) schedule(static, 64)
    for(int i = N_ghost-1; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost-1; j < Ny+N_ghost; ++j) {
            std::pair<double,double> rho_pair_x = weno3_axis_x(rho, i, j);
            rho_Lx[i][j] = rho_pair_x.first;
            rho_Rx[i][j] = rho_pair_x.second;
            std::pair<double,double> rho_u_pair_x = weno3_axis_x(rho_u, i, j);
            rho_u_Lx[i][j] = rho_u_pair_x.first;
            rho_u_Rx[i][j] = rho_u_pair_x.second;
            std::pair<double,double> rho_v_pair_x = weno3_axis_x(rho_v, i, j);
            rho_v_Lx[i][j] = rho_v_pair_x.first;
            rho_v_Rx[i][j] = rho_v_pair_x.second;
            std::pair<double,double> E_pair_x = weno3_axis_x(E, i, j);
            E_Lx[i][j] = E_pair_x.first;
            E_Rx[i][j] = E_pair_x.second;

            std::pair<double,double> rho_pair_y = weno3_axis_y(rho, i, j);
            rho_Ly[i][j] = rho_pair_y.first;
            rho_Ry[i][j] = rho_pair_y.second;
            std::pair<double,double> rho_u_pair_y = weno3_axis_y(rho_u, i, j);
            rho_u_Ly[i][j] = rho_u_pair_y.first;
            rho_u_Ry[i][j] = rho_u_pair_y.second;
            std::pair<double,double> rho_v_pair_y = weno3_axis_y(rho_v, i, j);
            rho_v_Ly[i][j] = rho_v_pair_y.first;
            rho_v_Ry[i][j] = rho_v_pair_y.second;
            std::pair<double,double> E_pair_y = weno3_axis_y(E, i, j);
            E_Ly[i][j] = E_pair_y.first;
            E_Ry[i][j] = E_pair_y.second;
        }
    } 
}


inline double pressure(double rho, double u, double v, double E, double gamma){
    double kinetic = 0.5*rho*(u*u + v*v);
    return (gamma - 1.0)*(E - kinetic);
}

inline double sound_speed(double gamma, double p, double rho) {
    return std::sqrt(gamma * p / rho);
}

void compute_hllc_flux(
    const Field& rho_L_, const Field& rho_R_, 
    const Field& rho_u_L_, const Field& rho_u_R_, 
    const Field& rho_v_L_, const Field& rho_v_R_, 
    const Field& E_L_, const Field& E_R_, 
    Field& flux_rho, Field& flux_rho_u, Field& flux_rho_v, Field& flux_E,
    double gamma, int Nx, int Ny, int N_ghost) {
    #pragma omp parallel for collapse(2) schedule(static, 64)
    for(int i = N_ghost-1; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost-1; j < Ny+N_ghost; ++j) { 
            double rho_L = rho_L_[i][j];
            double u_L = rho_u_L_[i][j]/rho_L;
            double v_L = rho_v_L_[i][j]/rho_L;
            double E_L = E_L_[i][j];
            double p_L = pressure(rho_L, u_L, v_L, E_L, gamma);
            double c_L = sound_speed(gamma, p_L, rho_L);

            double rho_R = rho_R_[i][j];
            double u_R = rho_u_R_[i][j]/rho_R;
            double v_R = rho_v_R_[i][j]/rho_R;
            double E_R = E_R_[i][j];
            double p_R = pressure(rho_R, u_R, v_R, E_R, gamma);
            double c_R = sound_speed(gamma, p_R, rho_R);
             
            double S_L = std::min(u_L - c_L, u_R - c_R);
            double S_R = std::max(u_L + c_L, u_R + c_R);
 
            double numer = p_R - p_L + rho_L*u_L*(S_L - u_L) - rho_R*u_R*(S_R - u_R);
            double denom = rho_L*(S_L - u_L) - rho_R*(S_R - u_R);
            double S_star = numer / denom;
 
 
            double rho_L_star = rho_L * (S_L - u_L) / (S_L - S_star);
            double rho_R_star = rho_R * (S_R - u_R) / (S_R - S_star);
 
            double rho_v_L_star = rho_L_star * v_L;
            double rho_v_R_star = rho_R_star * v_R;
             
            double p_star = p_L + rho_L*(S_L - u_L)*(S_star - u_L);
 
            double E_L_star = ((S_L - u_L)*E_L - p_L*u_L + p_star*S_star) / (S_L - S_star);
            double E_R_star = ((S_R - u_R)*E_R - p_R*u_R + p_star*S_star) / (S_R - S_star);
            
            double rho_u_L_star = rho_L_star * S_star;
            double rho_u_R_star = rho_R_star * S_star;
            
            if (S_L >= 0.0){
                flux_rho[i][j]   = rho_u_L_[i][j];
                flux_rho_u[i][j] = rho_u_L_[i][j] * u_L + p_L;
                flux_rho_v[i][j] = rho_u_L_[i][j] * v_L;
                flux_E[i][j]     = (E_L + p_L) * u_L;
            }
            else if (S_L <= 0.0 && S_star >= 0.0){
                flux_rho[i][j]   = rho_u_L_star;
                flux_rho_u[i][j] = rho_u_L_star * rho_u_L_star / rho_L_star + p_star;
                flux_rho_v[i][j] = rho_v_L_star * rho_u_L_star / rho_L_star;
                flux_E[i][j]     = (E_L_star + p_star) * rho_u_L_star / rho_L_star;
            }
            else if (S_star <= 0.0 && S_R >= 0.0){
                flux_rho[i][j]   = rho_u_R_star;
                flux_rho_u[i][j] = rho_u_R_star * rho_u_R_star / rho_R_star + p_star;
                flux_rho_v[i][j] = rho_v_R_star * rho_u_R_star / rho_R_star;
                flux_E[i][j]     = (E_R_star + p_star) * rho_u_R_star / rho_R_star;
            }
            else{  // S_R <= 0
                flux_rho[i][j]   = rho_u_R_[i][j];
                flux_rho_u[i][j] = rho_u_R_[i][j] * u_R + p_R;
                flux_rho_v[i][j] = rho_u_R_[i][j] * v_R;
                flux_E[i][j]     = (E_R + p_R) * u_R;
            }
        }
    } 
}



void compute_advection_weno3_HLLC(
    const Field& rho, const Field& rho_u, const Field& rho_v,const Field& E,
    Field& conv_rho, Field& conv_rho_u, Field& conv_rho_v, Field& conv_E, 
    double gamma, double dx, double dy,
    int Nx, int Ny, int N_ghost)
{ 
    static Field rho_Lx,   rho_Rx,   rho_u_Lx, rho_u_Rx,
                 rho_v_Lx, rho_v_Rx, E_Lx,     E_Rx,
                 rho_Ly,   rho_Ry,   rho_u_Ly, rho_u_Ry,
                 rho_v_Ly, rho_v_Ry, E_Ly,     E_Ry;

    static Field flux_rho_x,  flux_rho_u_x,  flux_rho_v_x,  flux_E_x;
    static Field flux_rho_y,  flux_rho_u_y,  flux_rho_v_y,  flux_E_y;

    bool first = (rho_Lx.empty());

    if (first){
        auto init = [&](Field& f){
            f.assign(Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost,0.0));
        }; 
        init(rho_Lx); init(rho_Rx); init(rho_u_Lx); init(rho_u_Rx);
        init(rho_v_Lx); init(rho_v_Rx); init(E_Lx); init(E_Rx);
        init(rho_Ly); init(rho_Ry); init(rho_u_Ly); init(rho_u_Ry);
        init(rho_v_Ly); init(rho_v_Ry); init(E_Ly); init(E_Ry); 
        init(flux_rho_x); init(flux_rho_u_x); init(flux_rho_v_x); init(flux_E_x);
        init(flux_rho_y); init(flux_rho_u_y); init(flux_rho_v_y); init(flux_E_y);
        // // sorties
        // init(conv_rho); init(conv_rho_u); init(conv_rho_v); init(conv_E);
    }
 
    reconstruct_weno3_states(
        rho, rho_u, rho_v, E,
        rho_Lx, rho_Rx, rho_u_Lx, rho_u_Rx, rho_v_Lx, rho_v_Rx, E_Lx, E_Rx,
        rho_Ly, rho_Ry, rho_u_Ly, rho_u_Ry, rho_v_Ly, rho_v_Ry, E_Ly, E_Ry,
        Nx, Ny, N_ghost);
 
    compute_hllc_flux(
        rho_Lx, rho_Rx,
        rho_u_Lx, rho_u_Rx,
        rho_v_Lx, rho_v_Rx,
        E_Lx, E_Rx,
        flux_rho_x, flux_rho_u_x, flux_rho_v_x, flux_E_x,
        gamma, Nx, Ny, N_ghost);
 
    //  -> on echange (u,v) 
    compute_hllc_flux(
        rho_Ly, rho_Ry,
        rho_v_Ly, rho_v_Ry,          // rho v  devient quantite de mvt normale
        rho_u_Ly, rho_u_Ry,          // rho u  devient quantite transverse
        E_Ly, E_Ry,
        flux_rho_y, flux_rho_v_y, flux_rho_u_y, flux_E_y,   // rho u <-> rho v  
        gamma, Nx, Ny, N_ghost);
 
    #pragma omp parallel for collapse(2) schedule(static,64)
    for(int i = N_ghost; i < Nx+N_ghost; ++i){
        for(int j = N_ghost; j < Ny+N_ghost; ++j){

            double dFdx_rho   = (flux_rho_x[i][j]   - flux_rho_x[i-1][j])   / dx;
            double dFdx_rho_u = (flux_rho_u_x[i][j] - flux_rho_u_x[i-1][j]) / dx;
            double dFdx_rho_v = (flux_rho_v_x[i][j] - flux_rho_v_x[i-1][j]) / dx;
            double dFdx_E     = (flux_E_x[i][j]     - flux_E_x[i-1][j])     / dx;

            double dGdy_rho   = (flux_rho_y[i][j]   - flux_rho_y[i][j-1])   / dy;
            double dGdy_rho_u = (flux_rho_u_y[i][j] - flux_rho_u_y[i][j-1]) / dy;
            double dGdy_rho_v = (flux_rho_v_y[i][j] - flux_rho_v_y[i][j-1]) / dy;
            double dGdy_E     = (flux_E_y[i][j]     - flux_E_y[i][j-1])     / dy;
 
            conv_rho[i][j]   = -(dFdx_rho   + dGdy_rho);
            conv_rho_u[i][j] = -(dFdx_rho_u + dGdy_rho_u);
            conv_rho_v[i][j] = -(dFdx_rho_v + dGdy_rho_v);
            conv_E[i][j]     = -(dFdx_E     + dGdy_E);
        }
    }
}



void compute_advection(
    const SimulationParams& params, 
    const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E, 
    Field& conv_rho, Field& conv_rho_u, Field& conv_rho_v, Field& conv_E, 
    double dx, double dy){
    if (params.advection_scheme == "compute_advection_weno3_HLLC") {
        compute_advection_weno3_HLLC(
            rho, rho_u, rho_v, E, 
            conv_rho, conv_rho_u, 
            conv_rho_v, conv_E, 
            params.gamma, 
            dx, dy,
            params.Nx, params.Ny,
            params.bc_ghost_layers
        );
    }
    else {
        throw std::invalid_argument(
            "Unknown advection scheme: " + params.advection_scheme
        );
    }
}





} // namespace navier_stokes



