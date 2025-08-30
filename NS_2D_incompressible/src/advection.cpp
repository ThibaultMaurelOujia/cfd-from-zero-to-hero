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

void compute_advection_upwind2ndOrder(const Field& u, const Field& v, 
                               Field& A_u, Field& A_v, 
                               double dx, double dy, 
                               int Nx, int Ny, int N_ghost) {
    #pragma omp parallel for collapse(2) schedule(static, 64)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            double du_dx = upwind2ndOrder_axis_x(u, u, dx, i, j);
            double du_dy = upwind2ndOrder_axis_y(u, v, dy, i, j);
            double dv_dx = upwind2ndOrder_axis_x(v, u, dx, i, j);
            double dv_dy = upwind2ndOrder_axis_y(v, v, dy, i, j);
            
            A_u[i][j] = u[i][j] * du_dx + v[i][j] * du_dy;
            A_v[i][j] = u[i][j] * dv_dx + v[i][j] * dv_dy;
        }
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

void compute_advection_weno3_rusanov(const Field& u, const Field& v, 
                               Field& A_u, Field& A_v, 
                               double dx, double dy, 
                               int Nx, int Ny, int N_ghost) {
    static Field flux_u_x, flux_v_x, flux_u_y, flux_v_y;
    static bool initialized_F_G_weno3 = false;
    if (!initialized_F_G_weno3) { 
        flux_u_x.assign(Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost, 0.0));
        flux_v_x.assign(Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost, 0.0));
        flux_u_y.assign(Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost, 0.0));
        flux_v_y.assign(Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost, 0.0));
        initialized_F_G_weno3 = true;
    }
    #pragma omp parallel for collapse(2) schedule(static, 64)
    for(int i = N_ghost-1; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost-1; j < Ny+N_ghost; ++j) {
            std::pair<double,double> u_pair_x = weno3_axis_x(u, i, j);
            double uLx = u_pair_x.first;
            double uRx = u_pair_x.second;
            std::pair<double,double> v_pair_x = weno3_axis_x(v, i, j);
            double vLx = v_pair_x.first;
            double vRx = v_pair_x.second;

            // u^2
            double Fu_L = uLx*uLx, Fu_R = uRx*uRx;
            double alpha_x_u = std::max(std::abs(uLx), std::abs(uRx));
            flux_u_x[i][j] = 0.5*(Fu_L + Fu_R) - 0.5*alpha_x_u*(uRx - uLx);
    
            // u*v
            double Fv_L = uLx*vLx, Fv_R = uRx*vRx;
            flux_v_x[i][j] = 0.5*(Fv_L + Fv_R) - 0.5*alpha_x_u*(vRx - vLx);


            std::pair<double,double> u_pair_y = weno3_axis_y(u, i, j);
            double uLy = u_pair_y.first;
            double uRy = u_pair_y.second;
            std::pair<double,double> v_pair_y = weno3_axis_y(v, i, j);
            double vLy = v_pair_y.first;
            double vRy = v_pair_y.second;

            // u*v
            double Gu_L = uLy*vLy, Gu_R = uRy*vRy;
            double alpha_y_u = std::max(std::abs(vLy), std::abs(vRy));
            flux_u_y[i][j] = 0.5*(Gu_L + Gu_R) - 0.5*alpha_y_u*(uRy - uLy);
            
            // v^2
            double Gv_L = vLy*vLy, Gv_R = vRy*vRy;
            flux_v_y[i][j] = 0.5*(Gv_L + Gv_R) - 0.5*alpha_y_u*(vRy - vLy);
        }
    } 
    #pragma omp parallel for collapse(2) schedule(static, 64)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) { 
            A_u[i][j] = (flux_u_x[i][j] - flux_u_x[i-1][j]) / dx + (flux_u_y[i][j] - flux_u_y[i][j-1]) / dy;
            A_v[i][j] = (flux_v_x[i][j] - flux_v_x[i-1][j]) / dx + (flux_v_y[i][j] - flux_v_y[i][j-1]) / dy;
        }
    } 
}



void compute_advection(
    const SimulationParams& params,
    const Field& u, const Field& v,
    Field& A_u, Field& A_v,
    double dx, double dy
) {
    if (params.advection_scheme == "Upwind2ndOrder") {
        compute_advection_upwind2ndOrder(u, v, A_u, A_v, dx, dy, params.Nx, params.Ny, params.bc_ghost_layers);
    }
    else if (params.advection_scheme == "Weno3_ConservativeRusanov") {
        compute_advection_weno3_rusanov(u, v, A_u, A_v, dx, dy, params.Nx, params.Ny, params.bc_ghost_layers);
    }
    else {
        throw std::invalid_argument(
            "Unknown advection scheme: " + params.advection_scheme
        );
    }
}








} // namespace navier_stokes




// no omp 
// [Timer] Poisson Upwind2ndOrder: 0.00806083 s
// omp 
// [Timer] Poisson Upwind2ndOrder: 0.0165022 s
// collapse(2)
// [Timer] Poisson Upwind2ndOrder: 0.00249154 s
// schedule(static, 8)
// [Timer] Poisson Upwind2ndOrder: 0.00217875 s
// collapse(2) schedule(static)
// [Timer] Poisson Upwind2ndOrder: 0.00217229 s