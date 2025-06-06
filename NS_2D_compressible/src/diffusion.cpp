#include "diffusion.hpp"

namespace navier_stokes {

inline double d_dx(const Field& q, double dx, int i, int j) {
    return (q[i+1][j] - q[i-1][j]) / (2.0*dx);
}
inline double d_dy(const Field& q, double dy, int i, int j) {
    return (q[i][j+1] - q[i][j-1]) / (2.0*dy);
}

inline double laplacian_centered_2ndOrder(const Field& q,
                           double dx2, double dy2,
                           int i, int j) {
    return ( q[i+1][j] - 2.0*q[i][j] + q[i-1][j] ) / dx2
         + ( q[i][j+1] - 2.0*q[i][j] + q[i][j-1] ) / dy2;
}

void compute_diffusion_centered_2ndOrder_compressible(
    const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
    Field& D_rho, Field& D_rho_u, Field& D_rho_v, Field& D_E,
    double dx, double dy,
    double mu, double Pr, double gamma,
    int Nx, int Ny, int N_ghost){
    const double dx2 = dx*dx, dy2 = dy*dy;
    const double cp  = 1005.0;                 // air // !!! 
    const double cv  = cp / gamma;
    const double k   = mu * cp / Pr;           // conduction thermique

    /* --- boucle principale --- */
    #pragma omp parallel for collapse(2) schedule(static,64)
    for(int i = N_ghost; i < Nx+N_ghost; ++i){
        for(int j = N_ghost; j < Ny+N_ghost; ++j){
            /* --- primitives --- */
            double r  = rho  [i][j];
            double ru = rho_u[i][j];
            double rv = rho_v[i][j];
            double Et = E    [i][j];
            double u = ru/r, v = rv/r;

            /* --- gradients de vitesse --- */
            double du_dx = d_dx(rho_u, dx, i, j)/r - u*d_dx(rho,dx,i,j)/r;
            double dv_dy = d_dy(rho_v, dy, i, j)/r - v*d_dy(rho,dy,i,j)/r;
            double du_dy = d_dy(rho_u, dy, i, j)/r - u*d_dy(rho,dy,i,j)/r;
            double dv_dx = d_dx(rho_v, dx, i, j)/r - v*d_dx(rho,dx,i,j)/r;

            /* --- tenseur visqueux de Stokes --- */
            double div_vel = du_dx + dv_dy;
            double tau_xx = 2.0*mu*du_dx - (2.0/3.0)*mu*div_vel;
            double tau_yy = 2.0*mu*dv_dy - (2.0/3.0)*mu*div_vel;
            double tau_xy =     mu*(du_dy + dv_dx);

            /* --- température & flux thermique --- */
            double kinetic = 0.5*(ru*ru + rv*rv)/r;
            double T = (Et - kinetic)/(r*cv);
            double dT_dx = d_dx(E,dx,i,j)/(r*cv) - T*d_dx(rho,dx,i,j)/r;
            double dT_dy = d_dy(E,dy,i,j)/(r*cv) - T*d_dy(rho,dy,i,j)/r;
            double qx = -k * dT_dx;
            double qy = -k * dT_dy;

            /* --- divergences (∂/∂x + ∂/∂y) --- */
            // On approxime ∂τ/∂x, ∂τ/∂y et ∂q/∂x,y par un laplacien 2ᵉ ordre.
            auto d_dx_c   = [&](double val_ip, double val_im){return (val_ip-val_im)/(2*dx);};
            auto d_dy_c   = [&](double val_jp, double val_jm){return (val_jp-val_jm)/(2*dy);};

            /* flux visqueux en x pour i±1 */
            double tau_xx_ip = 2.0*mu*d_dx(rho_u,dx,i+1,j)/rho[i+1][j]
                            - (2.0/3.0)*mu*(d_dx(rho_u,dx,i+1,j)/rho[i+1][j]
                                            + d_dy(rho_v,dy,i+1,j)/rho[i+1][j]);
            double tau_xx_im = 2.0*mu*d_dx(rho_u,dx,i-1,j)/rho[i-1][j]
                            - (2.0/3.0)*mu*(d_dx(rho_u,dx,i-1,j)/rho[i-1][j]
                                            + d_dy(rho_v,dy,i-1,j)/rho[i-1][j]);
            double tau_xy_ip =     mu*( d_dy(rho_u,dy,i+1,j)/rho[i+1][j]
                                    + d_dx(rho_v,dx,i+1,j)/rho[i+1][j] );
            double tau_xy_im =     mu*( d_dy(rho_u,dy,i-1,j)/rho[i-1][j]
                                    + d_dx(rho_v,dx,i-1,j)/rho[i-1][j] );

            double tau_xy_jp =     mu*( d_dy(rho_u,dy,i,j+1)/rho[i][j+1]
                                    + d_dx(rho_v,dx,i,j+1)/rho[i][j+1] );
            double tau_xy_jm =     mu*( d_dy(rho_u,dy,i,j-1)/rho[i][j-1]
                                    + d_dx(rho_v,dx,i,j-1)/rho[i][j-1] );
            double tau_yy_jp = 2.0*mu*d_dy(rho_v,dy,i,j+1)/rho[i][j+1]
                            - (2.0/3.0)*mu*(d_dx(rho_u,dx,i,j+1)/rho[i][j+1]
                                            + d_dy(rho_v,dy,i,j+1)/rho[i][j+1]);
            double tau_yy_jm = 2.0*mu*d_dy(rho_v,dy,i,j-1)/rho[i][j-1]
                            - (2.0/3.0)*mu*(d_dx(rho_u,dx,i,j-1)/rho[i][j-1]
                                            + d_dy(rho_v,dy,i,j-1)/rho[i][j-1]);

            double Fx_rho_u = tau_xx;
            double Fx_rho_v = tau_xy;
            double Fx_E     = u*tau_xx + v*tau_xy + qx;

            double Fy_rho_u = tau_xy;
            double Fy_rho_v = tau_yy;
            double Fy_E     = u*tau_xy + v*tau_yy + qy;

            double Fx_rho_u_dx = d_dx_c(tau_xx_ip, tau_xx_im);
            double Fx_rho_v_dx = d_dx_c(tau_xy_ip, tau_xy_im);
            double Fx_E_dx     = d_dx_c( u*tau_xx_ip + v*tau_xy_ip,
                                        u*tau_xx_im + v*tau_xy_im );

            double Fy_rho_u_dy = d_dy_c(tau_xy_jp, tau_xy_jm);
            double Fy_rho_v_dy = d_dy_c(tau_yy_jp, tau_yy_jm);
            double Fy_E_dy     = d_dy_c( u*tau_xy_jp + v*tau_yy_jp,
                                        u*tau_xy_jm + v*tau_yy_jm );

            /* --- Termes diffusionnels --- */
            D_rho  [i][j] = 0.0;
            D_rho_u[i][j] = Fx_rho_u_dx + Fy_rho_u_dy;
            D_rho_v[i][j] = Fx_rho_v_dx + Fy_rho_v_dy;
            D_E    [i][j] = Fx_E_dx     + Fy_E_dy;
        }
    }

    /* --- Zéro sur ghost layers --- */
    for(int i=0;i<N_ghost;++i) for(int j=0;j<Ny+2*N_ghost;++j){
        D_rho[i][j]=D_rho_u[i][j]=D_rho_v[i][j]=D_E[i][j]=0.0;
        D_rho[Nx+N_ghost+i][j]=D_rho_u[Nx+N_ghost+i][j]=D_rho_v[Nx+N_ghost+i][j]=D_E[Nx+N_ghost+i][j]=0.0;
    }
    for(int i=0;i<Nx+2*N_ghost;++i) for(int j=0;j<N_ghost;++j){
        D_rho[i][j]=D_rho_u[i][j]=D_rho_v[i][j]=D_E[i][j]=0.0;
        D_rho[i][Ny+N_ghost+j]=D_rho_u[i][Ny+N_ghost+j]=D_rho_v[i][Ny+N_ghost+j]=D_E[i][Ny+N_ghost+j]=0.0;
    }
}



void compute_diffusion(const SimulationParams& params,
                       const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
                       Field& D_rho, Field& D_rho_u, Field& D_rho_v, Field& D_E,
                       double dx, double dy) {
    const int Nx = params.Nx;
    const int Ny = params.Ny;
    const int N_ghost = params.bc_ghost_layers;
    const double mu = params.viscosity;
    const double Pr = params.Pr;
    const double gamma = params.gamma;
    
    compute_diffusion_centered_2ndOrder_compressible(rho, rho_u, rho_v, E, D_rho, D_rho_u, D_rho_v, D_E, dx, dy, mu, Pr, gamma, Nx, Ny, N_ghost);
}



} // namespace navier_stokes