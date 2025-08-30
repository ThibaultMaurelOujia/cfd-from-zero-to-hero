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
    const Field& rho , const Field& rho_u , const Field& rho_v , const Field& E,
          Field& D_rho , Field& D_rho_u , Field& D_rho_v , Field& D_E,
    double  dx , double  dy ,
    double  mu , double  Pr , double  gamma ,
    int     Nx , int     Ny , int     N_ghost )
{
    const double dx2 = dx*dx , dy2 = dy*dy ;
    const double cp  = 1005.0 ;   // air
    const double cv  = cp / gamma ;
    const double k   = mu * cp / Pr ;   

    auto d_dx = [&](const Field& q, int i,int j)
                { return (q[i+1][j] - q[i-1][j]) / (2.0*dx); };
    auto d_dy = [&](const Field& q, int i,int j)
                { return (q[i][j+1] - q[i][j-1]) / (2.0*dy); };

    #pragma omp parallel for collapse(2) schedule(static,64)
    for (int i = N_ghost ; i < Nx+N_ghost ; ++i)
    for (int j = N_ghost ; j < Ny+N_ghost ; ++j)
    {
        const double  r  = rho  [i][j];
        const double ru  = rho_u[i][j];
        const double rv  = rho_v[i][j];
        const double Et  = E    [i][j];

        const double u = ru / r;
        const double v = rv / r;

        const double du_dx = (d_dx(rho_u,i,j) - u*d_dx(rho,i,j)) / r;
        const double du_dy = (d_dy(rho_u,i,j) - u*d_dy(rho,i,j)) / r;
        const double dv_dx = (d_dx(rho_v,i,j) - v*d_dx(rho,i,j)) / r;
        const double dv_dy = (d_dy(rho_v,i,j) - v*d_dy(rho,i,j)) / r;

        const double div_uv = du_dx + dv_dy;
        const double tau_xx = 2.0*mu*du_dx - (2.0/3.0)*mu*div_uv;
        const double tau_yy = 2.0*mu*dv_dy - (2.0/3.0)*mu*div_uv;
        const double tau_xy =     mu*(du_dy + dv_dx);

        const double kinetic = 0.5*(ru*ru + rv*rv)/r;
        const double T       = (Et - kinetic)/(r*cv);

        const double dT_dx = (d_dx(E,i,j) - T*d_dx(rho,i,j)*cv*r) / (r*cv);
        const double dT_dy = (d_dy(E,i,j) - T*d_dy(rho,i,j)*cv*r) / (r*cv);

        const double qx = -k*dT_dx;
        const double qy = -k*dT_dy;

        auto tau_xx_ip = [&]{
            const double r_ip = rho[i+1][j];
            const double u_ip = rho_u[i+1][j]/r_ip, v_ip = rho_v[i+1][j]/r_ip;
            const double du_dx_ip = (rho_u[i+2][j]-rho_u[i][j])/(2.*dx*r_ip)
                                  - u_ip*(rho[i+2][j]-rho[i][j])/(2.*dx*r_ip);
            const double dv_dy_ip = (rho_v[i+1][j+1]-rho_v[i+1][j-1])/(2.*dy*r_ip)
                                  - v_ip*(rho[i+1][j+1]-rho[i+1][j-1])/(2.*dy*r_ip);
            return 2.*mu*du_dx_ip - (2./3.)*mu*(du_dx_ip+dv_dy_ip);}
        ();
        auto tau_xx_im = [&]{
            const double r_im = rho[i-1][j];
            const double u_im = rho_u[i-1][j]/r_im, v_im = rho_v[i-1][j]/r_im;
            const double du_dx_im = (rho_u[i][j]-rho_u[i-2][j])/(2.*dx*r_im)
                                  - u_im*(rho[i][j]-rho[i-2][j])/(2.*dx*r_im);
            const double dv_dy_im = (rho_v[i-1][j+1]-rho_v[i-1][j-1])/(2.*dy*r_im)
                                  - v_im*(rho[i-1][j+1]-rho[i-1][j-1])/(2.*dy*r_im);
            return 2.*mu*du_dx_im - (2./3.)*mu*(du_dx_im+dv_dy_im);}
        ();

        auto tau_xy_ip = [&]{
            const double r_ip = rho[i+1][j];
            const double du_dy_ip = (rho_u[i+1][j+1]-rho_u[i+1][j-1])/(2.*dy*r_ip)
                                  - (rho_u[i+1][j]/r_ip)*
                                    (rho[i+1][j+1]-rho[i+1][j-1])/(2.*dy*r_ip);
            const double dv_dx_ip = (rho_v[i+2][j]-rho_v[i][j])/(2.*dx*r_ip)
                                  - (rho_v[i+1][j]/r_ip)*
                                    (rho[i+2][j]-rho[i][j])/(2.*dx*r_ip);
            return mu*(du_dy_ip+dv_dx_ip);}
        ();
        auto tau_xy_im = [&]{
            const double r_im = rho[i-1][j];
            const double du_dy_im = (rho_u[i-1][j+1]-rho_u[i-1][j-1])/(2.*dy*r_im)
                                  - (rho_u[i-1][j]/r_im)*
                                    (rho[i-1][j+1]-rho[i-1][j-1])/(2.*dy*r_im);
            const double dv_dx_im = (rho_v[i][j]-rho_v[i-2][j])/(2.*dx*r_im)
                                  - (rho_v[i-1][j]/r_im)*
                                    (rho[i][j]-rho[i-2][j])/(2.*dx*r_im);
            return mu*(du_dy_im+dv_dx_im);}
        ();

        auto tau_xy_jp = [&]{
            const double r_jp = rho[i][j+1];
            const double du_dy_jp = (rho_u[i][j+2]-rho_u[i][j])/(2.*dy*r_jp)
                                  - (rho_u[i][j+1]/r_jp)*
                                    (rho[i][j+2]-rho[i][j])/(2.*dy*r_jp);
            const double dv_dx_jp = (rho_v[i+1][j+1]-rho_v[i-1][j+1])/(2.*dx*r_jp)
                                  - (rho_v[i][j+1]/r_jp)*
                                    (rho[i+1][j+1]-rho[i-1][j+1])/(2.*dx*r_jp);
            return mu*(du_dy_jp+dv_dx_jp);}
        ();
        auto tau_xy_jm = [&]{
            const double r_jm = rho[i][j-1];
            const double du_dy_jm = (rho_u[i][j]-rho_u[i][j-2])/(2.*dy*r_jm)
                                  - (rho_u[i][j-1]/r_jm)*
                                    (rho[i][j]-rho[i][j-2])/(2.*dy*r_jm);
            const double dv_dx_jm = (rho_v[i+1][j-1]-rho_v[i-1][j-1])/(2.*dx*r_jm)
                                  - (rho_v[i][j-1]/r_jm)*
                                    (rho[i+1][j-1]-rho[i-1][j-1])/(2.*dx*r_jm);
            return mu*(du_dy_jm+dv_dx_jm);}
        ();

        auto tau_yy_jp = [&]{
            const double r_jp = rho[i][j+1];
            const double dv_dy_jp = (rho_v[i][j+2]-rho_v[i][j])/(2.*dy*r_jp)
                                  - (rho_v[i][j+1]/r_jp)*
                                    (rho[i][j+2]-rho[i][j])/(2.*dy*r_jp);
            const double du_dx_jp = (rho_u[i+1][j+1]-rho_u[i-1][j+1])/(2.*dx*r_jp)
                                  - (rho_u[i][j+1]/r_jp)*
                                    (rho[i+1][j+1]-rho[i-1][j+1])/(2.*dx*r_jp);
            return 2.*mu*dv_dy_jp - (2./3.)*mu*(du_dx_jp+dv_dy_jp);}
        ();
        auto tau_yy_jm = [&]{
            const double r_jm = rho[i][j-1];
            const double dv_dy_jm = (rho_v[i][j]-rho_v[i][j-2])/(2.*dy*r_jm)
                                  - (rho_v[i][j-1]/r_jm)*
                                    (rho[i][j]-rho[i][j-2])/(2.*dy*r_jm);
            const double du_dx_jm = (rho_u[i+1][j-1]-rho_u[i-1][j-1])/(2.*dx*r_jm)
                                  - (rho_u[i][j-1]/r_jm)*
                                    (rho[i+1][j-1]-rho[i-1][j-1])/(2.*dx*r_jm);
            return 2.*mu*dv_dy_jm - (2./3.)*mu*(du_dx_jm+dv_dy_jm);}
        ();

        auto qx_ip = -k * ( (E[i+2][j]-E[i][j])/(2.*dx*cv*rho[i+1][j])
                          - T * (rho[i+2][j]-rho[i][j])/(2.*dx*rho[i+1][j]) );
        auto qx_im = -k * ( (E[i][j]-E[i-2][j])/(2.*dx*cv*rho[i-1][j])
                          - T * (rho[i][j]-rho[i-2][j])/(2.*dx*rho[i-1][j]) );
        auto qy_jp = -k * ( (E[i][j+2]-E[i][j])/(2.*dy*cv*rho[i][j+1])
                          - T * (rho[i][j+2]-rho[i][j])/(2.*dy*rho[i][j+1]) );
        auto qy_jm = -k * ( (E[i][j]-E[i][j-2])/(2.*dy*cv*rho[i][j-1])
                          - T * (rho[i][j]-rho[i][j-2])/(2.*dy*rho[i][j-1]) );

        const double d_tau_xx_dx = (tau_xx_ip - tau_xx_im) / (2.*dx);
        const double d_tau_xy_dx = (tau_xy_ip - tau_xy_im) / (2.*dx);
        const double d_tau_xy_dy = (tau_xy_jp - tau_xy_jm) / (2.*dy);
        const double d_tau_yy_dy = (tau_yy_jp - tau_yy_jm) / (2.*dy);

        const double d_qx_dx = (qx_ip - qx_im) / (2.*dx);
        const double d_qy_dy = (qy_jp - qy_jm) / (2.*dy);

        D_rho  [i][j] = 0.0;                  
        D_rho_u[i][j] = d_tau_xx_dx + d_tau_xy_dy;
        D_rho_v[i][j] = d_tau_xy_dx + d_tau_yy_dy;
        D_E    [i][j] = u*(d_tau_xx_dx + d_tau_xy_dy)
                      + v*(d_tau_xy_dx + d_tau_yy_dy)
                      + d_qx_dx + d_qy_dy;
    }
 
    for (int i = 0; i < N_ghost            ; ++i)
    for (int j = 0; j < Ny+2*N_ghost       ; ++j)
        D_rho[i][j] = D_rho_u[i][j] = D_rho_v[i][j] = D_E[i][j] = 0.0,
        D_rho[Nx+N_ghost+i][j] = D_rho_u[Nx+N_ghost+i][j] =
        D_rho_v[Nx+N_ghost+i][j] = D_E[Nx+N_ghost+i][j] = 0.0;

    for (int i = 0; i < Nx+2*N_ghost       ; ++i)
    for (int j = 0; j < N_ghost            ; ++j)
        D_rho[i][j] = D_rho_u[i][j] = D_rho_v[i][j] = D_E[i][j] = 0.0,
        D_rho[i][Ny+N_ghost+j] = D_rho_u[i][Ny+N_ghost+j] =
        D_rho_v[i][Ny+N_ghost+j] = D_E[i][Ny+N_ghost+j] = 0.0;
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