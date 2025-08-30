#include "diffusion.hpp"

namespace navier_stokes {

inline double laplacian_centered_2ndOrder(const Field& q,
                           double dx2, double dy2,
                           int i, int j) {
    return ( q[i+1][j] - 2.0*q[i][j] + q[i-1][j] ) / dx2
         + ( q[i][j+1] - 2.0*q[i][j] + q[i][j-1] ) / dy2;
}

void compute_diffusion_centered_2ndOrder(const Field& u, const Field& v,
                                         Field& D_u, Field& D_v,
                                         double dx, double dy,
                                         double nu,
                                         int Nx, int Ny, int N_ghost) {
    const double dx2 = dx*dx;
    const double dy2 = dy*dy;
    
    #pragma omp parallel for collapse(2) schedule(static, 64)
    for (int i = N_ghost; i < Nx+N_ghost; ++i){
        for (int j = N_ghost; j < Ny+N_ghost; ++j)
        {
            D_u[i][j] = nu * laplacian_centered_2ndOrder(u, dx2, dy2, i, j);
            D_v[i][j] = nu * laplacian_centered_2ndOrder(v, dx2, dy2, i, j);
        }
    }
}
//[Timer] compute_diffusion_centered_2ndOrder: 0.00507833 s
//[Timer] compute_diffusion_centered_2ndOrder: 0.00968904 s
// [Timer] compute_diffusion_centered_2ndOrder: 0.001356 s


void compute_diffusion(const SimulationParams& params,
                       const Field& u, const Field& v,
                       Field& D_u, Field& D_v,
                       double dx, double dy) {
    const int Nx = params.Nx;
    const int Ny = params.Ny;
    const int N_ghost = params.bc_ghost_layers;
    const double nu = params.viscosity;
    
    compute_diffusion_centered_2ndOrder(u, v, D_u, D_v, dx, dy, nu, Nx, Ny, N_ghost);
}





} // namespace navier_stokes