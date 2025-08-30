#include "boundary.hpp"
#include <cassert>

namespace navier_stokes {

static inline void copy_periodic_1field(const SimulationParams& params, Field& q){
    const int Nx = params.Nx;
    const int Ny = params.Ny;
    const int N_ghost  = params.bc_ghost_layers;

    #pragma omp parallel for
    for (int i = 0; i < N_ghost; ++i) {
        int il = i;     
        int ir = Nx + N_ghost + i; 
        int src_left  = Nx + i; 
        int src_right = N_ghost + i;  
        for (int j = 0; j < Ny + 2 * N_ghost; ++j) {
            q[il][j] = q[src_left ][j]; 
            q[ir][j] = q[src_right][j];
        }
    }
    #pragma omp parallel for
    for (int j = 0; j < N_ghost; ++j) {
        int jb = j;  
        int jt = Ny + N_ghost + j; 
        int src_bot = Ny + j;  
        int src_top = N_ghost + j;  
        for (int i = 0; i < Nx + 2 * N_ghost; ++i) {
            q[i][jb] = q[i][src_bot]; 
            q[i][jt] = q[i][src_top];
        }
    }
}

void apply_periodic_bc(const SimulationParams& params, Field& u, Field& v, Field& p){
    copy_periodic_1field(params, u);
    copy_periodic_1field(params, v);
    copy_periodic_1field(params, p);
}



static void copy_periodic_1field_y(const SimulationParams& params, Field& f)
{
    int Nx = params.Nx, Ny = params.Ny, N_ghost = params.bc_ghost_layers; 
    for (int i = N_ghost; i < Nx + N_ghost; ++i) { 
        for (int g = 0; g < N_ghost; ++g) {
            f[i][g] = f[i][Ny + g];
        } 
        for (int g = 0; g < N_ghost; ++g) {
            f[i][Ny + N_ghost + g] = f[i][N_ghost + g];
        }
    }
}

void apply_periodic_bc_y(const SimulationParams& params, Field& u, Field& v, Field& p){
    copy_periodic_1field_y(params, u);
    copy_periodic_1field_y(params, v);
    copy_periodic_1field_y(params, p);
}

static void apply_inflow_outflow_bc(const SimulationParams& params, Field& u, Field& v, Field& p)
{
    int Nx = params.Nx, Ny = params.Ny,  N_ghost = params.bc_ghost_layers;
    double u_in = params.inflow_velocity;
    
    copy_periodic_1field_y(params, u);
    copy_periodic_1field_y(params, v);
    copy_periodic_1field_y(params, p);

    for (int i = N_ghost; i <2*N_ghost; ++i) {
        for (int j = 0; j < Ny + 2 * N_ghost; ++j) {
            u[i][j] = u_in;           
            v[i][j] = 0.0;    
            p[i][j] = p[N_ghost][j];   
        }
    }

    for (int i = N_ghost; i < N_ghost*2; ++i) {
        for (int j = 0; j < Ny + 2 * N_ghost; ++j) {
            u[Nx + N_ghost - i][j] = u[Nx - 1][j]; 
            v[Nx + N_ghost - i][j] = v[Nx - 1][j]; 
            p[Nx + N_ghost - i][j] = p[Nx - 1][j];  
        }
    }
}



void apply_boundary_conditions(const SimulationParams& params, Field& u, Field& v, Field& p)
{
    if (params.boundary_conditions == "periodic") {
        apply_periodic_bc(params, u, v, p);
    }
    else if (params.boundary_conditions == "inflow_outflow_x") {
        apply_inflow_outflow_bc(params, u, v, p);
    }
}




static inline double coord(double L, int N, int idx){ return (idx + 0.5) * (L / N); }
 
static bool isSolidNone(int, int) { return false; }
 
static bool isSolidChannel(int i, int j, int Nx, int N_ghost){ return j < N_ghost || j >= Nx - N_ghost; }
 
static bool isSolidCircle(int i, int j,
                          double Lx, double Ly,
                          int Nx, int Ny,
                          double xc, double yc,
                          double radius){
    double x = coord(Lx, Nx, i);
    double y = coord(Ly, Ny, j);
    double dx = x - xc * Lx;
    double dy = y - yc * Ly;
    return dx*dx + dy*dy <= radius * radius;
}
 
static bool isSolidSquare(int i, int j,
                          double Lx, double Ly,
                          int Nx, int Ny,
                          double xc, double yc,
                          double half_side){
    double x = coord(Lx, Nx, i);
    double y = coord(Ly, Ny, j);
    return std::abs(x - xc * Lx) <= half_side
        && std::abs(y - yc * Ly) <= half_side;
}

static void ensure_min_thickness(std::vector<std::vector<bool>>& solid, int N_ghost){
    const int Nx = static_cast<int>(solid.size());
    const int Ny = static_cast<int>(solid[0].size());
    const int T  = 2 * N_ghost;       
 
    for (int i = 0; i < Nx; ++i) {
        int j = 0;
        while (j < Ny) {
            if (solid[i][j]) {      
                int j0 = j;
                while (j < Ny && solid[i][j]) ++j;
                int j1 = j;    
                int L  = j1 - j0;    
                if (L < T) {
                    int need  = T - L;
                    int left  = need / 2 + (need & 1);
                    int right = need / 2;
                    int newL  = std::max(0, j0 - left);
                    int newR  = std::min(Ny, j1 + right);
                    for (int jj = newL; jj < newR; ++jj) solid[i][jj] = true;
                    j = newR;    
                }
            } else {
                ++j;
            }
        }
    }
 
    for (int j = 0; j < Ny; ++j) {
        int i = 0;
        while (i < Nx) {
            if (solid[i][j]) {
                int i0 = i;
                while (i < Nx && solid[i][j]) ++i;
                int i1 = i;
                int L  = i1 - i0;
                if (L < T) {
                    int need   = T - L;
                    int top    = need / 2 + (need & 1);
                    int bottom = need / 2;
                    int newT   = std::max(0, i0 - top);
                    int newB   = std::min(Nx, i1 + bottom);
                    for (int ii = newT; ii < newB; ++ii) solid[ii][j] = true;
                    i = newB;
                }
            } else {
                ++i;
            }
        }
    }
}

ObstacleMask createMask(const SimulationParams& params){
    const double Lx =  params.Lx, Ly =  params.Ly;
    const int    Nx =  params.Nx, Ny =  params.Ny, N_ghost =  params.bc_ghost_layers;
    const std::string& type =  params.obstacle_type;
    const double xc =  params.obstacle_center_x;
    const double yc =  params.obstacle_center_y;
    const double size =  params.obstacle_size;  
 
    std::vector<std::vector<bool>> solid(Nx, std::vector<bool>(Ny, false));

    auto isSolid = [&](int i, int j) -> bool {
        if      (type == "none")     return false;
        else if (type == "channel")  return isSolidChannel(i, j, Ny, N_ghost);
        else if (type == "circular") return isSolidCircle(i, j, Lx, Ly, Nx, Ny,
                                                          xc, yc, size);
        else if (type == "square")   return isSolidSquare (i, j, Lx, Ly, Nx, Ny,
                                                           xc, yc, size);
        throw std::runtime_error("Obstacle type inconnu : " + type);
    };

    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            solid[i][j] = isSolid(i, j);
            // std::cout << "solid[i][j]= " << solid[i][j] << '\n' << std::flush;
        }
    }
 
    if (type != "none")
        ensure_min_thickness(solid, N_ghost);
 
    ObstacleMask mask;
    mask.solid = solid;
    mask.obstacle.reserve(Nx * Ny / 4);    // heuristique

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (!solid[i][j]) continue;
            // std::cout << "i=" << i << " j =" << j << '\n' << std::flush;

            const int I = i + N_ghost;    
            const int J = j + N_ghost;
            mask.obstacle.emplace_back(I, J);

            const bool fluid_left   = (i > 0    ) && !solid[i-1][j];
            const bool fluid_right  = (i < Nx-1 ) && !solid[i+1][j];
            const bool fluid_bottom = (j > 0    ) && !solid[i][j-1];
            const bool fluid_top    = (j < Ny-1 ) && !solid[i][j+1];

            if (fluid_left  ) mask.left  .emplace_back(I, J);
            if (fluid_right ) mask.right .emplace_back(I, J);
            if (fluid_bottom) mask.bottom.emplace_back(I, J);
            if (fluid_top   ) mask.top   .emplace_back(I, J);
        }
    }
    return mask;
}


void apply_immersed_boundary(const SimulationParams& params,
                             Field& u, Field& v,
                             const ObstacleMask& mask)
{
    const int  N_ghost     = params.bc_ghost_layers;

    //  no-slip 
    for (auto [I, J] : mask.obstacle) {
        u[I][J] = 0.0;
        v[I][J] = 0.0;
    }
 
    for (int r = 1; r <= N_ghost; ++r) { 
        // gauche 
        for (auto [I, J] : mask.left) {
            // miroir antisymetrique   u_bc[I][J] = - u_orig[I-r][J]
            u[I][J] = u[I - r][J];
        }
        // droite
        for (auto [I, J] : mask.right) {
            u[I][J] = u[I + r][J];
        }
 
        // bas
        for (auto [I, J] : mask.bottom) {
            v[I][J] = v[I][J - r];
        }
        // haut
        for (auto [I, J] : mask.top) {
            v[I][J] = v[I][J + r];
        }
    }
}




} // namespace navier_stokes








