#include "boundary.hpp"
#include <cassert>

namespace navier_stokes {



//---------------------------------------------------------------------------
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

static void apply_periodic_bc(const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E) {
    copy_periodic_1field(params, rho);
    copy_periodic_1field(params, rho_u);
    copy_periodic_1field(params, rho_v);
    copy_periodic_1field(params, E);
}



//---------------------------------------------------------------------------
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

static void apply_periodic_bc_y( const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E) {
    copy_periodic_1field_y(params, rho);
    copy_periodic_1field_y(params, rho_u);
    copy_periodic_1field_y(params, rho_v);
    copy_periodic_1field_y(params, E);
}

static void apply_inflow_outflow_bc(const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E) {
    int Nx      = params.Nx;
    int Ny      = params.Ny;
    int N_ghost = params.bc_ghost_layers;
    double gamma = params.gamma;
 
    apply_periodic_bc_y(params, rho, rho_u, rho_v, E);
 
    double rho0 = 1.0;
    double p0   = 1.0;
    double a0   = std::sqrt(gamma * p0 / rho0);
    double M    = params.inflow_velocity;
    double u_in = M * a0;

    for (int g = 0; g < N_ghost; ++g) { 
        int il = g;
        for (int j = 0; j < Ny + 2 * N_ghost; ++j) {
            rho[il][j]   = rho0;
            rho_u[il][j] = rho0 * u_in;
            rho_v[il][j] = 0.0;
            double kinetic = 0.5 * rho0 * u_in * u_in;
            E[il][j]     = p0/(gamma-1.0) + kinetic;
        }
    }
 
    for (int g = 0; g < N_ghost; ++g) { 
        int ir = Nx + N_ghost + g;
        int is = Nx + N_ghost - 1;
        // int ir = Nx + N_ghost - g;
        // int is = Nx - 1;
        for (int j = 0; j < Ny + 2 * N_ghost; ++j) {
            rho[ir][j]   = rho[is][j];
            rho_u[ir][j] = rho_u[is][j];
            rho_v[ir][j] = rho_v[is][j];
            E[ir][j]     = E[is][j];
        }
    }
}



//---------------------------------------------------------------------------
void apply_boundary_conditions(const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E) {
    if (params.boundary_conditions == "periodic") {
        apply_periodic_bc(params, rho, rho_u, rho_v, E);
    }
    else if (params.boundary_conditions == "inflow_outflow_x") {
        apply_inflow_outflow_bc(params, rho, rho_u, rho_v, E);
    }
    else {
        throw std::invalid_argument(
            "Unknown boundary condition: " + params.boundary_conditions
        );
    }
}



//---------------------------------------------------------------------------
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



//---------------------------------------------------------------------------
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

// NACA 2412
static bool isSolidNaca(
    int    i, int    j,
    double Lx, double Ly,
    int    Nx, int    Ny,
    double xc, double yc,  
    double chordFrac    
) { 
    const double m   = 0.02;  
    const double p   = 0.4;   
    const double t   = 0.12;  
    const double aoa = 0.0;  
 
    double dx = Lx / Nx;
    double dy = Ly / Ny;
    double x   = (i + 0.5) * dx;
    double y   = (j + 0.5) * dy;
 
    double chord = chordFrac * Lx;
    double x0    = xc * Lx;  
    double y0    = yc * Ly;  
    double x_le  = x0 - 0.5 * chord;  
 
    double x_rel = x - x_le;
    if (x_rel < 0.0 || x_rel > chord) return false;

    double xr = x_rel / chord;  
 
    double yc_rel;
    if (xr <= p) {
        yc_rel = m / (p*p) * (2*p*xr - xr*xr) * chord;
    } else {
        yc_rel = m / ((1-p)*(1-p)) * ((1 - 2*p) + 2*p*xr - xr*xr) * chord;
    }
 
    double yt_rel = 5.0 * t * chord * (
          0.2969 * std::sqrt(xr)
        - 0.1260 * xr
        - 0.3516 * xr*xr
        + 0.2843 * xr*xr*xr
        - 0.1015 * xr*xr*xr*xr
    );
 
    double y_upper = y0 + yc_rel + yt_rel;
    double y_lower = y0 + yc_rel - yt_rel;
 
    return (y >= y_lower && y <= y_upper);
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
        else if (type == "naca")     return isSolidNaca(i,j, Lx, Ly, Nx, Ny, xc, yc, size);
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
                             Field& rho, Field& rho_u, Field& rho_v, Field& E,
                             const ObstacleMask& mask)
{
    const int N_ghost = params.bc_ghost_layers;
 
    for (auto [I, J] : mask.obstacle) {
        rho_u[I][J] = 0.0;
        rho_v[I][J] = 0.0; 
    }
 
    for (int r = 1; r <= N_ghost; ++r)
    { 
        // gauche
        for (auto [I, J] : mask.left) {
            int Isrc = I - r;           
            rho  [I][J] = rho  [Isrc][J];
            rho_u[I][J] = -rho_u[Isrc][J];   
            rho_v[I][J] =  rho_v[Isrc][J];  
            E    [I][J] =  E    [Isrc][J];
        }
        // droite
        for (auto [I, J] : mask.right) {
            int Isrc = I + r;
            rho  [I][J] = rho  [Isrc][J];
            rho_u[I][J] = -rho_u[Isrc][J];
            rho_v[I][J] =  rho_v[Isrc][J];
            E    [I][J] =  E    [Isrc][J];
        }
 
        // bas
        for (auto [I, J] : mask.bottom) {
            int Jsrc = J - r;
            rho  [I][J] = rho  [I][Jsrc];
            rho_u[I][J] =  rho_u[I][Jsrc];
            rho_v[I][J] = -rho_v[I][Jsrc];  
            E    [I][J] =  E    [I][Jsrc];
        }
        // haut
        for (auto [I, J] : mask.top) {
            int Jsrc = J + r;
            rho  [I][J] = rho  [I][Jsrc];
            rho_u[I][J] =  rho_u[I][Jsrc];
            rho_v[I][J] = -rho_v[I][Jsrc];
            E    [I][J] =  E    [I][Jsrc];
        }
    }
}



} // namespace navier_stokes








// static void apply_inflow_outflow_bc(
//     const SimulationParams& params,
//     Field& rho,   Field& rho_u,
//     Field& rho_v, Field& E)
// {
//     int Nx = params.Nx, Ny = params.Ny, G = params.bc_ghost_layers;
//     double γ = params.gamma; 
//     double p0   = params.inflow_pressure;     
//     double rho0 = params.inflow_density;     
//     double a0   = std::sqrt(γ * p0 / rho0);
//     double u_in = params.inflow_velocity * a0;   
//     double v_in = 0.0;
 
//     copy_periodic_1field_y(params, rho);
//     copy_periodic_1field_y(params, rho_u);
//     copy_periodic_1field_y(params, rho_v);
//     copy_periodic_1field_y(params, E);
 
//     for(int i = 0; i < G; ++i){
//       for(int j = 0; j < Ny+2*G; ++j){ 
//         int i_int = G;
//         double ρi = rho [i_int][j];
//         double mxi= rho_u[i_int][j];
//         double myi= rho_v[i_int][j];
//         double Ei = E   [i_int][j];
//         double ui = mxi/ρi, vi = myi/ρi;
//         double pi = (γ-1.0)*(Ei - 0.5*(mxi*mxi+myi*myi)/ρi);

//         double Min = ui/a0;  
//         if (Min >= 1.0) { 
//           rho_u[i][j] = rho0*u_in;
//           rho_v[i][j] = rho0*v_in;
//           rho  [i][j] = rho0;
//           E    [i][j] = p0/(γ-1.0) + 0.5*rho0*(u_in*u_in + v_in*v_in);
//         } else { 
//           double p_in = p0;            
//           double ρ_ex = rho [i_int][j];   
//           rho  [i][j] = ρ_ex;
//           rho_u[i][j] = ρ_ex * u_in;
//           rho_v[i][j] = ρ_ex * v_in;
//           E    [i][j] = p_in/(γ-1.0) + 0.5*ρ_ex*(u_in*u_in + v_in*v_in);
//         }
//       }
//     }
 
//     for(int i = Nx+G; i < Nx+2*G; ++i){
//       for(int j = 0; j < Ny+2*G; ++j){
//         int i_int = Nx+G-1;
//         double ρi = rho [i_int][j];
//         double mxi= rho_u[i_int][j];
//         double myi= rho_v[i_int][j];
//         double Ei = E   [i_int][j];
//         double ui = mxi/ρi, vi = myi/ρi;
//         double pi = (γ-1.0)*(Ei - 0.5*(mxi*mxi+myi*myi)/ρi);

//         double Mout = ui/a0;
//         if (Mout <= 1.0) { 
//           double p_out = params.outflow_pressure;  
//           double ρ_ex  = rho [i_int][j];
//           rho  [i][j] = ρ_ex;
//           rho_u[i][j] = rho_u[i_int][j];
//           rho_v[i][j] = rho_v[i_int][j];
//           E    [i][j] = p_out/(γ-1.0) + 0.5*ρ_ex*(ui*ui + vi*vi);
//         } else { 
//           rho  [i][j] = rho  [i_int][j];
//           rho_u[i][j] = rho_u[i_int][j];
//           rho_v[i][j] = rho_v[i_int][j];
//           E    [i][j] = E    [i_int][j];
//         }
//       }
//     }
// }
