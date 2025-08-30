#include "init.hpp"

namespace navier_stokes {

void init_kelvin_helmholtz(
    const SimulationParams& params,
    Field& u, Field& v, Field& p,
    double delta,
    double amp,
    int    kx
) {
    const int Nx    = params.Nx;
    const int Ny    = params.Ny;
    const double Lx = params.Lx;
    const double Ly = params.Ly;
    const double dx = params.dx;
    const double dy = params.dy;
    const int N_ghost = params.bc_ghost_layers;
    const double U0 = params.inflow_velocity;

    const double two_delta2 = 2.0 * delta * delta;

    assert(int(u.size()) == Nx + 2 * N_ghost && int(v.size()) == Nx + 2 * N_ghost && int(p.size()) == Nx + 2 * N_ghost);
    assert(int(u[0].size()) == Ny + 2 * N_ghost && int(v[0].size()) == Ny + 2*N_ghost && int(p[0].size()) == Ny + 2 * N_ghost);

    for (int i = N_ghost; i < Nx + N_ghost; ++i) {
        double x = (i + 0.5) * dx;  
        for (int j = N_ghost; j < Ny + N_ghost; ++j) {
            double y = (j + 0.5) * dy;
 
            u[i][j] = U0 * std::tanh((y - 0.5 * Ly) / delta);
 
            v[i][j] = amp
                * std::sin(TWO_PI * kx * x / Lx)
                * std::exp(- ( (y - 0.5 * Ly)*(y - 0.5 * Ly) ) / two_delta2 );
 
            p[i][j] = 0.0;
        }
    }
}

void init_one_x(const SimulationParams& params, Field& u, Field& v, Field& p){
    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int N_ghost      = params.bc_ghost_layers;
    const double U0  = params.inflow_velocity;
 
    assert(int(u.size())        == Nx + 2*N_ghost);
    assert(int(u.front().size())== Ny + 2*N_ghost);
 
    for (int i = N_ghost; i < Nx + N_ghost; ++i) {
        for (int j = N_ghost; j < Ny + N_ghost; ++j) {
            u[i][j] = U0;  
            v[i][j] = 0.0; 
            p[i][j] = 0.0;    
        }
    }
}



static void add_velocity_noise(Field& u, Field& v, int Nx, int Ny, int N_ghost, double amplitude)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);

    for (int i = N_ghost; i < Nx + N_ghost; ++i) {
        for (int j = N_ghost; j < Ny + N_ghost; ++j) {
            u[i][j] += dist(gen);
            v[i][j] += dist(gen);
        }
    }
}



void initialize_flow_field(const SimulationParams& params, Field& u, Field& v, Field& p) {
    if (params.initial_condition == "kelvin_helmholtz") {
        init_kelvin_helmholtz(params, u, v, p);
    }
    else if (params.initial_condition == "one_x") {
        init_one_x(params, u, v, p);
    }
    else {
        throw std::invalid_argument(
            "Unknown initial condition: " + params.initial_condition
        );
    }
    
    constexpr double noise_level = 1e-2;
    add_velocity_noise(u, v,
        params.Nx, params.Ny, params.bc_ghost_layers,
        noise_level
    );
}





} // namespace navier_stokes








