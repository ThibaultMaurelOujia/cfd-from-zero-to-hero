#ifndef NAVIER_STOKES_CPP_INIT_HPP
#define NAVIER_STOKES_CPP_INIT_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

namespace navier_stokes {

void init_kelvin_helmholtz(
    const SimulationParams& params,
    Field& u, Field& v, Field& p,
    double delta = 0.005,
    double amp   = 0.1,
    int    kx    = 4
);

void init_one_x(const SimulationParams& params, Field& u, Field& v, Field& p);

void initialize_flow_field(const SimulationParams& params, Field& u, Field& v, Field& p);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_INIT_HPP