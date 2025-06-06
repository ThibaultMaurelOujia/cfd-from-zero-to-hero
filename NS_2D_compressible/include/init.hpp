#ifndef NAVIER_STOKES_CPP_INIT_HPP
#define NAVIER_STOKES_CPP_INIT_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

namespace navier_stokes {


void initialize_flow_field(const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_INIT_HPP