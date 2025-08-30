#ifndef NAVIER_STOKES_CPP_ADVECTION_HPP
#define NAVIER_STOKES_CPP_ADVECTION_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>
#include <stdexcept>

namespace navier_stokes {


 void compute_advection(const SimulationParams& params,
    const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
    Field& conv_rho, Field& conv_rho_u, Field& conv_rho_v, Field& conv_E,
    double dx, double dy);



} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_ADVECTION_HPP

