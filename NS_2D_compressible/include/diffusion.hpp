#ifndef NAVIER_STOKES_CPP_DIFFUSION_HPP
#define NAVIER_STOKES_CPP_DIFFUSION_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>
#include <stdexcept>

namespace navier_stokes {

void compute_diffusion(const SimulationParams& params,
                       const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
                       Field& D_rho, Field& D_rho_u, Field& D_rho_v, Field& D_E,
                       double dx, double dy);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_DIFFUSION_HPP

