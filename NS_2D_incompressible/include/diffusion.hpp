#ifndef NAVIER_STOKES_CPP_DIFFUSION_HPP
#define NAVIER_STOKES_CPP_DIFFUSION_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>
#include <stdexcept>

namespace navier_stokes {

void compute_diffusion(
    const SimulationParams& params,
    const Field& u, const Field& v,
    Field& D_u, Field& D_v,
    double dx, double dy
);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_DIFFUSION_HPP

