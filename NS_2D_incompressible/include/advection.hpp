#ifndef NAVIER_STOKES_CPP_ADVECTION_HPP
#define NAVIER_STOKES_CPP_ADVECTION_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>
#include <stdexcept>

namespace navier_stokes {

    
 void compute_advection(
    const SimulationParams& params,
    const Field& u, const Field& v,
    Field& A_u, Field& A_v,
    double dx, double dy
);



} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_ADVECTION_HPP

