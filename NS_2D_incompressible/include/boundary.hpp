#ifndef NAVIER_STOKES_CPP_BOUNDARY_HPP
#define NAVIER_STOKES_CPP_BOUNDARY_HPP

#include "utils.hpp"
#include "params.hpp"
#include <vector>

namespace navier_stokes {


using ObstacleBorderCell = std::pair<int,int>;


struct ObstacleMask {
    std::vector<ObstacleBorderCell> obstacle;

    std::vector<ObstacleBorderCell> left;
    std::vector<ObstacleBorderCell> right;
    std::vector<ObstacleBorderCell> bottom;
    std::vector<ObstacleBorderCell> top;

    std::vector<std::vector<bool>> solid;
};

ObstacleMask createMask(const SimulationParams& p);

void apply_immersed_boundary(const SimulationParams& params,
    Field& u, Field& v,
    const ObstacleMask& mask);

void apply_boundary_conditions(const SimulationParams& params,
                               Field& u,
                               Field& v,
                               Field& p);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_BOUNDARY_HPP