#ifndef NAVIER_STOKES_CPP_SIMULATION_HPP
#define NAVIER_STOKES_CPP_SIMULATION_HPP

#include "utils.hpp"
#include "params.hpp"
#include "init.hpp"
#include "boundary.hpp"
#include "advection.hpp"
#include "diffusion.hpp"

#include "renderer_colormap.hpp"
#include <SFML/Graphics.hpp>

#include <vector>


namespace navier_stokes {

/// Drives the whole 2D Navier-Stokes run: init, time-loop, output...
class Simulator {
public:
    explicit Simulator(const SimulationParams& params);

    void run();

private:
    SimulationParams params_;

    std::pair<double,double> display_size_;
    

    Field rho_;    // mass
    Field rho_u_;  // momentum x
    Field rho_v_;  // momentum y
    Field E_;      // total energy

    // Buffers RK3
    Field rho_star_;
    Field rho_u_star_;
    Field rho_v_star_;
    Field E_star_;

    Field conv_rho_, conv_rho_u_, conv_rho_v_, conv_E_;

    Field D_rho_, D_rho_u_, D_rho_v_, D_E_;   // termes diffusionnels compressibles

    ObstacleMask mask_;

    void initialize_fields();               // call init_*() and initial BCs
    void time_loop();                       // CFL, RK3 steps, Poisson solve, BCs, output
    void output_step(int iter, double t, double dt);   // write to screen / files every save_interval
    
    void step_SSP_RK3(double dt);
    void rk3_substep(double c0, double c1, double dt);

     // SFML objects for on-screen rendering 
    sf::RenderWindow               win_;       // fenetre unique
    sf::Image                      img_;
    sf::Texture                    tex_;
    sf::Sprite                     spr_;
    
};



} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_SIMULATION_HPP