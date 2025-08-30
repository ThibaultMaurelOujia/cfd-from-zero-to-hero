#ifndef NAVIER_STOKES_CPP_PARAMS_HPP
#define NAVIER_STOKES_CPP_PARAMS_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace navier_stokes {



struct SimulationParams {
    // - Domain & maillage -
    double Lx = 1.0;     ///< longueur du domaine en x
    double Ly = 1.0;     ///< longueur du domaine en y
    int    Nx = 128;     ///< nombre de mailles en x
    int    Ny = 128;     ///< nombre de mailles en y

    // - Pas de grille -
    double dx = 0.0;     ///< = Lx / Nx, a remplir via finalize()
    double dy = 0.0;     ///< = Ly / Ny, a remplir via finalize()

    // - Proprietes physiques & temps -
    double cfl_number     = 0.1;   ///< nombre de Courant
    double viscosity      = 1e-6;  ///< viscosite cinematique nu
    double final_time     = 5.0;   ///< temps final de simulation

    // - Schemas numeriques -
    std::string advection_scheme    = "Upwind2ndOrder";   ///< choix du schema d'advection
    /**  
     *   "Upwind2ndOrder"              : upwind d'ordre 2  
     *   "Weno3_ConservativeRusanov"   : WENO3 conservative + flux Rusanov  
     */
    std::string poisson_solver      = "direct";    ///< "direct","cholesky","amg","fft"
    std::string boundary_conditions = "periodic";  ///< "periodic","inflow_outflow_x",...

    // - Couches fantômes -
    int bc_ghost_layers = 0;  ///< 1 pour upwind2, 2 pour WENO3, calcule via finalize()

    // - Conditions initiales -
    std::string initial_condition = "one";  ///< "one","taylor_green","kelvin_helmholtz",...

    // - Obstacles / masque -
    std::string obstacle_type      = "none"; ///< "none","circular","square" 
    double      obstacle_center_x  = 0.5;    ///< coord. x du centre (normalise)
    double      obstacle_center_y  = 0.5;    ///< coord. y du centre (normalise)
    double      obstacle_size      = 0.1;    ///< taille circular et square

    // - Conditions aux limites inflow/outflow -
    double inflow_velocity = 1.0;  ///< vitesse imposee a l'entree

    // - Sortie & journalisation -
    std::string output_directory = "./results"; ///< dossier de sortie
    int         save_interval    = 1;           ///< tous les n pas de temps
    int         verbosity        = 1;           ///< 0=silence,1=info,2=debug

    // - Solveur de Poisson iteratif -
    double poisson_tol      = 1e-8;  ///< tolerance pour AMG/CG
    int    poisson_max_iter = 100;   ///< iterations max pour AMG/CG
 
    void finalize();
};
 
bool load_config(const std::string& filename, SimulationParams& p);

void print_params(const SimulationParams& p);

} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_PARAMS_HPP
