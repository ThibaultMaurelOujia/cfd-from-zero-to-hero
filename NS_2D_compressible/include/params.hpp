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
    double cfl_number     = 0.1;        ///< nombre de Courant
    double final_time     = 5.0;        ///< temps final de simulation
    double gamma          = 1.4;        ///< rapport des chaleurs specifiques (Cp/Cv)
    double viscosity      = 1e-6;       ///< viscosite dynamique mu [Pa.s]
    double Pr             = 0.71;       ///< Prandtl
    double cp             = 1005.0;     ///< capacite thermique a Cp constant [J/(kg K)]
    double cv             = 718.0;      ///< capacite thermique a Cv constant [J/(kg K)]
    double R              = 287.0;      ///< constante specifique du gaz parfait [J/(kg K)]

    // - Schemas numeriques -
    std::string advection_scheme    = "compute_advection_weno3_HLLC";   ///< choix du schema d'advection
    std::string boundary_conditions = "periodic";  ///< "periodic","inflow_outflow_x",...

    // - Couches fantomes -
    int bc_ghost_layers = 0;  ///< 1 pour upwind2, 2 pour WENO3, calcule via finalize()

    // - Conditions initiales -
    std::string initial_condition = "one";  ///< "one","taylor_green","kelvin_helmholtz",...
    double      noise_level       = 0.0;  ///< amplitude du bruit ajoute sur la vitesse initiale

    // - Obstacles / masque -
    std::string obstacle_type      = "none"; ///< "none","circular","square" 
    double      obstacle_center_x  = 0.5;    ///< coord. x du centre (normalise)
    double      obstacle_center_y  = 0.5;    ///< coord. y du centre (normalise)
    double      obstacle_size      = 0.1;    ///< taille circular et square

    // - Conditions aux limites inflow/outflow -
    double inflow_velocity = 1.0;       ///< vitesse imposee a l'entree
    double rho_ref = 1.225;             ///< densite de reference de l'air a 15 C, 1 atm [kg/m^3]
    double p_ref   = 101325.0;          ///< pression de reference de l'air a 15 C, 1 atm [Pa]


    // - Sortie & journalisation -
    std::string output_directory = "./results"; ///< dossier de sortie
    int         save_interval    = 1;           ///< tous les n pas de temps
    std::string output_quantity  = "schlieren"; ///< "density","pressure","schlieren","vorticity","rho_u","rho_v","E"
    int         verbosity        = 1;           ///< 0=silence,1=info,2=debug

    /// Calcule dx, dy et determine bc_ghost_layers selon advection_scheme.
    void finalize();
};


// Lit le fichier de configuration.
// false si on n'a pas pu ouvrir/parse le fichier.
bool load_config(const std::string& filename, SimulationParams& p);


void print_params(const SimulationParams& p);

} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_PARAMS_HPP
