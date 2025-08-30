#include "params.hpp"

namespace navier_stokes {

void SimulationParams::finalize() {
    dx = Lx / Nx;
    dy = Ly / Ny;
    if      (advection_scheme == "Upwind2ndOrder") {
        bc_ghost_layers = 2;
    }
    else if (advection_scheme == "Weno3_ConservativeRusanov") {
        bc_ghost_layers = 2;
    }
    else { 
        bc_ghost_layers = 2;
    }
}

// trim whitespace from both ends of a string
static inline std::string trim(const std::string &s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    auto last  = s.find_last_not_of (" \t\r\n");
    return s.substr(first, last - first + 1);
}

bool load_config(const std::string &filename, SimulationParams &p) {
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        // ignore comments and empty lines
        auto t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        auto eq = t.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(t.substr(0, eq));
        std::string val = trim(t.substr(eq + 1));
        std::istringstream ss(val);

        if      (key == "Lx")                   ss >> p.Lx;
        else if (key == "Ly")                   ss >> p.Ly;
        else if (key == "Nx")                   ss >> p.Nx;
        else if (key == "Ny")                   ss >> p.Ny;
        else if (key == "viscosity")            ss >> p.viscosity;
        else if (key == "cfl_number")           ss >> p.cfl_number;
        else if (key == "final_time")           ss >> p.final_time;

        else if (key == "advection_scheme")     p.advection_scheme    = val;
        else if (key == "poisson_solver")       p.poisson_solver      = val;
        else if (key == "boundary_conditions")  p.boundary_conditions = val;

        else if (key == "initial_condition")    p.initial_condition   = val;

        else if (key == "obstacle_type")        p.obstacle_type       = val;
        else if (key == "obstacle_center_x")    ss >> p.obstacle_center_x;
        else if (key == "obstacle_center_y")    ss >> p.obstacle_center_y;
        else if (key == "obstacle_size")        ss >> p.obstacle_size;

        else if (key == "inflow_velocity")      ss >> p.inflow_velocity;

        else if (key == "output_directory")     p.output_directory    = val;
        else if (key == "save_interval")        ss >> p.save_interval;
        else if (key == "verbosity")            ss >> p.verbosity;

        else if (key == "poisson_tol")          ss >> p.poisson_tol;
        else if (key == "poisson_max_iter")     ss >> p.poisson_max_iter;
 
    }
 
    p.finalize();
    return true;
}

void print_params(const SimulationParams& p) {
    std::cout
      << "=== Simulation parameters ===\n"
      << " Domain:      Lx=" << p.Lx << " ; Ly=" << p.Ly << "\n"
      << " Mesh:        Nx=" << p.Nx << " ; Ny=" << p.Ny << "\n"
      << " dx, dy:      " << p.dx << ", " << p.dy << "\n"
      << " CFL:         " << p.cfl_number << "\n"
      << " Viscosity:   " << p.viscosity << "\n"
      << " Final time:  " << p.final_time << "\n"
      << " Advection:   " << p.advection_scheme << "\n"
      << " Poisson:     " << p.poisson_solver << "\n"
      << " BCs:         " << p.boundary_conditions << "\n"
      << " Init:        " << p.initial_condition << "\n"
      << " Obstacle:    " << p.obstacle_type << "\n"
      << " Out dir:     " << p.output_directory << "\n"
      << " Save every:  " << p.save_interval << "\n"
      << " Verbosity:   " << p.verbosity << "\n"
      << "=============================\n"
      << std::flush;
  }
  
} // namespace navier_stokes


