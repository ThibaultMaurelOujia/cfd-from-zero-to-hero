#include "utils.hpp"

namespace navier_stokes {




using Clock = std::chrono::high_resolution_clock;
static Clock::time_point _timer_start;


//---------------------------------------------------------------------------
void time_tic() {
    _timer_start = Clock::now();
}

double time_toc(const std::string& msg, int verbosity) {
    auto now     = Clock::now();
    double elapsed = std::chrono::duration<double>(now - _timer_start).count();
    if (verbosity >= 2 && !msg.empty()) {
        std::cout << "[Timer] " << msg << ": " << elapsed << " s\n";
    }
    return elapsed;
}
// time_tic();
// time_toc("Poisson direct 1"); 



//---------------------------------------------------------------------------
double max_abs_interior(
    const Field& field,
    int Nx, int Ny, int N_ghost
) {

    double m = 0.0;
    for (int i = N_ghost; i < Nx + N_ghost; ++i) {
        for (int j = N_ghost; j < Ny + N_ghost; ++j) {
            m = std::max(m, std::abs(field[i][j]));
        }
    }
    return m;
}



//---------------------------------------------------------------------------
void write_field_binary(const std::string& filename,
                        const Field& field,
                        int Nx, int Ny,
                        int N_ghost)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Impossible d'ouvrir " + filename);
 
    for (int i = N_ghost; i < Nx + N_ghost; ++i) {
        for (int j = N_ghost; j < Ny + N_ghost; ++j) {
            double v = field[i][j];
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }
    ofs.close();
}


//---------------------------------------------------------------------------
std::pair<double,double> compute_display_size(
    double Lx, double Ly,
    double min_w, double min_h,
    double max_w, double max_h) {
    double ratio = Lx / Ly;
    double w, h;

    if (ratio >= 1.0) {
        // domaine plus large que haut
        h = min_h;
        w = ratio * h;
    } else {
        // domaine plus haut que large
        w = min_w;
        h = w / ratio;
    }

    // plafonner si necessaire
    if (w > max_w) {
        w = max_w;
        h = w / ratio;
    }
    if (h > max_h) {
        h = max_h;
        w = h * ratio;
    }
    return {w, h};
}


void compute_vorticity(
    const Field& rho, const Field& rho_u, const Field& rho_v, 
    Field& w, 
    int Nx, int Ny, int N_ghost,
    double dx, double dy) { 
    if ((int)rho.size()   != Nx+2*N_ghost ||
        (int)rho_u.size() != Nx+2*N_ghost ||
        (int)rho_v.size() != Nx+2*N_ghost ||
        (int)w.size()     != Nx+2*N_ghost ||
        (int)rho.front().size()   != Ny+2*N_ghost ||
        (int)rho_u.front().size() != Ny+2*N_ghost ||
        (int)rho_v.front().size() != Ny+2*N_ghost ||
        (int)w.front().size()     != Ny+2*N_ghost)
    {
        throw std::invalid_argument("Tailles de champs incorrectes dans compute_vorticity");
    }

    // differences centrales  
    #pragma omp parallel for collapse(2)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) { 
            double u_c  = rho_u[i][j]   / rho[i][j];
            double v_c  = rho_v[i][j]   / rho[i][j];
            double v_ip = rho_v[i+1][j] / rho[i+1][j];
            double v_im = rho_v[i-1][j] / rho[i-1][j];
            double u_jp = rho_u[i][j+1] / rho[i][j+1];
            double u_jm = rho_u[i][j-1] / rho[i][j-1];

            double dvdx = (v_ip - v_im) / (2.0 * dx);
            double dudy = (u_jp - u_jm) / (2.0 * dy);
            w[i][j]    = dvdx - dudy;
        }
    }
 
    for(int i = 0; i < N_ghost; ++i) {
        for(int j = 0; j < (int)w[i].size(); ++j) {
            w[i][j]               = 0.0;
            w[Nx+N_ghost][j]      = 0.0;
        }
    }
    for(int i = 0; i < (int)w.size(); ++i) {
        for(int j = 0; j < N_ghost; ++j) {
            w[i][j]               = 0.0;
            w[i][Ny+N_ghost]      = 0.0;
        }
    }
}

void compute_schlieren(
    const Field& rho,
    Field& schlieren,
    int Nx, int Ny, int N_ghost,
    double dx, double dy) { 
    if ((int)rho.size()     != Nx+2*N_ghost ||
        (int)schlieren.size() != Nx+2*N_ghost ||
        (int)rho.front().size()     != Ny+2*N_ghost ||
        (int)schlieren.front().size() != Ny+2*N_ghost)
    {
        throw std::invalid_argument("Tailles de champs incorrectes pour schlieren");
    }

    // derivees centrales  
    #pragma omp parallel for collapse(2)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            double drdx = (rho[i+1][j] - rho[i-1][j]) / (2.0 * dx);
            double drdy = (rho[i][j+1] - rho[i][j-1]) / (2.0 * dy);
            schlieren[i][j] = std::sqrt(drdx*drdx + drdy*drdy);
        }
    }
 
    for(int i = 0; i < N_ghost; ++i) {
        for(int j = 0; j < (int)schlieren[i].size(); ++j) {
            schlieren[i][j]          = 0.0;
            schlieren[Nx+N_ghost][j] = 0.0;
        }
    }
    for(int i = 0; i < (int)schlieren.size(); ++i) {
        for(int j = 0; j < N_ghost; ++j) {
            schlieren[i][j]          = 0.0;
            schlieren[i][Ny+N_ghost] = 0.0;
        }
    }
}

void compute_pressure(
    const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
    Field& p,
    int Nx, int Ny, int N_ghost,
    double gamma) { 
    if ((int)rho.size()     != Nx+2*N_ghost ||
        (int)p.size()       != Nx+2*N_ghost ||
        (int)rho.front().size()     != Ny+2*N_ghost ||
        (int)p.front().size()       != Ny+2*N_ghost)
    {
        throw std::invalid_argument("Tailles de champs incorrectes pour pressure");
    }
 
    #pragma omp parallel for collapse(2)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            double r = rho[i][j];
            double ru = rho_u[i][j];
            double rv = rho_v[i][j];
            double Et = E[i][j];
            double kinetic = 0.5 * (ru*ru + rv*rv) / r;
            p[i][j] = (gamma - 1.0) * (Et - kinetic);
        }
    }
 
    for(int i = 0; i < N_ghost; ++i) {
        for(int j = 0; j < (int)p[i].size(); ++j) {
            p[i][j]          = 0.0;
            p[Nx+N_ghost][j] = 0.0;
        }
    }
    for(int i = 0; i < (int)p.size(); ++i) {
        for(int j = 0; j < N_ghost; ++j) {
            p[i][j]          = 0.0;
            p[i][Ny+N_ghost] = 0.0;
        }
    }
}

void compute_mach(
    const Field& rho,
    const Field& rho_u,
    const Field& rho_v,
    const Field& E,
    Field&       mach,
    int          Nx,
    int          Ny,
    int          N_ghost,
    double       gamma)
{ 
    if ((int)rho.size()     != Nx+2*N_ghost ||
        (int)rho.front().size() != Ny+2*N_ghost ||
        (int)rho_u.size()   != Nx+2*N_ghost ||
        (int)rho_v.size()   != Nx+2*N_ghost ||
        (int)E.size()       != Nx+2*N_ghost ||
        (int)mach.size()    != Nx+2*N_ghost)
    {
        throw std::invalid_argument("Tailles incorrectes dans compute_mach");
    }

    #pragma omp parallel for collapse(2)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
      for(int j = N_ghost; j < Ny+N_ghost; ++j) {
        double r  = rho  [i][j];
        double ru = rho_u[i][j];
        double rv = rho_v[i][j];
        double Et = E    [i][j];
        double u = ru/r;
        double v = rv/r;
        double kinetic = 0.5*(ru*ru + rv*rv)/r;
        double p = (gamma - 1.0)*(Et - kinetic);
        double c = std::sqrt(gamma * p / r);
        mach[i][j] = std::sqrt(u*u + v*v) / c;
      }
    }

    for(int i = 0; i < N_ghost; ++i)
      for(int j = 0; j < (int)mach[i].size(); ++j)
        mach[i][j] = mach[Nx+N_ghost][j] = 0.0;
    for(int i = 0; i < (int)mach.size(); ++i)
      for(int j = 0; j < N_ghost; ++j)
        mach[i][j] = mach[i][Ny+N_ghost] = 0.0;
}











} // namespace navier_stokes