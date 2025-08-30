#include "utils.hpp"

namespace navier_stokes {




using Clock = std::chrono::high_resolution_clock;
static Clock::time_point _timer_start;


void time_tic() {
    _timer_start = Clock::now();
}

double time_toc(const std::string& msg) {
    auto now     = Clock::now();
    double elapsed = std::chrono::duration<double>(now - _timer_start).count();
    if (!msg.empty()) {
        std::cout << "[Timer] " << msg << ": " << elapsed << " s\n";
    }
    return elapsed;
}
// time_tic();
// time_toc("Poisson direct 1"); 




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

    // plafonner si nécessaire
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



void compute_vorticity(const Field& u, const Field& v, Field& w,
                       int Nx, int Ny, int N_ghost,
                       double dx, double dy) {
    if ((int)u.size()    != Nx+2*N_ghost ||
        (int)v.size()    != Nx+2*N_ghost ||
        (int)w.size() != Nx+2*N_ghost)
        throw std::invalid_argument("Tailles de champs incorrectes");

    //   différences centrales 
    #pragma omp parallel for
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            double dvdx = ( v[i+1][j] - v[i-1][j] ) / (2.0 * dx);
            double dudy = ( u[i][j+1] - u[i][j-1] ) / (2.0 * dy);
            w[i][j]  = dvdx - dudy;
        }
    }
 
    for(int i=0; i<N_ghost; ++i)
        for(int j=0; j<(int)w[i].size(); ++j)
            w[i][j] = w[Nx+N_ghost][j] = 0.0;
    for(int i=0; i<(int)w.size(); ++i)
        for(int j=0; j<N_ghost; ++j)
            w[i][j] = w[i][Ny+N_ghost] = 0.0;
}




} // namespace navier_stokes