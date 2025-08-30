#include "utils.hpp"
#include "boundary.hpp"


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

void compute_pressure_coefficient(
    const Field& p_field, 
    Field& Cp_on_wall,
    const ObstacleMask& mask,
    int Nx, int Ny, int N_ghost,
    double p_inf, double rho_inf, double U_inf) {
    // verification de taille
    if ((int)p_field.size()                   != Nx+2*N_ghost ||
        (int)Cp_on_wall.size()          != Nx+2*N_ghost ||
        (int)p_field.front().size()           != Ny+2*N_ghost ||
        (int)Cp_on_wall.front().size()  != Ny+2*N_ghost) {
        throw std::invalid_argument("Tailles de champs incorrectes pour pressure");
    }

    for(auto [I, J] : mask.left) {
        int i_fluid = I - 1;
        int j_fluid = J;
        double p_loc = p_field[i_fluid][j_fluid];
        double Cp_val = (p_loc - p_inf) / (0.5 * rho_inf * U_inf * U_inf);
        Cp_on_wall[i_fluid][j_fluid] = Cp_val;
    }

    for(auto [I, J] : mask.right) {
        int i_fluid = I + 1;
        int j_fluid = J;
        double p_loc = p_field[i_fluid][j_fluid];
        double Cp_val = (p_loc - p_inf) / (0.5 * rho_inf * U_inf * U_inf);
        Cp_on_wall[i_fluid][j_fluid] = Cp_val;
    }

    for(auto [I, J] : mask.bottom) {
        int i_fluid = I;
        int j_fluid = J - 1;
        double p_loc = p_field[i_fluid][j_fluid];
        double Cp_val = (p_loc - p_inf) / (0.5 * rho_inf * U_inf * U_inf);
        Cp_on_wall[i_fluid][j_fluid] = Cp_val;
    }

    for(auto [I, J] : mask.top) {
        int i_fluid = I;
        int j_fluid = J + 1;
        double p_loc = p_field[i_fluid][j_fluid];
        double Cp_val = (p_loc - p_inf) / (0.5 * rho_inf * U_inf * U_inf);
        Cp_on_wall[i_fluid][j_fluid] = Cp_val;
    }

    // #pragma omp parallel for collapse(2)
    // for(int i = N_ghost; i < Nx+N_ghost; ++i) {
    //   for(int j = N_ghost; j < Ny+N_ghost; ++j) {
    //     double p_loc = p_field[i][j];
    //     double Cp_val = (p_loc - p_inf) / (0.5 * rho_inf * U_inf * U_inf);
    //     Cp_on_wall[i][j] = Cp_val;
    //   }
    // }
}

void compute_viscous_stress_tensor(
    const Field& rho, const Field& rho_u, const Field& rho_v,  
    Field& tau_xx, Field& tau_xy, Field& tau_yx, Field& tau_yy, 
    int Nx, int Ny, int N_ghost,
    double dx, double dy,
    double mu) {

    #pragma omp parallel for collapse(2)
    for(int i = N_ghost; i < Nx + N_ghost; ++i) {
        for(int j = N_ghost; j < Ny + N_ghost; ++j) {
 
            double u_ip = rho_u[i+1][j] / rho[i+1][j];
            double u_im = rho_u[i-1][j] / rho[i-1][j];
            double v_ip = rho_v[i+1][j] / rho[i+1][j];
            double v_im = rho_v[i-1][j] / rho[i-1][j];
            double u_jp = rho_u[i][j+1] / rho[i][j+1];
            double u_jm = rho_u[i][j-1] / rho[i][j-1];
            double v_jp = rho_v[i][j+1] / rho[i][j+1];
            double v_jm = rho_v[i][j-1] / rho[i][j-1];

            double dudx = (u_ip - u_im) / (2.0 * dx);
            double dvdy = (v_jp - v_jm) / (2.0 * dy);
            double dudy = (u_jp - u_jm) / (2.0 * dy);
            double dvdx = (v_ip - v_im) / (2.0 * dx);

            // Divergence
            double divV = dudx + dvdy;

            // Tenseur
            double tau_xx_loc = 2.0 * mu * dudx - (2.0/3.0) * mu * divV;
            double tau_yy_loc = 2.0 * mu * dvdy - (2.0/3.0) * mu * divV;
            double tau_xy_loc =       mu * (dudy + dvdx);
            double tau_yx_loc = tau_xy_loc;

            // tau
            tau_xx[i][j] = tau_xx_loc;
            tau_xy[i][j] = tau_xy_loc;
            tau_yx[i][j] = tau_yx_loc;
            tau_yy[i][j] = tau_yy_loc;
        }
    }
}

void compute_lift_coefficient(
    const Field& p_field, const Field& tau_xx, const Field& tau_xy, const Field& tau_yx, const Field& tau_yy, 
    Field& C_L_on_wall,
    const ObstacleMask& mask,
    int Nx, int Ny, int N_ghost,
    double dx, double dy,
    double q_inf) {

    for(auto [I, J] : mask.bottom) {
        int i_fluid = I;
        int j_fluid = J - 1;
        // int n_x = 0;
        int n_y = -1;
        // C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_y + tau_yx[i_fluid][j_fluid]*n_x + tau_yy[i_fluid][j_fluid]*n_y) * dx / q_inf;
        C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_y + tau_yy[i_fluid][j_fluid]*n_y) * dx / q_inf;
    } // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! A CORRIGE ET FINIR

    for(auto [I, J] : mask.top) {
        int i_fluid = I;
        int j_fluid = J + 1;
        // int n_x = 0;
        int n_y = 1;
        // C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_y + tau_yx[i_fluid][j_fluid]*n_x + tau_yy[i_fluid][j_fluid]*n_y) * dx / q_inf;
        C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_y + tau_yy[i_fluid][j_fluid]*n_y) * dx / q_inf;
    }// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! A CORRIGE ET FINIR
}

void compute_drag_coefficient(
    const Field& p_field, const Field& tau_xx, const Field& tau_xy, const Field& tau_yx, const Field& tau_yy, 
    Field& C_L_on_wall,
    const ObstacleMask& mask,
    int Nx, int Ny, int N_ghost,
    double dx, double dy,
    double q_inf) {

    for(auto [I, J] : mask.left) {
        int i_fluid = I - 1;
        int j_fluid = J;
        int n_x = -1;
        // int n_y = 0;
        // C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_x + tau_xx[i_fluid][j_fluid]*n_x + tau_xy[i_fluid][j_fluid]*n_y) * dy / q_inf;
        C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_x + tau_xx[i_fluid][j_fluid]*n_x) * dy / q_inf;
    }// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! A CORRIGE ET FINIR

    for(auto [I, J] : mask.right) {
        int i_fluid = I + 1;
        int j_fluid = J;
        int n_x = 1;
        // int n_y = 0;
        // C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_x + tau_xx[i_fluid][j_fluid]*n_x + tau_xy[i_fluid][j_fluid]*n_y) * dy / q_inf;
        C_L_on_wall[i_fluid][j_fluid] = (-p_field[i_fluid][j_fluid]*n_x + tau_xx[i_fluid][j_fluid]*n_x) * dy / q_inf;
    }// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! A CORRIGE ET FINIR
}



} // namespace navier_stokes