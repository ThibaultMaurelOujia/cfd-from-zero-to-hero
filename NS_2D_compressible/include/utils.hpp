#ifndef NAVIER_STOKES_CPP_UTILS_HPP
#define NAVIER_STOKES_CPP_UTILS_HPP

#include <filesystem>
#include <cmath> 
#include <cassert>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>

#include <omp.h>


namespace navier_stokes {

inline constexpr double PI     = 3.14159265358979323846;
inline constexpr double TWO_PI = 2.0 * PI;


using Field = std::vector<std::vector<double>>;

struct ObstacleMask;


// DEmarre le chronometre
void time_tic();

// ArrEte le chrono, affiche (si msg non vide) et renvoie le temps ecoule en secondes
double time_toc(const std::string& msg = "", int verbosity=1);


 double max_abs_interior(
    const Field& field,
    int Nx, int Ny, int N_ghost
);


void compute_vorticity(
    const Field& rho, const Field& rho_u, const Field& rho_v, 
    Field& w, 
    int Nx, int Ny, int N_ghost,
    double dx, double dy);

void compute_schlieren(const Field& rho,
    Field& schlieren,
    int Nx, int Ny, int N_ghost,
    double dx, double dy);

void compute_pressure(const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
    Field& p,
    int Nx, int Ny, int N_ghost,
    double gamma);                       

void compute_mach(const Field& rho, const Field& rho_u, const Field& rho_v, const Field& E,
    Field& mach,
    int Nx, int Ny, int N_ghost,
    double gamma);

void compute_pressure_coefficient(const Field& p_field, Field& Cp_on_wall, const ObstacleMask& mask, 
    int Nx, int Ny, int N_ghost, double p_inf, double rho_inf, double U_inf);    

void compute_viscous_stress_tensor(
    const Field& rho, const Field& rho_u, const Field& rho_v,  
    Field& tau_xx, Field& tau_xy, Field& tau_yx, Field& tau_yy, 
    int Nx, int Ny, int N_ghost, double dx, double dy, double mu);

void compute_lift_coefficient(const Field& p_field, const Field& tau_xx, const Field& tau_xy, const Field& tau_yx, const Field& tau_yy,  Field& C_L_on_wall,
    const ObstacleMask& mask, int Nx, int Ny, int N_ghost, double dx, double dy, double q_inf);

void compute_drag_coefficient(const Field& p_field, const Field& tau_xx, const Field& tau_xy, const Field& tau_yx, const Field& tau_yy,  Field& C_L_on_wall,
    const ObstacleMask& mask, int Nx, int Ny, int N_ghost, double dx, double dy, double q_inf);
    
std::pair<double,double> compute_display_size(
    double Lx, double Ly,
    double min_w = 7.0, double min_h = 8.0,
    double max_w = 15.0, double max_h = 10.0);


void write_field_binary(const std::string& filename,
    const Field& field,
    int Nx, int Ny,
    int N_ghost);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_UTILS_HPP