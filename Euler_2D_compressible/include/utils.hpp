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

 
void time_tic();
 
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