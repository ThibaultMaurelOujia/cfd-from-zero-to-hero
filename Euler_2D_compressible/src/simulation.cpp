#include "simulation.hpp"

namespace navier_stokes {




Simulator::Simulator(const SimulationParams& params): params_{params}
                                                    , display_size_{ compute_display_size(params.Lx, params.Ly) }
                                                    , win_{
                                                        sf::VideoMode{
                                                            sf::Vector2u{
                                                                unsigned(display_size_.first  * 100),
                                                                unsigned(display_size_.second * 100)
                                                            }
                                                        },
                                                        "Navier-Stokes"
                                                    }
                                                    // , win_{
                                                    //     // on agrandit la fenêtre sans toucher à img_ ni tex_
                                                    //     sf::VideoMode{ sf::Vector2u{
                                                    //         static_cast<unsigned>(params.Nx * display_zoom),
                                                    //         static_cast<unsigned>(params.Ny * display_zoom)
                                                    //     } },
                                                    //     "u-field"
                                                    //     }
                                                    , img_{ sf::Vector2u{
                                                            static_cast<unsigned>(params.Nx),
                                                            static_cast<unsigned>(params.Ny)
                                                        },
                                                        sf::Color::Black
                                                        }
                                                    , tex_{ img_ }
                                                    , spr_{ tex_ }
                                                    { 
    int N_ghost = params_.bc_ghost_layers;
    int Nx = params_.Nx;
    int Ny = params_.Ny;

    rho_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    rho_u_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    rho_v_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));

    E_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    rho_star_.assign(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    rho_u_star_.assign(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    rho_v_star_.assign(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    E_star_.assign   (Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));

    conv_rho_.assign   (Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    conv_rho_u_.assign (Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    conv_rho_v_.assign (Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    conv_E_.assign     (Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));


    // fenetre SFML 
    // mise a l'echelle du sprite 
    spr_.setScale(
        sf::Vector2f{
            float(win_.getSize().x) / float(params.Nx),
            float(win_.getSize().y) / float(params.Ny)
        }
    );
    win_.setVerticalSyncEnabled(true);
}

void Simulator::run() {
    initialize_fields();     
    time_loop();
}

void Simulator::initialize_fields() { 
    initialize_flow_field(params_, rho_, rho_u_, rho_v_, E_);
 
    mask_ = createMask(params_);
 
    apply_immersed_boundary(params_, rho_, rho_u_, rho_v_, E_, mask_);
 
    apply_boundary_conditions(params_, rho_, rho_u_, rho_v_, E_);
}

void Simulator::time_loop() {
    double t = 0.0;
    int    it = 0;
    const int    Nx      = params_.Nx;
    const int    Ny      = params_.Ny;
    const double dx      = params_.dx;
    const double dy      = params_.dy;
    const int    N_ghost = params_.bc_ghost_layers;

    while (t < params_.final_time) { 
        double max_sx = 1e-8, max_sy = 1e-8;
        #pragma omp parallel for collapse(2) reduction(max:max_sx,max_sy)
        for(int i = N_ghost; i < Nx+N_ghost; ++i){
            for(int j = N_ghost; j < Ny+N_ghost; ++j){
                double r  = rho_[i][j];
                double ru = rho_u_[i][j];
                double rv = rho_v_[i][j];
                double Et = E_[i][j];
                double u = ru/r;
                double v = rv/r;
                double kinetic = 0.5*(ru*ru + rv*rv)/r;
                double p = (params_.gamma - 1.0)*(Et - kinetic);
                double c = std::sqrt(params_.gamma * p / r);

                max_sx = std::max(max_sx, std::abs(u) + c);
                max_sy = std::max(max_sy, std::abs(v) + c);
            }
        }
 
        double dt_x = dx / max_sx;
        double dt_y = dy / max_sy;
        double dt   = params_.cfl_number * std::min(dt_x, dt_y);
 
        step_SSP_RK3(dt);
 
        if (it % params_.save_interval == 0) {
            output_step(it, t, dt);
        }
 
        t += dt;
        ++it;
    }
}


void Simulator::step_SSP_RK3(double dt) {
 
    rk3_substep(0.0, 1.0, dt);
 
    rk3_substep(3.0/4.0, 1.0/4.0, dt);
 
    rk3_substep(1.0/3.0, 2.0/3.0, dt);
}


void Simulator::rk3_substep(double c0, double c1, double dt) {
 
    const int    Nx      = params_.Nx;
    const int    Ny      = params_.Ny;
    const double dx      = params_.dx;
    const double dy      = params_.dy;
    const int    N_ghost = params_.bc_ghost_layers;
    
    // 1) apply BCs and immersed boundary
    apply_boundary_conditions(params_, rho_, rho_u_, rho_v_, E_);
    apply_immersed_boundary(params_, rho_, rho_u_, rho_v_, E_, mask_);

    // 2) advection term
    time_tic();
    compute_advection(
        params_,
        rho_, rho_u_, rho_v_, E_,
        conv_rho_, conv_rho_u_, conv_rho_v_, conv_E_,
        dx, dy
    );
    time_toc("Advection", params_.verbosity); 

    // 3) diffusion term
    // time_tic();
    // compute_diffusion(params_, u_, v_, D_u_, D_v_, dx, dy);
    // time_toc("Diffusion"); 

    // 4) build provisional velocities 
    time_tic();
    #pragma omp parallel for collapse(2) //schedule(static,64)
    for(int i = N_ghost; i < Nx + N_ghost; ++i) {
        for(int j = N_ghost; j < Ny + N_ghost; ++j) {
            rho_star_[i][j]   = rho_[i][j]   + dt * conv_rho_[i][j];
            rho_u_star_[i][j] = rho_u_[i][j] + dt * conv_rho_u_[i][j];
            rho_v_star_[i][j] = rho_v_[i][j] + dt * conv_rho_v_[i][j];
            E_star_[i][j]     = E_[i][j]     + dt * conv_E_[i][j];
        }
    }
    time_toc("u_star_", params_.verbosity); 
    apply_boundary_conditions(params_, rho_, rho_u_, rho_v_, E_);
    apply_immersed_boundary(params_, rho_, rho_u_, rho_v_, E_, mask_); 

    // 7) RK3 linear combination
    time_tic();
    #pragma omp parallel for collapse(2) //schedule(static,64)
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            rho_[i][j] = c0*rho_[i][j] + c1*rho_star_[i][j]; 
            rho_u_[i][j] = c0*rho_u_[i][j] + c1*rho_u_star_[i][j];
            rho_v_[i][j] = c0*rho_v_[i][j] + c1*rho_v_star_[i][j];
            E_[i][j] = c0*E_[i][j] + c1*E_star_[i][j];
        }
    }
    time_toc("RK3 linear combination", params_.verbosity); 
}




void Simulator::output_step(int it, double t, double dt)
{
    if (params_.verbosity > 0)
        std::cout << "it=" << it
                  << "  t=" << t
                  << "  dt=" << dt << '\n' << std::flush; // std::endl

 
    const int Nx = params_.Nx;
    const int Ny = params_.Ny;
    const int N_ghost  = params_.bc_ghost_layers;


    Field field(Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost, 0.0));
    double fmax = 1e-16;
    std::string cmap = "";

    if (params_.output_quantity == "density") {
        field = rho_;   
        fmax  = max_abs_interior(rho_, Nx, Ny, N_ghost);
    }
    else if (params_.output_quantity == "pressure") {
        compute_pressure(rho_, rho_u_, rho_v_, E_,
                               field, Nx, Ny, N_ghost,
                               params_.gamma);
        fmax = max_abs_interior(field, Nx, Ny, N_ghost);
    }
    else if (params_.output_quantity == "schlieren") {
        compute_schlieren(rho_, field,
                          Nx, Ny, N_ghost,
                          params_.dx, params_.dy);
        fmax = max_abs_interior(field, Nx, Ny, N_ghost);
        cmap = "schlieren";
    }
    else if (params_.output_quantity == "vorticity") {
        compute_vorticity(rho_, rho_u_, rho_v_, field,
                          Nx, Ny, N_ghost,
                          params_.dx, params_.dy);
        fmax = max_abs_interior(field, Nx, Ny, N_ghost);
    }
    else if (params_.output_quantity == "rho_u") {
        field = rho_u_;
        fmax  = max_abs_interior(rho_u_, Nx, Ny, N_ghost);
    }
    else if (params_.output_quantity == "rho_v") {
        field = rho_v_;
        fmax  = max_abs_interior(rho_v_, Nx, Ny, N_ghost);
    }
    else if (params_.output_quantity == "mach") {
        compute_mach(rho_, rho_u_, rho_v_, E_, field, Nx, Ny, N_ghost,params_.gamma);
        fmax = max_abs_interior(field, Nx, Ny, N_ghost);
    }
    else if (params_.output_quantity == "E") {
        field = E_;
        fmax  = max_abs_interior(E_, Nx, Ny, N_ghost);
    }
    else {
        throw std::invalid_argument(
           "Unknown output_quantity: " + params_.output_quantity
        );
    }

    // 2) Rendu
    render_scalar_field(field,
                        mask_, img_,
                        Nx, Ny, N_ghost,
                        -fmax, fmax,
                        cmap);



    // on met a jour la texture, on reassocie au sprite
    tex_.update(img_);
    spr_.setTexture(tex_, true);

    // on dessine
    win_.clear();
    win_.draw(spr_);
    win_.display();

    // boucle d'evenements SFML3
    while (auto ev = win_.pollEvent())
    {
        if (ev->is<sf::Event::Closed>())
            win_.close();
    }



    // Noms de fichiers
    const std::string base = params_.output_directory + "/it_" + std::to_string(it);
    const std::string file_u = base + "_u.bin";
    const std::string file_v = base + "_v.bin";
    const std::string file_w = base + "_w.bin";

    // Écriture binaire de u_ et v_
    try {
        // write_field_binary(file_u, u_, Nx, Ny, N_ghost);
        // write_field_binary(file_v, v_, Nx, Ny, N_ghost);
        // write_field_binary(file_w, w, Nx, Ny, N_ghost);
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur ecriture binaire : " << e.what() << "\n";
    }

}

} // namespace navier_stokes













