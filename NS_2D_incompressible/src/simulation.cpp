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

    u_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    v_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    p_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));

    A_u_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    A_v_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));

    D_u_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));
    D_v_.assign(Nx + 2 * N_ghost, std::vector<double>(Ny + 2 * N_ghost, 0.0));

    u_star_.assign(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    v_star_.assign(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));

    p_corr_.assign(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));

    // fenetre SFML 
    // mise a l'echelle du sprite 
    spr_.setScale(
        sf::Vector2f{
            float(win_.getSize().x) / float(params.Nx),
            float(win_.getSize().y) / float(params.Ny)
        }
    );
    // spr_.setScale(
    //     sf::Vector2f{
    //       static_cast<float>(display_zoom),
    //       static_cast<float>(display_zoom)
    //     }
    // );
    // allocation des tableaux numeriques 
    win_.setVerticalSyncEnabled(true);
}

void Simulator::run() {
    initialize_fields();        // fill u,v,p,mask at t=0 
    time_loop();
}

void Simulator::initialize_fields() { 
    initialize_flow_field(params_, u_, v_, p_);
 
    mask_ = createMask(params_);
 
    apply_immersed_boundary(params_, u_, v_, mask_);
 
    apply_boundary_conditions(params_, u_, v_, p_);
}

void Simulator::time_loop() {
    double t = 0.0;
    int    it = 0;
    double dx = params_.dx;
    double dy = params_.dy;
    double nu = params_.viscosity;
    double CFL = params_.cfl_number;

    while (t < params_.final_time) {
 
        double u_v_max = std::max( 1e-8,
            std::max(
                max_abs_interior(u_, params_.Nx, params_.Ny, params_.bc_ghost_layers),
                max_abs_interior(v_, params_.Nx, params_.Ny, params_.bc_ghost_layers)
            )
        );
        double dt_adv = CFL * std::min(dx/u_v_max, dy/u_v_max);
        double dt_diff= 0.5 * std::min(dx*dx, dy*dy) / nu;
        double dt     = std::min(dt_adv, dt_diff); 

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


void Simulator::rk3_substep(double c0, double c1, double dt, int max_iters) {
 
    const int    Nx      = params_.Nx;
    const int    Ny      = params_.Ny;
    const double dx      = params_.dx;
    const double dy      = params_.dy;
    const double nu      = params_.viscosity;
    const int    N_ghost = params_.bc_ghost_layers;
    
    // 1) apply BCs and immersed boundary
    apply_boundary_conditions(params_, u_, v_, p_);
    apply_immersed_boundary(params_, u_, v_, mask_); 

    // 2) advection term
    compute_advection(params_, u_, v_, A_u_, A_v_, dx, dy);

    // 3) diffusion term
    compute_diffusion(params_, u_, v_, D_u_, D_v_, dx, dy);

    // 4) build provisional velocities 
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
      for(int j = N_ghost; j < Ny+N_ghost; ++j) {
        u_star_[i][j] = u_[i][j] + dt*(-A_u_[i][j] + D_u_[i][j]);
        v_star_[i][j] = v_[i][j] + dt*(-A_v_[i][j] + D_v_[i][j]);
      }
    }
    apply_boundary_conditions(params_, u_star_, v_star_, p_);
    apply_immersed_boundary(params_, u_star_, v_star_, mask_); 

    // 5) solve Poisson for pressure correction
    p_corr_ = solve_pressure_poisson(p_,
                                    compute_divergence(u_star_, v_star_, Nx, Ny, N_ghost, dx, dy, dt),
                                    Nx, Ny, N_ghost, 
                                    dx, dy,
                                    params_.poisson_solver,
                                    max_iters);

    // 6) project to get corrected velocities
    Field& u_corr = u_star_;
    Field& v_corr = v_star_;
    project(u_corr, v_corr, p_corr_, Nx, Ny, N_ghost, dx, dy, dt);
    apply_boundary_conditions(params_, u_corr, v_corr, p_corr_);
    apply_immersed_boundary(params_, u_corr, v_corr, mask_); 

    // 7) RK3 linear combination
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            u_[i][j] = c0*u_[i][j] + c1*u_corr[i][j]; 
            v_[i][j] = c0*v_[i][j] + c1*v_corr[i][j];
            p_[i][j] = c0*p_[i][j] + c1*p_corr_[i][j];
        }
    }
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


    Field w(u_.size(), std::vector<double>(u_[0].size(), 0.0));
    compute_vorticity(u_, v_, w,
                      params_.Nx, params_.Ny,
                      params_.bc_ghost_layers,
                      params_.dx, params_.dy);
     
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
        std::cerr << "Erreur écriture binaire : " << e.what() << "\n";
    }


    double u_max = max_abs_interior(u_, Nx, Ny, N_ghost);
    // std::cout << "u_max=" << u_max << '\n';
    // render_scalar_field(u_, mask_, img_, Nx, Ny, N_ghost, -u_max*2/3, u_max*2/3);

    double v_max = max_abs_interior(v_, Nx, Ny, N_ghost);
    // std::cout << "v_max=" << v_max << '\n';
    // render_scalar_field(v_, mask_, img_, Nx, Ny, N_ghost, -v_max*2/3, v_max*2/3);

    double p_max = max_abs_interior(p_, Nx, Ny, N_ghost);
    // std::cout << "p_max=" << p_max << '\n';
    // render_scalar_field(p_, mask_, img_, Nx, Ny, N_ghost, -p_max*2/3, p_max*2/3);

    double w_max = max_abs_interior(w, Nx, Ny, N_ghost);
    // std::cout << "w_max=" << w_max << '\n';
    render_scalar_field(w, mask_, img_, Nx, Ny, N_ghost, -w_max*2/3, w_max*2/3);


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
}




} // namespace navier_stokes


