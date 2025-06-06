#include "renderer_colormap.hpp"




namespace navier_stokes {

    void render_scalar_field(
        const Field& field,
        const ObstacleMask& mask,
        sf::Image& img,
        int Nx, int Ny, int N_ghost,
        double vmin, double vmax,
        const std::string& cmap
    )
    {
        double span = vmax - vmin;
        for(int j = 0; j < Ny; ++j) {
            for(int i = 0; i < Nx; ++i) {
                if (mask.solid[i][j]) {
                    img.setPixel({unsigned(i), unsigned(Ny - 1 - j)}, sf::Color::Black);
                } else {
                    double w = field[i + N_ghost][j + N_ghost];
                    double t = (w - vmin) / span;  // normalisation lineaire
    
                    sf::Color c;
                    if (cmap == "schlieren") {
                        t = w / vmax; // 2.0/3.0 // 3.0/4.0
                        t = std::sqrt(t);
                        c = colormap_schlieren_bluewhite(t);
                    } else {
                        c = colormap_B_W_R(t);
                    }
    
                    img.setPixel({unsigned(i), unsigned(Ny - 1 - j)}, c);
                }
            }
        }
    }

} // namespace navier_stokes



