#include "renderer_colormap.hpp"




namespace navier_stokes {

void render_scalar_field(const Field& field,
                        const ObstacleMask& mask,
                        sf::Image& img,
                        int Nx, int Ny, int N_ghost,
                        double vmin, double vmax)
{
    double span = vmax - vmin;
    for(int j=0; j<Ny; ++j){
        for(int i=0; i<Nx; ++i){
        if (mask.solid[i][j]) {
            img.setPixel({unsigned(i), unsigned(Ny - 1 - j)}, sf::Color::Black);
        } else {
            double w = field[i+N_ghost][j+N_ghost];
            // normalisation linéaire sur [vmin,vmax]
            double t = (w - vmin)/span;
            // sf::Color c = colormap_jet(t);
            sf::Color c = colormap_B_W_R(t);
            img.setPixel({unsigned(i), unsigned(Ny-1-j)}, c);
            }
        }
    }
}

} // namespace navier_stokes



