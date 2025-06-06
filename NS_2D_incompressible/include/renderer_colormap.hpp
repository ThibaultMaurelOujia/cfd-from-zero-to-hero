#ifndef NAVIER_STOKES_CPP_COLORMAP_HPP
#define NAVIER_STOKES_CPP_COLORMAP_HPP

#include "utils.hpp"
#include "boundary.hpp"
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Image.hpp> 

#include <cstdint> 

namespace navier_stokes {

inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// t in [0,1] -> degrade defini par vos segments
inline sf::Color colormap_B_W_R(double t_)
{
    // clamp
    double t = std::max(0.0, std::min(1.0, t_));

    // bleu :
    // 0.0 -> 0.3, 0.25 -> 1.0, 0.5 -> 1.0, 0.75 -> 0.0, 1.0 -> 0.0
    double b;
    if (t < 0.25) {
        b = lerp(0.3, 1.0, t / 0.25);
    } else if (t < 0.5) {
        b = 1.0;
    } else if (t < 0.75) {
        b = lerp(1.0, 0.0, (t - 0.5) / 0.25);
    } else {
        b = 0.0;
    }

    // vert :
    // 0.0 -> 0.0, 0.25 -> 0.0, 0.5 -> 1.0, 0.75 -> 0.0, 1.0 -> 0.0
    double g;
    if (t < 0.25) {
        g = 0.0;
    } else if (t < 0.5) {
        g = lerp(0.0, 1.0, (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        g = lerp(1.0, 0.0, (t - 0.5) / 0.25);
    } else {
        g = 0.0;
    }

    // rouge :
    // 0.0 -> 0.0, 0.25 -> 0.0, 0.5 -> 1.0, 0.75 -> 1.0, 1.0 -> 0.5
    double r;
    if (t < 0.25) {
        r = 0.0;
    } else if (t < 0.5) {
        r = lerp(0.0, 1.0, (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        r = 1.0;
    } else {
        r = lerp(1.0, 0.5, (t - 0.75) / 0.25);
    }

    // passage en 0–255
    auto to8 = [&](double x){
        return static_cast<std::uint8_t>(std::max(0.0, std::min(1.0, x)) * 255.0 + 0.5);
    };

    return sf::Color{ to8(r), to8(g), to8(b) };
}



void render_scalar_field(const Field& field,
    const ObstacleMask& mask,
    sf::Image&   img,
    int          Nx,
    int          Ny,
    int          N_ghost,
    double       vmin,
    double       vmax);

    
} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_COLORMAP_HPP