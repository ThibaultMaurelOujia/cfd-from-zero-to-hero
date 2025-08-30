#include "init.hpp"

namespace navier_stokes {




inline double sound_speed_ideal(double gamma, double p, double rho){ return std::sqrt(gamma * p / rho); }
 
inline void primitives_to_conserved(double rho, double u, double v, double p, double gamma,
                                    double& rho_u, double& rho_v, double& E){
    rho_u = rho * u;
    rho_v = rho * v;
    double kinetic = 0.5 * rho * (u*u + v*v);
    E = p/(gamma-1.0) + kinetic;
}


//---------------------------------------------------------------------------
void init_kelvin_helmholtz(      
    const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E,
    double delta   = 0.005,   
    double amp     = 1e-1,  
    int    kx      = 4)   
{
    const int Nx = params.Nx, Ny = params.Ny, N = params.bc_ghost_layers;
    const double Lx = params.Lx, Ly = params.Ly, dx = params.dx, dy = params.dy;
    const double gamma = params.gamma;

    double rho0 = params.rho_ref;   
    double p0   = params.p_ref;    
    const double a0   = sound_speed_ideal(gamma,p0,rho0);
 
    double M  = params.inflow_velocity;   
    double U0 = M * a0;         
    
    const double two_delta2 = 2.0 * delta * delta;

    for(int i=N;i<Nx+N;++i){
        double x = (i+0.5)*dx;
        for(int j=N;j<Ny+N;++j){
            double y = (j+0.5)*dy;
 
            double u   = U0 * std::tanh((y-0.5*Ly)/delta);    
            double v   = amp * std::sin(2*M_PI*kx*x/Lx)
                              * std::exp(-(y-0.5*Ly)*(y-0.5*Ly)/two_delta2);
 
            double rho_u_, rho_v_, E_;
            primitives_to_conserved(rho0,u,v,p0,gamma,rho_u_,rho_v_,E_);

            rho[i][j]   = rho0;
            rho_u[i][j] = rho_u_;
            rho_v[i][j] = rho_v_;
            E  [i][j]   = E_;
        }
    }
}



//---------------------------------------------------------------------------
void init_one_x(        
    const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E)
{
    const int Nx=params.Nx, Ny=params.Ny, N=params.bc_ghost_layers;
    const double gamma=params.gamma;
 
    double rho0 = params.rho_ref;
    double p0   = params.p_ref;
    const double a0   = sound_speed_ideal(gamma,p0,rho0);

    double M  = params.inflow_velocity;     
    double u0 = M * a0;

    for(int i=N;i<Nx+N;++i){
        for(int j=N;j<Ny+N;++j){
            double rho_u_,rho_v_,E_;
            primitives_to_conserved(rho0,u0,0.0,p0,gamma,rho_u_,rho_v_,E_);

            rho[i][j]   = rho0;
            rho_u[i][j] = rho_u_;
            rho_v[i][j] = rho_v_;      // = 0
            E  [i][j]   = E_;
        }
    }
}



//---------------------------------------------------------------------------
void init_sod_x(const SimulationParams& p,
                Field& rho,Field& rho_u,Field& rho_v,Field& E) 
{
    const int Nx=p.Nx, Ny=p.Ny, N=p.bc_ghost_layers;
    double gamma=p.gamma;

    const double rho_L=1.0,  p_L=1.0;
    const double rho_R=0.125,p_R=0.1;
    const double a_L=sound_speed_ideal(gamma,p_L,rho_L);
    const double a_R=sound_speed_ideal(gamma,p_R,rho_R);

    for(int i=N;i<Nx+N;++i){
        bool left = (i < N + Nx/2);
        for(int j=N;j<Ny+N;++j){
            double rho0 = left?rho_L:rho_R;
            double p0   = left?p_L  :p_R;

            double rho_u_,rho_v_,E_;
            primitives_to_conserved(rho0,0.0,0.0,p0,gamma,rho_u_,rho_v_,E_);

            rho[i][j]=rho0; rho_u[i][j]=rho_u_;
            rho_v[i][j]=rho_v_; E[i][j]=E_;
        }
    }
}



//---------------------------------------------------------------------------
void init_isentropic_vortex(
    const SimulationParams& params,
    Field& rho,   Field& rho_u,
    Field& rho_v, Field& E)
{
    const int Nx      = params.Nx;
    const int Ny      = params.Ny;
    const int N       = params.bc_ghost_layers;
    const double Lx   = params.Lx;
    const double Ly   = params.Ly;
    const double dx   = params.dx;
    const double dy   = params.dy;
    const double gamma= params.gamma;
 
    const double U0     = 1.0;  
    const double V0     = 0.0;
    const double beta   = 5.0;    
    const double x0     = 0.5 * Lx; 
    const double y0     = 0.5 * Ly;

    for(int i = N; i < Nx+N; ++i) {
        double x = (i + 0.5) * dx;
        for(int j = N; j < Ny+N; ++j) {
            double y = (j + 0.5) * dy;
            double xd = x - x0;
            double yd = y - y0;
            double r2 = xd*xd + yd*yd;
            double expf = std::exp((1.0 - r2)/2.0);
 
            double u = U0 - beta/(2*M_PI) * yd * expf;
            double v = V0 + beta/(2*M_PI) * xd * expf;
 
            double T = 1.0
                - (gamma - 1.0)/(2.0*gamma)
                  * (beta*beta/(4.0*M_PI*M_PI)) * expf*expf;
            double rho0 = std::pow(T, 1.0/(gamma - 1.0));
            double p0   = std::pow(T, gamma/(gamma - 1.0));
 
            double ru, rv, Et;
            primitives_to_conserved(rho0, u, v, p0, gamma, ru, rv, Et);

            rho  [i][j] = rho0;
            rho_u[i][j] = ru;
            rho_v[i][j] = rv;
            E    [i][j] = Et;
        }
    }
}



//---------------------------------------------------------------------------
void init_blast_2d(
    const SimulationParams& params,
    Field& rho,   Field& rho_u,
    Field& rho_v, Field& E)
{
    const int Nx      = params.Nx;
    const int Ny      = params.Ny;
    const int N       = params.bc_ghost_layers;
    const double Lx   = params.Lx;
    const double Ly   = params.Ly;
    const double dx   = params.dx;
    const double dy   = params.dy;
    const double gamma= params.gamma;
 
    const double rho_in  = 1.0,  p_in  = 1000.0;
    const double rho_out = 1.0,  p_out = 0.1;
    const double x0      = 0.5 * Lx;
    const double y0      = 0.5 * Ly;
    const double r0      = 0.1 * std::min(Lx, Ly);

    for(int i = N; i < Nx+N; ++i) {
        double x = (i + 0.5) * dx;
        for(int j = N; j < Ny+N; ++j) {
            double y = (j + 0.5) * dy;
            double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

            double rho0 = (r < r0 ? rho_in : rho_out);
            double p0   = (r < r0 ? p_in   : p_out);
 
            double u = 0.0, v = 0.0;
            double ru, rv, Et;
            primitives_to_conserved(rho0, u, v, p0, gamma, ru, rv, Et);

            rho  [i][j] = rho0;
            rho_u[i][j] = ru;
            rho_v[i][j] = rv;
            E    [i][j] = Et;
        }
    }
}



//---------------------------------------------------------------------------
static void add_velocity_noise(Field& rho_u, Field& rho_v,
                               int Nx,int Ny,int N,double amplitude)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> d(-amplitude,amplitude);

    for(int i=N;i<Nx+N;++i)
      for(int j=N;j<Ny+N;++j){
          rho_u[i][j] += d(gen);
          rho_v[i][j] += d(gen);
      }
}



//---------------------------------------------------------------------------
void initialize_flow_field(
    const SimulationParams& params,
    Field& rho, Field& rho_u, Field& rho_v, Field& E)
{
    if (params.initial_condition == "kelvin_helmholtz") {
        init_kelvin_helmholtz(params, rho, rho_u, rho_v, E);
    }
    else if (params.initial_condition == "one_x") {
        init_one_x(params, rho, rho_u, rho_v, E);
    }
    else if (params.initial_condition == "sod_x") {
        init_sod_x(params, rho, rho_u, rho_v, E);
    }
    else if (params.initial_condition == "isentropic_vortex") {
        init_isentropic_vortex(params, rho, rho_u, rho_v, E);
    }
    else if (params.initial_condition == "blast_2d") {
        init_blast_2d(params, rho, rho_u, rho_v, E);
    }
    else {
        throw std::invalid_argument(
            "Unknown initial condition: " + params.initial_condition
        );
    }
 
    double noise = params.noise_level;
    add_velocity_noise(rho_u, rho_v,
        params.Nx, params.Ny, params.bc_ghost_layers,
        noise);
}

// isentropic_vortex, blast_2d, riemann_2d



} // namespace navier_stokes








