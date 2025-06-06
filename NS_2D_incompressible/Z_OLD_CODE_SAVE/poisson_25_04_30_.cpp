
#include "poisson.hpp"

namespace navier_stokes {


void project(Field& u, Field& v, const Field& p, 
             int Nx, int Ny, int N_ghost, 
             double dx, double dy, double dt) {
    // #pragma omp parallel // DON'T WORK
    for (int i = N_ghost; i < Nx+N_ghost; ++i){
        for (int j = N_ghost; j < Ny+N_ghost; ++j) {
            u[i][j] = u[i][j] - dt * (p[i+1][j] - p[i-1][j]) / (2 * dx);
            v[i][j] = v[i][j] - dt * (p[i][j+1] - p[i][j-1]) / (2 * dy);
        }
    }
}



Field compute_divergence_centered_2ndOrder(const Field& u, const Field& v,
                                           int Nx, int Ny, int N_ghost,
                                           double dx, double dy, double dt) {
    Field div(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost,0.0));
    #pragma omp parallel
    for(int i = N_ghost; i < Nx+N_ghost; ++i) {
        for(int j = N_ghost; j < Ny+N_ghost; ++j) {
            double dudx = (u[i+1][j] - u[i-1][j]) / (2*dx);
            double dvdy = (v[i][j+1] - v[i][j-1]) / (2*dy);
            div[i][j] = (dudx + dvdy)/dt;
        }
    }
    return div;
}

Field compute_divergence(const Field& u, const Field& v,
                                           int Nx, int Ny, int N_ghost,
                                           double dx, double dy, double dt) {
    return compute_divergence_centered_2ndOrder(u, v, Nx, Ny, N_ghost, dx, dy, dt);
}



Field solve_pressure_poisson(const Field& p_old, const Field& b,
    int Nx, int Ny, int N_ghost,
    double dx, double dy,
    const std::string& method,
    bool full_solve){

    // return p_old; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (method == "direct") {
        return solve_pressure_poisson_direct(p_old, b,
                                             Nx, Ny, N_ghost,
                                             dx, dy);
    }
    // else if (method == "cholesky") {
    //     return solve_pressure_poisson_cholesky(p_old, b,
    //                                            Nx, Ny, N_ghost,
    //                                            dx, dy);
    // }
    else if (method == "amg") {
        return solve_pressure_poisson_amg(p_old, b,
                                        Nx, Ny, N_ghost,
                                        dx, dy,
                                        full_solve);
    }
    // else if (method == "fft") {
    //     return solve_pressure_poisson_fft(p_old, b,
    //                                       Nx, Ny, N_ghost,
    //                                       dx, dy);
    // }
    else {
        throw std::invalid_argument("solve_pressure_poisson : methode inconnue " + method);
    }
}









Field solve_pressure_poisson_direct(
    const Field& p_old,
    const Field& b, 
    int           Nx,
    int           Ny,
    int           N_ghost,
    double        dx,
    double        dy) {

    // // debug : combien de threads OpenMP et Eigen ?
    // int omp_thr = omp_get_max_threads();
    // int eigen_thr = Eigen::nbThreads();
    // std::cout << "[solve_pressure_poisson_direct] OpenMP threads = "
    //         << omp_thr
    //         << "   Eigen nbThreads = "
    //         << eigen_thr
    //         << std::endl;
            
    // 1) Construire le RHS // !!!!!!!!!!!!!!
    const int N = Nx*Ny;
    Eigen::VectorXd rhs(N);
    for(int j = 0, k = 0; j < Ny; ++j)
        for(int i = 0; i < Nx; ++i, ++k)
            rhs(k) = b[i+N_ghost][j+N_ghost];
    double mean_rhs = rhs.mean();
    rhs.array() -= mean_rhs;
    
    // 2) Clé de cache (Nx, Ny, dx*1e8, dy*1e8)
    Key key{
        Nx, Ny,
        int(std::round(dx*1e8)),
        int(std::round(dy*1e8))
    };

    // 3) Chercher dans le cache
    auto it = LU_cache.find(key);
    if (it == LU_cache.end()) {
        // --- première fois : on monte A et on factorise ---

        std::cout << "[solve_pressure_poisson_direct] Cache miss, LU construction for (Nx,Ny,dx,dy)=("
            << Nx << "," << Ny << ","
            << dx << "," << dy << ")\n";

        // --- Tx ---
        SpMat Tx(Nx,Nx);
        {
            std::vector<Eigen::Triplet<double>> T;
            T.reserve(3*Nx);
            for(int i=0; i<Nx; ++i){
                T.emplace_back(i, i, -2.0);
                int next = (i+1) % Nx;
                int prev = (i-1 + Nx) % Nx;
                if (next != prev) {
                    T.emplace_back(i, next, 1.0);
                    T.emplace_back(i, prev, 1.0);
                } else {
                    T.emplace_back(i, next, 1.0); // Cas où Nx est pair
                }
            }
            Tx.setFromTriplets(T.begin(), T.end());
        }

        // --- Ty ---
        SpMat Ty(Ny,Ny);
        {
            std::vector<Eigen::Triplet<double>> T;
            T.reserve(3*Ny);
            for(int j=0; j<Ny; ++j){
                T.emplace_back(j, j, -2.0);
                int next = (j+1) % Ny;
                int prev = (j-1 + Ny) % Ny;
                if (next != prev) {
                    T.emplace_back(j, next, 1.0);
                    T.emplace_back(j, prev, 1.0);
                } else {
                    T.emplace_back(j, next, 1.0); // Cas où Ny est pair
                }
            }
            Ty.setFromTriplets(T.begin(), T.end());
        }


        // 3b) Identités et 2D laplacien
        SpMat Ix(Nx,Nx); Ix.setIdentity();
        SpMat Iy(Ny,Ny); Iy.setIdentity();
        // SpMat A = Eigen::kroneckerProduct(Tx, Iy)/(dx*dx)
        //         + Eigen::kroneckerProduct(Ix, Ty)/(dy*dy);
        SpMat A = Eigen::kroneckerProduct(Iy, Tx)/(dx*dx)
                + Eigen::kroneckerProduct(Ty, Ix)/(dy*dy);
        A.makeCompressed();

        
        // 3c) Construire et stocker la factorisation
        auto solver = std::make_unique<LU>();
        solver->compute(A);  // fait analyse+factorise et initialise tout
        if (solver->info() != Eigen::Success) {
            std::cerr << "Erreur de factorisation LU\n";
            std::exit(EXIT_FAILURE);
        }

        it = LU_cache.emplace(key, std::move(solver)).first;
    }

    // 4) Résoudre
    Eigen::VectorXd sol = it->second->solve(rhs);
    if (it->second->info() != Eigen::Success) {
        std::cerr << "Erreur lors de la résolution\n";
        std::exit(EXIT_FAILURE);
    }

    // 5) Transférer dans p_new (et enlever la moyenne)
    Field p_new = p_old;
    double mean = sol.mean();
    for(int j=0,k=0;j<Ny;++j)
        for(int i=0;i<Nx;++i,++k)
            p_new[i+N_ghost][j+N_ghost] = sol(k) - mean;

    return p_new;
}


























template <typename T>
class MultigridPoissonSolver {
    public:
        MultigridPoissonSolver(int nx, int ny, double dx, double dy) 
            : nx_(nx), ny_(ny), dx_(dx), dy_(dy) {
            init_grids();
        }

    void solve(std::vector<T>& p, const std::vector<T>& b, int max_cycles=50, double tol=1e-6) {
        bool done = false;
        int cycle = 0;
        
        #pragma omp parallel shared(done, cycle)
        {
            while (!done && cycle < max_cycles) {
                // Calcul de la norme résiduelle
                double res_norm = v_cycle(p, b);
                
                // Section critique pour la condition d'arrêt
                #pragma omp single nowait
                {
                    if (res_norm < tol) done = true;
                    cycle++;
                }
                
                // Barrière implicite pour synchroniser les threads
                #pragma omp barrier
            }
        }
    }

    private:
        struct GridLevel {
            std::vector<T> values;
            std::vector<T> rhs;
            int nx, ny;
            double hx, hy;
        };

    int nx_, ny_;
    double dx_, dy_;
    std::vector<GridLevel> grids_;

    void init_grids() {
        int levels = std::floor(std::log2(std::min(nx_, ny_))) - 1;
        grids_.reserve(levels);

        // Création hiérarchie multigrille
        for (int l = 0; l < levels; ++l) {
            int nx = nx_ >> l;
            int ny = ny_ >> l;
            grids_.push_back({
                std::vector<T>((nx+2)*(ny+2), 0),
                std::vector<T>((nx+2)*(ny+2), 0),
                nx, ny,
                dx_ * (1 << l), 
                dy_ * (1 << l)
            });
        }
    }

    double v_cycle(std::vector<T>& p, const std::vector<T>& b) {
        // Niveau le plus fin
        grids_[0].rhs = b;

        // Descente
        for (size_t l = 0; l < grids_.size()-1; ++l) {
            smooth(grids_[l], 3);  // Pré-lissage
            restrict(residual(grids_[l]), grids_[l+1].rhs);
        }

        // Solveur direct au niveau le plus grossier
        direct_solve(grids_.back());

        // Remontée
        for (int l = grids_.size()-2; l >= 0; --l) {
            prolongate(grids_[l+1].values, grids_[l].values);
            smooth(grids_[l], 3);  // Post-lissage
        }

        return residual_norm(grids_[0]);
    }

    void smooth(GridLevel& grid, int iterations) {
        const double hx2 = 1.0/(grid.hx*grid.hx);
        const double hy2 = 1.0/(grid.hy*grid.hy);
        const double factor = 0.5/(hx2 + hy2);

        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= grid.nx; ++i) {
            for (int j = 1; j <= grid.ny; ++j) {
                grid.values[idx(i,j,grid)] = factor * (
                    hx2*(grid.values[idx(i+1,j,grid)] + grid.values[idx(i-1,j,grid)]) +
                    hy2*(grid.values[idx(i,j+1,grid)] + grid.values[idx(i,j-1,grid)]) -
                    grid.rhs[idx(i,j,grid)]
                );
            }
        }
    }

    
    // Convertit les indices 2D (i,j) en index 1D pour les tableaux plats
    inline int idx(int i, int j, const GridLevel& grid) const {
        return i * (grid.ny + 2) + j; // +2 pour les ghost cells
    }

    // Calcule le résidu r = b - A*p
    std::vector<T> residual(const GridLevel& grid) const {
        std::vector<T> res(grid.values.size(), 0);
        const double hx2 = 1.0/(grid.hx * grid.hx);
        const double hy2 = 1.0/(grid.hy * grid.hy);
        
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= grid.nx; ++i) {
            for (int j = 1; j <= grid.ny; ++j) {
                res[idx(i,j,grid)] = grid.rhs[idx(i,j,grid)] - (
                    hx2*(grid.values[idx(i+1,j,grid)] - 2*grid.values[idx(i,j,grid)] + grid.values[idx(i-1,j,grid)]) +
                    hy2*(grid.values[idx(i,j+1,grid)] - 2*grid.values[idx(i,j,grid)] + grid.values[idx(i,j-1,grid)])
                );
            }
        }
        return res;
    }

    // Restriction full-weighting (fine -> grossier)
    void restrict(const std::vector<T>& fine, std::vector<T>& coarse) {
        const GridLevel& f_grid = grids_[0];
        GridLevel& c_grid = grids_[1];
        
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= c_grid.nx; ++i) {
            for (int j = 1; j <= c_grid.ny; ++j) {
                int fi = 2*i;
                int fj = 2*j;
                coarse[idx(i,j,c_grid)] = 0.25 * (
                    fine[idx(fi, fj, f_grid)] + 
                    fine[idx(fi+1, fj, f_grid)] + 
                    fine[idx(fi, fj+1, f_grid)] + 
                    fine[idx(fi+1, fj+1, f_grid)]
                );
            }
        }
    }

    // Prolongation linéaire (grossier -> fin)
    void prolongate(const std::vector<T>& coarse, std::vector<T>& fine) {
        const GridLevel& c_grid = grids_[1];
        GridLevel& f_grid = grids_[0];
        
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= f_grid.nx; ++i) {
            for (int j = 1; j <= f_grid.ny; ++j) {
                int ci = i/2;
                int cj = j/2;
                fine[idx(i,j,f_grid)] += coarse[idx(ci,cj,c_grid)];
            }
        }
    }

    // Solveur direct Gauss-Seidel pour le niveau grossier
    void direct_solve(GridLevel& grid) {
        const double hx2 = 1.0/(grid.hx * grid.hx);
        const double hy2 = 1.0/(grid.hy * grid.hy);
        const double tol = 1e-8;
        const int max_iter = 1000;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            double max_err = 0.0;
            for (int i = 1; i <= grid.nx; ++i) {
                for (int j = 1; j <= grid.ny; ++j) {
                    double new_val = (grid.rhs[idx(i,j,grid)] 
                        + hx2*(grid.values[idx(i+1,j,grid)] + grid.values[idx(i-1,j,grid)])
                        + hy2*(grid.values[idx(i,j+1,grid)] + grid.values[idx(i,j-1,grid)]))
                        / (2*(hx2 + hy2));
                    
                    max_err = std::max(max_err, std::abs(new_val - grid.values[idx(i,j,grid)]));
                    grid.values[idx(i,j,grid)] = new_val;
                }
            }
            if (max_err < tol) break;
        }
    }

    // Norme du résidu (max absolu)
    double residual_norm(const GridLevel& grid) const {
        double max_res = 0.0;
        #pragma omp parallel for reduction(max:max_res)
        for (int i = 1; i <= grid.nx; ++i) {
            for (int j = 1; j <= grid.ny; ++j) {
                max_res = std::max(max_res, std::abs(grid.rhs[idx(i,j,grid)]));
            }
        }
        return max_res;
    }
};



Field solve_pressure_poisson_amg(
    const Field& p_old, const Field& b,
    int Nx, int Ny, int N_ghost,
    double dx, double dy,
    bool full_solve
) {
    // Dimensions intérieures sans ghost cells
    const int interior_Nx = Nx;
    const int interior_Ny = Ny;
    
    // Initialisation du solveur (à faire une fois)
    static MultigridPoissonSolver<double> mg_solver(
        interior_Nx, interior_Ny, dx, dy
    );
    
    // Conversion Field -> vector 1D (sans ghost cells)
    std::vector<double> b_flat;
    b_flat.reserve(interior_Nx * interior_Ny);
    for(int i = N_ghost; i < Nx + N_ghost; ++i)
        for(int j = N_ghost; j < Ny + N_ghost; ++j)
            b_flat.push_back(b[i][j]);
    
    // Résolution
    std::vector<double> p_flat(interior_Nx * interior_Ny, 0.0);
    mg_solver.solve(p_flat, b_flat, full_solve ? 50 : 10, 1e-6);
    
    // Conversion vector 1D -> Field (avec ghost cells)
    Field p_new = p_old;  // Préserve les ghost cells existants
    int idx = 0;
    for(int i = N_ghost; i < Nx + N_ghost; ++i)
        for(int j = N_ghost; j < Ny + N_ghost; ++j)
            p_new[i][j] = p_flat[idx++];
    
    return p_new;
}


















} // namespace navier_stokes



















// #include <vector>
// #include <cmath>
// #include <omp.h>

// template <typename T>
// class MultigridPoissonSolver {
// public:
//     MultigridPoissonSolver(int nx, int ny, double dx, double dy) 
//         : nx_(nx), ny_(ny), dx_(dx), dy_(dy) {
//         init_grids();
//     }

//     void solve(std::vector<T>& p, const std::vector<T>& b, int max_cycles=50, double tol=1e-6) {
//         #pragma omp parallel
//         {
//             for (int cycle = 0; cycle < max_cycles; ++cycle) {
//                 double res_norm = v_cycle(p, b);
                
//                 #pragma omp single
//                 {
//                     if (res_norm < tol) break;
//                 }
//             }
//         }
//     }

// private:
//     struct GridLevel {
//         std::vector<T> values;
//         std::vector<T> rhs;
//         int nx, ny;
//         double hx, hy;
//     };

//     int nx_, ny_;
//     double dx_, dy_;
//     std::vector<GridLevel> grids_;

//     void init_grids() {
//         int levels = std::floor(std::log2(std::min(nx_, ny_))) - 1;
//         grids_.reserve(levels);

//         // Création hiérarchie multigrille
//         for (int l = 0; l < levels; ++l) {
//             int nx = nx_ >> l;
//             int ny = ny_ >> l;
//             grids_.push_back({
//                 std::vector<T>((nx+2)*(ny+2), 0),
//                 std::vector<T>((nx+2)*(ny+2), 0),
//                 nx, ny,
//                 dx_ * (1 << l), 
//                 dy_ * (1 << l)
//             });
//         }
//     }

//     double v_cycle(std::vector<T>& p, const std::vector<T>& b) {
//         // Niveau le plus fin
//         grids_[0].rhs = b;

//         // Descente
//         for (size_t l = 0; l < grids_.size()-1; ++l) {
//             smooth(grids_[l], 3);  // Pré-lissage
//             restrict(residual(grids_[l]), grids_[l+1].rhs);
//         }

//         // Solveur direct au niveau le plus grossier
//         direct_solve(grids_.back());

//         // Remontée
//         for (int l = grids_.size()-2; l >= 0; --l) {
//             prolongate(grids_[l+1].values, grids_[l].values);
//             smooth(grids_[l], 3);  // Post-lissage
//         }

//         return residual_norm(grids_[0]);
//     }

//     void smooth(GridLevel& grid, int iterations) {
//         const double hx2 = 1.0/(grid.hx*grid.hx);
//         const double hy2 = 1.0/(grid.hy*grid.hy);
//         const double factor = 0.5/(hx2 + hy2);

//         #pragma omp parallel for collapse(2)
//         for (int i = 1; i <= grid.nx; ++i) {
//             for (int j = 1; j <= grid.ny; ++j) {
//                 grid.values[idx(i,j,grid)] = factor * (
//                     hx2*(grid.values[idx(i+1,j,grid)] + grid.values[idx(i-1,j,grid)]) +
//                     hy2*(grid.values[idx(i,j+1,grid)] + grid.values[idx(i,j-1,grid)]) -
//                     grid.rhs[idx(i,j,grid)]
//                 );
//             }
//         }
//     }

//     // Autres méthodes: restrict, prolongate, residual, direct_solve...
// };



// find_package(OpenMP REQUIRED)

// add_executable(navier_stokes ${NS_SOURCES})

// target_link_libraries(navier_stokes 
//     PRIVATE 
//     SFML::Graphics SFML::Window SFML::System 
//     OpenMP::OpenMP_CXX
// )

// target_compile_options(navier_stokes 
//     PRIVATE 
//     ${OpenMP_CXX_FLAGS} 
//     -march=native -O3 -ffast-math
// )








// https://github.com/ddemidov/amgcl
// https://github.com/ddemidov/amgcl/blob/master/docs/examples.rst



// // Choix du backend double-precision
// using Backend    = amgcl::backend::builtin<double>;

// // 2) Composants AMG
// using Coarsening = amgcl::coarsening::smoothed_aggregation<Backend>;
// using Relaxation = amgcl::relaxation::spai0<Backend>;

// // 3) Preconditionneur AMG : l’ordre des paramètres est (Backend, Coarsening, Relaxation)
// using Precond = amgcl::amg<Backend, Coarsening, Relaxation>;

// // 4) Solveur iteratif + preconditionneur
// using Solver = amgcl::make_solver<
//     Precond,
//     amgcl::solver::bicgstab<Backend>
// >;  // <- N’OUBLIEZ PAS le point-virgule ici

// // 5) Cache global
// static std::map<
//     std::tuple<int,int,int,int>,
//     std::unique_ptr<Solver>
// > amg_cache;

// Field solve_pressure_poisson_amg(
//     const Field& p_old,
//     const Field& b,
//     int Nx,
//     int Ny,
//     int N_ghost,
//     double dx,
//     double dy,
//     bool full_solve
// ) {
//     const int N = Nx * Ny;

//     // Utilisation de ptrdiff_t pour les indices de colonne
//     std::vector<size_t> ptr(N + 1, 0);
//     std::vector<ptrdiff_t> col; // Change de int à ptrdiff_t
//     std::vector<double> val;

//     for (int j = 0, k = 0; j < Ny; ++j) {
//         for (int i = 0; i < Nx; ++i, ++k) {
//             const int ip = i + N_ghost;
//             const int jp = j + N_ghost;

//             const double diag = -2.0/(dx*dx) - 2.0/(dy*dy);
            
//             val.push_back(diag);
//             col.push_back(k);
            
//             // Voisins
//             auto add_neighbor = [&](double coeff, int i2, int j2) {
//                 val.push_back(coeff);
//                 col.push_back((i2 + Nx) % Nx + ((j2 + Ny) % Ny) * Nx);
//             };

//             add_neighbor(1.0/(dx*dx), i-1, j);
//             add_neighbor(1.0/(dx*dx), i+1, j);
//             add_neighbor(1.0/(dy*dy), i, j-1);
//             add_neighbor(1.0/(dy*dy), i, j+1);

//             ptr[k+1] = val.size();
//         }
//     }

//     // Vecteur RHS
//     std::vector<double> rhs(N);
//     for (int j = 0, k = 0; j < Ny; ++j) {
//         for (int i = 0; i < Nx; ++i, ++k) {
//             const int ip = i + N_ghost;
//             const int jp = j + N_ghost;
//             rhs[k] = b[ip][jp];
//         }
//     }

//     auto key = std::make_tuple(
//         Nx, Ny,
//         static_cast<int>(dx * 1e8),
//         static_cast<int>(dy * 1e8)
//     );

//     auto& solver_ptr = amg_cache[key];
//     if (!solver_ptr) {
//         typename Solver::params prm;
//         prm.precond.coarsening.aggr.eps_strong = 0.08;
//         prm.precond.relax.type = amgcl::relaxation::spai0; // Correction du type
        
//         // Adaptation avec les bons types
//         auto A = amgcl::adapter::zero_copy(
//             N, 
//             ptr.data(),
//             col.data(), 
//             val.data()
//         );

//         solver_ptr = std::make_unique<Solver>(A, prm);
//     }

//     std::vector<double> sol(N, 0.0);
//     std::tie(std::ignore, std::ignore) = (*solver_ptr)(rhs, sol);

//     double mean = std::accumulate(sol.begin(), sol.end(), 0.0) / N;
    
//     Field p_new = p_old;
//     for (int j = 0, k = 0; j < Ny; ++j) {
//         for (int i = 0; i < Nx; ++i, ++k) {
//             p_new[i + N_ghost][j + N_ghost] = sol[k] - mean;
//         }
//     }

//     return p_new;
// }

















