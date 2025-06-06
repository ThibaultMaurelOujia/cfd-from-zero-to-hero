
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



Field solve_pressure_poisson(Field& p_old, const Field& b,
    int Nx, int Ny, int N_ghost,
    double dx, double dy,
    const std::string& method,
    int max_iters){

    // return p_old; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (method == "direct") {
        return solve_pressure_poisson_direct(p_old, b,
                                             Nx, Ny, N_ghost,
                                             dx, dy);
    }
    // else if (method == "cholesky") {
    //     return solve_pressure_poisson_cholesky(b,
    //                                            Nx, Ny, N_ghost,
    //                                            dx, dy);
    // }
    else if (method == "ginkgo") {
        return solve_pressure_poisson_ginkgo(p_old, b, Nx, Ny, N_ghost, dx, dy, max_iters);
    }
    // else if (method == "hypre") {
    //     return solve_pressure_poisson_hypre(b, Nx, Ny, dx, dy);
    // }
    // else if (method == "amg") {
    //     return solve_pressure_poisson_amg(b,
    //                                     Nx, Ny, N_ghost,
    //                                     dx, dy);
    // }
    // else if (method == "fft") {
    //     return solve_pressure_poisson_fft(b,
    //                                       Nx, Ny, N_ghost,
    //                                       dx, dy);
    // }
    else {
        throw std::invalid_argument("solve_pressure_poisson : methode inconnue " + method);
    }
}









Field solve_pressure_poisson_direct(
    Field& p_old,  const Field& b, 
    int Nx, int Ny, int N_ghost,
    double dx, double dy) {

    // // debug : combien de threads OpenMP et Eigen ?
    // int omp_thr = omp_get_max_threads();
    // int eigen_thr = Eigen::nbThreads();
    // std::cout << "[solve_pressure_poisson_direct] OpenMP threads = "
    //         << omp_thr
    //         << "   Eigen nbThreads = "
    //         << eigen_thr
    //         << std::endl;
    
    
    // 1) Construire le RHS 
    const int N = Nx*Ny;
    Eigen::VectorXd rhs(N);
    for(int i = 0, k = 0; i < Nx; ++i)
        for(int j = 0; j < Ny; ++j, ++k)
            rhs(k) = b[i+N_ghost][j+N_ghost];
    double mean_rhs = rhs.mean();
    rhs.array() -= mean_rhs;
    
    // 2) Cle de cache (Nx, Ny, dx*1e8, dy*1e8)
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

        // 3b) Identites et 2D laplacien
        SpMat Ix(Nx,Nx); Ix.setIdentity();
        SpMat Iy(Ny,Ny); Iy.setIdentity();
        SpMat A = Eigen::kroneckerProduct(Tx, Iy)/(dx*dx)
                + Eigen::kroneckerProduct(Ix, Ty)/(dy*dy);
        A.makeCompressed();

        
        // 3c) Construire et stime_tocker la factorisation
        auto solver = std::make_unique<LU>();
        solver->compute(A);  // fait analyse+factorise et initialise tout
        if (solver->info() != Eigen::Success) {
            std::cerr << "Erreur de factorisation LU\n";
            std::exit(EXIT_FAILURE);
        }

        it = LU_cache.emplace(key, std::move(solver)).first;
    }

    // 4) Resoudre
    Eigen::VectorXd sol = it->second->solve(rhs);
    if (it->second->info() != Eigen::Success) {
        std::cerr << "Erreur lors de la resolution\n";
        std::exit(EXIT_FAILURE);
    }

    // 5) Transferer dans p_new (et enlever la moyenne)
    // Field p_new(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    Field& p_new = p_old;        // !!!!!!!!!! on a pas besoin de p_old 
    // Field p_new(Nx + 2*N_ghost, std::vector<double>(Ny + 2*N_ghost, 0.0));
    double mean = sol.mean();
    for(int i=0,k=0;i<Nx;++i)
        for(int j=0;j<Ny;++j,++k)
            p_new[i+N_ghost][j+N_ghost] = sol(k) - mean;

    return p_new;
}











Field solve_pressure_poisson_ginkgo(Field& p_old, const Field& b,
                                    int Nx, int Ny, int N_ghost,
                                    double dx, double dy,
                                    int max_iters) {
    // const int N = Nx * Ny;                       // nombre d'inconnues
    int N = Nx * Ny; // !!!!!!!!!!!!!!!!

    using Csr  = gko::matrix::Csr<double, int>;  // <val, index>
    using Vec  = gko::matrix::Dense<double>;
    using cg   = gko::solver::Cg<>;
    using jac  = gko::preconditioner::Jacobi<>;
    using ic   = gko::preconditioner::Ic<>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<>;
    const double tol = 1e-8;
    // using amg = gko::preconditioner::Amg<>;

    //--------------------------------------------------------------------------
    // 1) executeurs : 'app' = reference, 'exec' = OpenMP (multithread)
    //--------------------------------------------------------------------------
    auto app_exec  = gko::ReferenceExecutor::create();
    auto exec      = gko::OmpExecutor::create();
    
    //--------------------------------------------------------------------------
    // 2) assemblage Laplacien 5-points periodique sur app_exec
    //--------------------------------------------------------------------------
    // time_tic();
    std::vector<int>    row_ptr(N + 1);
    std::vector<int>    col_idx;
    std::vector<double> vals;
    col_idx.reserve(5 * N);
    vals.reserve(5 * N);

    for (int idx = 0; idx < N; ++idx) {
        int i = idx / Ny, j = idx % Ny;
        row_ptr[idx] = static_cast<int>(col_idx.size());

        double c0 = - (2.0/(dx*dx) + 2.0/(dy*dy));
        double cx =   1.0/(dx*dx);
        double cy =   1.0/(dy*dy);

        // centre
        col_idx.push_back(idx);                 vals.push_back(c0);
        // +x, -x
        col_idx.push_back(((i + 1) % Nx) * Ny + j);      vals.push_back(cx);
        col_idx.push_back(((i - 1 + Nx) % Nx) * Ny + j); vals.push_back(cx);
        // +y, -y
        col_idx.push_back(i * Ny + ((j + 1) % Ny));      vals.push_back(cy);
        col_idx.push_back(i * Ny + ((j - 1 + Ny) % Ny)); vals.push_back(cy);
    }
    row_ptr[N] = static_cast<int>(col_idx.size());

    auto A_host = Csr::create(app_exec,
                              gko::dim<2>(N, N),
                              gko::array<double>::view(app_exec,
                                                       vals.size(),
                                                       vals.data()),
                              gko::array<int>::view(app_exec,
                                                    col_idx.size(),
                                                    col_idx.data()),
                              gko::array<int>::view(app_exec,
                                                    row_ptr.size(),
                                                    row_ptr.data()));
    // time_toc("A "); 

    // // --- Debug : affichage dense de A_host (attention, N×N entries !) ---
    // {
    //     std::cout << "Nx " << Nx << "Ny " << Ny << "dx " << dx << "dy " << dy << '\n';
    //     std::cout << "Affichage dense de A (N = " << N << "):\n";
    //     // 1) Construire un tableau dense initialisé à zéro
    //     std::vector<double> A_dense(N * N, 0.0);
    //     // 2) Remplir à partir du CSR
    //     for (int i = 0; i < N; ++i) {
    //         for (int idx = row_ptr[i]; idx < row_ptr[i+1]; ++idx) {
    //             int j = col_idx[idx];
    //             A_dense[i * N + j] = vals[idx];
    //         }
    //     }
    //     // 3) Afficher ligne par ligne
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             std::cout << A_dense[i * N + j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << std::endl;
    // }


    //--------------------------------------------------------------------------
    // 3) second membre aplati (host)
    //--------------------------------------------------------------------------
    // time_tic();
    std::vector<double> rhs_flat(N);
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            rhs_flat[i * Ny + j] = b[i + N_ghost][j + N_ghost];

    double mean_b = std::accumulate(rhs_flat.begin(), rhs_flat.end(), 0.0)/N;
    for(auto& v: rhs_flat) v -= mean_b;

    auto rhs_host = Vec::create(app_exec,
                                gko::dim<2>(N, 1),
                                gko::array<double>::view(app_exec,
                                                         rhs_flat.size(),
                                                         rhs_flat.data()),
                                1);
    // time_toc("B "); 

    //--------------------------------------------------------------------------
    // 4) clonage vers exec
    //--------------------------------------------------------------------------
    // time_tic(); 
    auto A   = gko::clone(exec, A_host);   // unique_ptr<Csr>
    auto rhs = gko::clone(exec, rhs_host); // unique_ptr<Vec>
    auto x   = Vec::create(exec, gko::dim<2>(N, 1));     // initialise à 0
    // time_toc("clone "); 

    //--------------------------------------------------------------------------
    // 5) generation solveur CG + Jacobi, puis solve
    //--------------------------------------------------------------------------
    // time_tic(); 
    auto logger = gko::share(gko::log::Convergence<double>::create());
    // using amg = gko::preconditioner::SmoothedAggregation<>;

    auto solver = cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(max_iters).on(exec),
            gko::stop::ResidualNorm<>::build().with_reduction_factor(tol).on(exec)
        )
        .with_preconditioner(jac::build().on(exec))
        // .with_preconditioner(schwarz::build().on(exec))
        // .with_preconditioner(ic::build().on(exec)) // nul
        .on(exec)
        ->generate(std::move(A));
    // time_toc("Solve build "); 

    // time_tic(); 
    solver->add_logger(logger);
    solver->apply(rhs, x);
    // time_toc("Solve apply "); 




    // #include <ginkgo/experimental/multigrid/pgm.hpp>
    // using Pgm = gko::experimental::multigrid::Pgm<>;
    
    // auto solver = cg::build()
    //     .with_criteria(/*…*/)
    //     .with_preconditioner(
    //         Pgm::build()
    //            .with_max_levels(3)
    //            .with_cycle_type(gko::experimental::multigrid::Cycle::V)
    //            .on(exec)
    //     )
    //     .on(exec)
    //     ->generate(A);
    






    // Recuperation correcte du residu final
    // Recuperer le LinOp contenant le residu
    auto linop_res = logger->get_residual_norm();
    // Caster dynamiquement en Dense<double>
    auto dense_res = gko::as<gko::matrix::Dense<double>>(linop_res);
    // Cloner vers l'executeur maitre pour acceder aux donnees sur CPU
    auto dense_res_host = gko::clone(app_exec, dense_res);
    // Lire la seule valeur [0,0]
    double residual_norm = dense_res_host->at(0, 0);
    // Nombre d'iterations (retourne directement comme entier)
    auto num_iters = logger->get_num_iterations();
    // std::cout << "Iterations : " << num_iters
    //         << ", Residu final : " << residual_norm << "\n";

    //--------------------------------------------------------------------------
    // 6) transfert resultat → p_old (retrait moyenne pour periodicite)
    //--------------------------------------------------------------------------
    // time_tic(); 
    auto x_host = gko::clone(app_exec, x);               // back to CPU
    const double* xv = x_host->get_const_values();
    double mean = std::accumulate(xv, xv + N, 0.0) / N;

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            p_old[i + N_ghost][j + N_ghost] = xv[i * Ny + j] - mean;
    // time_toc("p_old "); 



    // auto Ax_host = Vec::create(app_exec, gko::dim<2>(N,1));
    // A_host->apply(x_host, Ax_host);

    // // Validation : résidu infini
    // const double* bvals  = rhs_host->get_const_values();
    // const double* Axvals = Ax_host->get_const_values();
    // double max_res = 0.0;
    // for (int k = 0; k < N; ++k) {
    //     max_res = std::max(max_res, std::abs(bvals[k] - Axvals[k]));
    // }
    // std::cout << "______________________________________________________" << std::endl;
    // std::cout << "[Validation] ‖b - A x‖∞ = " << max_res << std::endl;
    // std::cout << "______________________________________________________" << std::endl;


    return p_old;
}



























// // alias pour ton Field
// using Field = std::vector<std::vector<double>>;

// // Solveur de Poisson par Jacobi
// Field solve_poisson_jacobi(
//     const Field& b,
//     int           Nx,
//     int           Ny,
//     int           N_ghost,
//     double        dx,
//     double        dy,
//     int           max_iter = 10000,
//     double        tol      = 1e-6
// ) {
//     // Cree p et p_new (tout à zero, ghost inclus)
//     Field p    (Nx+2*N_ghost, std::vector<double>(Ny+2*N_ghost, 0.0));
//     Field p_new = p;

//     const double idx2 = 1.0/(dx*dx);
//     const double idy2 = 1.0/(dy*dy);
//     const double denom = 2*(idx2 + idy2);

//     for(int iter = 0; iter < max_iter; ++iter) {
//         // 1) appliquer les conditions periodiques sur p
//         for(int j = N_ghost; j < Ny+N_ghost; ++j) {
//             p[0         + j][j] = p[Nx +0 + j][j]; // gauche ← droite interieure
//             p[Nx+N_ghost+ j][j] = p[N_ghost + j][j]; // droite ← gauche interieure
//         }
//         for(int i = N_ghost; i < Nx+N_ghost; ++i) {
//             p[i][0          ] = p[i][Ny + N_ghost]; // bas ← haut interieure
//             p[i][Ny+N_ghost ] = p[i][N_ghost];      // haut ← bas interieure
//         }

//         double max_diff = 0.0;

//         // 2) Jacobi update (interieur uniquement)
//         for(int i = N_ghost; i < Nx+N_ghost; ++i) {
//             for(int j = N_ghost; j < Ny+N_ghost; ++j) {
//                 double lap  = ( p[i+1][j] + p[i-1][j] ) * idx2
//                             + ( p[i][j+1] + p[i][j-1] ) * idy2;
//                 double rhs  = b[i][j];
//                 double pnew = (lap - rhs) / denom;
//                 max_diff = std::max(max_diff, std::abs(pnew - p[i][j]));
//                 p_new[i][j] = pnew;
//             }
//         }

//         // 3) swap et test de convergence
//         p.swap(p_new);
//         if (max_diff < tol) {
//             std::cout << "[Jacobi] converged in " << iter << " iters, diff=" << max_diff << "\n";
//             break;
//         }
//         if (iter == max_iter-1) {
//             std::cout << "[Jacobi] max_iter reached, diff=" << max_diff << "\n";
//         }
//     }

//     return p;
// }









// Field solve_pressure_poisson_amg(
//     const Field&  b,
//     int           Nx,
//     int           Ny,
//     int           N_ghost,
//     double        dx,
//     double        dy
//     ) {
//     using Backend    = amgcl::backend::builtin<double>;               // OpenMP inside
//     using Coarsening = amgcl::coarsening::smoothed_aggregation<Backend>;
//     using Relaxation = amgcl::relaxation::spai0<Backend>;
//     // using Precond    = amgcl::amg<Coarsening, Relaxation, Backend>;
//     using Precond    = amgcl::amg<Backend, Coarsening, Relaxation>;
//     using Solver     = amgcl::make_solver<
//                            Precond,
//                            amgcl::solver::bicgstab<Backend>
//                        >;

//     const ptrdiff_t N = statime_tic_cast<ptrdiff_t>(Nx) * Ny;

//     // ---------------------------------------------------------------------
//     // 1) Construction du CRS (periodic 5-points)
//     // ---------------------------------------------------------------------
//     std::vector<size_t>   ptr(N + 1, 0);
//     std::vector<ptrdiff_t> col;
//     std::vector<double>   val;
//     col.reserve(5 * N);
//     val.reserve(5 * N);

//     auto idx = [Nx](int i, int j){ return j * Nx + i; };

//     double dx2 = 1.0 / (dx * dx);
//     double dy2 = 1.0 / (dy * dy);

//     for (int j = 0, k = 0; j < Ny; ++j) {
//         for (int i = 0; i < Nx; ++i, ++k) {

//             // diag
//             col.push_back(k);
//             val.push_back(-2.0 * (dx2 + dy2));

//             // voisins periodiques
//             col.push_back(idx((i+1)%Nx, j  )); val.push_back(dx2);
//             col.push_back(idx((i-1+Nx)%Nx,j)); val.push_back(dx2);
//             col.push_back(idx( i, (j+1)%Ny)); val.push_back(dy2);
//             col.push_back(idx( i, (j-1+Ny)%Ny)); val.push_back(dy2);

//             ptr[k+1] = val.size();
//         }
//     }

//     // ---------------------------------------------------------------------
//     // 2) RHS
//     // ---------------------------------------------------------------------
//     std::vector<double> rhs(N);
//     for (int j = 0, k = 0; j < Ny; ++j)
//         for (int i = 0; i < Nx; ++i, ++k)
//             rhs[k] = b[i + N_ghost][j + N_ghost];

//     // ---------------------------------------------------------------------
//     // 3) Pre-conditionneur + solveur
//     // ---------------------------------------------------------------------
//     Solver::params prm;
//     prm.precond.coarsening.aggr.eps_strong = 0.08;
//     prm.solver.maxiter = 200;
//     prm.solver.tol     = 1e-8;

//     Solver solve(
//         amgcl::adapter::zero_copy(N, ptr.data(), col.data(), val.data()),
//         prm
//     );

//     std::vector<double> sol(N, 0.0);
//     auto [iters, err] = solve(rhs, sol);

//     // (facultatif) affichage info solveur
//     std::cout << "[AMG] iters=" << iters << "  resid=" << err << '\n';

//     // ---------------------------------------------------------------------
//     // 4) Copie dans Field + retrait de la moyenne
//     // ---------------------------------------------------------------------
//     Field p_new(Nx + 2*N_ghost,
//                 std::vector<double>(Ny + 2*N_ghost, 0.0));

//     double mean = std::accumulate(sol.begin(), sol.end(), 0.0) / N;

//     for (int j = 0, k = 0; j < Ny; ++j)
//         for (int i = 0; i < Nx; ++i, ++k)
//             p_new[i + N_ghost][j + N_ghost] = sol[k] - mean;

//     return p_new;
// }















// HYPRE_StructPFMGSetRelaxType(solver, 1); // 0=Jacobi, 1=Gauss-Seidel
// HYPRE_StructPFMGSetNumPreRelax(solver, 2);
// HYPRE_StructPFMGSetNumPostRelax(solver, 2);


// Field solve_pressure_poisson_hypre(const Field& b, int Nx, int Ny, double dx, double dy) {
//     // 1. Initialisation HYPRE
//     HYPRE_Int ilower[2] = {0, 0};
//     HYPRE_Int iupper[2] = {Nx-1, Ny-1};
//     HYPRE_Int periodic[2] = {1, 1};
//     HYPRE_Int entries[5] = {0, 1, 2, 3, 4};
    
//     // 2. Creation du grid
//     HYPRE_StructGrid grid;
//     HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);
//     HYPRE_StructGridSetExtents(grid, ilower, iupper);
//     HYPRE_StructGridSetPeriodic(grid, periodic);
//     HYPRE_StructGridAssemble(grid);

//     // 3. Definition du stencil
//     HYPRE_StructStencil stencil;
//     HYPRE_Int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
//     HYPRE_StructStencilCreate(2, 5, &stencil);
//     for (HYPRE_Int i=0; i<5; ++i)
//         HYPRE_StructStencilSetElement(stencil, i, offsets[i]);

//     // 4. Initialisation matrice
//     HYPRE_StructMatrix A;
//     HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
//     HYPRE_StructMatrixInitialize(A);

//     // 5. Remplissage matrice
//     const HYPRE_Int nvalues = Nx*Ny;
//     std::vector<HYPRE_Complex> values(nvalues*5);
//     double hx2 = 1.0/(dx*dx), hy2 = 1.0/(dy*dy);
    
//     for (HYPRE_Int i=0; i<Nx; ++i) {
//         for (HYPRE_Int j=0; j<Ny; ++j) {
//             const HYPRE_Int idx = i*Ny + j;
//             values[idx*5 + 0] = -2.0*(hx2 + hy2); // Diag
//             values[idx*5 + 1] = hx2; // Ouest
//             values[idx*5 + 2] = hx2; // Est
//             values[idx*5 + 3] = hy2; // Sud
//             values[idx*5 + 4] = hy2; // Nord
//         }
//     }
//     HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 5, entries, values.data());
//     HYPRE_StructMatrixAssemble(A);

//     // 6. Vecteur b
//     HYPRE_StructVector b_hypre;
//     HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b_hypre);
//     HYPRE_StructVectorInitialize(b_hypre);
    
//     std::vector<HYPRE_Complex> b_values(nvalues);
//     for (HYPRE_Int i=0; i<Nx; ++i)
//         for (HYPRE_Int j=0; j<Ny; ++j)
//             b_values[i*Ny + j] = b[i+1][j+1]; // Ghost cells à [0] et [N+1]
    
//     HYPRE_StructVectorSetBoxValues(b_hypre, ilower, iupper, b_values.data());
//     HYPRE_StructVectorAssemble(b_hypre);

//     // 7. Solveur PFMG
//     HYPRE_StructSolver solver;
//     HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
//     HYPRE_StructPFMGSetMaxIter(solver, 50);
//     HYPRE_StructPFMGSetTol(solver, 1e-6);
//     HYPRE_StructPFMGSetRelaxType(solver, 1); // Gauss-Seidel
//     HYPRE_StructPFMGSetLogging(solver, 1);

//     // 8. Resolution
//     HYPRE_StructVector x_hypre;
//     HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x_hypre);
//     HYPRE_StructVectorInitialize(x_hypre);
    
//     HYPRE_StructPFMGSetup(solver, A, b_hypre, x_hypre);
//     HYPRE_StructPFMGSolve(solver, A, b_hypre, x_hypre);

//     // 9. Recuperation solution
//     Field p(Nx + 2, std::vector<double>(Ny + 2, 0.0));
//     std::vector<HYPRE_Complex> sol(nvalues);
//     HYPRE_StructVectorGetBoxValues(x_hypre, ilower, iupper, sol.data());
    
//     for (HYPRE_Int i=0; i<Nx; ++i)
//         for (HYPRE_Int j=0; j<Ny; ++j)
//             p[i+1][j+1] = sol[i*Ny + j];

//     // 10. Nettoyage
//     HYPRE_StructGridDestroy(grid);
//     HYPRE_StructStencilDestroy(stencil);
//     HYPRE_StructMatrixDestroy(A);
//     HYPRE_StructVectorDestroy(b_hypre);
//     HYPRE_StructVectorDestroy(x_hypre);
//     HYPRE_StructPFMGDestroy(solver);

//     return p;
// }




























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

//         // Creation hierarchie multigrille
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
//             smooth(grids_[l], 3);  // Pre-lissage
//             restrict(residual(grids_[l]), grids_[l+1].rhs);
//         }

//         // Solveur direct au niveau le plus grossier
//         direct_solve(grids_.back());

//         // Remontee
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

//     // Autres methodes: restrict, prolongate, residual, direct_solve...
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

// // 3) Preconditionneur AMG : l'ordre des paramètres est (Backend, Coarsening, Relaxation)
// using Precond = amgcl::amg<Backend, Coarsening, Relaxation>;

// // 4) Solveur iteratif + preconditionneur
// using Solver = amgcl::make_solver<
//     Precond,
//     amgcl::solver::bicgstab<Backend>
// >;  // <- N'OUBLIEZ PAS le point-virgule ici

// // 5) Cache global
// statime_tic std::map<
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
//         statime_tic_cast<int>(dx * 1e8),
//         statime_tic_cast<int>(dy * 1e8)
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

















