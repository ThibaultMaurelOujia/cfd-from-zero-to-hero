
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
        //   première fois : on monte A et on factorise  

        std::cout << "[solve_pressure_poisson_direct] Cache miss, LU construction for (Nx,Ny,dx,dy)=("
            << Nx << "," << Ny << ","
            << dx << "," << dy << ")\n";

        //   Tx  
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
                    T.emplace_back(i, next, 1.0); // Cas ou Nx est pair
                }
            }
            Tx.setFromTriplets(T.begin(), T.end());
        }

        //   Ty  
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
                    T.emplace_back(j, next, 1.0); // Cas ou Ny est pair
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
    int N = Nx * Ny; // !!!!!!!!!!!!!!!!

    using Csr  = gko::matrix::Csr<double, int>;  // <val, index>
    using Vec  = gko::matrix::Dense<double>;
    using cg   = gko::solver::Cg<>;
    using jac  = gko::preconditioner::Jacobi<>;
    using ic   = gko::preconditioner::Ic<>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<>;
    const double tol = 1e-8;
    // using amg = gko::preconditioner::Amg<>;

    // 
    //  executeurs : 'app' = reference, 'exec' = OpenMP (multithread)
    // 
    auto app_exec  = gko::ReferenceExecutor::create();
    auto exec      = gko::OmpExecutor::create();
     
    // assemblage Laplacien 5-points periodique sur app_exec 
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
 


    // 
    // second membre aplati  
    // 
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
 
    // clonage vers exec 
    // time_tic(); 
    auto A   = gko::clone(exec, A_host);   // unique_ptr<Csr>
    auto rhs = gko::clone(exec, rhs_host); // unique_ptr<Vec>
    auto x   = Vec::create(exec, gko::dim<2>(N, 1));     // initialise à 0
    // time_toc("clone "); 
 
    // generation solveur CG + Jacobi et puis solve 
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

 


    // Recuperation correcte du residu final 
    auto linop_res = logger->get_residual_norm(); 
    auto dense_res = gko::as<gko::matrix::Dense<double>>(linop_res); 
    auto dense_res_host = gko::clone(app_exec, dense_res); 
    double residual_norm = dense_res_host->at(0, 0); 
    auto num_iters = logger->get_num_iterations();
    // std::cout << "Iterations : " << num_iters << ", Residu final : " << residual_norm << "\n";
 
    // transfert resultat -> p_old (retrait moyenne pour periodicite) 
    // time_tic(); 
    auto x_host = gko::clone(app_exec, x);               // back to CPU
    const double* xv = x_host->get_const_values();
    double mean = std::accumulate(xv, xv + N, 0.0) / N;

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            p_old[i + N_ghost][j + N_ghost] = xv[i * Ny + j] - mean;
    // time_toc("p_old "); 


    return p_old;
}





























} // namespace navier_stokes




















