#ifndef NAVIER_STOKES_CPP_POISSON_HPP
#define NAVIER_STOKES_CPP_POISSON_HPP

#include "utils.hpp"

#include <numeric>

// #include <amgcl/amg.hpp>
// #include <amgcl/relaxation/spai0.hpp>
// #include <amgcl/backend/builtin.hpp>
// #include <amgcl/adapter/zero_copy.hpp>
// #include <amgcl/adapter/crs_tuple.hpp>
// #include <amgcl/make_solver.hpp>
// #include <amgcl/coarsening/smoothed_aggregation.hpp>
// #include <amgcl/solver/bicgstab.hpp>
// brew install llvm    # installe clang, clang++, libomp dans /opt/homebrew/opt/llvm
// export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
// export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
// export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
// which clang        # doit renvoyer /opt/homebrew/opt/llvm/bin/clang
// clang --version    # version Homebrew LLVM
// cd third_party/amgcl
// rm -rf build-openmp && mkdir build-openmp
// cd build-openmp
// cmake .. \
//   -DCMAKE_BUILD_TYPE=Release \
//   -DAMGCL_WITH_OPENMP=ON \
//   -DCMAKE_C_COMPILER=clang \
//   -DCMAKE_CXX_COMPILER=clang++ \
//   -DCMAKE_C_FLAGS="-fopenmp" \
//   -DCMAKE_CXX_FLAGS="-fopenmp" \
//   -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
//   -DCMAKE_POLICY_VERSION=3.10
// make -j$(sysctl -n hw.ncpu)

#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>
#include <unordered_map>


#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>
// #include <ginkgo/experimental/preconditioner/schwarz.hpp>




// git clone https://github.com/ginkgo-project/ginkgo.git
// cd ginkgo
// mkdir build && cd build

// brew install gflags
// cmake .. \
//   -DGINKGO_BUILD_OMP=ON \
//   -DGINKGO_BUILD_CUDA=OFF \
//   -DGINKGO_BUILD_HIP=OFF \
//   -DCMAKE_INSTALL_PREFIX=../../extern/ginkgo-install \
//   -Dgflags_DIR="$(brew --prefix gflags)/lib/cmake/gflags" \
//   \
//   -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
//   -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
//   -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"

// export CC=/opt/homebrew/opt/llvm/bin/clang
// export CXX=/opt/homebrew/opt/llvm/bin/clang++

// cmake .. \
//   -DGINKGO_BUILD_OMP=ON \
//   -DGINKGO_BUILD_CUDA=OFF \
//   -DGINKGO_BUILD_HIP=OFF \
//   -DGINKGO_BUILD_REFERENCE=ON \
//   -DGINKGO_BUILD_SYCL=OFF \
//   -DGINKGO_BUILD_MULTIGRID=ON \
//   -DCMAKE_INSTALL_PREFIX=../../extern/ginkgo-install \
//   -Dgflags_DIR="$(brew --prefix gflags)/lib/cmake/gflags" \
//   -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
//   -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
//   -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
// cmake --build . --target install





// brew install hypre --build-from-source --without-mpi
// #include "HYPRE_struct_mv.h"
// #include "HYPRE_struct_ls.h"
// # -- HYPRE --
// set(HYPRE_ROOT_DIR "/opt/homebrew/opt/hypre")
// include_directories(${HYPRE_ROOT_DIR}/include)
// link_directories(${HYPRE_ROOT_DIR}/lib)
// find_library(HYPRE_LIB NAMES HYPRE PATHS ${HYPRE_ROOT_DIR}/lib REQUIRED)
// target_link_libraries(navier_stokes PRIVATE ${HYPRE_LIB})
// #find_package(HYPRE REQUIRED)
// #target_link_libraries(navier_stokes PRIVATE HYPRE::HYPRE)











using DenseMat = Eigen::MatrixXd;
using SpMat    = Eigen::SparseMatrix<double>;



namespace navier_stokes {


void project(Field& u, Field& v, const Field& p, 
             int Nx, int Ny, int N_ghost, 
             double dx, double dy, double dt);
                    


Field compute_divergence(const Field& u, const Field& v,
                         int Nx, int Ny, int N_ghost,
                         double dx, double dy, double dt); 



/// Alias pour la matrice creuse et sa factorisation LU
using LU    = Eigen::SparseLU<SpMat>;

/// Clé de cache : (Nx,Ny,dx*1e8,dy*1e8)
using Key = std::tuple<int,int,int,int>;
struct KeyHash {
    std::size_t operator()(Key const& k) const noexcept {
        auto [Nx,Ny,dx8,dy8] = k;
        return (Nx * 73856093u)
             ^ (Ny * 19349663u)
             ^ (static_cast<unsigned>(dx8) * 83492791u)
             ^ static_cast<unsigned>(dy8);
    }
};

// Cache global des factorisations
static std::unordered_map<Key, std::unique_ptr<LU>, KeyHash> LU_cache;

// Solveur direct LU pour Lap p = b
Field solve_pressure_poisson_direct(Field& p_old, const Field& b,
                                    int Nx, int Ny, int N_ghost,
                                    double dx, double dy);

Field solve_pressure_poisson_ginkgo(Field& p_old, const Field& b,
                                    int Nx, int Ny, int N_ghost,
                                    double dx, double dy, 
                                    int max_iters);

// // Solveur par factorisation de Cholesky (ou LU creuse)
// Field solve_pressure_poisson_cholesky(const Field& b,
//                                 int          Nx,
//                                 int          Ny,
//                                 int          N_ghost,
//                                 double       dx,
//                                 double       dy);

// // Solveur AMG (V-cycle ou solve complet selon full_solve)
// Field solve_pressure_poisson_amg(const Field&  b,
//                                 int           Nx,
//                                 int           Ny,
//                                 int           N_ghost,
//                                 double        dx,
//                                 double        dy);

// // Solveur par FFT (domaines périodiques)
// Field solve_pressure_poisson_fft(const Field& b,
//                                 int          Nx,
//                                 int          Ny,
//                                 int          N_ghost,
//                                 double       dx,
//                                 double       dy);


// Field solve_pressure_poisson_hypre(const Field& b, 
//                                 int Nx, int Ny, 
//                                 double dx, double dy);




Field solve_pressure_poisson(Field& p_old, const Field& b,
                             int Nx, int Ny, int N_ghost,
                             double dx, double dy,
                             const std::string& method,
                             int max_iters=10000);


} // namespace navier_stokes

#endif // NAVIER_STOKES_CPP_POISSON_HPP













// PETSc : https://petsc.org

// Trilinos : https://trilinos.github.io

// Hypre : https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods

// amgcl  




