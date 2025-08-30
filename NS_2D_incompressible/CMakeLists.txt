cmake_minimum_required(VERSION 3.10)
project(NS_2D_incompressible VERSION 1.0 LANGUAGES CXX)

# -- C++ standard --
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Debug or Release)" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling AddressSanitizer & UBSan in Debug")
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()



# -- Répertoires include --
# Ajout du dossier include/ au chemin de recherche des headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# -- Rassembler tous les .cpp de src/ --
# Recherche récursive de tous les fichiers .cpp sous src/
file(GLOB_RECURSE NS_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# -- Exécutable principal --
# Déclaration de l’exécutable principal à partir des sources collectées
add_executable(navier_stokes
    ${NS_SOURCES}
)
set_target_properties(navier_stokes PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)







if(APPLE)
  # détecte automatiquement le préfix Homebrew pour libomp
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # dossiers d'include et de bibliothèque pour libomp
  set(OMP_INC "${LIBOMP_PREFIX}/include")
  set(OMP_LIB "${LIBOMP_PREFIX}/lib/libomp.dylib")

  # on ajoute omp.h
  target_include_directories(navier_stokes PUBLIC ${OMP_INC})

  # indique à clang d'utiliser OpenMP au pré-processing
  target_compile_options(navier_stokes PUBLIC
    "-Xpreprocessor" "-fopenmp"
  )

  # on linke la runtime OpenMP
  target_link_libraries(navier_stokes PUBLIC ${OMP_LIB})
endif()





# -- AMGCL --
#add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/amgcl)
#target_link_libraries(navier_stokes PRIVATE amgcl)
#target_include_directories(navier_stokes PRIVATE ${PROJECT_SOURCE_DIR}/third_party/amgcl/amgcl)

#set(AMGCL_ENABLE_OPENMP ON)                  # active le support OpenMP
#add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/amgcl)      # pointe vers le dossier cloné
#target_link_libraries(navier_stokes PRIVATE amgcl)




# -- GINKGO --
set(GINKGO_ROOT "${CMAKE_SOURCE_DIR}/third_party/extern/ginkgo-install")
find_package(Ginkgo CONFIG REQUIRED
             PATHS "${GINKGO_ROOT}/lib/cmake/Ginkgo")
target_link_libraries(navier_stokes PRIVATE Ginkgo::ginkgo)




# -- EIGEN --
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(navier_stokes PRIVATE Eigen3::Eigen)


# -- SFML --
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
target_link_libraries(navier_stokes PRIVATE
  SFML::Graphics
  SFML::Window
  SFML::System
)





# -- Cible rebuild -- 
add_custom_target(rebuild
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMENT "Rebuild entire project"
)

# -------------------------------------------------------------------
# TODO: activer plus tard l’installation de l’exécutable
# install(TARGETS navier_stokes
#     RUNTIME DESTINATION bin
# )
# -------------------------------------------------------------------

# -- Pour afficher la liste des cibles personnalisées --
set_property(GLOBAL PROPERTY RULE_LAUNCH_HELP TRUE)



# make all
# make clean
# make rebuild
# make help
# make install
