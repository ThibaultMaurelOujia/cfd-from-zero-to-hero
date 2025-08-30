cmake_minimum_required(VERSION 3.10)
project(Euler_2D_compressible VERSION 1.0 LANGUAGES CXX)

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
add_executable(euler
    ${NS_SOURCES}
)
set_target_properties(euler PROPERTIES
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
  target_include_directories(euler PUBLIC ${OMP_INC})

  # indique à clang d'utiliser OpenMP au pré-processing
  target_compile_options(euler PUBLIC
    "-Xpreprocessor" "-fopenmp"
  )

  # on linke la runtime OpenMP
  target_link_libraries(euler PUBLIC ${OMP_LIB})
endif()




# -- SFML --
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
target_link_libraries(euler PRIVATE
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
# install(TARGETS euler
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
