cmake_minimum_required(VERSION 3.23)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
project(
	rexlib-cuda
	VERSION ${PROJECT_VERSION}
	LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

include(CTest)
include(GNUInstallDirs)
include(fetch_boost)
include(fetch_spdlog)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

# Its own option rather than BUILD_TESTING: dependencies of dependencies
# write that one into the cache, and this project's tests should not be at
# their mercy.
option(REXLIB_CUDA_BUILD_TESTING "Build this project's tests" ${BUILD_TESTING})

option(REXLIB_CUDA_INSTALL "Generate install rules" ON)

set(REXLIB_CUDA_REXLIB_VERSION 0.1 CACHE STRING
	"Version of rexlib this plugin requires")

# The CUDA runtime is linked by SONAME, so it is found wherever the loader
# already looks. A wheel is the exception: it declares a dependency on
# NVIDIA's runtime wheel, which unpacks beside the package rather than onto
# any search path, and the directory it unpacks into differs per CUDA
# major. Whoever builds such a wheel knows that path; nothing here can.
set(REXLIB_CUDA_RUNTIME_RPATH "" CACHE STRING
	"RPATH entry reaching the CUDA runtime, for example \
$ORIGIN/../../../nvidia/cuda_runtime/lib")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# CUDA is enabled here rather than in project() because the architectures
# have to be settled first: left to itself nvcc picks a default by looking
# for a local GPU, which no build machine has. One fat binary per major
# instead. Spelling the list out would rot, as CUDA 12 dropped sm_35 and
# sm_37 and CUDA 13 dropped Maxwell, Pascal and Volta.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
enable_language(CUDA)

# Only a real toolkit can build this. NVIDIA's runtime wheels ship
# libcudart.so.<major> with no development symlink, so find_library never
# matches them, and their nvcc package carries a compiler driver only from
# CUDA 13 on. They are a way to ship the runtime, not to compile with.
find_package(CUDAToolkit 11 REQUIRED)

fetch_boost(VERSION 1.92.0)
fetch_spdlog(VERSION 1.17.0)

if(NOT TARGET rexlib::rexlib)
	find_package(rexlib ${REXLIB_CUDA_REXLIB_VERSION} REQUIRED)
endif()

# ---------------------------------------------------------------------------
# The plugin
# ---------------------------------------------------------------------------

add_subdirectory(src)

if(REXLIB_CUDA_BUILD_TESTING)
	add_subdirectory(tests)
endif()
