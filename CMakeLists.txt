cmake_minimum_required(VERSION 3.23)

# Define the project. CUDA is enabled further down, once the toolchain has
# been resolved: CMAKE_CUDA_COMPILER cannot be set after project() has
# probed for one, and the compiler may come from a pip wheel rather than
# from a system installation.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
project(
	rexlib-cuda
	VERSION ${PROJECT_VERSION}
	LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# Include CMake packages
include(GNUInstallDirs)
include(CTest)

# Several of Boost's sub-libraries write BUILD_TESTING into the cache, so
# fetching it turns this project's own tests off behind our back. Capture
# what was asked for before that can happen.
set(REXLIB_CUDA_BUILD_TESTING ${BUILD_TESTING})

include(fetch_boost)
include(fetch_spdlog)
include(find_cuda_toolchain)
include(find_rexlib)

# Find rexlib
if(NOT TARGET rexlib::rexlib)
	find_rexlib()
endif()

# Find dependencies
find_cuda_toolchain()
fetch_boost(VERSION 1.90.0)
fetch_spdlog(VERSION 1.17.0)

# One fat binary per CUDA major, covering the architectures that major's
# own compiler supports plus PTX for the newest. Spelling the list out
# would rot: CUDA 12 dropped sm_35 and sm_37, CUDA 13 dropped Maxwell,
# Pascal and Volta.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
enable_language(CUDA)

# Set project-wide compilation options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

# Add source directory
add_subdirectory(src)

# Only build tests if it is the main project
if(REXLIB_CUDA_BUILD_TESTING)
	add_subdirectory(tests)
endif()
