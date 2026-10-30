cmake_minimum_required(VERSION 3.18)

include(CTest)
include(FetchContent)

# Declared here rather than beside the other toggles, because these two
# dependencies exist only while the tests are built.
cmake_dependent_option(
	REXLIB_CUDA_BUILD_CATCH2 "Fetch Catch2 and build it with the plugin" ON
	"REXLIB_CUDA_BUILD_DEPENDENCIES" OFF
)
cmake_dependent_option(
	REXLIB_CUDA_BUILD_TROMPELOEIL
	"Fetch trompeloeil and build it with the plugin"
	ON
	"REXLIB_CUDA_BUILD_DEPENDENCIES" OFF
)

if(REXLIB_CUDA_BUILD_CATCH2)
	include(rexlib_cuda_fetch_catch2)
	rexlib_cuda_fetch_catch2(
		COMMIT 08092139210881f01a34e60c5f2fd6afa0ea4024
	) # v3.16.1
else()
	find_package(Catch2 3 REQUIRED CONFIG)
endif()

if(REXLIB_CUDA_BUILD_TROMPELOEIL)
	include(rexlib_cuda_fetch_trompeloeil)
	rexlib_cuda_fetch_trompeloeil(
		COMMIT eaeb89c1ce9d354b0ba7eb921fd5712cdbd78adf
	) # v49
else()
	find_package(trompeloeil REQUIRED CONFIG)
endif()

include(Catch)

add_subdirectory(unitary)
add_subdirectory(integration)
