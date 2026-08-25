cmake_minimum_required(VERSION 3.23)

# Resolves a CUDA toolchain, preferring a system installation and falling
# back to the wheels NVIDIA publishes on PyPI, and defines
# rexlib-cuda::cudart for the runtime.
#
# Call this before enable_language(CUDA): CMAKE_CUDA_COMPILER cannot be
# set once project() has probed for one, which is why CUDA is not in this
# project's LANGUAGES list.
#
# The pip layout has to be wired by hand because FindCUDAToolkit cannot
# read it: the wheels ship libcudart.so.<major> with no libcudart.so link
# symlink, so find_library() never matches. torch and JAX do the same.
function(find_cuda_toolchain)
	find_package(CUDAToolkit QUIET)
	if(CUDAToolkit_FOUND)
		message(STATUS "CUDA toolkit: system, ${CUDAToolkit_VERSION}")
		add_library(rexlib-cuda::cudart ALIAS CUDA::cudart)
		return()
	endif()

	_rexlib_cuda_find_pip_toolkit(RUNTIME INCLUDE_DIRS NVCC SITE_PACKAGES)
	if(NOT RUNTIME)
		message(FATAL_ERROR
			"No CUDA toolkit found. Install one, or add the wheels with "
			"'pip install cuda-toolkit[cudart,nvcc]==<major>.*'."
		)
	endif()

	add_library(rexlib-cuda::cudart SHARED IMPORTED GLOBAL)
	set_target_properties(
		rexlib-cuda::cudart
		PROPERTIES
			IMPORTED_LOCATION "${RUNTIME}"
			INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}"
	)

	get_filename_component(RUNTIME_DIR "${RUNTIME}" DIRECTORY)
	set(REXLIB_CUDA_RUNTIME_DIR "${RUNTIME_DIR}" PARENT_SCOPE)
	set(REXLIB_CUDA_SITE_PACKAGES "${SITE_PACKAGES}" PARENT_SCOPE)

	if(NVCC)
		set(CMAKE_CUDA_COMPILER "${NVCC}" PARENT_SCOPE)
		message(STATUS "CUDA toolkit: pip wheels, with nvcc at ${NVCC}")
	else()
		# Only the CUDA 13 wheels carry the nvcc driver; the -cu11 and
		# -cu12 ones ship ptxas and libdevice for runtime JIT only, so a
		# system compiler is still needed to build device code there.
		message(STATUS "CUDA toolkit: pip wheels at ${RUNTIME_DIR}, no nvcc")
	endif()
endfunction()

# The toolkit is spread over several wheels whose directory names changed
# at CUDA 13: up to 12 they are nvidia-*-cu<major> unpacking into
# nvidia/cuda_runtime/ and nvidia/cuda_nvcc/, from 13 they are unsuffixed
# and share nvidia/cu<major>/. Rather than encode either shape, take every
# include directory NVIDIA laid down and let the compiler see all of them.
# cuda_runtime.h includes crt/host_config.h, which ships in the nvcc
# package rather than the runtime one, so this really is needed.
function(_rexlib_cuda_find_pip_toolkit RUNTIME_VAR INCLUDES_VAR NVCC_VAR SITE_VAR)
	find_package(Python COMPONENTS Interpreter QUIET)
	if(NOT Python_Interpreter_FOUND)
		return()
	endif()

	execute_process(
		COMMAND "${Python_EXECUTABLE}" -c
			"import importlib.util as u;s=u.find_spec('nvidia');print(s.submodule_search_locations[0] if s else '')"
		OUTPUT_VARIABLE ROOT
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET
	)
	if(NOT ROOT OR NOT IS_DIRECTORY "${ROOT}")
		return()
	endif()

	get_filename_component(SITE_PACKAGES "${ROOT}" DIRECTORY)
	set(${SITE_VAR} "${SITE_PACKAGES}" PARENT_SCOPE)

	file(GLOB COMPONENTS "${ROOT}/*")
	set(INCLUDES "")
	foreach(COMPONENT IN LISTS COMPONENTS)
		if(IS_DIRECTORY "${COMPONENT}/include")
			list(APPEND INCLUDES "${COMPONENT}/include")
		endif()

		file(GLOB RUNTIMES
			"${COMPONENT}/lib/libcudart.so.*"
			"${COMPONENT}/lib/libcudart.*.dylib"
			"${COMPONENT}/bin/cudart64_*.dll"
		)
		if(RUNTIMES AND NOT RUNTIME)
			list(GET RUNTIMES 0 RUNTIME)
		endif()

		if(NOT NVCC AND EXISTS "${COMPONENT}/bin/nvcc")
			set(NVCC "${COMPONENT}/bin/nvcc")
		endif()
	endforeach()

	set(${RUNTIME_VAR} "${RUNTIME}" PARENT_SCOPE)
	set(${INCLUDES_VAR} "${INCLUDES}" PARENT_SCOPE)
	set(${NVCC_VAR} "${NVCC}" PARENT_SCOPE)
endfunction()

# Gives an installed plugin an RPATH reaching the CUDA runtime that
# find_cuda_toolchain() resolved, when that runtime came from pip.
#
# Both the plugin and the runtime live under site-packages, so the hop is
# expressed relative to it: up out of the plugin directory, then down into
# the nvidia package. A system toolkit needs nothing, since it is already
# on the loader's search path.
function(rexlib_cuda_add_runtime_rpath TARGET)
	if(NOT REXLIB_CUDA_SITE_PACKAGES OR WIN32)
		return()
	endif()

	file(RELATIVE_PATH DOWN
		"${REXLIB_CUDA_SITE_PACKAGES}" "${REXLIB_CUDA_RUNTIME_DIR}"
	)

	# One "../" per component of the plugin's install directory, which is
	# itself relative to site-packages in a wheel.
	string(REPLACE "/" ";" COMPONENTS "${REXLIB_PLUGINS_INSTALL_DIR}")
	set(UP "")
	foreach(COMPONENT IN LISTS COMPONENTS)
		string(APPEND UP "../")
	endforeach()

	if(APPLE)
		set(ORIGIN "@loader_path")
	else()
		set(ORIGIN "$ORIGIN")
	endif()

	set_property(
		TARGET ${TARGET}
		APPEND PROPERTY INSTALL_RPATH "${ORIGIN}/${UP}${DOWN}"
	)
endfunction()
