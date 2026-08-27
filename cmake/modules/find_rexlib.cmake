cmake_minimum_required(VERSION 3.23)

# Finds rexlib, including the copy that ships inside the rexlib Python
# package.
#
# That copy is a complete install prefix nested in site-packages rather
# than a system one, so CMake has no reason to look there. The package
# says where it is through get_cmake_dir(), which is how a plugin builds
# against a pip-installed rexlib without anything having to pass paths
# around.
#
# A macro rather than a function: find_package() defines variables in the
# calling scope, and rexlib's package configuration carries
# REXLIB_PLUGINS_INSTALL_DIR, which the plugin's install rule needs.
macro(find_rexlib)
	find_package(rexlib QUIET)

	if(NOT rexlib_FOUND)
		find_package(Python COMPONENTS Interpreter QUIET)
		if(Python_Interpreter_FOUND)
			execute_process(
				COMMAND "${Python_EXECUTABLE}" -c
					"import rexlib; print(rexlib.get_cmake_dir())"
				OUTPUT_VARIABLE REXLIB_PYTHON_CMAKE_DIR
				OUTPUT_STRIP_TRAILING_WHITESPACE
				ERROR_QUIET
			)
			if(REXLIB_PYTHON_CMAKE_DIR AND IS_DIRECTORY "${REXLIB_PYTHON_CMAKE_DIR}")
				find_package(rexlib REQUIRED
					PATHS "${REXLIB_PYTHON_CMAKE_DIR}"
					NO_DEFAULT_PATH
				)
				message(STATUS
					"rexlib: Python package at ${REXLIB_PYTHON_CMAKE_DIR}"
				)
			endif()
		endif()
	endif()

	if(NOT rexlib_FOUND)
		message(FATAL_ERROR
			"rexlib not found. Install it system-wide, point CMAKE_PREFIX_PATH "
			"at a release archive, or install it with 'pip install rexlib'."
		)
	endif()
endmacro()
