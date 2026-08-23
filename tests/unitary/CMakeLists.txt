cmake_minimum_required(VERSION 3.16)

# Test excecutable name
set(TESTS_NAME "${PROJECT_NAME}_unitary_tests")

# Register all source and header files
file(
	GLOB_RECURSE
	TEST_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)

# The plugin itself is a MODULE and cannot be linked against, so the tests
# embed the same objects it is built from and pick up their usage requirements
# from the shared interface target.
add_executable(
	${TESTS_NAME}
	${TEST_SOURCES}
	$<TARGET_OBJECTS:${PROJECT_NAME}-objects>
)
set_target_properties(
	${TESTS_NAME} PROPERTIES
	CXX_STANDARD 20
)
target_link_libraries(
	${TESTS_NAME}
	PRIVATE
		${PROJECT_NAME}-deps
		Catch2::Catch2WithMain
		trompeloeil::trompeloeil
)

# The embedded objects still import xmipp4-core, whose DLL lives in its own
# install prefix and is not on the loader's search path. Unlike the integration
# suite, which loads the plugin from its working directory, this executable is
# started by the test discovery step, so the library has to sit next to it.
if(WIN32)
	add_custom_command(TARGET ${TESTS_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
			$<TARGET_FILE:xmipp4-core> $<TARGET_FILE_DIR:${TESTS_NAME}>
		COMMENT "Copy xmipp4-core DLL next to ${TESTS_NAME}" VERBATIM
	)
endif()

# Discover tests
catch_discover_tests(${TESTS_NAME})

# Install
install(
	TARGETS ${TESTS_NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
