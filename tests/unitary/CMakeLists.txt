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

# Add the test executable
add_executable(
    ${TESTS_NAME}
    ${TEST_SOURCES}
)
set_target_properties(
	${TESTS_NAME} PROPERTIES
	CXX_STANDARD 20
)
# The plugin itself is a MODULE and cannot be linked against, so the tests
# consume the same objects it is built from.
target_link_libraries(
	${TESTS_NAME}
	PRIVATE
		${PROJECT_NAME}-objects
		Catch2::Catch2WithMain
		trompeloeil::trompeloeil
)

# Discover tests
catch_discover_tests(${TESTS_NAME})

# Install
install(
	TARGETS ${TESTS_NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
