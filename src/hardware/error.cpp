// SPDX-License-Identifier: GPL-3.0-only

// Based on: https://leimao.github.io/blog/Proper-CUDA-Error-Checking/

#include "error.hpp"

#include <iostream>
#include <sstream>

namespace xmipp4
{
namespace cuda
{

static std::string format_error(
	cudaError_t code,
	const char* call,
	const char* file,
	int line
)
{
	std::ostringstream oss;
	oss << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
	oss << cudaGetErrorString(code) << " " << call << std::endl;
	return oss.str();
}

void check(
	cudaError_t code,
	const char* call,
	const char* file,
	int line
)
{
	if (code != cudaSuccess)
	{
		throw error(format_error(code, call, file, line));
	}
}

void check_no_throw(
	cudaError_t code,
	const char* call,
	const char* file,
	int line
) noexcept
{
	if (code == cudaSuccess || code == cudaErrorCudartUnloading)
	{
		return;
	}

	try
	{
		std::cerr << format_error(code, call, file, line);
	}
	catch (...)
	{
		// There is no other way left to report the error.
	}
}

} // namespace cuda
} // namespace xmipp4
