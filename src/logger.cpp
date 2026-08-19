// SPDX-License-Identifier: GPL-3.0-only

#include "logger.hpp"

#include <cstdio>

namespace xmipp4
{
namespace cuda
{

void log_error(const char *message) noexcept
{
	std::fprintf(stderr, "[xmipp4-cuda] %s\n", message);
}

} // namespace cuda
} // namespace xmipp4
