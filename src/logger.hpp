// SPDX-License-Identifier: GPL-3.0-only

#pragma once

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Report a message the plugin cannot act upon.
 *
 * The core's logger is a private header, so it is not reachable from here.
 */
void log_error(const char *message) noexcept;

} // namespace cuda
} // namespace xmipp4
