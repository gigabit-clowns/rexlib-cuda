// SPDX-License-Identifier: GPL-3.0-only

#pragma once

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Scoped selection of the calling thread's current CUDA device.
 *
 * Most of the CUDA runtime API acts on the current device of the calling
 * thread rather than on an explicitly named one: allocations, stream and
 * event creation all belong to whichever device was selected last. Every
 * code path in this plugin that reaches the runtime must therefore select
 * its device first, and restore the previous selection afterwards so that
 * callers outside the plugin are left undisturbed.
 *
 * Selecting the device that is already current is a no-op, so guarding a
 * path that is already on the right device costs nothing but a query.
 */
class device_guard
{
public:
	/**
	 * @brief Make @p ordinal the current device until destruction.
	 *
	 * @param ordinal The CUDA device ordinal to select.
	 *
	 * @throws error If the current device cannot be queried or @p ordinal
	 * cannot be selected.
	 */
	explicit device_guard(int ordinal);
	device_guard(const device_guard &other) = delete;
	device_guard(device_guard &&other) = delete;

	/**
	 * @brief Restore the device that was current at construction.
	 *
	 * Failures are reported rather than thrown.
	 */
	~device_guard();

	device_guard& operator=(const device_guard &other) = delete;
	device_guard& operator=(device_guard &&other) = delete;

private:
	int m_previous;
};

} // namespace cuda
} // namespace xmipp4
