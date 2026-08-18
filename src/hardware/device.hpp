// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/device.hpp>

#include <memory>

namespace xmipp4
{
namespace cuda
{

class device_memory_resource;
class pinned_memory_resource;

/**
 * @brief CUDA implementation of @ref xmipp4::device.
 *
 * The handle is a thin wrapper around a device ordinal: it owns no driver
 * resource of its own, so several handles may refer to the same physical
 * device without interfering with each other.
 */
class device final
	: public xmipp4::device
{
public:
	device(
		int ordinal,
		std::shared_ptr<const device_memory_resource> device_memory,
		std::shared_ptr<const pinned_memory_resource> pinned_memory
	) noexcept;
	device(const device &other) = delete;
	device(device &&other) = delete;
	~device() override;

	device& operator=(const device &other) = delete;
	device& operator=(device &&other) = delete;

	int get_ordinal() const noexcept;

	const xmipp4::memory_resource&
	get_memory_resource(memory_resource_affinity affinity) const override;

	std::shared_ptr<xmipp4::command_queue> create_command_queue() const override;

	std::shared_ptr<xmipp4::event>
	create_event(event_usage_flags usage) const override;

private:
	int m_ordinal;
	std::shared_ptr<const device_memory_resource> m_device_memory;
	std::shared_ptr<const pinned_memory_resource> m_pinned_memory;
};

} // namespace cuda
} // namespace xmipp4
