// SPDX-License-Identifier: GPL-3.0-only

#include "device.hpp"

#include "command_queue.hpp"
#include "event.hpp"
#include "memory/device_memory_resource.hpp"
#include "memory/pinned_memory_resource.hpp"

#include <xmipp4/core/platform/assert.hpp>

#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

device::device(
	int ordinal,
	std::shared_ptr<const device_memory_resource> device_memory,
	std::shared_ptr<const pinned_memory_resource> pinned_memory
) noexcept
	: m_ordinal(ordinal)
	, m_device_memory(std::move(device_memory))
	, m_pinned_memory(std::move(pinned_memory))
{
}

device::~device() = default;

int device::get_ordinal() const noexcept
{
	return m_ordinal;
}

const xmipp4::memory_resource&
device::get_memory_resource(memory_resource_affinity affinity) const
{
	XMIPP4_ASSERT( m_device_memory );
	XMIPP4_ASSERT( m_pinned_memory );

	switch (affinity)
	{
	case memory_resource_affinity::device:
		return *m_device_memory;

	case memory_resource_affinity::host:
		return *m_pinned_memory;

	default:
		throw std::invalid_argument("Unknown memory resource affinity");
	}
}

std::shared_ptr<xmipp4::command_queue> device::create_command_queue() const
{
	return std::make_shared<command_queue>(m_ordinal);
}

std::shared_ptr<xmipp4::event>
device::create_event(event_usage_flags /*usage*/) const
{
	return std::make_shared<event>(m_ordinal);
}

} // namespace cuda
} // namespace xmipp4
