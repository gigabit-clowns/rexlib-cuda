// SPDX-License-Identifier: GPL-3.0-only

#include "device.hpp"

#include "command_queue.hpp"
#include "event.hpp"

#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

device::device(int ordinal) noexcept
	: m_ordinal(ordinal)
{
}

device::~device() = default;

int device::get_ordinal() const noexcept
{
	return m_ordinal;
}

const memory_resource&
device::get_memory_resource(memory_resource_affinity /*affinity*/) const
{
	// TODO: Return the device local resource for the device affinity and the
	// pinned host resource for the host affinity. Until then, no device
	// session can be created for a CUDA device.
	throw std::runtime_error(
		"The CUDA backend does not expose any memory resource yet."
	);
}

std::shared_ptr<xmipp4::command_queue> device::create_command_queue() const
{
	return std::make_shared<command_queue>(m_ordinal);
}

std::shared_ptr<xmipp4::event>
device::create_event(event_usage_flags /*usage*/) const
{
	// CUDA events support every capability, so the requested subset does not
	// change which primitive is picked.
	return std::make_shared<event>(m_ordinal);
}

} // namespace cuda
} // namespace xmipp4
