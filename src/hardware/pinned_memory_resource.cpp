// SPDX-License-Identifier: GPL-3.0-only

#include "pinned_memory_resource.hpp"

#include "caching_allocator/caching_memory_allocator.hpp"
#include "caching_allocator/pinned_memory_source.hpp"
#include "caching_allocator/pooled_event_recorder.hpp"

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Check whether every CUDA device shares its memory with the host.
 *
 * Only then is page locked memory the same thing the device computes out of,
 * rather than a staging area to transfer through. A machine with no device at
 * all has nothing to share with, and a machine with a mix is answered
 * conservatively: the devices that do share get the ordinary staging answer
 * and lose an optimization, rather than being promised an accessibility that
 * does not hold for all of them.
 */
static bool all_devices_share_host_memory() noexcept
{
	int count;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
	{
		return false;
	}

	for (int ordinal = 0; ordinal < count; ++ordinal)
	{
		cudaDeviceProp properties;
		if (cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess)
		{
			return false;
		}

		if (!properties.integrated)
		{
			return false;
		}
	}

	return true;
}



pinned_memory_resource::pinned_memory_resource()
	: m_kind(
		all_devices_share_host_memory()
			? memory_resource_kind::unified
			: memory_resource_kind::host_staging
	)
{
}

pinned_memory_resource::~pinned_memory_resource() = default;

memory_resource_kind pinned_memory_resource::get_kind() const noexcept
{
	return m_kind;
}

std::shared_ptr<memory_allocator>
pinned_memory_resource::create_allocator() const
{
	return std::make_shared<caching_memory_allocator>(
		*this,
		// Mapping page locked memory into the device address space is only
		// worth what it costs where the device would otherwise have to
		// transfer it, which is exactly where it does not share it.
		std::make_unique<pinned_memory_source>(
			m_kind == memory_resource_kind::unified
		),
		std::make_shared<pooled_event_recorder>()
	);
}

const pinned_memory_resource& pinned_memory_resource::get()
{
	// Built on first use rather than at load time, since it asks the driver
	// what devices there are, and a plugin is loaded long before anyone has
	// said they want one.
	static const pinned_memory_resource instance;
	return instance;
}

} // namespace cuda
} // namespace xmipp4
