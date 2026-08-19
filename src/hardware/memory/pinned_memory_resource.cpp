// SPDX-License-Identifier: GPL-3.0-only

#include "pinned_memory_resource.hpp"

#include "caching_memory_allocator.hpp"
#include "memory_heap.hpp"

#include "../../config.hpp"

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

static memory_resource_kind probe_kind() noexcept
{
	// Where the host and the device share physical memory there is nothing to
	// stage through. Device zero speaks for the system: a machine mixing an
	// integrated device with a discrete one is not worth modelling here.
	cudaDeviceProp properties;
	if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess &&
		properties.integrated)
	{
		return memory_resource_kind::unified;
	}

	return memory_resource_kind::host_staging;
}



pinned_memory_resource::pinned_memory_resource() noexcept
	: m_kind(probe_kind())
{
}

std::size_t pinned_memory_resource::get_max_alignment() const noexcept
{
	return XMIPP4_CUDA_COALESCE_ALIGN_BYTES;
}

std::shared_ptr<memory_heap>
pinned_memory_resource::create_heap(std::size_t size) const
{
	return memory_heap::create_pinned_memory(size);
}

memory_resource_kind pinned_memory_resource::get_kind() const noexcept
{
	return m_kind;
}

std::shared_ptr<memory_allocator>
pinned_memory_resource::create_allocator() const
{
	auto result = m_allocator.lock();
	if (!result)
	{
		result = std::make_shared<
			caching_memory_allocator<pinned_memory_resource>
		>(shared_from_this(), XMIPP4_CUDA_PINNED_HEAP_STEP_BYTES);
		m_allocator = result;
	}

	return result;
}

} // namespace cuda
} // namespace xmipp4
