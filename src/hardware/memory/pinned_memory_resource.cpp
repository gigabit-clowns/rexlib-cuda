// SPDX-License-Identifier: GPL-3.0-only

#include "pinned_memory_resource.hpp"

#include "memory_heap.hpp"

#include "../device_guard.hpp"
#include "../error.hpp"
#include "../../config.hpp"

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

pinned_memory_resource::pinned_memory_resource(int ordinal)
	: memory_resource(ordinal)
	, m_kind(memory_resource_kind::host_staging)
{
	cudaDeviceProp properties;
	XMIPP4_CUDA_CHECK( cudaGetDeviceProperties(&properties, ordinal) );
	if (properties.integrated)
	{
		m_kind = memory_resource_kind::unified;
	}
}

memory_resource_kind pinned_memory_resource::get_kind() const noexcept
{
	return m_kind;
}

std::size_t pinned_memory_resource::get_max_alignment() const noexcept
{
	return XMIPP4_CUDA_COALESCE_ALIGN_BYTES;
}

std::shared_ptr<memory_heap>
pinned_memory_resource::create_heap(std::size_t size) const
{
	return std::make_shared<pinned_memory_heap>(get_ordinal(), size);
}

} // namespace cuda
} // namespace xmipp4
