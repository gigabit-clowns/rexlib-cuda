// SPDX-License-Identifier: GPL-3.0-only

#include "device_memory_resource.hpp"

#include "memory_heap.hpp"

#include "../../config.hpp"

namespace xmipp4
{
namespace cuda
{

memory_resource_kind device_memory_resource::get_kind() const noexcept
{
	return memory_resource_kind::device_local;
}

std::size_t device_memory_resource::get_max_alignment() const noexcept
{
	return XMIPP4_CUDA_COALESCE_ALIGN_BYTES;
}

std::shared_ptr<memory_heap>
device_memory_resource::create_heap(std::size_t size) const
{
	return std::make_shared<device_memory_heap>(get_ordinal(), size);
}

} // namespace cuda
} // namespace xmipp4
