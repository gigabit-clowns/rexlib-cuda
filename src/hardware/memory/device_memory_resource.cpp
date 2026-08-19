// SPDX-License-Identifier: GPL-3.0-only

#include "device_memory_resource.hpp"

#include "caching_memory_allocator.hpp"
#include "memory_heap.hpp"

#include "../../config.hpp"

namespace xmipp4
{
namespace cuda
{

device_memory_resource::device_memory_resource(int ordinal) noexcept
	: m_ordinal(ordinal)
{
}

int device_memory_resource::get_ordinal() const noexcept
{
	return m_ordinal;
}

std::size_t device_memory_resource::get_max_alignment() const noexcept
{
	return XMIPP4_CUDA_COALESCE_ALIGN_BYTES;
}

std::shared_ptr<memory_heap>
device_memory_resource::create_heap(std::size_t size) const
{
	return memory_heap::create_device_memory(m_ordinal, size);
}

memory_resource_kind device_memory_resource::get_kind() const noexcept
{
	return memory_resource_kind::device_local;
}

std::shared_ptr<memory_allocator>
device_memory_resource::create_allocator() const
{
	auto result = m_allocator.lock();
	if (!result)
	{
		result = std::make_shared<
			caching_memory_allocator<device_memory_resource>
		>(shared_from_this(), XMIPP4_CUDA_DEVICE_HEAP_STEP_BYTES);
		m_allocator = result;
	}

	return result;
}

} // namespace cuda
} // namespace xmipp4
