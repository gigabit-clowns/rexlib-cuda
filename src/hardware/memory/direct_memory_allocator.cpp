// SPDX-License-Identifier: GPL-3.0-only

#include "direct_memory_allocator.hpp"

#include "buffer.hpp"
#include "memory_heap.hpp"
#include "memory_resource.hpp"

#include <xmipp4/core/memory/align.hpp>
#include <xmipp4/core/platform/assert.hpp>

#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

direct_memory_allocator::direct_memory_allocator(
	std::shared_ptr<const memory_resource> resource
) noexcept
	: m_resource(std::move(resource))
{
}

direct_memory_allocator::~direct_memory_allocator() = default;

const xmipp4::memory_resource&
direct_memory_allocator::get_memory_resource() const noexcept
{
	XMIPP4_ASSERT( m_resource );
	return *m_resource;
}

std::size_t direct_memory_allocator::get_max_alignment() const noexcept
{
	return m_resource->get_max_alignment();
}

std::shared_ptr<xmipp4::buffer> direct_memory_allocator::allocate(
	std::size_t size,
	std::size_t alignment,
	command_queue* /*queue_hint*/
)
{
	if (alignment == 0 || (alignment & (alignment - 1)) != 0)
	{
		throw std::invalid_argument("alignment must be a power of two");
	}
	if (alignment > get_max_alignment())
	{
		throw std::invalid_argument(
			"alignment parameter exceeds the maximum alignment of this "
			"allocator"
		);
	}

	// The driver already returns a suitably aligned pointer, so a heap of the
	// rounded up size is all a buffer needs.
	size = align_ceil(size, alignment);
	auto heap = m_resource->create_heap(size);
	XMIPP4_ASSERT( heap );

	return std::make_shared<buffer>(m_resource, std::move(heap), 0UL, size);
}

} // namespace cuda
} // namespace xmipp4
