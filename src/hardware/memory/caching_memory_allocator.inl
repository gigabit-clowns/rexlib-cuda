// SPDX-License-Identifier: GPL-3.0-only

#include "caching_memory_allocator.hpp"

#include "buffer.hpp"
#include "memory_heap.hpp"

#include "../command_queue.hpp"
#include "../../logger.hpp"

#include <xmipp4/core/memory/align.hpp>
#include <xmipp4/core/platform/assert.hpp>

#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

template <typename Resource>
caching_memory_allocator<Resource>::caching_memory_allocator(
	std::shared_ptr<const Resource> resource,
	std::size_t heap_step
) noexcept
	: m_resource(std::move(resource))
	, m_heap_step(heap_step)
{
}

template <typename Resource>
caching_memory_allocator<Resource>::~caching_memory_allocator()
{
	try
	{
		m_deferred_release.wait_pending_free(m_pool);
	}
	catch (const std::exception &e)
	{
		log_error(e.what());
	}
}

template <typename Resource>
const xmipp4::memory_resource&
caching_memory_allocator<Resource>::get_memory_resource() const noexcept
{
	XMIPP4_ASSERT( m_resource );
	return *m_resource;
}

template <typename Resource>
std::size_t
caching_memory_allocator<Resource>::get_max_alignment() const noexcept
{
	return m_resource->get_max_alignment();
}

template <typename Resource>
std::shared_ptr<xmipp4::buffer> caching_memory_allocator<Resource>::allocate(
	std::size_t size,
	std::size_t alignment,
	xmipp4::command_queue *queue_hint
)
{
	const auto max_alignment = get_max_alignment();
	if (alignment == 0 || (alignment & (alignment - 1)) != 0)
	{
		throw std::invalid_argument("alignment must be a power of two");
	}
	if (alignment > max_alignment)
	{
		throw std::invalid_argument(
			"alignment parameter exceeds the maximum alignment of this "
			"allocator"
		);
	}

	const auto *queue =
		queue_hint ? command_queue::try_cast(*queue_hint) : nullptr;
	if (queue_hint && !queue)
	{
		throw std::invalid_argument(
			"The provided queue was not created by the CUDA backend."
		);
	}

	// Rounding up to the maximum alignment keeps the remainder of a
	// partitioned block aligned too.
	size = align_ceil(size, max_alignment);

	const std::lock_guard<std::mutex> lock(m_mutex);
	m_deferred_release.process_pending_free(m_pool);

	auto *block = obtain_block(size, queue);
	XMIPP4_ASSERT( block );

	const auto remaining = block->get_size() - size;
	if (remaining >= max_alignment)
	{
		std::tie(block, std::ignore) =
			m_pool.partition_block(block, size, remaining);
	}

	XMIPP4_ASSERT( block );
	return create_buffer(*block);
}

template <typename Resource>
void caching_memory_allocator<Resource>::recycle_block(
	memory_block &block,
	span<command_queue *const> queues
)
{
	const std::lock_guard<std::mutex> lock(m_mutex);
	if (queues.empty())
	{
		m_pool.release(block);
	}
	else
	{
		m_deferred_release.defer_release(block, queues);
	}
}

template <typename Resource>
memory_block* caching_memory_allocator<Resource>::obtain_block(
	std::size_t size,
	const command_queue *queue
)
{
	auto *block = m_pool.find_suitable_block(size, queue);
	if (block)
	{
		return block;
	}

	const auto request_size = align_ceil(size, m_heap_step);
	std::shared_ptr<memory_heap> heap;
	try
	{
		heap = m_resource->create_heap(request_size);
	}
	catch (const std::exception&)
	{
		// Everything held back is waited for and every whole free heap is
		// given back before deciding that the request cannot be served.
		m_deferred_release.wait_pending_free(m_pool);
		m_pool.release_unused_heaps();

		block = m_pool.find_suitable_block(size, queue);
		if (block)
		{
			return block;
		}

		heap = m_resource->create_heap(request_size);
	}

	XMIPP4_ASSERT( heap );
	return m_pool.register_heap(std::move(heap), queue);
}

template <typename Resource>
std::shared_ptr<xmipp4::buffer>
caching_memory_allocator<Resource>::create_buffer(memory_block &block)
{
	XMIPP4_ASSERT( block.is_free() );
	m_pool.acquire(block);

	return std::make_shared<buffer>(
		m_resource,
		shared_from_this(),
		block
	);
}

} // namespace cuda
} // namespace xmipp4
