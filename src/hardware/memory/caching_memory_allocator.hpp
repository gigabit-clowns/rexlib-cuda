// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "memory_block_deferred_release.hpp"
#include "memory_block_pool.hpp"

#include <xmipp4/core/hardware/memory_allocator.hpp>
#include <xmipp4/core/span.hpp>

#include <memory>
#include <mutex>

namespace xmipp4
{
namespace cuda
{

class command_queue;
class memory_block;

/**
 * @brief What a @ref buffer needs from the allocator that produced it.
 *
 * Keeps the buffer independent of which resource the allocator was
 * instantiated for.
 */
class block_recycler
	: public std::enable_shared_from_this<block_recycler>
{
public:
	virtual ~block_recycler();

	virtual void recycle_block(
		memory_block &block,
		span<command_queue *const> queues
	) = 0;
};

/**
 * @brief Allocator that keeps the memory it is given back.
 *
 * Released blocks return to a pool instead of to the driver, and are merged
 * with their free neighbours so that later requests can be served from them.
 * The driver is only asked for more when nothing suitable is left, and only
 * given memory back when it refuses a request.
 *
 * Blocks are segregated by the queue they were allocated for, which is what
 * makes reuse safe without synchronizing: a block returned by a queue can be
 * handed out again to that same queue right away, because the work is ordered
 * within it. Any other queue that touched the buffer forces the release to be
 * held back until an event says it is done.
 */
template <typename Resource>
class caching_memory_allocator final
	: public memory_allocator
	, public block_recycler
{
public:
	caching_memory_allocator(
		std::shared_ptr<const Resource> resource,
		std::size_t heap_step
	) noexcept;
	~caching_memory_allocator() override;

	const xmipp4::memory_resource& get_memory_resource() const noexcept override;

	std::size_t get_max_alignment() const noexcept override;

	std::shared_ptr<xmipp4::buffer> allocate(
		std::size_t size,
		std::size_t alignment,
		xmipp4::command_queue *queue_hint
	) override;

	void recycle_block(
		memory_block &block,
		span<command_queue *const> queues
	) override;

private:
	std::shared_ptr<const Resource> m_resource;
	std::size_t m_heap_step;
	std::mutex m_mutex;
	memory_block_pool m_pool;
	memory_block_deferred_release m_deferred_release;

	memory_block* obtain_block(
		std::size_t size,
		const command_queue *queue
	);
	std::shared_ptr<xmipp4::buffer> create_buffer(memory_block &block);
};

} // namespace cuda
} // namespace xmipp4

#include "caching_memory_allocator.inl"
