// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "deferred_release.hpp"
#include "memory_block_pool.hpp"
#include "queue_handle.hpp"

#include <xmipp4/core/hardware/memory_allocator.hpp>
#include <xmipp4/core/span.hpp>

#include <cstddef>
#include <memory>
#include <mutex>

namespace xmipp4
{
namespace cuda
{

class event_recorder;
class memory_block;
class memory_source;

/**
 * @brief A caching, queue aware @ref xmipp4::memory_allocator.
 *
 * Reaching the driver for every allocation is far too slow to do per operation,
 * so memory taken from it is kept and handed out again. What makes that safe
 * is knowing, for each block, what has to have finished before it can be
 * handed out once more.
 *
 * Blocks belong to the queue they were last handed to. That queue runs its
 * work in order, so giving one of its own blocks straight back to it costs
 * nothing. Taking one over from another queue costs a wait, which is scheduled
 * on the device rather than on the calling thread. Only when neither is
 * possible does the pool grow, doubling each time so that a workload settles
 * into asking the driver for nothing at all.
 *
 * A buffer that outlives its own queue's work, because some other queue was
 * also given work referencing it, does not block its own destructor: the block
 * is held back until those queues have caught up.
 *
 * Thread safe.
 */
class caching_memory_allocator final
	: public memory_allocator
	, public std::enable_shared_from_this<caching_memory_allocator>
{
public:
	/**
	 * @brief Build an allocator over a source of memory.
	 *
	 * @param resource The resource the memory belongs to. Must outlive the
	 * allocator.
	 * @param source Where the memory is taken from. Can not be nullptr.
	 * @param recorder Used to capture the points that say when a block can be
	 * handed out again. Can not be nullptr.
	 *
	 * @throws std::invalid_argument If @p source or @p recorder is nullptr.
	 *
	 * @note The buffers it hands out keep it alive, so it has to be owned by
	 * a @c shared_ptr. Allocating from one that is not throws
	 * @c std::bad_weak_ptr.
	 */
	caching_memory_allocator(
		const memory_resource &resource,
		std::unique_ptr<memory_source> source,
		std::shared_ptr<event_recorder> recorder
	);
	caching_memory_allocator(
		const caching_memory_allocator &other
	) = delete;
	caching_memory_allocator(caching_memory_allocator &&other) = delete;
	~caching_memory_allocator() override;

	caching_memory_allocator&
	operator=(const caching_memory_allocator &other) = delete;
	caching_memory_allocator&
	operator=(caching_memory_allocator &&other) = delete;

	const memory_resource& get_memory_resource() const noexcept override;

	std::size_t get_max_alignment() const noexcept override;

	std::shared_ptr<xmipp4::buffer> allocate(
		std::size_t size,
		std::size_t alignment,
		xmipp4::command_queue *queue_hint = nullptr
	) override;

	/**
	 * @brief Allocate a buffer for a queue that is already resolved.
	 *
	 * What @ref allocate does once it has worked out which queue the hint
	 * names. Callers inside this backend usually hold a CUDA queue already,
	 * and this spares them the downcast.
	 *
	 * @param size Requested minimum size, in bytes.
	 * @param alignment Requested alignment, in bytes. Must be a power of two
	 * and not greater than @ref get_max_alignment.
	 * @param queue The queue the buffer is expected to be used on. May stand
	 * for no queue, in which case the buffer can be used on any of them.
	 * @return std::shared_ptr<xmipp4::buffer> The buffer. Never null.
	 *
	 * @throws std::bad_alloc If the device had no memory left.
	 * @throws std::invalid_argument If @p alignment is not a valid power of
	 * two within @ref get_max_alignment.
	 */
	std::shared_ptr<xmipp4::buffer> allocate(
		std::size_t size,
		std::size_t alignment,
		const queue_handle &queue
	);

	/**
	 * @brief Take a block back from the buffer that owned it.
	 *
	 * @param block The block to take back.
	 * @param queues The queues besides its own that were given work
	 * referencing it. The block is held back until they have caught up.
	 */
	void recycle(
		memory_block &block,
		span<const queue_handle> queues
	) noexcept;

	/**
	 * @brief Give back to the driver whatever is not being used.
	 *
	 * Waits for every queue that has memory waiting on it, which is what makes
	 * the blocks scattered across them add back up to whole heaps.
	 *
	 * @return std::size_t Number of bytes given back.
	 *
	 * @throws error If a queue could not be waited for.
	 */
	std::size_t trim();

	/**
	 * @brief Get how much memory the allocator is holding.
	 *
	 * @return std::size_t Total size of its heaps, in bytes, whether handed
	 * out or not.
	 */
	std::size_t get_pool_size() const;

private:
	const memory_resource *m_resource;
	std::unique_ptr<memory_source> m_source;
	std::shared_ptr<event_recorder> m_recorder;

	std::mutex m_mutex;

	// The deferred release gives blocks back to the pool as it drains, and the
	// pool gives heaps back to the source. Declared in the order that makes
	// them die the other way around.
	memory_block_pool m_pool;
	deferred_release m_deferred;

	memory_block& acquire_block(
		std::size_t size,
		const queue_handle &queue
	);
	memory_block* take_over_foreign_block(
		std::size_t size,
		const queue_handle &queue
	);
	memory_block& grow(std::size_t size);
	std::size_t get_heap_size(std::size_t size) const noexcept;
	std::size_t trim_locked();
};

} // namespace cuda
} // namespace xmipp4
