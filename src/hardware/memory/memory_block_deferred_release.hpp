// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/span.hpp>

#include <forward_list>
#include <memory>
#include <utility>
#include <vector>

namespace xmipp4
{
namespace cuda
{

class command_queue;
class event;
class memory_block;
class memory_block_pool;

/**
 * @brief Holds back the blocks that queues other than their own still use.
 *
 * A block is only safe to reuse once every queue that touched it has passed
 * the point where it did. An event is recorded on each of them, and the block
 * goes back to the pool when they have all been signaled.
 */
class memory_block_deferred_release
{
public:
	memory_block_deferred_release() = default;
	memory_block_deferred_release(
		const memory_block_deferred_release &other
	) = delete;
	memory_block_deferred_release(
		memory_block_deferred_release &&other
	) = default;
	~memory_block_deferred_release() = default;

	memory_block_deferred_release&
	operator=(const memory_block_deferred_release &other) = delete;
	memory_block_deferred_release&
	operator=(memory_block_deferred_release &&other) = default;

	/**
	 * @brief Block until every pending release has completed.
	 */
	void wait_pending_free(memory_block_pool &pool);

	/**
	 * @brief Return the blocks whose events have all been signaled.
	 */
	void process_pending_free(memory_block_pool &pool);

	/**
	 * @brief Hold a block back until @p queues reach their current point.
	 *
	 * @note Passing the same block twice before it has returned to the pool
	 * leads to undefined behavior.
	 */
	void defer_release(
		memory_block &block,
		span<command_queue *const> queues
	);

private:
	using event_list = std::forward_list<std::shared_ptr<event>>;

	event_list m_event_pool;
	std::vector<
		std::pair<std::reference_wrapper<memory_block>, event_list>
	> m_pending_free;

	void pop_completed_events(event_list &events);
	void record_event(event_list &events, command_queue &queue);
};

} // namespace cuda
} // namespace xmipp4
