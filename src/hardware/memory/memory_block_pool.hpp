// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "memory_block.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/unordered/unordered_set.hpp>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Keeps track of the blocks of every heap the allocator owns.
 *
 * Blocks are never handed back to the driver on release: they are returned
 * here, merged with their free neighbours, and reused. Heaps only go back
 * through @ref release_unused_heaps, which the allocator calls when the
 * driver refuses a request.
 */
class memory_block_pool
{
public:
	memory_block_pool() = default;
	memory_block_pool(const memory_block_pool &other) = delete;
	memory_block_pool(memory_block_pool &&other) = delete;
	~memory_block_pool();

	memory_block_pool& operator=(const memory_block_pool &other) = delete;
	memory_block_pool& operator=(memory_block_pool &&other) = delete;

	/**
	 * @brief Mark a free block as occupied.
	 */
	void acquire(memory_block &block) noexcept;

	/**
	 * @brief Mark an occupied block as free, merging it with its neighbours.
	 */
	void release(memory_block &block) noexcept;

	/**
	 * @brief Find a free block of at least @p size bytes on @p queue.
	 *
	 * @return memory_block* The best fit, or nullptr if there is none.
	 */
	memory_block* find_suitable_block(
		std::size_t size,
		const command_queue *queue
	);

	/**
	 * @brief Split a free block in two.
	 *
	 * @return The two resulting partitions, both free.
	 */
	std::pair<memory_block*, memory_block*> partition_block(
		memory_block *block,
		std::size_t size1,
		std::size_t size2
	);

	/**
	 * @brief Take ownership of a heap and return the block spanning it.
	 */
	memory_block* register_heap(
		std::shared_ptr<memory_heap> heap,
		const command_queue *queue
	);

	/**
	 * @brief Return every heap that is whole and free to the driver.
	 */
	void release_unused_heaps();

private:
	class free_memory_block_compare
	{
	public:
		bool operator()(
			const memory_block &lhs,
			const memory_block &rhs
		) const noexcept;
	};

	using memory_block_list_type = boost::intrusive::list<
		memory_block,
		boost::intrusive::member_hook<
			memory_block,
			memory_block::block_list_hook_type,
			&memory_block::block_list_hook
		>
	>;
	using free_memory_block_set_type = boost::intrusive::multiset<
		memory_block,
		boost::intrusive::member_hook<
			memory_block,
			memory_block::free_block_set_hook_type,
			&memory_block::free_block_set_hook
		>,
		boost::intrusive::compare<free_memory_block_compare>
	>;
	using heap_set_type = boost::unordered::unordered_set<
		std::shared_ptr<memory_heap>,
		std::hash<std::shared_ptr<memory_heap>>
	>;

	memory_block_list_type m_blocks;
	free_memory_block_set_type m_free_blocks;
	heap_set_type m_heaps;

	void consider_merging_block(memory_block &block) noexcept;
	void consider_merging_forwards(memory_block &block) noexcept;
	void consider_merging_backwards(memory_block &block) noexcept;
	bool is_partition(const memory_block &block) const noexcept;
	void release(const memory_heap &heap);
};

} // namespace cuda
} // namespace xmipp4
