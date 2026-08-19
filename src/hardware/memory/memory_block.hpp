// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <cstddef>

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>

namespace xmipp4
{
namespace cuda
{

class command_queue;
class memory_heap;

/**
 * @brief A range of a heap tracked by the caching allocator.
 *
 * Blocks are linked into two intrusive containers at once: the list keeps
 * every block of every heap in address order, so neighbours can be merged,
 * and the set holds only the free ones, ordered so that a best fit is the
 * first candidate found.
 */
class memory_block
{
public:
	using block_list_hook_type = boost::intrusive::list_member_hook<>;
	using free_block_set_hook_type = boost::intrusive::set_member_hook<>;

	block_list_hook_type block_list_hook;
	free_block_set_hook_type free_block_set_hook;

	memory_block(
		const command_queue *queue,
		std::size_t size,
		memory_heap *heap,
		std::size_t offset
	) noexcept;

	const command_queue* get_queue() const noexcept;

	void set_size(std::size_t size) noexcept;
	std::size_t get_size() const noexcept;

	memory_heap* get_heap() const noexcept;

	void set_offset(std::size_t offset) noexcept;
	std::size_t get_offset() const noexcept;

	bool is_free() const noexcept;

private:
	const command_queue *m_queue;
	std::size_t m_size;
	memory_heap *m_heap;
	std::size_t m_offset;
};

bool operator==(const memory_block &lhs, const memory_block &rhs) noexcept;
bool operator!=(const memory_block &lhs, const memory_block &rhs) noexcept;

} // namespace cuda
} // namespace xmipp4

#include "memory_block.inl"
