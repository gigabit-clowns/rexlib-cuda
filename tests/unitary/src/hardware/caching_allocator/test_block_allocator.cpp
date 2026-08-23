// SPDX-License-Identifier: GPL-3.0-only

#include "test_block_allocator.hpp"

#include "test_event_recorder_reference.hpp"
#include "test_memory_source_reference.hpp"

#include <hardware/caching_allocator/memory_block.hpp>

#include <utility>

namespace xmipp4
{
namespace cuda
{

block_allocator_fixture::block_allocator_fixture(
	std::size_t capacity,
	bool host_accessible
)
	: m_arena(capacity, 256)
	, m_reached_by_default(true)
{
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_source, get_base_alignment())
			.LR_RETURN(m_arena.get_alignment())
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_source, is_host_accessible())
			.RETURN(host_accessible)
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_source, allocate(trompeloeil::_))
			.LR_RETURN(m_arena.take(_1))
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_source, deallocate(trompeloeil::_))
			.LR_SIDE_EFFECT(m_arena.give_back(_1))
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_recorder, record(trompeloeil::_))
			.LR_RETURN(this->capture(_1))
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_recorder, release(trompeloeil::_))
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_recorder, is_complete(trompeloeil::_))
			.LR_RETURN(m_reached[_1 - 1])
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_recorder, wait(trompeloeil::_))
			.LR_SIDE_EFFECT(m_reached[_1 - 1] = true)
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(
			m_recorder,
			enqueue_wait(trompeloeil::_, trompeloeil::_)
		)
	);

	m_allocator = std::make_shared<memory_block_allocator>(
		std::make_unique<memory_source_reference>(m_source),
		std::make_shared<event_recorder_reference>(m_recorder)
	);
}

block_allocator_fixture::~block_allocator_fixture()
{
	// The allocator has to be gone before the mocks it forwards to, since it
	// gives every heap it is still holding back on the way out.
	m_allocator.reset();
}

memory_block_allocator& block_allocator_fixture::get() noexcept
{
	return *m_allocator;
}

const std::shared_ptr<memory_block_allocator>&
block_allocator_fixture::get_shared() noexcept
{
	return m_allocator;
}

void* block_allocator_fixture::allocate_and_drop(
	std::size_t size,
	const queue_handle &queue
)
{
	auto &block = m_allocator->allocate(size, 256, queue);
	auto *result = block.get_data();
	m_allocator->recycle(block, span<const queue_handle>());
	return result;
}

mock_memory_source& block_allocator_fixture::get_source() noexcept
{
	return m_source;
}

mock_event_recorder& block_allocator_fixture::get_recorder() noexcept
{
	return m_recorder;
}

test_arena& block_allocator_fixture::get_arena() noexcept
{
	return m_arena;
}

void block_allocator_fixture::reach(event_recorder::ticket ticket) noexcept
{
	m_reached[ticket - 1] = true;
}

void block_allocator_fixture::set_reached_by_default(bool reached) noexcept
{
	m_reached_by_default = reached;
}

const queue_handle& block_allocator_fixture::get_captured_queue(
	event_recorder::ticket ticket
) const noexcept
{
	return m_captured_queues[ticket - 1];
}

event_recorder::ticket block_allocator_fixture::capture(const queue_handle &queue)
{
	m_captured_queues.push_back(queue);
	m_reached.push_back(m_reached_by_default);
	return m_reached.size();
}

} // namespace cuda
} // namespace xmipp4
