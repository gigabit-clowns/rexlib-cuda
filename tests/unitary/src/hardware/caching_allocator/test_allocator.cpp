// SPDX-License-Identifier: GPL-3.0-only

#include "test_allocator.hpp"

#include "test_event_recorder_reference.hpp"
#include "test_memory_source_reference.hpp"

#include <xmipp4/core/hardware/memory_resource_kind.hpp>

#include <utility>

namespace xmipp4
{
namespace cuda
{

allocator_fixture::allocator_fixture(
	std::size_t capacity,
	memory_resource_kind kind
)
	: m_arena(capacity, 256)
	, m_reached_by_default(true)
{
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_resource, get_kind())
			.RETURN(kind)
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_source, get_base_alignment())
			.LR_RETURN(m_arena.get_alignment())
	);
	m_expectations.push_back(
		NAMED_ALLOW_CALL(m_source, is_host_accessible())
			.RETURN(xmipp4::is_host_accessible(kind))
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

	m_allocator = std::make_shared<caching_memory_allocator>(
		m_resource,
		std::make_unique<memory_source_reference>(m_source),
		std::make_shared<event_recorder_reference>(m_recorder)
	);
}

allocator_fixture::~allocator_fixture()
{
	// The allocator has to be gone before the mocks it forwards to, since it
	// gives every heap it is still holding back on the way out.
	m_allocator.reset();
}

caching_memory_allocator& allocator_fixture::get() noexcept
{
	return *m_allocator;
}

const std::shared_ptr<caching_memory_allocator>&
allocator_fixture::get_shared() noexcept
{
	return m_allocator;
}

mock_memory_source& allocator_fixture::get_source() noexcept
{
	return m_source;
}

mock_event_recorder& allocator_fixture::get_recorder() noexcept
{
	return m_recorder;
}

const mock_memory_resource& allocator_fixture::get_resource() const noexcept
{
	return m_resource;
}

test_arena& allocator_fixture::get_arena() noexcept
{
	return m_arena;
}

void allocator_fixture::reach(event_recorder::ticket ticket) noexcept
{
	m_reached[ticket - 1] = true;
}

void allocator_fixture::set_reached_by_default(bool reached) noexcept
{
	m_reached_by_default = reached;
}

const queue_handle& allocator_fixture::get_captured_queue(
	event_recorder::ticket ticket
) const noexcept
{
	return m_captured_queues[ticket - 1];
}

event_recorder::ticket allocator_fixture::capture(const queue_handle &queue)
{
	m_captured_queues.push_back(queue);
	m_reached.push_back(m_reached_by_default);
	return m_reached.size();
}

} // namespace cuda
} // namespace xmipp4
