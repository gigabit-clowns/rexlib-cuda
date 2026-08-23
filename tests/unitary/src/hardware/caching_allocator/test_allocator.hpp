// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "mock/mock_event_recorder.hpp"
#include "mock/mock_memory_resource.hpp"
#include "mock/mock_memory_source.hpp"
#include "test_arena.hpp"

#include <hardware/caching_allocator/caching_memory_allocator.hpp>

#include <xmipp4/core/hardware/memory_resource_kind.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief A caching allocator over ordinary host memory.
 *
 * Everything the allocator reaches out to is mocked, so a test can say what
 * the device has left and when its queues have caught up, and can ask what was
 * demanded of either.
 *
 * Taking memory and giving it back are allowed throughout, since almost every
 * test causes some of both; a test that is about those calls says so itself,
 * which takes precedence. Capturing a point is answered with a fresh ticket
 * that starts out already reached, so that a test only has to say anything
 * when it wants work to still be running.
 */
class allocator_fixture
{
public:
	/**
	 * @brief Build an allocator.
	 *
	 * @param capacity Number of bytes the device has.
	 * @param kind Kind of memory the allocator hands out.
	 */
	explicit allocator_fixture(
		std::size_t capacity = test_arena::unlimited,
		memory_resource_kind kind = memory_resource_kind::device_local
	);
	allocator_fixture(const allocator_fixture &other) = delete;
	allocator_fixture(allocator_fixture &&other) = delete;
	~allocator_fixture();

	allocator_fixture& operator=(const allocator_fixture &other) = delete;
	allocator_fixture& operator=(allocator_fixture &&other) = delete;

	caching_memory_allocator& get() noexcept;
	const std::shared_ptr<caching_memory_allocator>& get_shared() noexcept;
	mock_memory_source& get_source() noexcept;
	mock_event_recorder& get_recorder() noexcept;
	const mock_memory_resource& get_resource() const noexcept;
	test_arena& get_arena() noexcept;

	/**
	 * @brief Report the point a ticket stands for as reached.
	 *
	 * @param ticket The ticket.
	 */
	void reach(event_recorder::ticket ticket) noexcept;

	/**
	 * @brief Say whether points start out reached.
	 *
	 * @param reached Whether a point captured from now on counts as reached
	 * straight away.
	 */
	void set_reached_by_default(bool reached) noexcept;

	/**
	 * @brief Get the queue a ticket captured a point of.
	 *
	 * @param ticket The ticket.
	 * @return const queue_handle& The queue.
	 */
	const queue_handle& get_captured_queue(
		event_recorder::ticket ticket
	) const noexcept;

private:
	mock_memory_resource m_resource;
	mock_memory_source m_source;
	mock_event_recorder m_recorder;
	test_arena m_arena;

	std::vector<queue_handle> m_captured_queues;
	std::vector<bool> m_reached;
	bool m_reached_by_default;

	std::vector<std::unique_ptr<trompeloeil::expectation>> m_expectations;
	std::shared_ptr<caching_memory_allocator> m_allocator;

	event_recorder::ticket capture(const queue_handle &queue);
};

} // namespace cuda
} // namespace xmipp4
