// SPDX-License-Identifier: GPL-3.0-only

#include "memory_block_deferred_release.hpp"

#include "memory_block_pool.hpp"

#include "../command_queue.hpp"
#include "../error.hpp"
#include "../event.hpp"

#include <xmipp4/core/platform/assert.hpp>

#include <algorithm>
#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

void memory_block_deferred_release::wait_pending_free(memory_block_pool &pool)
{
	for (auto &item : m_pending_free)
	{
		auto &events = item.second;
		for (const auto &pending : events)
		{
			XMIPP4_ASSERT( pending );
			pending->wait();
		}

		pool.release(item.first);
		m_event_pool.splice_after(m_event_pool.cbefore_begin(), events);
	}

	m_pending_free.clear();
}

void memory_block_deferred_release::process_pending_free(
	memory_block_pool &pool
)
{
	const auto last = std::remove_if(
		m_pending_free.begin(), m_pending_free.end(),
		[this, &pool] (auto &item) -> bool
		{
			auto &events = item.second;
			pop_completed_events(events);

			const auto completed = events.empty();
			if (completed)
			{
				pool.release(item.first);
			}

			return completed;
		}
	);

	m_pending_free.erase(last, m_pending_free.end());
}

void memory_block_deferred_release::defer_release(
	memory_block &block,
	span<command_queue *const> queues
)
{
	if (queues.empty())
	{
		throw std::invalid_argument(
			"No queues were provided to defer the release"
		);
	}

	m_pending_free.emplace_back(
		std::piecewise_construct,
		std::forward_as_tuple(block),
		std::forward_as_tuple()
	);

	auto &events = m_pending_free.back().second;
	for (command_queue *queue : queues)
	{
		XMIPP4_ASSERT( queue );
		record_event(events, *queue);
	}
}

void memory_block_deferred_release::pop_completed_events(event_list &events)
{
	auto prev_ite = events.cbefore_begin();
	event_list::const_iterator ite;
	while ((ite = std::next(prev_ite)) != events.cend())
	{
		const auto &pending = *ite;
		XMIPP4_ASSERT( pending );
		if (pending->is_signaled())
		{
			m_event_pool.splice_after(
				m_event_pool.cbefore_begin(),
				events,
				prev_ite
			);
		}
		else
		{
			++prev_ite;
		}
	}
}

void memory_block_deferred_release::record_event(
	event_list &events,
	command_queue &queue
)
{
	if (m_event_pool.empty())
	{
		events.emplace_front(std::make_shared<event>(queue.get_ordinal()));
	}
	else
	{
		events.splice_after(
			events.cbefore_begin(),
			m_event_pool,
			m_event_pool.cbefore_begin()
		);
	}

	XMIPP4_ASSERT( events.front() );
	XMIPP4_CUDA_CHECK(
		cudaEventRecord(events.front()->get_handle(), queue.get_handle())
	);
}

} // namespace cuda
} // namespace xmipp4
