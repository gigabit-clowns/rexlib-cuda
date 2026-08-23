// SPDX-License-Identifier: GPL-3.0-only

#include "pooled_event_recorder.hpp"

#include "queue_handle.hpp"

#include "../device_guard.hpp"
#include "../error.hpp"

#include <cstddef>
#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

pooled_event_recorder::~pooled_event_recorder()
{
	for (const auto &item : m_events)
	{
		// Acts on the device that owns the event, so the current one is
		// irrelevant here.
		XMIPP4_CUDA_CHECK_NO_THROW( cudaEventDestroy(item.event) );
	}
}

event_recorder::ticket
pooled_event_recorder::record(const queue_handle &queue)
{
	if (!queue)
	{
		throw std::invalid_argument(
			"pooled_event_recorder::record: A queue is required to capture a "
			"point of its timeline."
		);
	}

	const auto result = acquire(queue.get_ordinal());
	try
	{
		XMIPP4_CUDA_CHECK(
			cudaEventRecord(get_event(result), queue.get_stream())
		);
	}
	catch (...)
	{
		release(result);
		throw;
	}

	return result;
}

void pooled_event_recorder::release(ticket ticket) noexcept
{
	if (ticket == no_ticket)
	{
		return;
	}

	// Room was reserved when the event was created, so this cannot throw.
	const auto ordinal = m_events[ticket - 1].ordinal;
	m_available[static_cast<std::size_t>(ordinal)].push_back(ticket);
}

bool pooled_event_recorder::is_complete(ticket ticket)
{
	const auto code = cudaEventQuery(get_event(ticket));

	bool result;
	switch (code)
	{
	case cudaSuccess:
		result = true;
		break;

	case cudaErrorNotReady:
		result = false;
		break;

	default:
		XMIPP4_CUDA_CHECK(code);
		result = false; // To avoid warnings. The above line should throw.
		break;
	}
	return result;
}

void pooled_event_recorder::wait(ticket ticket)
{
	XMIPP4_CUDA_CHECK( cudaEventSynchronize(get_event(ticket)) );
}

void pooled_event_recorder::enqueue_wait(
	const queue_handle &queue,
	ticket ticket
)
{
	if (!queue)
	{
		throw std::invalid_argument(
			"pooled_event_recorder::enqueue_wait: A queue is required to defer "
			"its work."
		);
	}

	XMIPP4_CUDA_CHECK(
		cudaStreamWaitEvent(
			queue.get_stream(),
			get_event(ticket),
			cudaEventWaitDefault
		)
	);
}

cudaEvent_t pooled_event_recorder::get_event(ticket ticket) const noexcept
{
	return m_events[ticket - 1].event;
}

std::vector<event_recorder::ticket>&
pooled_event_recorder::get_available(int ordinal)
{
	const auto index = static_cast<std::size_t>(ordinal);
	if (index >= m_available.size())
	{
		m_available.resize(index + 1);
	}

	return m_available[index];
}

event_recorder::ticket pooled_event_recorder::acquire(int ordinal)
{
	auto &available = get_available(ordinal);
	if (available.empty())
	{
		return create(ordinal);
	}

	const auto result = available.back();
	available.pop_back();
	return result;
}

event_recorder::ticket pooled_event_recorder::create(int ordinal)
{
	// Every bit of room is made before the event exists, so that nothing can
	// throw once it does and leave it stranded. Reserving for the whole pool
	// rather than for this one event also keeps giving a ticket back
	// allocation free, which it has to be: it happens where a failure can no
	// longer be handled.
	const auto count = m_events.size() + 1;
	if (count > m_events.capacity())
	{
		m_events.reserve(2 * count);
	}
	get_available(ordinal).reserve(m_events.capacity());

	// Timing is never read from these events, and no host thread ever blocks
	// on one for long enough for spinning to be the wrong trade.
	cudaEvent_t event;
	{
		const device_guard guard(ordinal);
		XMIPP4_CUDA_CHECK(
			cudaEventCreateWithFlags(&event, cudaEventDisableTiming)
		);
	}

	m_events.push_back(pooled_event{event, ordinal});
	return m_events.size();
}

} // namespace cuda
} // namespace xmipp4
