// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "event_recorder.hpp"

#include <vector>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Captures points using recycled CUDA events.
 *
 * Recording a point is on the path of every release of a buffer that outlived
 * its own queue, so the events are pooled and re-recorded rather than created
 * and destroyed each time.
 *
 * The pool is kept per device, because an event can only be recorded on a
 * stream belonging to the device it was created on.
 *
 * Not thread safe; callers are expected to hold the lock of whatever owns it.
 */
class pooled_event_recorder final
	: public event_recorder
{
public:
	pooled_event_recorder() noexcept = default;
	~pooled_event_recorder() override;

	ticket record(const queue_handle &queue) override;
	void release(ticket ticket) noexcept override;
	bool is_complete(ticket ticket) override;
	void wait(ticket ticket) override;
	void enqueue_wait(const queue_handle &queue, ticket ticket) override;

private:
	struct pooled_event
	{
		cudaEvent_t event;
		int ordinal;
	};

	/// Every event ever created, in use or not. Tickets index into it, offset
	/// by one so that zero can stand for no ticket.
	std::vector<pooled_event> m_events;

	/// Tickets of the events not currently handed out, per device ordinal.
	std::vector<std::vector<ticket>> m_available;

	cudaEvent_t get_event(ticket ticket) const noexcept;
	std::vector<ticket>& get_available(int ordinal);
	ticket acquire(int ordinal);
	ticket create(int ordinal);
};

} // namespace cuda
} // namespace xmipp4
