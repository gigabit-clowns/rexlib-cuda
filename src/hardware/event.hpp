// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/event.hpp>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief CUDA implementation of @ref xmipp4::event, backed by a cudaEvent_t.
 *
 * The underlying event is created with timing disabled, as the framework
 * never reads elapsed times from it, and with blocking synchronization, so
 * that a host wait yields the CPU instead of spinning. Host waits only
 * happen on the deferred release path of the allocator, which is rare
 * enough that yielding is the right trade.
 *
 * A freshly created CUDA event that has never been recorded reports itself
 * as complete, which is exactly the initially signaled state that
 * @ref xmipp4::event requires.
 */
class event final
	: public xmipp4::event
{
public:
	using handle = cudaEvent_t;

	/**
	 * @brief Create an event on the given device.
	 *
	 * @param ordinal The CUDA device ordinal that owns the event.
	 *
	 * @throws error If the event cannot be created.
	 */
	explicit event(int ordinal);
	event(const event &other) = delete;
	event(event &&other) = delete;
	~event() override;

	event& operator=(const event &other) = delete;
	event& operator=(event &&other) = delete;

	/**
	 * @brief Get the underlying CUDA event.
	 *
	 * @return handle The CUDA event. Never null.
	 */
	handle get_handle() const noexcept;

	/**
	 * @brief Get the device that owns this event.
	 *
	 * @return int The CUDA device ordinal.
	 */
	int get_ordinal() const noexcept;

	event_usage_flags get_supported_usage() const noexcept override;

	void wait() const override;
	bool is_signaled() const override;

	/**
	 * @brief Downcast an event to this backend's implementation.
	 *
	 * @param ev The event to be cast.
	 * @return event& The same event, as a CUDA event.
	 *
	 * @throws std::invalid_argument If the event was not created by this
	 * backend.
	 */
	static event& cast(xmipp4::event &ev);

	/**
	 * @brief Downcast an event to this backend's implementation.
	 *
	 * Const overload of @ref cast(xmipp4::event&).
	 *
	 * @param ev The event to be cast.
	 * @return const event& The same event, as a CUDA event.
	 *
	 * @throws std::invalid_argument If the event was not created by this
	 * backend.
	 */
	static const event& cast(const xmipp4::event &ev);

private:
	handle m_event;
	int m_ordinal;
};

} // namespace cuda
} // namespace xmipp4
