// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <rexlib/core/hardware/command_queue.hpp>

#include <memory>

#include <cuda_runtime.h>

namespace rexlib
{
namespace cuda
{

class event_recorder;
class stream_timeline;

/**
 * @brief CUDA implementation of @ref rexlib::command_queue, backed by a
 * cudaStream_t.
 *
 * The queue owns the stream. Its @ref stream_timeline is shared with the
 * tokens of its commands, and outlives it.
 */
class command_queue final
	: public rexlib::command_queue
{
public:
	using handle = cudaStream_t;

	/**
	 * @brief Create a queue on a device.
	 *
	 * @param ordinal Ordinal of the device.
	 * @param recorder Captures the points of the stream that say when a
	 * command has finished. Must serve the same device. Can not be nullptr.
	 *
	 * @throws std::invalid_argument If @p recorder is nullptr.
	 * @throws error If the stream could not be created.
	 */
	command_queue(int ordinal, std::shared_ptr<event_recorder> recorder);
	command_queue(const command_queue &other) = delete;
	command_queue(command_queue &&other) = delete;
	~command_queue() override;

	command_queue& operator=(const command_queue &other) = delete;
	command_queue& operator=(command_queue &&other) = delete;

	handle get_handle() const noexcept;
	int get_ordinal() const noexcept;

	/**
	 * @brief Get the timeline of the commands of this queue.
	 *
	 * @return const std::shared_ptr<stream_timeline>& The timeline. Never
	 * null.
	 */
	const std::shared_ptr<stream_timeline>& get_timeline() const noexcept;

	/**
	 * @brief Block the calling thread until the stream is idle.
	 *
	 * @throws error If the stream cannot be synchronized.
	 */
	void synchronize() const;

	command_token submit(command cmd) override;

	/**
	 * @brief Downcast a queue to this backend's implementation.
	 *
	 * @param queue The queue to be cast.
	 * @return command_queue& The same queue, as a CUDA queue.
	 *
	 * @throws std::invalid_argument If the queue was not created by this
	 * backend.
	 */
	static command_queue& cast(rexlib::command_queue &queue);

	/**
	 * @brief Downcast a queue to this backend's implementation.
	 *
	 * @param queue The queue to be cast.
	 * @return const command_queue& The same queue, as a CUDA queue.
	 *
	 * @throws std::invalid_argument If the queue was not created by this
	 * backend.
	 */
	static const command_queue& cast(const rexlib::command_queue &queue);

private:
	std::shared_ptr<stream_timeline> m_timeline;
	handle m_stream;
	int m_ordinal;
};

} // namespace cuda
} // namespace rexlib
