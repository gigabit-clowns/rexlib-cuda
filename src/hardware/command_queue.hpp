// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/command_queue.hpp>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief CUDA implementation of @ref xmipp4::command_queue, backed by a
 * cudaStream_t.
 *
 * The stream is created with cudaStreamNonBlocking so that it never
 * implicitly synchronizes with the legacy default stream: queues are only
 * ordered against each other through @ref event objects, which is the
 * contract the framework describes.
 */
class command_queue final
	: public xmipp4::command_queue
{
public:
	using handle = cudaStream_t;

	/**
	 * @brief Create a queue on the given device.
	 *
	 * @param ordinal The CUDA device ordinal that owns the stream.
	 *
	 * @throws error If the stream cannot be created.
	 */
	explicit command_queue(int ordinal);
	command_queue(const command_queue &other) = delete;
	command_queue(command_queue &&other) = delete;
	~command_queue() override;

	command_queue& operator=(const command_queue &other) = delete;
	command_queue& operator=(command_queue &&other) = delete;

	/**
	 * @brief Get the underlying CUDA stream.
	 *
	 * @return handle The CUDA stream. Never null.
	 */
	handle get_handle() const noexcept;

	/**
	 * @brief Get the device that owns this queue.
	 *
	 * @return int The CUDA device ordinal.
	 */
	int get_ordinal() const noexcept;

	/**
	 * @brief Block the calling thread until the stream is idle.
	 *
	 * Used as the fallback path of the memory allocator when a block cannot
	 * be recycled through events.
	 *
	 * @throws error If the stream cannot be synchronized.
	 */
	void synchronize() const;

	void submit(const command &cmd) override;
	void signal(xmipp4::event &ev) override;
	void wait(const xmipp4::event &ev) override;

private:
	handle m_stream;
	int m_ordinal;
};

} // namespace cuda
} // namespace xmipp4
