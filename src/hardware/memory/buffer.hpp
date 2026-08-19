// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "../../config.hpp"

#include <xmipp4/core/hardware/buffer.hpp>

#include <memory>

#include <boost/container/small_vector.hpp>

namespace xmipp4
{
namespace cuda
{

class block_recycler;
class command_queue;
class memory_block;

/**
 * @brief A range of a memory heap handed out as a framework buffer.
 *
 * The buffer also records which queues have been given work on it, so that
 * the allocator knows what it has to wait for before handing the range out
 * again. Queues are recorded by @ref command_queue::submit and only when
 * they are not the one the range was allocated for.
 */
class buffer final
	: public xmipp4::buffer
{
public:
	buffer(
		std::shared_ptr<const xmipp4::memory_resource> resource,
		std::shared_ptr<block_recycler> recycler,
		memory_block &block
	) noexcept;
	~buffer() override;

	void* get_host_ptr() noexcept override;
	const void* get_host_ptr() const noexcept override;
	std::size_t get_size() const noexcept override;
	const xmipp4::memory_resource& get_memory_resource() const noexcept override;

	void* get_device_ptr() const noexcept;

	/**
	 * @brief Take note that @p queue has been given work on this buffer.
	 */
	void record_queue(command_queue &queue);

	/**
	 * @brief Downcast a buffer to this backend's implementation.
	 *
	 * @return buffer* The same buffer, or nullptr when another backend
	 * produced it.
	 */
	static buffer* try_cast(xmipp4::buffer &buf) noexcept;
	static const buffer* try_cast(const xmipp4::buffer &buf) noexcept;

private:
	using queue_vector_type = boost::container::small_vector<
		command_queue*,
		XMIPP4_CUDA_SMALL_QUEUE_COUNT
	>;

	std::shared_ptr<const xmipp4::memory_resource> m_resource;
	std::shared_ptr<block_recycler> m_recycler;
	memory_block *m_block;
	queue_vector_type m_queues;
};

} // namespace cuda
} // namespace xmipp4
