// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <cstddef>

namespace xmipp4
{
namespace cuda
{

/**
 * @brief A single CUDA allocation that buffers are carved out of.
 *
 * Buffers refer to a range of a heap rather than to an allocation of their
 * own, so that a caching allocator can hand out several of them from one
 * call to the driver without the buffer knowing about it.
 */
class memory_heap
{
public:
	memory_heap() noexcept = default;
	memory_heap(const memory_heap &other) = delete;
	memory_heap(memory_heap &&other) = delete;
	virtual ~memory_heap();

	memory_heap& operator=(const memory_heap &other) = delete;
	memory_heap& operator=(memory_heap &&other) = delete;

	virtual void* get_device_ptr() const noexcept = 0;

	/**
	 * @brief Get the host accessible pointer to the start of the heap.
	 *
	 * @return void* The pointer, or nullptr when the heap cannot be reached
	 * from the host.
	 */
	virtual void* get_host_ptr() const noexcept = 0;

	virtual std::size_t get_size() const noexcept = 0;
};

/**
 * @brief Heap backed by cudaMalloc. Not reachable from the host.
 */
class device_memory_heap final
	: public memory_heap
{
public:
	device_memory_heap(int ordinal, std::size_t size);
	~device_memory_heap() override;

	void* get_device_ptr() const noexcept override;
	void* get_host_ptr() const noexcept override;
	std::size_t get_size() const noexcept override;

private:
	void *m_data;
	std::size_t m_size;
	int m_ordinal;
};

/**
 * @brief Heap backed by cudaHostAlloc. Reachable from both sides.
 */
class pinned_memory_heap final
	: public memory_heap
{
public:
	pinned_memory_heap(int ordinal, std::size_t size);
	~pinned_memory_heap() override;

	void* get_device_ptr() const noexcept override;
	void* get_host_ptr() const noexcept override;
	std::size_t get_size() const noexcept override;

private:
	void *m_data;
	std::size_t m_size;
};

} // namespace cuda
} // namespace xmipp4
