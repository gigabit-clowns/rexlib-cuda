// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <cstddef>
#include <memory>

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
	memory_heap(const memory_heap &other) = delete;
	memory_heap(memory_heap &&other) = delete;
	~memory_heap();

	memory_heap& operator=(const memory_heap &other) = delete;
	memory_heap& operator=(memory_heap &&other) = delete;

	void* get_device_ptr() const noexcept;

	/**
	 * @brief Get the host accessible pointer to the start of the heap.
	 *
	 * @return void* The pointer, or nullptr when the heap cannot be reached
	 * from the host.
	 */
	void* get_host_ptr() const noexcept;

	std::size_t get_size() const noexcept;

	/**
	 * @brief Allocate device memory on the given device.
	 *
	 * @throws error If the allocation fails.
	 */
	static std::shared_ptr<memory_heap>
	create_device_memory(int ordinal, std::size_t size);

	/**
	 * @brief Allocate page locked host memory, usable from every device.
	 *
	 * @throws error If the allocation fails.
	 */
	static std::shared_ptr<memory_heap> create_pinned_memory(std::size_t size);

private:
	memory_heap(void *data, std::size_t size, int ordinal) noexcept;

	/// Held by pinned heaps, which do not belong to any single device.
	static constexpr int no_device = -1;

	void *m_data;
	std::size_t m_size;
	int m_ordinal;
};

} // namespace cuda
} // namespace xmipp4
