// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/buffer.hpp>

#include <memory>

namespace xmipp4
{
namespace cuda
{

class memory_heap;
class memory_resource;

/**
 * @brief A range of a @ref memory_heap handed out as a framework buffer.
 *
 * The buffer keeps both the heap and the resource alive, so it stays valid
 * for as long as it is held, no matter what happens to the device or the
 * session it came from.
 */
class buffer final
	: public xmipp4::buffer
{
public:
	buffer(
		std::shared_ptr<const memory_resource> resource,
		std::shared_ptr<memory_heap> heap,
		std::size_t offset,
		std::size_t size
	) noexcept;
	~buffer() override;

	void* get_host_ptr() noexcept override;
	const void* get_host_ptr() const noexcept override;
	std::size_t get_size() const noexcept override;
	const xmipp4::memory_resource& get_memory_resource() const noexcept override;

	/**
	 * @brief Get the device accessible pointer to the start of the buffer.
	 */
	void* get_device_ptr() const noexcept;

	/**
	 * @brief Downcast a buffer to this backend's implementation.
	 *
	 * @return buffer* The same buffer, or nullptr when another backend
	 * produced it.
	 */
	static buffer* try_cast(xmipp4::buffer &buf) noexcept;
	static const buffer* try_cast(const xmipp4::buffer &buf) noexcept;

private:
	std::shared_ptr<const memory_resource> m_resource;
	std::shared_ptr<memory_heap> m_heap;
	std::size_t m_offset;
	std::size_t m_size;
};

} // namespace cuda
} // namespace xmipp4
