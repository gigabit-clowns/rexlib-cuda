// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_resource.hpp>

#include <cstddef>
#include <memory>

namespace xmipp4
{
namespace cuda
{

class memory_heap;

/**
 * @brief Base of the memory resources this backend exposes.
 *
 * Adds what the allocators need on top of the framework interface: a way to
 * obtain heaps and the strictest alignment they can be asked for. Resources
 * are always held through a shared_ptr, so that a buffer outliving the
 * device it came from keeps its own resource alive.
 */
class memory_resource
	: public xmipp4::memory_resource
	, public std::enable_shared_from_this<memory_resource>
{
public:
	explicit memory_resource(int ordinal) noexcept;
	~memory_resource() override;

	int get_ordinal() const noexcept;

	virtual std::size_t get_max_alignment() const noexcept = 0;

	/**
	 * @brief Obtain a new heap on this resource.
	 *
	 * @param size Size of the heap, in bytes.
	 * @return The new heap. Never null.
	 *
	 * @throws std::bad_alloc If the driver cannot satisfy the request.
	 */
	virtual std::shared_ptr<memory_heap> create_heap(std::size_t size) const = 0;

	std::shared_ptr<memory_allocator> create_allocator() const override;

private:
	int m_ordinal;
	mutable std::weak_ptr<memory_allocator> m_allocator;
};

} // namespace cuda
} // namespace xmipp4
