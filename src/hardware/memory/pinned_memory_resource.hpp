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
 * @brief Page locked host memory, which every device can transfer to and from.
 *
 * Not tied to any device: the allocations are portable across contexts, so a
 * single resource serves the whole process. Reports itself as unified where
 * host and device share the same physical memory, and as host staging
 * everywhere else.
 */
class pinned_memory_resource final
	: public memory_resource
	, public std::enable_shared_from_this<pinned_memory_resource>
{
public:
	std::size_t get_max_alignment() const noexcept;
	std::shared_ptr<memory_heap> create_heap(std::size_t size) const;

	memory_resource_kind get_kind() const noexcept override;
	std::shared_ptr<memory_allocator> create_allocator() const override;

private:
	mutable std::weak_ptr<memory_allocator> m_allocator;
};

} // namespace cuda
} // namespace xmipp4
