// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "memory_resource.hpp"

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Page locked host memory, which the device can transfer to and from.
 *
 * Reports itself as unified on integrated devices, where host and device
 * share the same physical memory, and as host staging everywhere else.
 */
class pinned_memory_resource final
	: public memory_resource
{
public:
	explicit pinned_memory_resource(int ordinal);

	memory_resource_kind get_kind() const noexcept override;
	std::size_t get_max_alignment() const noexcept override;
	std::shared_ptr<memory_heap> create_heap(std::size_t size) const override;

private:
	memory_resource_kind m_kind;
};

} // namespace cuda
} // namespace xmipp4
