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
 * @brief Memory that lives on the device and cannot be reached from the host.
 */
class device_memory_resource final
	: public memory_resource
	, public std::enable_shared_from_this<device_memory_resource>
{
public:
	explicit device_memory_resource(int ordinal) noexcept;

	int get_ordinal() const noexcept;
	std::size_t get_max_alignment() const noexcept;
	std::shared_ptr<memory_heap> create_heap(std::size_t size) const;

	memory_resource_kind get_kind() const noexcept override;
	std::shared_ptr<memory_allocator> create_allocator() const override;

private:
	int m_ordinal;
	mutable std::weak_ptr<memory_allocator> m_allocator;
};

} // namespace cuda
} // namespace xmipp4
