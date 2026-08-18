// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "memory_resource.hpp"

namespace xmipp4
{
namespace cuda
{

/**
 * @brief Memory that lives on the device and cannot be reached from the host.
 */
class device_memory_resource final
	: public memory_resource
{
public:
	using memory_resource::memory_resource;

	memory_resource_kind get_kind() const noexcept override;
	std::size_t get_max_alignment() const noexcept override;
	std::shared_ptr<memory_heap> create_heap(std::size_t size) const override;
};

} // namespace cuda
} // namespace xmipp4
