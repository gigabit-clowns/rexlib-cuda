// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_allocator.hpp>

#include <memory>

namespace xmipp4
{
namespace cuda
{

class memory_resource;

/**
 * @brief Allocator that asks the driver for one heap per buffer.
 *
 * Every allocation reaches the driver and every release returns to it, which
 * makes it a poor fit for anything allocating in a loop. It exists as the
 * straightforward implementation to compare the caching allocator against.
 */
class direct_memory_allocator final
	: public memory_allocator
{
public:
	explicit direct_memory_allocator(
		std::shared_ptr<const memory_resource> resource
	) noexcept;
	~direct_memory_allocator() override;

	const xmipp4::memory_resource& get_memory_resource() const noexcept override;

	std::size_t get_max_alignment() const noexcept override;

	std::shared_ptr<xmipp4::buffer> allocate(
		std::size_t size,
		std::size_t alignment,
		command_queue *queue_hint
	) override;

private:
	std::shared_ptr<const memory_resource> m_resource;
};

} // namespace cuda
} // namespace xmipp4
