// SPDX-License-Identifier: GPL-3.0-only

#include "memory_resource.hpp"

#include "direct_memory_allocator.hpp"

namespace xmipp4
{
namespace cuda
{

memory_resource::memory_resource(int ordinal) noexcept
	: m_ordinal(ordinal)
{
}

memory_resource::~memory_resource() = default;

int memory_resource::get_ordinal() const noexcept
{
	return m_ordinal;
}

std::shared_ptr<memory_allocator> memory_resource::create_allocator() const
{
	// One allocator per resource, so that a device draws from a single pool.
	auto result = m_allocator.lock();
	if (!result)
	{
		result = std::make_shared<direct_memory_allocator>(
			std::static_pointer_cast<const memory_resource>(shared_from_this())
		);
		m_allocator = result;
	}

	return result;
}

} // namespace cuda
} // namespace xmipp4
