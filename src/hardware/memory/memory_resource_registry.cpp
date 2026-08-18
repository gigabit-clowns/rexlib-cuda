// SPDX-License-Identifier: GPL-3.0-only

#include "memory_resource_registry.hpp"

#include "device_memory_resource.hpp"
#include "pinned_memory_resource.hpp"

namespace xmipp4
{
namespace cuda
{

template <typename T, typename M>
static std::shared_ptr<const T> get_or_create(M &map, int ordinal)
{
	auto &slot = map[ordinal];
	auto result = slot.lock();
	if (!result)
	{
		result = std::make_shared<T>(ordinal);
		slot = result;
	}

	return result;
}

std::shared_ptr<const device_memory_resource>
memory_resource_registry::get_device_memory(int ordinal)
{
	const std::lock_guard<std::mutex> lock(m_mutex);
	return get_or_create<device_memory_resource>(m_device_memory, ordinal);
}

std::shared_ptr<const pinned_memory_resource>
memory_resource_registry::get_pinned_memory(int ordinal)
{
	const std::lock_guard<std::mutex> lock(m_mutex);
	return get_or_create<pinned_memory_resource>(m_pinned_memory, ordinal);
}

} // namespace cuda
} // namespace xmipp4
