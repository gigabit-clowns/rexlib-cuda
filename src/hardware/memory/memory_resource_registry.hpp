// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <map>
#include <memory>
#include <mutex>

namespace xmipp4
{
namespace cuda
{

class device_memory_resource;
class pinned_memory_resource;

/**
 * @brief Hands out the memory resources of each device, one per ordinal.
 *
 * Two device handles on the same GPU must share their resources, or the
 * caching allocator behind them would cache the same memory twice. The
 * registry only keeps weak references, so the resources live exactly as long
 * as the devices, allocators and buffers using them, and free their memory
 * while the CUDA runtime is still up.
 */
class memory_resource_registry
{
public:
	std::shared_ptr<const device_memory_resource> get_device_memory(int ordinal);
	std::shared_ptr<const pinned_memory_resource> get_pinned_memory(int ordinal);

private:
	std::mutex m_mutex;
	std::map<int, std::weak_ptr<device_memory_resource>> m_device_memory;
	std::map<int, std::weak_ptr<pinned_memory_resource>> m_pinned_memory;
};

} // namespace cuda
} // namespace xmipp4
