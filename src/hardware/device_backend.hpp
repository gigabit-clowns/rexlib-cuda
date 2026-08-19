// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/device_backend.hpp>

#include <map>
#include <memory>
#include <mutex>

namespace xmipp4
{

class device_manager;

namespace cuda
{

class device_memory_resource;
class pinned_memory_resource;

/**
 * @brief Implementation of the @ref xmipp4::device_backend interface to
 * retrieve the CUDA capable devices.
 */
class device_backend final
	: public xmipp4::device_backend
{
public:
	std::string get_name() const override;
	version get_version() const override;

	void enumerate_devices(std::vector<std::size_t> &ids) const override;

	bool get_device_properties(
		std::size_t id,
		device_properties &desc
	) const override;

	std::shared_ptr<xmipp4::device> create_device(std::size_t id) const override;

	static bool register_at(xmipp4::device_manager &manager);

private:
	// Every handle on a device shares its resources, or the allocators behind
	// them would each hold their own cache of the same memory.
	mutable std::mutex m_mutex;
	mutable std::map<int, std::shared_ptr<device_memory_resource>>
		m_device_memory;
	mutable std::shared_ptr<pinned_memory_resource> m_pinned_memory;
};

} // namespace cuda
} // namespace xmipp4
