// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/device_backend.hpp>

namespace xmipp4
{

class device_manager;

namespace cuda
{

/**
 * @brief Implementation of the @ref xmipp4::device_backend interface to
 * expose the CUDA capable devices present in the system.
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

	/**
	 * @brief Register this backend at the provided manager.
	 *
	 * @param manager The manager where the backend is registered.
	 * @return true The backend was registered.
	 * @return false A backend with the same name was already registered.
	 */
	static bool register_at(xmipp4::device_manager &manager);
};

} // namespace cuda
} // namespace xmipp4
