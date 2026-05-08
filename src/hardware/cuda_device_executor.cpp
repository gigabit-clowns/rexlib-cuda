// SPDX-License-Identifier: GPL-3.0-only

#include <xmipp4/cuda/hardware/cuda_device_executor.hpp>

#include <xmipp4/cuda/hardware/cuda_error.hpp>
#include <xmipp4/cuda/hardware/cuda_device.hpp>

#include <utility>

namespace xmipp4
{
namespace hardware
{

cuda_device_executor::cuda_device_executor(cuda_device &device)
{
	XMIPP4_CUDA_CHECK( cudaSetDevice(device.get_index()) );
	XMIPP4_CUDA_CHECK( cudaStreamCreate(&m_stream) );
}

cuda_device_executor::cuda_device_executor(cuda_device_executor &&other) noexcept
	: m_stream(other.m_stream)
{
	other.m_stream = nullptr;
}

cuda_device_executor::~cuda_device_executor()
{
	reset();
}

cuda_device_executor&
cuda_device_executor::operator=(cuda_device_executor &&other) noexcept
{
	swap(other);
	other.reset();
	return *this;
}

void cuda_device_executor::swap(cuda_device_executor &other) noexcept
{
	std::swap(m_stream, other.m_stream);
}

void cuda_device_executor::reset() noexcept
{
	if (m_stream)
	{
		XMIPP4_CUDA_CHECK( cudaStreamDestroy(m_stream) );
	}
}


cuda_device_executor::handle cuda_device_executor::get_handle() noexcept
{
	return m_stream;
}

void cuda_device_executor::wait_until_completed() const
{
	XMIPP4_CUDA_CHECK( cudaStreamSynchronize(m_stream) );
}

bool cuda_device_executor::is_idle() const noexcept
{
	const auto code = cudaStreamQuery(m_stream);

	bool result;
	switch (code)
	{
	case cudaSuccess:
		result = true;
		break;

	case cudaErrorNotReady:
		result = false;
		break;
	
	default:
		XMIPP4_CUDA_CHECK(code);
		result = false; // To avoid warnings. The line above should throw.
		break;
	}
	return result;
}

} // namespace hardware
} // namespace xmipp4
