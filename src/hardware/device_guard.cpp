// SPDX-License-Identifier: GPL-3.0-only

#include "device_guard.hpp"

#include "error.hpp"

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

device_guard::device_guard(int ordinal)
	: m_previous(ordinal)
{
	XMIPP4_CUDA_CHECK( cudaGetDevice(&m_previous) );

	if (m_previous != ordinal)
	{
		XMIPP4_CUDA_CHECK( cudaSetDevice(ordinal) );
	}
}

device_guard::~device_guard()
{
	int current;
	if (cudaGetDevice(&current) == cudaSuccess && current != m_previous)
	{
		XMIPP4_CUDA_CHECK_NO_THROW( cudaSetDevice(m_previous) );
	}
}

} // namespace cuda
} // namespace xmipp4
