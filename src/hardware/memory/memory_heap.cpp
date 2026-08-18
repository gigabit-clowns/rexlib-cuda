// SPDX-License-Identifier: GPL-3.0-only

#include "memory_heap.hpp"

#include "../device_guard.hpp"
#include "../error.hpp"

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

memory_heap::~memory_heap() = default;



device_memory_heap::device_memory_heap(int ordinal, std::size_t size)
	: m_data(nullptr)
	, m_size(size)
	, m_ordinal(ordinal)
{
	const device_guard guard(ordinal);
	XMIPP4_CUDA_CHECK( cudaMalloc(&m_data, size) );
}

device_memory_heap::~device_memory_heap()
{
	// device_guard is not usable here, as it reports failures by throwing.
	int previous = m_ordinal;
	const bool restore =
		cudaGetDevice(&previous) == cudaSuccess && previous != m_ordinal;
	if (restore)
	{
		XMIPP4_CUDA_CHECK_NO_THROW( cudaSetDevice(m_ordinal) );
	}

	XMIPP4_CUDA_CHECK_NO_THROW( cudaFree(m_data) );

	if (restore)
	{
		XMIPP4_CUDA_CHECK_NO_THROW( cudaSetDevice(previous) );
	}
}

void* device_memory_heap::get_device_ptr() const noexcept
{
	return m_data;
}

void* device_memory_heap::get_host_ptr() const noexcept
{
	return nullptr;
}

std::size_t device_memory_heap::get_size() const noexcept
{
	return m_size;
}



pinned_memory_heap::pinned_memory_heap(int ordinal, std::size_t size)
	: m_data(nullptr)
	, m_size(size)
{
	const device_guard guard(ordinal);
	XMIPP4_CUDA_CHECK( cudaHostAlloc(&m_data, size, cudaHostAllocDefault) );
}

pinned_memory_heap::~pinned_memory_heap()
{
	// Pinned memory is process wide, so no device needs to be selected.
	XMIPP4_CUDA_CHECK_NO_THROW( cudaFreeHost(m_data) );
}

void* pinned_memory_heap::get_device_ptr() const noexcept
{
	// Unified addressing makes the host pointer valid on the device too.
	return m_data;
}

void* pinned_memory_heap::get_host_ptr() const noexcept
{
	return m_data;
}

std::size_t pinned_memory_heap::get_size() const noexcept
{
	return m_size;
}

} // namespace cuda
} // namespace xmipp4
