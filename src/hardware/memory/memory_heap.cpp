// SPDX-License-Identifier: GPL-3.0-only

#include "memory_heap.hpp"

#include "../device_guard.hpp"
#include "../error.hpp"

#include <cuda_runtime.h>

namespace xmipp4
{
namespace cuda
{

memory_heap::memory_heap(
	construction_key,
	void *data,
	std::size_t size,
	int ordinal
) noexcept
	: m_data(data)
	, m_size(size)
	, m_ordinal(ordinal)
{
}

memory_heap::~memory_heap()
{
	if (m_ordinal == no_device)
	{
		XMIPP4_CUDA_CHECK_NO_THROW( cudaFreeHost(m_data) );
		return;
	}

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

void* memory_heap::get_device_ptr() const noexcept
{
	return m_data;
}

void* memory_heap::get_host_ptr() const noexcept
{
	return m_ordinal == no_device ? m_data : nullptr;
}

std::size_t memory_heap::get_size() const noexcept
{
	return m_size;
}

std::shared_ptr<memory_heap>
memory_heap::create_device_memory(int ordinal, std::size_t size)
{
	void *data = nullptr;
	const device_guard guard(ordinal);
	XMIPP4_CUDA_CHECK( cudaMalloc(&data, size) );

	return std::make_shared<memory_heap>(
		construction_key(), data, size, ordinal
	);
}

std::shared_ptr<memory_heap> memory_heap::create_pinned_memory(std::size_t size)
{
	// Portable, so that the memory stays pinned for every device and not only
	// for whichever one happened to be current here.
	void *data = nullptr;
	XMIPP4_CUDA_CHECK(
		cudaHostAlloc(&data, size, cudaHostAllocPortable)
	);

	return std::make_shared<memory_heap>(
		construction_key(), data, size, no_device
	);
}

} // namespace cuda
} // namespace xmipp4
