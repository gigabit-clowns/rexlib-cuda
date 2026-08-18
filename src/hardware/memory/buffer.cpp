// SPDX-License-Identifier: GPL-3.0-only

#include "buffer.hpp"

#include "memory_heap.hpp"
#include "memory_resource.hpp"

#include <xmipp4/core/platform/assert.hpp>

namespace xmipp4
{
namespace cuda
{

static void* offset_pointer(void *base, std::size_t offset) noexcept
{
	return base ? static_cast<std::byte*>(base) + offset : nullptr;
}



buffer::buffer(
	std::shared_ptr<const memory_resource> resource,
	std::shared_ptr<memory_heap> heap,
	std::size_t offset,
	std::size_t size
) noexcept
	: m_resource(std::move(resource))
	, m_heap(std::move(heap))
	, m_offset(offset)
	, m_size(size)
{
}

buffer::~buffer() = default;

void* buffer::get_host_ptr() noexcept
{
	return offset_pointer(m_heap->get_host_ptr(), m_offset);
}

const void* buffer::get_host_ptr() const noexcept
{
	return offset_pointer(m_heap->get_host_ptr(), m_offset);
}

std::size_t buffer::get_size() const noexcept
{
	return m_size;
}

const xmipp4::memory_resource& buffer::get_memory_resource() const noexcept
{
	XMIPP4_ASSERT( m_resource );
	return *m_resource;
}

void* buffer::get_device_ptr() const noexcept
{
	return offset_pointer(m_heap->get_device_ptr(), m_offset);
}

buffer* buffer::try_cast(xmipp4::buffer &buf) noexcept
{
	return dynamic_cast<buffer*>(&buf);
}

const buffer* buffer::try_cast(const xmipp4::buffer &buf) noexcept
{
	return dynamic_cast<const buffer*>(&buf);
}

} // namespace cuda
} // namespace xmipp4
