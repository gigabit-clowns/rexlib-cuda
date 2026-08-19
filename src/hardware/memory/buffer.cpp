// SPDX-License-Identifier: GPL-3.0-only

#include "buffer.hpp"

#include "caching_memory_allocator.hpp"
#include "memory_block.hpp"
#include "memory_heap.hpp"

#include "../command_queue.hpp"
#include "../../logger.hpp"

#include <xmipp4/core/platform/assert.hpp>

#include <algorithm>

namespace xmipp4
{
namespace cuda
{

static void* offset_pointer(void *base, std::size_t offset) noexcept
{
	return base ? static_cast<char*>(base) + offset : nullptr;
}



buffer::buffer(
	std::shared_ptr<const xmipp4::memory_resource> resource,
	std::shared_ptr<block_recycler> recycler,
	memory_block &block
) noexcept
	: m_resource(std::move(resource))
	, m_recycler(std::move(recycler))
	, m_block(&block)
{
}

buffer::~buffer()
{
	XMIPP4_ASSERT( m_recycler );
	try
	{
		m_recycler->recycle_block(
			*m_block,
			span<command_queue *const>(m_queues.data(), m_queues.size())
		);
	}
	catch (const std::exception &e)
	{
		log_error(e.what());
	}
}

void* buffer::get_host_ptr() noexcept
{
	return offset_pointer(
		m_block->get_heap()->get_host_ptr(),
		m_block->get_offset()
	);
}

const void* buffer::get_host_ptr() const noexcept
{
	return offset_pointer(
		m_block->get_heap()->get_host_ptr(),
		m_block->get_offset()
	);
}

std::size_t buffer::get_size() const noexcept
{
	return m_block->get_size();
}

const xmipp4::memory_resource& buffer::get_memory_resource() const noexcept
{
	XMIPP4_ASSERT( m_resource );
	return *m_resource;
}

void* buffer::get_device_ptr() const noexcept
{
	return offset_pointer(
		m_block->get_heap()->get_device_ptr(),
		m_block->get_offset()
	);
}

void buffer::record_queue(command_queue &queue)
{
	// The queue the range was allocated for orders its own work, so only the
	// others have to be waited for.
	if (&queue == m_block->get_queue())
	{
		return;
	}

	const auto ite = std::find(m_queues.cbegin(), m_queues.cend(), &queue);
	if (ite == m_queues.cend())
	{
		m_queues.push_back(&queue);
	}
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
