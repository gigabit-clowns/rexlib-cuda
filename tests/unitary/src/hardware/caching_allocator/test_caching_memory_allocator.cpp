// SPDX-License-Identifier: GPL-3.0-only

#include <catch2/catch_test_macros.hpp>

#include <hardware/caching_allocator/caching_memory_allocator.hpp>

#include <hardware/caching_allocator/buffer.hpp>

#include "test_allocator.hpp"
#include "test_queue.hpp"

#include <config.hpp>

#include <xmipp4/core/hardware/buffer.hpp>
#include <xmipp4/core/memory/align.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

using namespace xmipp4;

namespace
{

constexpr std::size_t alignment = XMIPP4_CUDA_CACHING_ALLOCATOR_MAX_ALIGNMENT_BYTES;
constexpr std::size_t min_heap = XMIPP4_CUDA_CACHING_ALLOCATOR_MIN_HEAP_BYTES;

void* device_ptr_of(const std::shared_ptr<xmipp4::buffer> &buf)
{
	return cuda::buffer::cast(*buf).get_device_ptr();
}

} // namespace

TEST_CASE(
	"caching_memory_allocator should report what it is backed by",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;

	CHECK( &fixture.get().get_memory_resource() == &fixture.get_resource() );
	CHECK( fixture.get().get_max_alignment() == alignment );
}

TEST_CASE(
	"caching_memory_allocator should refuse an alignment it cannot promise",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	FORBID_CALL(fixture.get_source(), allocate(trompeloeil::_));

	SECTION( "when it is stricter than the maximum" )
	{
		CHECK_THROWS_AS(
			fixture.get().allocate(1024, 2 * alignment, queue),
			std::invalid_argument
		);
	}

	SECTION( "when it is not a power of two" )
	{
		CHECK_THROWS_AS(
			fixture.get().allocate(1024, 24, queue),
			std::invalid_argument
		);
	}

	SECTION( "when it is zero" )
	{
		CHECK_THROWS_AS(
			fixture.get().allocate(1024, 0, queue),
			std::invalid_argument
		);
	}
}

TEST_CASE(
	"a buffer from a caching_memory_allocator should be aligned and at least "
	"as big as asked for",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	auto buf = fixture.get().allocate(1000, alignment, queue);

	REQUIRE( buf != nullptr );
	CHECK( is_aligned(device_ptr_of(buf), alignment) );
	CHECK( &buf->get_memory_resource() == &fixture.get_resource() );

	// Rounding up is what keeps the remainder of a block usable, so the size
	// the buffer reports is the rounded one rather than what was asked for.
	CHECK( buf->get_size() == align_ceil(std::size_t(1000), alignment) );
}

TEST_CASE(
	"a buffer from a caching_memory_allocator should only offer a host "
	"pointer when the host can reach it",
	"[caching_memory_allocator]"
)
{
	const auto queue = cuda::make_test_queue(0);

	SECTION( "not when it is device local" )
	{
		cuda::allocator_fixture fixture(
			cuda::test_arena::unlimited,
			memory_resource_kind::device_local
		);

		auto buf = fixture.get().allocate(1024, alignment, queue);
		CHECK( buf->get_host_ptr() == nullptr );
		CHECK( device_ptr_of(buf) != nullptr );
	}

	SECTION( "but yes when it is page locked host memory" )
	{
		cuda::allocator_fixture fixture(
			cuda::test_arena::unlimited,
			memory_resource_kind::host_staging
		);

		auto buf = fixture.get().allocate(1024, alignment, queue);
		CHECK( buf->get_host_ptr() != nullptr );
		CHECK( buf->get_host_ptr() == device_ptr_of(buf) );
	}
}

TEST_CASE(
	"a caching_memory_allocator should stop reaching the driver once it has "
	"memory of its own",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	REQUIRE_CALL(fixture.get_source(), allocate(min_heap))
		.LR_RETURN(fixture.get_arena().take(_1));

	void *first = nullptr;
	for (std::size_t i = 0; i < 1000; ++i)
	{
		auto buf = fixture.get().allocate(1024, alignment, queue);
		if (i == 0)
		{
			first = device_ptr_of(buf);
		}

		// Handed straight back to the queue it came from, whose own work it is
		// already ordered against, so it lands in the same place every time.
		CHECK( device_ptr_of(buf) == first );
	}
}

TEST_CASE(
	"a caching_memory_allocator should cut one heap into the buffers that are "
	"alive at once",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	REQUIRE_CALL(fixture.get_source(), allocate(min_heap))
		.LR_RETURN(fixture.get_arena().take(_1));

	std::vector<std::shared_ptr<xmipp4::buffer>> buffers;
	for (std::size_t i = 0; i < 4; ++i)
	{
		buffers.push_back(fixture.get().allocate(1024, alignment, queue));
	}

	// Consecutive, since each one is cut off the front of what the last one
	// left behind.
	for (std::size_t i = 1; i < buffers.size(); ++i)
	{
		const auto previous =
			reinterpret_cast<std::uintptr_t>(device_ptr_of(buffers[i - 1]));
		const auto current =
			reinterpret_cast<std::uintptr_t>(device_ptr_of(buffers[i]));
		CHECK( current == previous + 1024 );
	}
}

TEST_CASE(
	"a caching_memory_allocator should grow by doubling",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	trompeloeil::sequence sequence;

	// The first heap is the smallest one worth taking, and each one after it
	// is as big as everything taken so far.
	REQUIRE_CALL(fixture.get_source(), allocate(min_heap))
		.LR_RETURN(fixture.get_arena().take(_1))
		.IN_SEQUENCE(sequence);
	REQUIRE_CALL(fixture.get_source(), allocate(min_heap))
		.LR_RETURN(fixture.get_arena().take(_1))
		.IN_SEQUENCE(sequence);
	REQUIRE_CALL(fixture.get_source(), allocate(2 * min_heap))
		.LR_RETURN(fixture.get_arena().take(_1))
		.IN_SEQUENCE(sequence);

	// Held, so that each request has to be served out of new memory.
	std::vector<std::shared_ptr<xmipp4::buffer>> buffers;
	buffers.push_back(fixture.get().allocate(min_heap, alignment, queue));
	buffers.push_back(fixture.get().allocate(min_heap, alignment, queue));
	buffers.push_back(fixture.get().allocate(2 * min_heap, alignment, queue));
}

TEST_CASE(
	"a caching_memory_allocator should ask for less rather than give up",
	"[caching_memory_allocator]"
)
{
	// Enough for the request, but nowhere near what growing by doubling would
	// like to take.
	cuda::allocator_fixture fixture(3 * min_heap);
	const auto queue = cuda::make_test_queue(0);

	auto held = fixture.get().allocate(2 * min_heap, alignment, queue);

	trompeloeil::sequence sequence;

	// Asks for as much again as it already holds, is refused, and halves until
	// what the device has left is enough.
	REQUIRE_CALL(fixture.get_source(), allocate(2 * min_heap))
		.THROW(std::bad_alloc())
		.IN_SEQUENCE(sequence);
	REQUIRE_CALL(fixture.get_source(), allocate(min_heap))
		.LR_RETURN(fixture.get_arena().take(_1))
		.IN_SEQUENCE(sequence);

	auto buf = fixture.get().allocate(min_heap, alignment, queue);

	REQUIRE( buf != nullptr );
	CHECK( buf->get_size() >= min_heap );
}

TEST_CASE(
	"a caching_memory_allocator should give memory back rather than fail",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture(2 * min_heap);
	const auto queue = cuda::make_test_queue(0);

	// Taken and dropped, so the allocator is holding a heap it is not using.
	fixture.get().allocate(min_heap, alignment, queue);
	REQUIRE( fixture.get().get_pool_size() == min_heap );

	// More than what is left, but not more than the device has. The only way
	// through is to hand the idle heap back first.
	auto buf = fixture.get().allocate(2 * min_heap, alignment, queue);

	REQUIRE( buf != nullptr );
	CHECK( buf->get_size() >= 2 * min_heap );
	CHECK( fixture.get_arena().get_region_count() == 1 );
}

TEST_CASE(
	"a caching_memory_allocator should give up when the device really is full",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture(min_heap);
	const auto queue = cuda::make_test_queue(0);

	// Held, so there is nothing idle to hand back.
	auto held = fixture.get().allocate(min_heap, alignment, queue);

	CHECK_THROWS_AS(
		fixture.get().allocate(min_heap, alignment, queue),
		std::bad_alloc
	);
}

TEST_CASE(
	"a caching_memory_allocator should give a large request a heap of its own",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);
	const std::size_t size = XMIPP4_CUDA_CACHING_ALLOCATOR_LARGE_ALLOCATION_BYTES;

	// Sized to the request rather than rounded up, since a heap this big would
	// otherwise sit there holding memory nothing else is small enough to use.
	REQUIRE_CALL(fixture.get_source(), allocate(size))
		.LR_RETURN(fixture.get_arena().take(_1));

	auto buf = fixture.get().allocate(size, alignment, queue);

	CHECK( buf->get_size() == size );
}

TEST_CASE(
	"a caching_memory_allocator should take a block over from another queue "
	"rather than grow",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto first = cuda::make_test_queue(0);
	const auto second = cuda::make_test_queue(1);

	// Taken by one queue and dropped, so it belongs to that queue's timeline.
	REQUIRE_CALL(fixture.get_source(), allocate(min_heap))
		.LR_RETURN(fixture.get_arena().take(_1));
	fixture.get().allocate(min_heap, alignment, first);

	// Reusing it costs a wait on the device rather than a call to the driver,
	// which is the whole point of knowing which queue a block belongs to.
	REQUIRE_CALL(fixture.get_recorder(), record(first))
		.LR_RETURN(1);
	REQUIRE_CALL(fixture.get_recorder(), enqueue_wait(second, 1u));
	FORBID_CALL(fixture.get_recorder(), wait(trompeloeil::_));

	auto buf = fixture.get().allocate(min_heap, alignment, second);

	CHECK( buf != nullptr );
}

TEST_CASE(
	"a caching_memory_allocator should not take a block over for an "
	"allocation that named no queue",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	fixture.get().allocate(min_heap, alignment, queue);

	// There is nowhere to put the wait that taking the block over would need,
	// so the only way to serve this is out of memory nothing is running
	// against.
	REQUIRE_CALL(fixture.get_source(), allocate(trompeloeil::_))
		.LR_RETURN(fixture.get_arena().take(_1));

	auto buf = fixture.get().allocate(min_heap, alignment, cuda::queue_handle());

	CHECK( buf != nullptr );
}

TEST_CASE(
	"trimming a caching_memory_allocator should give back what is idle",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto queue = cuda::make_test_queue(0);

	SECTION( "all of it when nothing is in use" )
	{
		fixture.get().allocate(1024, alignment, queue);
		REQUIRE( fixture.get().get_pool_size() == min_heap );

		CHECK( fixture.get().trim() == min_heap );
		CHECK( fixture.get().get_pool_size() == 0 );
		CHECK( fixture.get_arena().get_region_count() == 0 );
	}

	SECTION( "and none of it while a buffer is alive" )
	{
		auto held = fixture.get().allocate(1024, alignment, queue);

		CHECK( fixture.get().trim() == 0 );
		CHECK( fixture.get().get_pool_size() == min_heap );
	}
}

TEST_CASE(
	"trimming a caching_memory_allocator should merge back what the queues "
	"kept apart",
	"[caching_memory_allocator]"
)
{
	cuda::allocator_fixture fixture;
	const auto first = cuda::make_test_queue(0);
	const auto second = cuda::make_test_queue(1);

	// Two halves of one heap, dropped while belonging to different queues, so
	// neither can absorb the other.
	{
		auto a = fixture.get().allocate(min_heap / 2, alignment, first);
		auto b = fixture.get().allocate(min_heap / 2, alignment, second);
	}

	REQUIRE( fixture.get().get_pool_size() == min_heap );

	// Waiting for both queues makes the two answers to "when is this safe" the
	// same one, and the halves add back up to the heap they came from.
	REQUIRE_CALL(fixture.get_recorder(), wait(trompeloeil::_))
		.TIMES(2);

	CHECK( fixture.get().trim() == min_heap );
	CHECK( fixture.get().get_pool_size() == 0 );
}
