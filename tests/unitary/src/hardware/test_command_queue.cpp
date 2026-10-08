// SPDX-License-Identifier: GPL-3.0-only

#include <catch2/catch_test_macros.hpp>

#include <hardware/command_queue.hpp>

#include <hardware/pooled_event_recorder.hpp>
#include <hardware/stream_timeline.hpp>

#include "mock/mock_command_queue.hpp"
#include "test_driver.hpp"
#include "test_stream_blocker.hpp"

#include <rexlib/core/hardware/command.hpp>
#include <rexlib/core/hardware/command_token.hpp>

#include <memory>
#include <stdexcept>

using namespace rexlib;

TEST_CASE(
	"a command_queue should refuse to be built without an event recorder",
	"[command_queue]"
)
{
	CHECK_THROWS_AS(
		cuda::command_queue(0, nullptr),
		std::invalid_argument
	);
}

TEST_CASE(
	"a command_queue should own a stream on its device",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto recorder = std::make_shared<cuda::pooled_event_recorder>(0);
	const cuda::command_queue first(0, recorder);
	const cuda::command_queue second(0, recorder);

	CHECK( first.get_ordinal() == 0 );
	CHECK( first.get_handle() != nullptr );
	CHECK( first.get_handle() != second.get_handle() );
}

TEST_CASE(
	"a command_queue should have a timeline of its own",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto recorder = std::make_shared<cuda::pooled_event_recorder>(0);
	const cuda::command_queue first(0, recorder);
	const cuda::command_queue second(0, recorder);

	REQUIRE( first.get_timeline() != nullptr );
	CHECK( first.get_timeline() != second.get_timeline() );
}

TEST_CASE(
	"the timeline of a command_queue should follow its stream",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto recorder = std::make_shared<cuda::pooled_event_recorder>(0);
	const cuda::command_queue queue(0, recorder);
	const auto &timeline = queue.get_timeline();

	cuda::stream_blocker blocker(queue.get_handle());
	const command_token token(
		timeline,
		timeline->record(queue.get_handle())
	);

	CHECK_FALSE( token.is_complete() );

	blocker.release();
	token.wait();

	CHECK( token.is_complete() );
}

TEST_CASE(
	"the timeline of a command_queue should outlive it",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	command_token token;
	{
		const cuda::command_queue queue(
			0,
			std::make_shared<cuda::pooled_event_recorder>(0)
		);
		const auto &timeline = queue.get_timeline();
		token = command_token(
			timeline,
			timeline->record(queue.get_handle())
		);
	}

	// An array may hold the token of a command long after the queue that
	// ran it is gone.
	CHECK_NOTHROW( token.wait() );
	CHECK( token.is_complete() );
}

TEST_CASE(
	"synchronizing a command_queue should wait for its stream",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const cuda::command_queue queue(
		0,
		std::make_shared<cuda::pooled_event_recorder>(0)
	);
	const auto &timeline = queue.get_timeline();
	const command_token token(
		timeline,
		timeline->record(queue.get_handle())
	);

	queue.synchronize();

	CHECK( token.is_complete() );
}

TEST_CASE(
	"a command_queue should refuse a command without a program",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	cuda::command_queue queue(
		0,
		std::make_shared<cuda::pooled_event_recorder>(0)
	);

	CHECK_THROWS_AS( queue.submit(command()), std::invalid_argument );
}

TEST_CASE(
	"casting a command_queue should reject one this backend did not make",
	"[command_queue]"
)
{
	if (!cuda::has_test_device())
	{
		SKIP( "No CUDA capable device is available." );
	}

	cuda::command_queue queue(
		0,
		std::make_shared<cuda::pooled_event_recorder>(0)
	);
	rexlib::command_queue &base = queue;

	CHECK( &cuda::command_queue::cast(base) == &queue );

	cuda::mock_command_queue foreign;
	CHECK_THROWS_AS(
		cuda::command_queue::cast(foreign),
		std::invalid_argument
	);

	const rexlib::command_queue &const_foreign = foreign;
	CHECK_THROWS_AS(
		cuda::command_queue::cast(const_foreign),
		std::invalid_argument
	);
}
