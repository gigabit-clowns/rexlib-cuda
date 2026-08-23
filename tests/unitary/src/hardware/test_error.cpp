// SPDX-License-Identifier: GPL-3.0-only

#include <catch2/catch_test_macros.hpp>

#include <hardware/error.hpp>

#include <new>
#include <string>

using namespace xmipp4;

TEST_CASE(
	"checking a CUDA return code should only throw on failure",
	"[error]"
)
{
	CHECK_NOTHROW( XMIPP4_CUDA_CHECK(cudaSuccess) );
	CHECK_THROWS_AS(
		XMIPP4_CUDA_CHECK(cudaErrorInvalidValue),
		cuda::error
	);
}

TEST_CASE(
	"a CUDA error should say what failed and where",
	"[error]"
)
{
	try
	{
		XMIPP4_CUDA_CHECK(cudaErrorInvalidValue);
		FAIL( "The check should have thrown." );
	}
	catch (const cuda::error &e)
	{
		const std::string message = e.what();

		// The call is what says which of the many in a function failed, and
		// the driver's own wording is what says why.
		CHECK( message.find("cudaErrorInvalidValue") != std::string::npos );
		CHECK(
			message.find(cudaGetErrorString(cudaErrorInvalidValue)) !=
			std::string::npos
		);
		CHECK( message.find(__FILE__) != std::string::npos );
	}
}

TEST_CASE(
	"checking an allocation should report running out of memory as such",
	"[error]"
)
{
	CHECK_NOTHROW( XMIPP4_CUDA_CHECK_ALLOCATION(cudaSuccess) );

	// The allocators tell "the device has no memory left", which they answer
	// by asking for less or by giving back what they are holding, apart from
	// every other failure, by its type alone.
	CHECK_THROWS_AS(
		XMIPP4_CUDA_CHECK_ALLOCATION(cudaErrorMemoryAllocation),
		std::bad_alloc
	);

	SECTION( "and anything else as an ordinary error" )
	{
		CHECK_THROWS_AS(
			XMIPP4_CUDA_CHECK_ALLOCATION(cudaErrorInvalidValue),
			cuda::error
		);

		// Which is not a bad_alloc, so it is not answered by retrying smaller.
		CHECK_THROWS_AS(
			XMIPP4_CUDA_CHECK_ALLOCATION(cudaErrorInvalidValue),
			std::runtime_error
		);
	}
}

TEST_CASE(
	"checking a CUDA return code without throwing should never throw",
	"[error]"
)
{
	// Meant for destructors, where there is nowhere left to report a failure
	// to and throwing would end the process.
	CHECK_NOTHROW( XMIPP4_CUDA_CHECK_NO_THROW(cudaSuccess) );
	CHECK_NOTHROW( XMIPP4_CUDA_CHECK_NO_THROW(cudaErrorInvalidValue) );
	CHECK_NOTHROW( XMIPP4_CUDA_CHECK_NO_THROW(cudaErrorMemoryAllocation) );
}
