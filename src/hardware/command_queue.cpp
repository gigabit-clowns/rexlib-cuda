// SPDX-License-Identifier: GPL-3.0-only

#include "command_queue.hpp"

#include "device_guard.hpp"
#include "error.hpp"
#include "event.hpp"

#include <xmipp4/core/exceptions/invalid_operation_error.hpp>
#include <xmipp4/core/hardware/command.hpp>

#include <stdexcept>

namespace xmipp4
{
namespace cuda
{

command_queue::command_queue(int ordinal)
	: m_stream(nullptr)
	, m_ordinal(ordinal)
{
	const device_guard guard(ordinal);
	XMIPP4_CUDA_CHECK(
		cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking)
	);
}

command_queue::~command_queue()
{
	// cudaStreamDestroy acts on the device that owns the stream, so it does
	// not need the current device to be changed. It returns immediately and
	// the stream is released once the pending work completes.
	XMIPP4_CUDA_CHECK_NO_THROW( cudaStreamDestroy(m_stream) );
}

command_queue::handle command_queue::get_handle() const noexcept
{
	return m_stream;
}

int command_queue::get_ordinal() const noexcept
{
	return m_ordinal;
}

void command_queue::synchronize() const
{
	XMIPP4_CUDA_CHECK( cudaStreamSynchronize(m_stream) );
}

void command_queue::submit(const command &cmd)
{
	if (!cmd.get_program())
	{
		throw std::invalid_argument(
			"command_queue::submit: Provided command does not have an "
			"associated program."
		);
	}

	// TODO: Execute the program and record this queue on every buffer bound
	// to the command, so that the allocator can defer their release. The
	// CUDA backend does not provide any program yet.
	throw invalid_operation_error(
		"The CUDA backend does not implement any program yet."
	);
}

void command_queue::signal(xmipp4::event &ev)
{
	auto &cuda_event = event::cast(ev);
	XMIPP4_CUDA_CHECK(
		cudaEventRecord(cuda_event.get_handle(), m_stream)
	);
}

void command_queue::wait(const xmipp4::event &ev)
{
	const auto &cuda_event = event::cast(ev);
	XMIPP4_CUDA_CHECK(
		cudaStreamWaitEvent(
			m_stream,
			cuda_event.get_handle(),
			cudaEventWaitDefault
		)
	);
}

} // namespace cuda
} // namespace xmipp4
