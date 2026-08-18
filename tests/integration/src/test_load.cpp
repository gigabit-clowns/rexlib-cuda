// SPDX-License-Identifier: GPL-3.0-only


#include <catch2/catch_test_macros.hpp>

#include <xmipp4/core/exceptions/invalid_operation_error.hpp>
#include <xmipp4/core/hardware/buffer.hpp>
#include <xmipp4/core/hardware/command.hpp>
#include <xmipp4/core/hardware/command_queue.hpp>
#include <xmipp4/core/hardware/device.hpp>
#include <xmipp4/core/hardware/device_backend.hpp>
#include <xmipp4/core/hardware/device_manager.hpp>
#include <xmipp4/core/hardware/device_properties.hpp>
#include <xmipp4/core/hardware/device_type.hpp>
#include <xmipp4/core/hardware/device_session.hpp>
#include <xmipp4/core/hardware/event.hpp>
#include <xmipp4/core/hardware/memory_allocator.hpp>
#include <xmipp4/core/hardware/memory_resource.hpp>
#include <xmipp4/core/hardware/memory_resource_kind.hpp>
#include <xmipp4/core/platform/operating_system.h>
#include <xmipp4/core/plugin.hpp>
#include <xmipp4/core/plugin_manager.hpp>
#include <xmipp4/core/service_catalog.hpp>

#include <algorithm>
#include <stdexcept>

using namespace xmipp4;


static std::string get_cuda_plugin_path()
{
	#if XMIPP4_WINDOWS
		return "xmipp4-cuda.dll";
	#elif XMIPP4_LINUX
		return "./libxmipp4-cuda.so";
	#elif XMIPP4_APPLE
		return "./libxmipp4-cuda.dylib";
	#else
		#error "Unknown platform"
	#endif
}

TEST_CASE( "load and register xmipp4-cuda plugin", "[cuda]" )
{
	plugin_manager manager;

	const auto* cuda_plugin =
		manager.load_plugin(get_cuda_plugin_path());

	REQUIRE( cuda_plugin != nullptr );
	REQUIRE( cuda_plugin->get_name() == "xmipp4-cuda" );
}

TEST_CASE( "register the CUDA device backend", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();
	REQUIRE( device_manager != nullptr );

	std::vector<std::string> names;
	device_manager->enumerate_backends(names);
	REQUIRE(
		std::find(names.cbegin(), names.cend(), "cuda") != names.cend()
	);

	auto *backend = device_manager->get_backend("cuda");
	REQUIRE( backend != nullptr );
	REQUIRE( backend->get_version().get_major() >= 11 );
}

TEST_CASE( "enumerate the CUDA devices", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();
	auto *backend = device_manager->get_backend("cuda");
	REQUIRE( backend != nullptr );

	// Enumeration must succeed even when no CUDA capable device or driver is
	// present, which is the case on the CI runners.
	std::vector<std::size_t> ids;
	REQUIRE_NOTHROW( backend->enumerate_devices(ids) );

	for (const auto id : ids)
	{
		device_properties properties;
		REQUIRE( backend->get_device_properties(id, properties) );
		REQUIRE_FALSE( properties.get_name().empty() );
		REQUIRE( properties.get_total_memory_bytes() > 0 );
		REQUIRE(
			(properties.get_type() == device_type::gpu ||
			 properties.get_type() == device_type::integrated_gpu)
		);

		REQUIRE( backend->create_device(id) != nullptr );
	}

	// Unknown ids are rejected regardless of whether there is any device.
	device_properties properties;
	REQUIRE_FALSE( backend->get_device_properties(ids.size(), properties) );
}

TEST_CASE( "synchronize a CUDA queue through an event", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();
	auto *backend = device_manager->get_backend("cuda");
	REQUIRE( backend != nullptr );

	std::vector<std::size_t> ids;
	backend->enumerate_devices(ids);
	if (ids.empty())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto dev = backend->create_device(ids.front());
	REQUIRE( dev != nullptr );

	const auto queue = dev->create_command_queue();
	REQUIRE( queue != nullptr );

	const auto ev = dev->create_event(event_usage_flag_bits::host_wait);
	REQUIRE( ev != nullptr );

	// An event that has never been recorded starts out signaled.
	REQUIRE( ev->is_signaled() );
	REQUIRE_NOTHROW( ev->wait() );

	// Recording on an idle queue completes right away, and the queue can be
	// made to wait on the event it just signaled.
	REQUIRE_NOTHROW( queue->signal(*ev) );
	REQUIRE_NOTHROW( ev->wait() );
	REQUIRE( ev->is_signaled() );
	REQUIRE_NOTHROW( queue->wait(*ev) );

	// Queues reject events that this backend did not create.
	class foreign_event final : public xmipp4::event
	{
	public:
		event_usage_flags get_supported_usage() const noexcept override
		{
			return {};
		}
		void wait() const override {}
		bool is_signaled() const override { return true; }
	};

	foreign_event other;
	REQUIRE_THROWS_AS( queue->signal(other), std::invalid_argument );
}

TEST_CASE( "the CUDA backend has no program to submit yet", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();
	auto *backend = device_manager->get_backend("cuda");
	REQUIRE( backend != nullptr );

	std::vector<std::size_t> ids;
	backend->enumerate_devices(ids);
	if (ids.empty())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto dev = backend->create_device(ids.front());

	// Submitting requires a program, and this backend provides none.
	const auto queue = dev->create_command_queue();
	REQUIRE_THROWS_AS( queue->submit(command()), std::invalid_argument );
}

TEST_CASE( "expose the CUDA memory resources", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();
	auto *backend = device_manager->get_backend("cuda");
	REQUIRE( backend != nullptr );

	std::vector<std::size_t> ids;
	backend->enumerate_devices(ids);
	if (ids.empty())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto dev = backend->create_device(ids.front());

	const auto &device_memory =
		dev->get_memory_resource(memory_resource_affinity::device);
	REQUIRE( device_memory.get_kind() == memory_resource_kind::device_local );
	REQUIRE( is_device_accessible(device_memory.get_kind()) );

	const auto &host_memory =
		dev->get_memory_resource(memory_resource_affinity::host);
	REQUIRE( is_host_accessible(host_memory.get_kind()) );

	// Two handles on the same device must share their resources, or the
	// allocators behind them would cache the same memory twice.
	const auto other = backend->create_device(ids.front());
	REQUIRE(
		&other->get_memory_resource(memory_resource_affinity::device) ==
		&device_memory
	);

	// A resource hands out one allocator, not one per request.
	const auto allocator = device_memory.create_allocator();
	REQUIRE( allocator != nullptr );
	REQUIRE( device_memory.create_allocator() == allocator );
	REQUIRE( &allocator->get_memory_resource() == &device_memory );
}

TEST_CASE( "allocate on a CUDA device", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();
	auto *backend = device_manager->get_backend("cuda");
	REQUIRE( backend != nullptr );

	std::vector<std::size_t> ids;
	backend->enumerate_devices(ids);
	if (ids.empty())
	{
		SKIP( "No CUDA capable device is available." );
	}

	const auto dev = backend->create_device(ids.front());

	SECTION( "device local memory is not reachable from the host" )
	{
		const auto allocator =
			dev->get_memory_resource(memory_resource_affinity::device)
			   .create_allocator();

		const auto buf = allocator->allocate(1024, 256, nullptr);
		REQUIRE( buf != nullptr );
		REQUIRE( buf->get_size() >= 1024 );
		REQUIRE( buf->get_host_ptr() == nullptr );
	}

	SECTION( "pinned memory can be written through" )
	{
		const auto allocator =
			dev->get_memory_resource(memory_resource_affinity::host)
			   .create_allocator();

		const auto buf = allocator->allocate(1024, 256, nullptr);
		REQUIRE( buf != nullptr );
		REQUIRE( buf->get_size() >= 1024 );

		auto *data = static_cast<unsigned char*>(buf->get_host_ptr());
		REQUIRE( data != nullptr );
		data[0] = 42;
		data[1023] = 7;
		REQUIRE( data[0] == 42 );
		REQUIRE( data[1023] == 7 );
	}

	SECTION( "invalid alignments are rejected" )
	{
		const auto allocator =
			dev->get_memory_resource(memory_resource_affinity::device)
			   .create_allocator();

		REQUIRE_THROWS_AS( allocator->allocate(16, 3, nullptr), std::invalid_argument );
		REQUIRE_THROWS_AS(
			allocator->allocate(16, allocator->get_max_alignment() * 2, nullptr),
			std::invalid_argument
		);
	}
}

TEST_CASE( "build a device session on a CUDA device", "[cuda]" )
{
	plugin_manager manager;
	REQUIRE( manager.load_plugin(get_cuda_plugin_path()) != nullptr );

	service_catalog catalog;
	register_all_plugins_at(manager, catalog);

	const auto device_manager = catalog.get_service_manager<xmipp4::device_manager>();

	std::vector<device_index> indices;
	device_manager->enumerate_devices(indices);

	const auto ite = std::find_if(
		indices.cbegin(), indices.cend(),
		[] (const device_index &index)
		{
			return index.get_backend_name() == "cuda";
		}
	);
	if (ite == indices.cend())
	{
		SKIP( "No CUDA capable device is available." );
	}

	// This is the entry point the rest of the framework goes through, and it
	// needs both allocators and a default queue to succeed.
	const auto session = device_manager->create_device_session(*ite);
	REQUIRE( session != nullptr );
	REQUIRE( session->get_device() != nullptr );
	REQUIRE( session->get_default_queue() != nullptr );
	REQUIRE( session->get_allocator(memory_resource_affinity::device) != nullptr );
	REQUIRE( session->get_allocator(memory_resource_affinity::host) != nullptr );
	REQUIRE_FALSE( session->get_properties().get_name().empty() );

	const auto buf = session
		->get_allocator(memory_resource_affinity::device)
		->allocate(4096, 256, session->get_default_queue().get());
	REQUIRE( buf != nullptr );
	REQUIRE( buf->get_size() >= 4096 );
}
