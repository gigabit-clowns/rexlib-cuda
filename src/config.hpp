// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#ifndef XMIPP4_CUDA_COALESCE_ALIGN_BYTES
	#define XMIPP4_CUDA_COALESCE_ALIGN_BYTES 256UL
#endif

/// Alignment that the CUDA allocation entry points guarantee for whatever they
/// return. 
#ifndef XMIPP4_CUDA_ALLOCATION_ALIGN_BYTES
	#define XMIPP4_CUDA_ALLOCATION_ALIGN_BYTES 256UL
#endif

/// Strictest alignment the caching allocators can satisfy. Every allocation
/// size is rounded up to it, so that splitting a block never leaves the
/// remainder misaligned. Raising it above what the driver guarantees makes the
/// heaps over-allocate and align their base by hand.
#ifndef XMIPP4_CUDA_MAX_ALIGNMENT_BYTES
	#define XMIPP4_CUDA_MAX_ALIGNMENT_BYTES 256UL
#endif

/// Smallest heap the pool ever requests from the driver. A first allocation of
/// a handful of bytes still pays this, which is what keeps the following
/// thousand from reaching the driver at all.
#ifndef XMIPP4_CUDA_MIN_HEAP_BYTES
	#define XMIPP4_CUDA_MIN_HEAP_BYTES (2UL << 20)
#endif

/// Largest heap the pool ever requests from the driver. The pool keeps growing
/// past it by adding heaps, rather than by asking for one huge contiguous
/// region the driver is increasingly unlikely to find.
#ifndef XMIPP4_CUDA_MAX_HEAP_BYTES
	#define XMIPP4_CUDA_MAX_HEAP_BYTES (1UL << 30)
#endif

/// Allocations of at least this size get a heap of their own, sized exactly to
/// the request. They are never split, so the whole heap goes back to the driver
/// as soon as the allocation is released and the pool is trimmed.
#ifndef XMIPP4_CUDA_LARGE_ALLOCATION_BYTES
	#define XMIPP4_CUDA_LARGE_ALLOCATION_BYTES (4UL << 30)
#endif

/// Number of command queues a buffer can be used on before its record of them
/// stops fitting inline and reaches the heap.
#ifndef XMIPP4_CUDA_SMALL_QUEUE_COUNT
	#define XMIPP4_CUDA_SMALL_QUEUE_COUNT 16UL
#endif
