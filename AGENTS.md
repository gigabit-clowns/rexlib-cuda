# xmipp4-cuda — AI Rules & Project Guide
> Reference document for LLM assistants working on this codebase.
> Read this first before making any changes.
> If any change makes this file obsolete or inaccurate, update it to keep it synced.

## TL;DR (for assistants)
- This is a **runtime-loaded plugin** for [xmipp4-core](https://github.com/gigabit-clowns/xmipp4-core). It has **no public headers** and exports exactly one symbol, `xmipp4_get_plugin`.
- **xmipp4-core is the style authority.** Before finishing any change, audit your comments against §5. Unnecessary comments are the single most common review finding on this repo.
- Types live in `namespace xmipp4::cuda`. Files carry **no `cuda_` prefix** — the namespace already says it.
- The reference for "how a backend is written today" is the core's own CPU backend: `src/backends/cpu/hardware/` and `src/core/hardware/host_memory/`. Mirror it.
- **No `.cu` files exist and none are planned yet.** Everything reaches the GPU through the runtime API, which is plain C++. See §9 before adding device code.
- The plugin is **mid-rework**. `command_queue::submit` throws: there are no programs yet. See §10 for what is done and what is not.
- Boost (`intrusive`, `unordered`, `container`) comes through `cmake/modules/fetch_boost.cmake`, copied from the core. Keep the version in step with it.

---
## 1. Project Overview

**xmipp4-cuda** exposes NVIDIA GPUs to the xmipp4 framework. The core discovers it at runtime through its plugin manager, and everything the plugin offers is reached through interfaces the core declares.

The plugin registers a single service: a `device_backend` named `"cuda"`. From there the framework enumerates devices, builds a `device_session`, and allocates through it.

```
core                                    plugin
────────────────────────────────────────────────────────────────
plugin_manager::load_plugin  ────────>  xmipp4_get_plugin
plugin::register_at          ────────>  cuda::device_backend
device_manager
  ::create_device_session    ────────>  cuda::device_backend::create_device
                                          └─> cuda::device
  memory_allocator_table     ────────>  device::get_memory_resource(affinity)
                                          └─> cuda::{device,pinned}_memory_resource
                                                └─> create_allocator()
```

`create_device_session` is the entry point that matters: if it throws, the plugin is unusable no matter what else works.

---
## 2. Architecture

```
src/
├── config.hpp                  # "Magic" constants, mirrors the core's config.hpp
├── plugin.{hpp,cpp}            # Registers device_backend and nothing else
├── plugin_hook.cpp             # The one exported symbol
└── hardware/
    ├── error.{hpp,cpp}         # cuda::error, XMIPP4_CUDA_CHECK[_NO_THROW]
    ├── device_guard.{hpp,cpp}  # RAII cudaSetDevice
    ├── device_backend.{hpp,cpp}
    ├── device.{hpp,cpp}
    ├── command_queue.{hpp,cpp} # cudaStream_t
    ├── event.{hpp,cpp}         # cudaEvent_t
    └── memory/
        ├── memory_heap.{hpp,cpp}                 # One CUDA allocation
        ├── device_memory_resource.{hpp,cpp}      # cudaMalloc, device_local
        ├── pinned_memory_resource.{hpp,cpp}      # cudaHostAlloc, process wide
        ├── memory_block.{hpp,inl}                # A range of a heap
        ├── memory_block_pool.{hpp,cpp}           # Free lists, merging, heap release
        ├── memory_block_deferred_release.{hpp,cpp}
        ├── caching_memory_allocator.{hpp,inl,cpp}
        └── buffer.{hpp,cpp}
```

### Naming
| Core interface | Our type |
|---|---|
| `xmipp4::device_backend` | `xmipp4::cuda::device_backend` |
| `xmipp4::command_queue` | `xmipp4::cuda::command_queue` |
| `xmipp4::buffer` | `xmipp4::cuda::buffer` |

Same name, different namespace, exactly as the core's CPU backend does. When both are in scope, qualify the core one as `xmipp4::` — a member function returning `const memory_resource&` inside `namespace cuda` resolves to **our** `memory_resource` and will not match the override.

---
## 3. Build

The plugin needs an installed xmipp4-core. There is no pinned copy in-tree.

```bash
# Build and install the core once
cmake -S ../xmipp4-core -B ../xmipp4-core/build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DCMAKE_INSTALL_PREFIX=<prefix>
cmake --build ../xmipp4-core/build -j && cmake --install ../xmipp4-core/build

# Then the plugin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
      -Dxmipp4-core_DIR=<prefix>/lib/cmake/xmipp4-core
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Point `CUDAToolkit_ROOT` at a specific toolkit when several are installed; `/usr/local/cuda` is a symlink and whichever it targets will be picked up silently otherwise.

### CMake shape
- `LANGUAGES C CXX CUDA`. There is no device code yet, but the language stays declared so it is not forgotten later. It makes a working `nvcc` a configure-time requirement.
- Sources compile into an **object library**, and the plugin is a `MODULE` built from it. A `MODULE` cannot be linked against, and the unitary tests will need to.
- Everything is `PRIVATE`. No headers are installed, no CMake config is exported.

---
## 4. Testing

- **Integration tests** (`tests/integration/`) load the built plugin through the core's `plugin_manager`, exactly as the framework would.
- Tests that need a GPU must call `SKIP(...)` when `enumerate_devices` comes back empty. CI runners have no GPU, and the suite has to stay green there.
- Verify both paths before calling a change done:
  ```bash
  ctest --test-dir build --output-on-failure
  cd build/tests/integration && CUDA_VISIBLE_DEVICES="" ./xmipp4-cuda_integration_tests --reporter compact
  ```
  The second one must report skips, not failures.
- `tests/unitary/` is wired but not enabled. See §10.

---
## 5. Code Style

**xmipp4-core is the authority. When in doubt, open the equivalent core file and copy what it does.** Tabs for indentation, SPDX header on every file.

### Comments — read this before every commit
This is where reviews on this repo get spent. The rules, taken from what the core actually does:

- **No TODOs.** There is not a single one in the whole core. When something is unimplemented, the thrown exception's message says so and no comment is added on top of it.
- **Private backend headers carry a class brief and nothing else.** See `src/backends/cpu/hardware/device_backend.hpp` in the core: not even `register_at` is documented. Document a method only when its contract is not evident from the signature — what it throws, when it returns null.
- **But internal types with non-obvious contracts do get full docs.** `src/core/hardware/memory_allocator_table.hpp` is the example. The rule is not "always terse", it is "terse when implementing an interface that is already documented".
- **Inline comments explain a _why_ the code cannot state.** Never a _what_. No section markers like `// Convert` / `// Write`. The whole core carries under a hundred lines of them across a hundred source files, clustered in genuinely tricky code.
- **Never restate** something already said in a class docstring, a sibling header, or an exception message.
- **One line is usually the whole comment**, in any language, CI files included. The reasoning behind a decision belongs in the commit message or the pull request, not next to the code. A three-line comment is nearly always two lines too long.
- **Say the fact, not the chain of reasoning that led to it.** When the value being commented already carries the meaning, write nothing.

The last two were settled by review on this repo: a three-line note on the Windows CUDA version was cut to `# Visual Studio 2026 is only supported from CUDA 13.2 onwards`, and a three-line note explaining the trimmed CUDA sub-packages was deleted outright, because `["nvcc", "cudart-dev"]` already says it.

Before opening a PR, grep the diff for `^\s*//` and `^\s*#` and for every docstring added, and justify each against the list.

### Types
Use `xmipp4::byte*` for raw memory, not `void*`. The core defines it in `memory/byte.hpp`. `void*` is acceptable only where an outside interface imposes it — `xmipp4::buffer::get_host_ptr`, and the CUDA runtime writing through a `void**`.

---
## 6. Key Implementation Details

### Every path into the driver sets the device
`cudaMalloc`, `cudaFree`, stream and event creation all act on the calling thread's current device, not on one they are told about. Use `device_guard`. Missing it is harmless with one GPU and silently wrong with two.

`device_guard` reports failures by throwing, so it cannot be used in a destructor. `device_memory_heap::~device_memory_heap` does the switch by hand for that reason.

### Events
Created with `cudaEventDisableTiming | cudaEventBlockingSync`: timing is never read, and a host wait should yield rather than spin. A CUDA event that was never recorded queries as complete, which is exactly the initially-signaled state the core's `event` requires.

CUDA events satisfy every `event_usage_flag_bits`, so `create_event` ignores the requested subset.

### Queues
Streams are created with `cudaStreamNonBlocking`, so queues are only ever ordered against each other through events and never implicitly through the legacy default stream.

### Buffers name a range of a heap
A `buffer` holds a `memory_block`, which names a range of a heap — not an allocation of its own. That is what lets the allocator hand out several buffers from one call to the driver. Do not "simplify" it into one allocation per buffer.

A buffer keeps its resource and its allocator alive, so it stays valid however the device or session that produced it is disposed of.

### The allocator caches, and segregates by queue
Released blocks return to `memory_block_pool`, are merged with their free neighbours, and are handed out again. The driver only sees a request when nothing suitable is left, and only gets memory back through `release_unused_heaps`, which runs when it refuses one.

Blocks belong to the queue they were allocated for. That queue can be given a block it just released without any synchronization, because its own work is ordered. Any **other** queue that was given work on the buffer is recorded on it, and the release is held behind an event per queue until they have all passed the point.

It is written for CUDA, not for every backend: concrete streams and events, no virtual dispatch in the pool, and the resource reaches the allocator as a template parameter rather than through a base class.

### Resources are shared per device
Every handle on a GPU shares its resources, or the allocators behind them would each hold a cache of the same memory. The backend owns them: a map by ordinal for device memory, and a single one for pinned memory, which is process-wide and portable across contexts rather than tied to any device.

**Do not make these process-wide statics.** It was proposed and rejected in review: holding managed objects past a forceful destruction is not a pattern to build on.

---
## 7. CI

`.github/workflows/build-and-test.yml`, one job per major toolkit version:

| OS | Compiler | CUDA |
|---|---|---|
| ubuntu-22.04 | gcc | 11.8.0 |
| ubuntu-latest | gcc, clang | 12.8.1 |
| windows-latest | MSVC | **13.3.1** |

**Windows needs CUDA ≥ 13.2.** The runner ships Visual Studio 2026, and no earlier toolkit installs MSBuild integration for it. Declaring the CUDA language then fails at the `project()` call with `No CUDA toolset found`. This is why that row differs; do not "align" it downwards.

Consequence to remember when device code arrives: CUDA 13 dropped offline compilation below compute capability 7.5, so a Windows build cannot target Maxwell, Pascal or Volta. The published wheels are manylinux only (CUDA 11.4/11.8/12.1) and are unaffected.

The CUDA install asks for `nvcc` and `cudart-dev` alone. Asking for `toolkit` pulls ~3.66 GB of profilers and math libraries that nothing here links, which is slow and has stalled jobs on a bad CDN edge.

### SonarCloud
The quality gate blocks on **zero new issues**, so every finding has to be fixed or dismissed. Two recurring false positives, both with precedent:
- `cpp:S5008` on `buffer::get_host_ptr` — the signature is imposed by `xmipp4::buffer`. The core marked the identical ones on its own `host_buffer` as `WONTFIX`.
- `githubactions:S8544` on the `pip install --no-index --find-links=dist` step — nothing is resolved from the network there, and pinning the version would break tracking of the `development` tag.

---
## 8. Gotchas

- **`cudaRuntimeGetVersion` goes through the driver** and fails with `cudaErrorInsufficientDriver` on a machine without a usable one. Nothing in the backend may throw just because there is no GPU: enumeration must report zero devices and the version falls back to `CUDART_VERSION`.
- **`--only-binary :all:` needs quoting in YAML.** `run: pip install --only-binary :all: x` does not parse — the colon-space inside `:all:` reads as a mapping. It is fine inside a `run: |` block.
- **Editing `deploy.yml` triggers the release workflow on the PR**, through its `paths` filter. Also: it is CRLF. Preserve the line endings or the diff becomes the whole file.
- **The `xmipp4-allocator-test` bundle target is stale**, still written against the pre-rework API.

---
## 9. Writing device code

There is no `.cu` in this repo and the runtime API is plain C++, which covers memory, streams, events and transfers — `copy_operation` comes out of `cudaMemcpyAsync` with no kernel at all.

Custom kernels are the line: `__global__`, `<<<>>>` and device intrinsics need a CUDA compiler. Three ways to get one, in preference order:
1. **NVIDIA's libraries** (cuFFT, cuBLAS, cuSOLVER) — callable from plain C++.
2. **NVRTC** — kernel source as a string, compiled at runtime. This fits the core's design: `program_builder::build` receives a `program_cache*` documented as being for "FFT plans, compiled kernels, workspaces".
3. **`.cu` compiled by nvcc** — the conventional route; needs `/Zc:preprocessor` under MSVC 2026 + CUDA 13.

`fill_operation` note: `cudaMemsetAsync` only fills bytes. The driver API's `cuMemsetD8/D16/D32Async` cover 8/16/32-bit patterns, which is enough for `float` and `int32`; 64-bit types would need a kernel.

---
## 10. Status and roadmap

The plugin is being brought back in line with the core after a long drift. Done:

- **Hardware layer** (#120) — backend, device, queue, event, error handling, `device_guard`.
- **Memory resources and the caching allocator** (#121) — heaps, blocks, pool, deferred release, buffers, both resources. `device_session` builds and allocations are served from the cache.

Next, in order:

1. **`copy_operation` and `fill_operation` builders.** These replace the `memory_transfer_*` classes deleted in #120 and are what make the plugin usable end to end. They also close the last gap in the allocator: `command_queue::submit` has to call `buffer::record_queue` on every bound buffer, which is what feeds the deferred release. Until a program exists there is nothing to record.
2. **Unitary tests.** `memory_block_pool` is pure bookkeeping and its merging and heap release deserve tests that do not need a GPU. It currently reaches CUDA only through `memory_heap`, so it needs a way to build a pool over heaps that are not real allocations.
3. **Benchmark.** `xmipp4-allocator-test` in the bundle is still written against the pre-rework API.
