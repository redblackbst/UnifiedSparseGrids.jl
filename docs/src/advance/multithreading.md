# Multi-threading

`apply_unidirectional!` uses two internal threaded backends:

- **fiber-level parallelism** inside one last-dimension sweep, and
- **term-level parallelism** for `UpDownTensorOp`, where different split terms are processed concurrently.

All internal tasks are created with `Threads.@spawn :default`. Scratch buffers are **task-local**,
not indexed by `threadid()`, because Julia tasks may migrate between worker threads.

## Execution backends

### Fiber-level queue inside one sweep

The low-level primitive `apply_lastdim_cycled!` applies one 1D line operator family to all
last-dimension fibers of the current cyclic orientation. When more than one worker is available,
the sweep uses a dynamic queue over an overdecomposed set of contiguous fiber chunks:

1. the layout is split into `min(nfibers, 4 * pool)` contiguous chunks,
2. each chunk stores its precomputed fused-scatter row starts,
3. worker tasks pull the next chunk index from an atomic counter,
4. each worker reuses its own `bufA`, `bufB`, `work`, and `rowptr` scratch.

The chunking is **layout-only**. It does not depend on the particular `AbstractLineOp` family.
This keeps the extension surface for custom line operators small: defining a new line operator
does not require teaching the scheduler about its cost model.

### Term-level queue for `UpDownTensorOp`

For `UpDownTensorOp`, the operator is expanded into split terms determined by the effective split
mask. In term mode, those terms are processed by a dynamic queue over the term-mask list:

1. worker tasks pull the next term index from an atomic counter,
2. each worker owns one `CyclicLayoutWorkspace` and accumulates into its private `acc_buf`,
3. the inner unidirectional sweeps inside one term run with `nworkers = 1`,
4. after all terms finish, the worker accumulators are combined by a parallel striped reduction.

The reduction partitions the output vector into contiguous stripes. Each reduction task owns one
stripe, copies the first worker accumulator into that stripe, and adds the remaining worker
accumulators on top.

Before the term queue starts, the implementation prewarms the required entries of
`meta.lineplans` on the caller thread. This avoids concurrent mutation of the shared plan cache.

## Backend selection and user guidance

### Automatic backend selection

The backend choice matters only for `UpDownTensorOp`.

Let

- `pool = Threads.threadpoolsize(:default)`,
- `omit_eff` be the effective omission dimension,
- `split_mask_eff` be the split mask after removing `omit_eff`, and
- `nterms = 2^k`, where `k = count_ones(split_mask_eff)`.

The current rule is

```math
\text{use term mode} \iff pool > 1 \text{ and } 2 n_{\mathrm{terms}} \ge pool.
```

Otherwise fiber mode is used.

`omit_eff` is obtained from `op.omit_dim` as follows:

- `omit_dim = 0` means **do not omit any dimension**,
- `1 <= omit_dim <= D` requests omission of that physical dimension,
- if the requested dimension is not actually split in `op.split_mask`, the effective omission is
  normalized to `0`.

The constructor default is `UpDownTensorOp(ops; omit_dim=1)`. So, by default, the first physical
dimension is omitted from the split when that dimension actually has both lower and upper parts.

### How the user should choose `omit_dim`

The main user-visible control is `omit_dim`.

If `k` dimensions are split before omission, then:

- `omit_dim = 0` keeps all split dimensions and produces `2^k` terms,
- omitting one split dimension reduces this to `2^(k-1)` terms.

So, in practice:

- choose `omit_dim = 0` when you want the largest amount of outer term concurrency,
- choose an explicit physical dimension when you want fewer terms or you already know that one
  dimension should stay unsplit,
- keep the default `omit_dim = 1` when the first dimension is the natural unsplit direction in
  your formulation.

If you want fully serial execution, start Julia with one default worker thread.

## Shared metadata vs. mutable workspace

The multithreaded implementation is organized around a shared plan plus task-local workspace.

### Fiber descriptors and queue plans

```julia
struct LastDimFiber{D}
    src_offset::Int      # 1-based offset into the recursive-layout coefficient vector
    len::Int             # number of coefficients in this fiber
    last_refinement::Int # largest active refinement index in the last dimension for this fiber
end
```

```julia
struct OrientationLayout{D,Ti<:Integer}
    perm::SVector{D,Int}      # cyclic storage order represented by this layout
    first_offsets::Vector{Ti} # fused-scatter row starts for this orientation
    maxlen::Ti                # maximum fiber length in this orientation
    nfibers::Ti               # number of last-dimension fibers in this orientation
    fibers::Vector{LastDimFiber{D}} # contiguous fiber descriptors in recursive-layout order
end
```

```julia
struct FiberChunkPlan{Ti<:Integer}
    ranges::Vector{UnitRange{Int}} # contiguous fiber chunks queued to worker tasks
    startptrs::Matrix{Ti}          # precomputed row-pointer starts, size (maxlen, nchunks)
end
```

```julia
struct FiberWorkerBuffers{T<:Number,Ti<:Integer}
    bufA::Vector{Vector{T}} # first per-worker ping-pong buffer for 1D pipelines
    bufB::Vector{Vector{T}} # second per-worker ping-pong buffer for 1D pipelines
    work::Vector{Vector{T}} # per-worker temporary work buffer passed to apply_line!
    rowptr::Matrix{Ti}      # per-worker row-pointer state, size (maxlen, pool)
end
```

### Shared metadata

```julia
struct CyclicLayoutMeta{D,Ti<:Integer}
    layouts::NTuple{D,OrientationLayout{D,Ti}} # one cached layout per cyclic orientation
    refinement_caps::SVector{D,Int}            # largest active refinement index per dimension
    lineplans::Dict{Tuple,AbstractVector}      # cached 1D plan vectors keyed by operator family
    fiber_queue_plans::NTuple{D,FiberChunkPlan{Ti}} # one prebuilt fiber queue plan per orientation
    pool::Int                                  # captured size of Threads.threadpoolsize(:default)
end
```

### Mutable workspace

```julia
struct CyclicLayoutWorkspace{Ti<:Integer,T<:Number}
    write_ptr::Vector{Ti} # serial row-pointer state for scatter passes
    scratch1::Vector{T}   # first serial 1D scratch buffer
    scratch2::Vector{T}   # second serial 1D scratch buffer
    scratch3::Vector{T}   # serial temporary work buffer passed to apply_line!
    unidir_buf::Vector{T} # full-size ping-pong buffer for generic tensor sweeps
    x_buf::Vector{T}      # full-size rotated input buffer used when the source orientation changes
    work_buf::Vector{T}   # term-local full-size work buffer
    acc_buf::Vector{T}    # full-size output or worker-local accumulation buffer
end
```

```julia
struct CyclicLayoutPlan{D,Ti<:Integer,T<:Number}
    meta::CyclicLayoutMeta{D,Ti}          # shared layouts, cached plans, and pool size
    workspace::CyclicLayoutWorkspace{Ti,T} # mutable scratch for the caller context
    fiber_workers::FiberWorkerBuffers{T,Ti} # reusable per-worker scratch for fiber queues
    term_workspaces::Vector{CyclicLayoutWorkspace{Ti,T}} # reusable per-worker scratch for term queues
end
```
