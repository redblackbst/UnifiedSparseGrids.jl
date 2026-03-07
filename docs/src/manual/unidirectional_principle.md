# Unidirectional principle

Many core algorithms on sparse grids are *dimension-wise tensor operators*:

- hierarchization / dehierarchization,
- nodal–modal transforms (FFT/DCT),
- basis conversions,
- matrix-free Galerkin operators built from 1D building blocks.

On a full tensor grid, applying such an operator is straightforward: sweep one dimension at a time.
On a sparse grid, we must be more careful:

- coefficients live in a *restricted* multi-index set,
- intermediate states can leave the representable space if we apply a tensor operator naively,
- performance hinges on streaming access rather than random memory access.

`UnifiedSparseGrids` implements the **unidirectional principle** in `src/unidirectional.jl`,
with the goal of applying separable operators on sparse grids using cache-friendly 1D fiber sweeps.

## Key invariant: nested-prefix fibers

The unidirectional engine assumes that each 1D axis family is **nested** and stored in a
**nested-prefix order**:

- refining from refinement index $r$ to $r+1$ appends new entries **after** the old ones,
- along any 1D fiber, the data for refinement index $r$ occupies the first `totalsize(axis, r)` entries.

This is why many examples use `LevelOrder()` axis orderings.

## Recursive layout and fibers

The unidirectional engine operates on the package's **recursive layout** (`RecursiveLayout()`).
In this layout, the coefficient vector is arranged so that for a chosen storage orientation
one physical dimension becomes the **contiguous fiber dimension**.

A *fiber* here means:

- fix all coordinates except one physical dimension,
- the coefficients along the remaining dimension form a contiguous segment in memory.

Internally, `LastDimFiberIterator` (stored in the plan as `fiber_iter`) streams all such fibers.

## Cyclic unidirectional sweeps

A `D`-dimensional tensor operator is applied as `D` consecutive 1D sweeps.
To avoid materializing `D` different coefficient layouts, the engine **cyclically rotates**
which physical dimension is treated as the contiguous fiber dimension.

### Orientation tracking with `OrientedCoeffs`

`OrientedCoeffs{D,T}` is a lightweight wrapper around a coefficient buffer:

- `data::Vector{T}` stores coefficients in recursive layout,
- `perm::SVector{D,Int}` maps *storage dimensions* to *physical dimensions*.

A single step performs:

1. apply a 1D operator along the current last physical dimension `perm[end]`,
2. rotate the permutation `perm = cycle_last_to_front(perm)`.

After `D` cyclic steps, the permutation returns to the original orientation.

### Cached layout metadata: `CyclicLayoutPlan`

Creating the cyclic layouts efficiently requires precomputed metadata.
`CyclicLayoutPlan(grid, T)` caches:

- per-orientation fiber offsets and maximum fiber lengths,
- DP tables for fast rank/offset computations,
- scratch buffers sized to the worst-case fiber,
- cached per-refinement plans for line operators (FFT/DCT plans, hierarchization patterns, …).

For performance-sensitive code you typically construct a plan once and reuse it across many sweeps.

## Line operators and tensor operators

The unidirectional engine is driven by a small set of interfaces.

### 1D building blocks: `AbstractLineOp`

A 1D operator implements

- `lineop_style(::Type{Op})` to declare in-place vs out-of-place semantics,
- `apply_line!(op, buf, work, axis, r, plan)` (in-place), or
- `apply_line!(outbuf, op, inp, work, axis, r, plan)` (out-of-place).

Some line operators benefit from per-refinement cached data (e.g. FFT plans). The hook is:

- `needs_plan(::Type{Op}) = Val(true)`
- `lineplan(op, axis, rmax, T) -> planvec`

The plan is cached inside `CyclicLayoutPlan` and passed to `apply_line!`.

### Tensor composition: `AbstractTensorOp`

Line operators are lifted to `D` dimensions via tensor wrappers:

- `tensorize(op, Val(D))` broadcasts the same 1D op to all dimensions.
- `TensorOp((op1, op2, ..., opD))` applies possibly different ops per dimension.
- `compose(opA, opB)` creates a `CompositeTensorOp` that runs several full sweeps sequentially.

All of these are executed by the single entry point:

```julia
apply_unidirectional!(u::OrientedCoeffs, grid::SparseGrid, op, plan)
```

## Built-in transforms

Many transforms used in sparse grid spectral methods are exposed directly as tensor operators:

- `ForwardTransform(Val(D))`: nodal values → standard modal coefficients
- `InverseTransform(Val(D))`: standard modal coefficients → nodal values

These are implemented as a `CompositeTensorOp{D}` built from 1D pieces:

- `LineTransform(:forward/:inverse)` (FFT/DCT/etc, depending on the axis family),
- `LineHierarchize` / `LineDehierarchize` for the hierarchical basis conversion.

They can be applied like any other operator:

```julia
u = OrientedCoeffs{D}(copy(x))
plan = CyclicLayoutPlan(grid, eltype(x))
apply_unidirectional!(u, grid, ForwardTransform(Val(D)), plan)
```

## Why sparse grids need "up/down" splitting

On a full tensor grid, applying $A_1 \otimes \cdots \otimes A_D$ stays inside the full space.
On a sparse grid, this is not guaranteed:

- the sparse index set contains only a subset of tensor-product subspaces,
- a 1D operator that couples refinement blocks can create intermediate contributions at multi-indices
  that are *not present* in the sparse index set.

To keep intermediate states representable, `UnifiedSparseGrids` provides the
**up/down splitting** mechanism:

- `updown(op) -> (down, up)` returns additive pieces of a 1D operator,
- `UpDownTensorOp` expands the tensor operator into a sum of terms and evaluates each term
  by applying all "down" parts before all "up" parts.

This is used, for example, by `HatMassLineOp` (hierarchical hat mass operator) where the
mass matrix is split into lower/upper contributions and applied matrix-free and in-place (also see Shen–Yu 2010 for a spectral analogue).
This is why sparse grid unidirectional operators are **refinement-aware** and why the order of operations
matters: we are not applying an invertible tensor operator on a full space, but a restricted operator on a sparse subspace.

## Minimal usage pattern

A typical workflow looks like:

```julia
using UnifiedSparseGrids

D = 2
I = SmolyakIndexSet(D, 4)
axes = ntuple(_ -> ChebyshevGaussLobattoNodes(), D)

grid = SparseGrid(SparseGridSpec(axes, I))

# Coefficients in recursive layout (the unidirectional engine's native layout).
u = OrientedCoeffs{D,Float64}(randn(length(grid)))

# Reusable plan (owns scratch buffers and cached per-refinement line plans).
plan = CyclicLayoutPlan(grid, Float64)

# Example sweep: hierarchize in every dimension.
op = tensorize(LineHierarchize(), Val(D))
apply_unidirectional!(u, grid, op, plan)
```

If your coefficients are stored in `SubspaceLayout()` order, convert them first:

```julia
x_rec = subspace_to_recursive(grid, x_sub)
```

and convert back at the end if needed.

## Related reading

- D. Holzmüller and D. Pflüger (2021), *Fast Sparse Grid Operations Using the Unidirectional Principle: A Generalized and Unified Framework*.
