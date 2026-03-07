# Development notes

## Optional plotting extension

The core package does not depend on plotting libraries.
Plotting helpers live in the `UnifiedSparseGridsPlotsExt` extension (see `ext/`).

To use it:

```julia
using UnifiedSparseGrids
using Plots  # triggers the extension

plot_subspace_layout(grid)
```

## Extending line operators and tensor operators

The unidirectional engine is intended to be extensible: users can implement custom 1D operators
and lift them to sparse grid tensor operators.

### 1D: define an `AbstractLineOp`

1\. Define a new operator type:

```julia
struct MyLineOp <: AbstractLineOp
    # parameters (matrices, scalars, precomputed tables, …)
end
```

2\. Declare whether the operator is applied in-place or out-of-place:

```julia
lineop_style(::Type{MyLineOp}) = InPlaceOp()      # or OutOfPlaceOp()
```

3\. Implement `apply_line!` on a single fiber.

For an in-place operator:

```julia
function apply_line!(op::MyLineOp, buf::AbstractVector, work::AbstractVector,
                    axis::AbstractAxisFamily, r::Int, plan)
    # update buf[1:totalsize(axis, r)]
    return buf
end
```

For an out-of-place operator:

```julia
function apply_line!(outbuf::AbstractVector, op::MyLineOp, inp::AbstractVector, work::AbstractVector,
                    axis::AbstractAxisFamily, r::Int, plan)
    # write outbuf[1:totalsize(axis, r)]
    return outbuf
end
```

### Optional: per-refinement planning

If your operator needs cached per-refinement data (FFT/DCT plans, hierarchy metadata, …), define:

```julia
needs_plan(::Type{MyLineOp}) = Val(true)

make_plan_shared(op::MyLineOp, axis::AbstractAxisFamily, rmax::Int, ::Type{T}) where {T} = nothing

function make_plan_entry(op::MyLineOp, axis::AbstractAxisFamily,
                         n::Int, r::Int, ::Type{T}, shared) where {T}
    # return the plan entry for size n = totalsize(axis, r)
end
```

`lineplan(op, axis, rmax, T)` then builds the cached `planvec` generically from
`make_plan_shared` / `make_plan_entry`, and `planvec[r+1]` is passed into `apply_line!`.

The `work` argument is a caller-provided scratch buffer. Planned operators must not mutate
internal scratch; use `work` instead to remain thread-safe.

### Lift to a sparse grid operator and apply

Lift to ``D`` dimensions with `tensorize` or `TensorOp`, then apply via `apply_unidirectional!`:

```julia
op = tensorize(MyLineOp(...), Val(D))
plan = CyclicLayoutPlan(grid, Float64)
u = OrientedCoeffs{D,Float64}(copy(x))
apply_unidirectional!(u, grid, op, plan)
```

If a line operator supports an additive triangular split `A = L + U`, define `updown(op)` and
use `UpDownTensorOp` to keep intermediate states representable on sparse grids.
