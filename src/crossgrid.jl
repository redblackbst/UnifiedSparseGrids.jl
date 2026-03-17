# ----------------------------------------------------------------------------
# Cross-grid tensor operators and transfer utilities

"""Internal: build an index map from `it_small` into `it_big`.

`it_small` must generate an ordered subsequence of the coordinates produced by
`it_big`. The matching coordinates in `it_big` are given by

    target = coord_map(coords_small).

The returned `map` satisfies `coords_big[map[i]] == coord_map(coords_small[i])`.
"""
function _subsequence_index_map(it_small, it_big;
                                coord_map=identity,
                                Ti::Type{<:Integer}=Int)
    map = Vector{Ti}(undef, length(it_small))

    big_i = 1
    small_i = 1
    big_nxt = iterate(it_big)
    small_nxt = iterate(it_small)

    while small_nxt !== nothing
        coords_s, st_s = small_nxt
        target = coord_map(coords_s)

        while true
            big_nxt === nothing && throw(ArgumentError(
                "it_small is not a subsequence of it_big (failed at small index $small_i)"))
            coords_b, st_b = big_nxt
            if coords_b == target
                map[small_i] = Ti(big_i)
                small_i += 1
                small_nxt = iterate(it_small, st_s)
                big_i += 1
                big_nxt = iterate(it_big, st_b)
                break
            else
                big_i += 1
                big_nxt = iterate(it_big, st_b)
            end
        end
    end

    return map
end

"""Return an index map from a *subgrid* to a *covering grid*.

`map[i]` is the index in `grid_big` corresponding to index `i` in `grid_small`.

This is valid when `traverse(grid_small)` produces an ordered subsequence of
`traverse(grid_big)`.

The returned indices refer to the recursive-layout ordering of each grid.
"""
function subgrid_index_map(grid_small::SparseGrid{<:SparseGridSpec{D}},
                           grid_big::SparseGrid{<:SparseGridSpec{D}};
                           Ti::Type{<:Integer}=Int) where {D}
    dim(grid_big) == D || throw(DimensionMismatch("grid dimension mismatch"))
    return _subsequence_index_map(traverse(grid_small), traverse(grid_big);
                                  coord_map=identity, Ti=Ti)
end

@inline function _gather!(y_small::AbstractVector,
                          x_big::AbstractVector,
                          map::AbstractVector{<:Integer})
    length(y_small) == length(map) || throw(DimensionMismatch("destination length mismatch"))
    @inbounds for i in eachindex(map)
        y_small[i] = x_big[map[i]]
    end
    return y_small
end

@inline function _scatter_zero!(y_big::AbstractVector,
                                x_small::AbstractVector,
                                map::AbstractVector{<:Integer})
    fill!(y_big, zero(eltype(y_big)))
    length(x_small) == length(map) || throw(DimensionMismatch("source length mismatch"))
    @inbounds for i in eachindex(map)
        y_big[map[i]] = x_small[i]
    end
    return y_big
end

"""Plan for transferring vectors between a *small* grid and a *big* grid.

The plan stores a map from indices in the small grid traversal into indices in
the big grid traversal.

This is intended for cheap gather/scatter operations when `traverse(grid_small)`
is an ordered subsequence of `traverse(grid_big)` up to a coordinate mapping.
"""
struct TransferPlan{Ti<:Integer}
    map::Vector{Ti}
end

function TransferPlan(grid_small::SparseGrid{<:SparseGridSpec{D}},
                      grid_big::SparseGrid{<:SparseGridSpec{D}};
                      coord_map=identity,
                      Ti::Type{<:Integer}=Int) where {D}
    dim(grid_big) == D || throw(DimensionMismatch("grid dimension mismatch"))
    map = _subsequence_index_map(traverse(grid_small), traverse(grid_big);
                                 coord_map=coord_map, Ti=Ti)
    return TransferPlan(map)
end

"""Restrict values from a *covering* grid vector into a *subgrid* vector.

This is a convenience wrapper around the internal gather kernel.

`plan` must be a [`TransferPlan`](@ref) from the small grid to the big grid.
"""
restrict!(y_small::AbstractVector, x_big::AbstractVector, plan::TransferPlan) =
    _gather!(y_small, x_big, plan.map)

"""Embed a *subgrid* vector into a *covering* grid vector (missing entries set to zero).

This is a convenience wrapper around the internal scatter kernel.

`plan` must be a [`TransferPlan`](@ref) from the small grid to the big grid.
"""
embed!(y_big::AbstractVector, x_small::AbstractVector, plan::TransferPlan) =
    _scatter_zero!(y_big, x_small, plan.map)

# ----------------------------------------------------------------------------
# Cross-grid tensor operators

"""Plan for applying a (possibly) cross-grid tensor-product operator.

The plan chooses an intermediate *covering grid* `gridW` that provides a stable
storage layout for unidirectional sweeps.

- If `gridX === gridW`, input embedding is skipped.
- If `gridY === gridW`, output restriction is skipped.

Otherwise, `x_to_w` and/or `y_to_w` hold index maps into `gridW`.
"""
struct CrossGridPlan{D,Ti<:Integer,ElT,GridWT,PlanWT}
    gridW::GridWT
    planW::PlanWT
    x_to_w::Union{Nothing,Vector{Ti}}
    y_to_w::Union{Nothing,Vector{Ti}}
end

function CrossGridPlan(gridY::SparseGrid{<:SparseGridSpec{D}},
                       gridX::SparseGrid{<:SparseGridSpec{D}},
                       ::Type{ElT}=Float64;
                       Ti::Type{<:Integer}=Int,
                       gridW::Union{Nothing,SparseGrid}=nothing) where {D,ElT}
    gridW === nothing && (gridW = length(gridX) >= length(gridY) ? gridX : gridY)

    planW = CyclicLayoutPlan(gridW, ElT; Ti=Ti)

    x_to_w = gridW === gridX ? nothing :
             _subsequence_index_map(traverse(gridX), traverse(gridW);
                                    coord_map=identity, Ti=Ti)
    y_to_w = gridW === gridY ? nothing :
             _subsequence_index_map(traverse(gridY), traverse(gridW);
                                    coord_map=identity, Ti=Ti)

    return CrossGridPlan{D,Ti,ElT,typeof(gridW),typeof(planW)}(gridW, planW, x_to_w, y_to_w)
end

"""Apply a tensor-product operator between two possibly different sparse grids.

    mul!(y, gridY, op, x, gridX; plan=nothing)

This computes

    y := R_Y * A * R_X^T * x,

where `A` is a tensor-product operator applied on the *covering grid* `gridW`
stored in `plan` (default: the larger of `gridX` and `gridY`).

`op` must be a square `AbstractTensorOp` compatible with `gridW`.
"""
function LinearAlgebra.mul!(y::AbstractVector{ElT},
                            gridY::SparseGrid{<:SparseGridSpec{D}},
                            op::AbstractTensorOp{D},
                            x::AbstractVector{ElT},
                            gridX::SparseGrid{<:SparseGridSpec{D}};
                            plan::Union{Nothing,CrossGridPlan}=nothing) where {D,ElT}
    length(y) == length(gridY) || throw(DimensionMismatch("destination length mismatch"))
    length(x) == length(gridX) || throw(DimensionMismatch("source length mismatch"))

    plan === nothing && (plan = CrossGridPlan(gridY, gridX, ElT))
    gridW = plan.gridW

    w = plan.planW.workspace.work_buf

    if plan.x_to_w === nothing
        length(w) == length(x) || throw(DimensionMismatch("plan/grid mismatch"))
        copyto!(w, x)
    else
        _scatter_zero!(w, x, plan.x_to_w)
    end

    u = OrientedCoeffs{D}(w)
    apply_unidirectional!(u, gridW, op, plan.planW)

    if plan.y_to_w === nothing
        copyto!(y, u.data)
    else
        _gather!(y, u.data, plan.y_to_w)
    end

    return y
end
