# ----------------------------------------------------------------------------
# Sparse grid specification + coefficient layouts

struct SparseGridSpec{D,AxesT,IndexT<:AbstractIndexSet{D}}
    axes::AxesT
    indexset::IndexT
end

"""Build a `SparseGridSpec` using the same 1D axis family in every dimension."""
function SparseGridSpec(axis::A, indexset::I) where {D,A<:AbstractAxisFamily,I<:AbstractIndexSet{D}}
    axes_tup = ntuple(_ -> axis, D)
    return SparseGridSpec{D,typeof(axes_tup),I}(axes_tup, indexset)
end

"""Build a `SparseGridSpec` with potentially different 1D axis families per dimension."""
function SparseGridSpec(axes::NTuple{D,Any}, indexset::I) where {D,I<:AbstractIndexSet{D}}
    all(ax -> ax isa AbstractAxisFamily, axes) || throw(ArgumentError("all axes must be AbstractAxisFamily instances"))
    return SparseGridSpec{D,typeof(axes),I}(axes, indexset)
end

@inline function Base.getproperty(S::SparseGridSpec, s::Symbol)
    if s === :nodes
        return getfield(S, :axes)
    end
    return getfield(S, s)
end

Base.propertynames(::SparseGridSpec, private::Bool=false) =
    private ? (:axes, :nodes, :indexset) : (:axes, :nodes, :indexset)

dim(S::SparseGridSpec) = dim(S.indexset)

# ----------------------------------------------------------------------------
# Layout tags

abstract type AbstractLayout end

"""Recursive (pseudorecursive) coefficient order.

This is the coefficient ordering induced by the package's dimension-recursive traversal.
"""
struct RecursiveLayout <: AbstractLayout end

"""Subspace (block) coefficient order.

Coefficients are grouped by hierarchical increment blocks indexed by refinement vectors
`r = (r_1,\\ldots,r_D)`. Each block is stored contiguously as a tensor-product block,
and the blocks are concatenated in a deterministic (sum, colex) order.
"""
struct SubspaceLayout <: AbstractLayout end

# ----------------------------------------------------------------------------
# SparseGrid container

"""Sparse grid container.

A `SparseGrid` stores only its specification (`SparseGridSpec`). Layouts are views
over the same coefficient set and are selected explicitly when iterating.
"""
struct SparseGrid{SpecT}
    spec::SpecT
end

SparseGrid(spec::SparseGridSpec) = SparseGrid{typeof(spec)}(spec)

dim(grid::SparseGrid) = dim(grid.spec)

"""Length of the sparse grid coefficient vector (independent of layout)."""
Base.length(grid::SparseGrid) = length(traverse(grid; layout=RecursiveLayout()))
Base.size(grid::SparseGrid) = (length(grid),)
Base.ndims(::SparseGrid) = 1

# ----------------------------------------------------------------------------
# Generic refinement-index helpers

"""Per-index-set refinement-cap vector used to bound iteration."""
refinement_caps(grid::SparseGrid) = refinement_caps(grid.spec.indexset)

@inline _zero_refinement(::Val{D}) where {D} = SVector{D,Int}(ntuple(_ -> 0, Val(D)))

@inline function _prefix_from_refinements(levels, perm::SVector{D,Int}, dim::Int) where {D}
    vals = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    @inbounds for k in 1:(dim - 1)
        pd = perm[k]
        vals[pd] = Int(levels[pd])
    end
    return SVector{D,Int}(vals)
end

@inline _prefix_from_levels(args...) = _prefix_from_refinements(args...)

@inline function _maxadmissible(I::AbstractIndexSet{D}, caps::SVector{D,Int}, prefix::SVector{D,Int}, pd::Int) where {D}
    capd = caps[pd]
    maxr = -1
    @inbounds for r in 0:capd
        contains(I, setindex(prefix, r, pd)) || break
        maxr = r
    end
    return maxr
end

@inline _block_start(axis, r::Integer) = (r == 0 ? 1 : totalsize(axis, r - 1) + 1)

# ----------------------------------------------------------------------------
# Subspace layout (Murarasu/FastSG): store each subspace W_ℓ contiguously and
# concatenate subspaces in a deterministic "strong" order.

"""Build the subspace blocks of a sparse grid.

Returns `(subspaces, offsets, extents, total_length, caps)` where:

* `subspaces[b]` is the refinement vector `r` of block `b` (one block per increment tensor block).
* `offsets[b]` is the 1-based start index of that block in the concatenated layout.
* `extents[b]` is the per-dimension Δ-size of the block (product equals block length).
* `total_length` equals the sum of all block lengths.
* `caps` is the per-dimension refinement-cap vector used to enumerate subspaces.

This is intentionally minimal so it can be reused by both `SubspaceLayoutIterator`
and `SubspaceBlockIterator` without computing lookup tables.
"""
function _build_subspace_blocks(spec::SparseGridSpec{D}) where {D}
    axes = spec.axes
    I = spec.indexset
    caps = refinement_caps(I)

    @inbounds for d in 1:D
        is_nested(axes[d]) || throw(ArgumentError("SubspaceLayout requires nested 1D node rules (dim=$d, got $(typeof(axes[d])))"))
    end

    subspaces = _enumerate_subspaces(I, caps)
    sort!(subspaces; by=_sortkey_colex)

    extents = SVector{D,Int}[]
    offsets = Int[]
    kept = SVector{D,Int}[]
    offset = 1
    for ℓ in subspaces
        m = SVector{D,Int}(ntuple(d -> blocksize(axes[d], ℓ[d]), D))
        blocklen = prod(m)
        blocklen == 0 && continue
        push!(kept, ℓ)
        push!(extents, m)
        push!(offsets, offset)
        offset += blocklen
    end
    total_length = offset - 1

    return kept, offsets, extents, total_length, caps
end

@inline function _build_totalsize_by_refinement(axes::NTuple{D,Any},
                                        caps::SVector{D,Int}) where {D}
    all(ax -> ax isa AbstractAxisFamily, axes) || throw(ArgumentError("all axes must be AbstractAxisFamily instances"))
    return ntuple(d -> begin
        L = caps[d]
        v = Vector{Int}(undef, L + 1)
        @inbounds for ℓ in 0:L
            v[ℓ + 1] = totalsize(axes[d], ℓ)
        end
        v
    end, Val(D))
end

# ----------------------------------------------------------------------------
# Subspace enumeration (generic across index sets)

function _enumerate_subspaces(I::AbstractIndexSet{D}, caps::SVector{D,Int}) where {D}
    out = SVector{D,Int}[]
    r = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    perm = SVector{D,Int}(ntuple(identity, Val(D)))

    function rec(dim::Int)
        if dim > D
            push!(out, SVector{D,Int}(r))
            return
        end

        prefix = _prefix_from_refinements(r, perm, dim)
        maxr = _maxadmissible(I, caps, prefix, dim)
        maxr < 0 && return

        for t in 0:maxr
            r[dim] = t
            rec(dim + 1)
        end
        r[dim] = 0
        return
    end

    rec(1)
    return out
end

@inline function _sortkey_colex(ℓ::SVector{D,Int}) where {D}
    return (sum(ℓ), Tuple(reverse(ℓ))...)
end

# ----------------------------------------------------------------------------
# SubspaceLayout iterator (plan flattened into the iterator)

struct SubspaceLayoutIterator{D}
    subspaces::Vector{SVector{D,Int}}
    offsets::Vector{Int}
    extents::Vector{SVector{D,Int}}
    lookup::Dict{SVector{D,Int},Int}
    totalsize_by_refinement::NTuple{D,Vector{Int}}
    total_length::Int
end

mutable struct _SubspaceIterState{D,ItT}
    it::ItT
    bid::Int
    locals::MVector{D,Int}
    starts::MVector{D,Int}
    extents::MVector{D,Int}
end

Base.IteratorEltype(::Type{<:SubspaceLayoutIterator}) = Base.HasEltype()
Base.eltype(::Type{<:SubspaceLayoutIterator{D}}) where {D} = SVector{D,Int}
Base.IteratorSize(::Type{<:SubspaceLayoutIterator}) = Base.HasLength()
Base.length(it::SubspaceLayoutIterator) = Int(it.total_length)

@inline function _load_block!(st::_SubspaceIterState{D}, bid::Int) where {D}
    st.bid = bid
    ℓ = st.it.subspaces[bid]
    m = st.it.extents[bid]
    @inbounds for d in 1:D
        st.locals[d] = 0
        st.extents[d] = m[d]
        st.starts[d] = (ℓ[d] == 0) ? 1 : (st.it.totalsize_by_refinement[d][ℓ[d]] + 1)
    end
    return nothing
end

@inline function _emit(st::_SubspaceIterState{D}) where {D}
    return SVector{D,Int}(ntuple(d -> st.starts[d] + st.locals[d], Val(D)))
end

@inline function _advance!(st::_SubspaceIterState{D}) where {D}
    # Column-major: dim 1 is the fastest varying.
    @inbounds for d in 1:D
        v = st.locals[d] + 1
        if v < st.extents[d]
            st.locals[d] = v
            return true
        end
        st.locals[d] = 0
    end

    bid2 = st.bid + 1
    bid2 > length(st.it.subspaces) && return false
    _load_block!(st, bid2)
    return true
end

function Base.iterate(it::SubspaceLayoutIterator{D}) where {D}
    isempty(it.subspaces) && return nothing

    locals = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    starts = MVector{D,Int}(ntuple(_ -> 1, Val(D)))
    extents = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    st = _SubspaceIterState{D,typeof(it)}(it, 1, locals, starts, extents)
    _load_block!(st, 1)
    return (_emit(st), st)
end

function Base.iterate(it::SubspaceLayoutIterator{D}, st::_SubspaceIterState{D}) where {D}
    _advance!(st) || return nothing
    return (_emit(st), st)
end

# ----------------------------------------------------------------------------
# Block iterator API for subspace layout

struct SubspaceBlock{D}
    refinement::SVector{D,Int}
    offset::Int
    len::Int
    extents::SVector{D,Int}
end

Base.range(b::SubspaceBlock) = b.offset:(b.offset + b.len - 1)

struct SubspaceBlockIterator{D}
    subspaces::Vector{SVector{D,Int}}
    offsets::Vector{Int}
    extents::Vector{SVector{D,Int}}
end

Base.IteratorEltype(::Type{<:SubspaceBlockIterator}) = Base.HasEltype()
Base.eltype(::Type{<:SubspaceBlockIterator{D}}) where {D} = SubspaceBlock{D}
Base.IteratorSize(::Type{<:SubspaceBlockIterator}) = Base.HasLength()
Base.length(it::SubspaceBlockIterator) = length(it.subspaces)

SubspaceBlockIterator(spec::SparseGridSpec{D}) where {D} = begin
    subspaces, offsets, extents, _len, _caps = _build_subspace_blocks(spec)
    SubspaceBlockIterator{D}(subspaces, offsets, extents)
end

SubspaceBlockIterator(it::SubspaceLayoutIterator{D}) where {D} =
    SubspaceBlockIterator{D}(it.subspaces, it.offsets, it.extents)

each_subspace_block(spec::SparseGridSpec) = SubspaceBlockIterator(spec)
each_subspace_block(grid::SparseGrid{<:SparseGridSpec}) = each_subspace_block(grid.spec)
each_subspace_block(it::SubspaceLayoutIterator) = SubspaceBlockIterator(it)

function Base.iterate(it::SubspaceBlockIterator{D}, bid::Int=1) where {D}
    bid > length(it.subspaces) && return nothing
    ℓ = it.subspaces[bid]
    off = it.offsets[bid]
    ext = it.extents[bid]
    len = prod(ext)
    return (SubspaceBlock{D}(ℓ, off, len, ext), bid + 1)
end

# ----------------------------------------------------------------------------
# Subspace layout: coordinate -> linear index

@inline function _refinement_local(totalsize_by_refinement::Vector{Int}, coord::Int)
    ℓidx = searchsortedfirst(totalsize_by_refinement, coord)
    ℓidx == 0 && throw(ArgumentError("invalid coordinate $coord"))
    ℓ = ℓidx - 1
    prev = (ℓidx == 1) ? 0 : totalsize_by_refinement[ℓidx - 1]
    local0 = coord - prev - 1
    return ℓ, local0
end

@inline function _subspace_linear_index(it::SubspaceLayoutIterator{D}, coords::SVector{D,Int}) where {D}
    ℓ = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    k = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    @inbounds for d in 1:D
        ld, kd = _refinement_local(it.totalsize_by_refinement[d], coords[d])
        ℓ[d] = ld
        k[d] = kd
    end

    ℓs = SVector{D,Int}(ℓ)
    bid = get(it.lookup, ℓs, 0)
    bid == 0 && throw(ArgumentError("subspace layout does not contain refinement vector $ℓs"))
    ext = it.extents[bid]

    # Column-major linearization: dim 1 is the fastest varying.
    inblock = 1
    stride = 1
    @inbounds for d in 1:D
        inblock += k[d] * stride
        stride *= ext[d]
    end
    return it.offsets[bid] + (inblock - 1)
end

# ----------------------------------------------------------------------------
# Layout transformations: recursive <-> subspace

function recursive_to_subspace(grid::SparseGrid, coeffs::AbstractVector)
    length(coeffs) == length(grid) ||
        throw(DimensionMismatch("expected length $(length(grid)), got $(length(coeffs))"))

    it_sub = traverse(grid; layout=SubspaceLayout())
    it_sub.total_length == length(grid) ||
        error("internal error: subspace plan length $(it_sub.total_length) != grid length $(length(grid))")

    out = similar(coeffs)
    i = 1
    for coords in traverse(grid; layout=RecursiveLayout())
        dst = _subspace_linear_index(it_sub, coords)
        @inbounds out[dst] = coeffs[i]
        i += 1
    end
    return out
end

function subspace_to_recursive(grid::SparseGrid, coeffs::AbstractVector)
    length(coeffs) == length(grid) ||
        throw(DimensionMismatch("expected length $(length(grid)), got $(length(coeffs))"))

    it_sub = traverse(grid; layout=SubspaceLayout())
    it_sub.total_length == length(grid) ||
        error("internal error: subspace plan length $(it_sub.total_length) != grid length $(length(grid))")

    out = similar(coeffs)
    i = 1
    for coords in traverse(grid; layout=RecursiveLayout())
        src = _subspace_linear_index(it_sub, coords)
        @inbounds out[i] = coeffs[src]
        i += 1
    end
    return out
end

"""Plot the 2D subspace layout (requires `using Plots` to activate the extension)."""
function plot_subspace_layout end

plot_subspace_layout(args...; kwargs...) =
    error("plot_subspace_layout requires loading Plots to activate UnifiedSparseGridsPlotsExt")

"""Plot the combination-technique block structure (requires `using Plots`)."""
function plot_combination_technique end

plot_combination_technique(args...; kwargs...) =
    error("plot_combination_technique requires loading Plots to activate UnifiedSparseGridsPlotsExt")

"""Plot the sparse grid point set (requires `using Plots` to activate the extension)."""
function plot_sparse_grid end

plot_sparse_grid(args...; kwargs...) =
    error("plot_sparse_grid requires loading Plots to activate UnifiedSparseGridsPlotsExt")

"""Plot the sparse grid index set in integer coordinates (requires `using Plots`)."""
function plot_sparse_indexset end

plot_sparse_indexset(args...; kwargs...) =
    error("plot_sparse_indexset requires loading Plots to activate UnifiedSparseGridsPlotsExt")

# ----------------------------------------------------------------------------
# Recursive layout iterator (generic prefix recursion)

struct GenericSubtreeCount{D}
    memo::Dict{Tuple{Int,SVector{D,Int}},Int}
    total::Int
end

@inline _subtree_size(S::GenericSubtreeCount{D}, dim::Int, prefix::SVector{D,Int}) where {D} =
    get(S.memo, (dim, prefix), dim == D + 1 ? 1 : 0)

"""Build suffix subtree counts for a generic downward-closed refinement-index set."""
function _build_subtree_count(spec::SparseGridSpec{D}, perm::SVector{D,Int}, I::AbstractIndexSet{D}) where {D}
    axes = spec.axes
    caps = refinement_caps(I)

    deltacounts = ntuple(d -> begin
        n = caps[d]
        is_nested(axes[d]) || throw(ArgumentError("RecursiveLayout requires nested 1D axis families (dim=$d, got $(typeof(axes[d])))"))
        v = Vector{Int}(undef, n + 1)
        @inbounds for r in 0:n
            v[r + 1] = blocksize(axes[d], r)
        end
        v
    end, Val(D))

    memo = Dict{Tuple{Int,SVector{D,Int}},Int}()
    z = _zero_refinement(Val(D))

    function rec(dim::Int, prefix::SVector{D,Int})
        key = (dim, prefix)
        if haskey(memo, key)
            return memo[key]
        end
        if dim == D + 1
            memo[key] = 1
            return 1
        end

        pd = perm[dim]
        maxr = _maxadmissible(I, caps, prefix, pd)
        acc = 0
        if maxr >= 0
            @inbounds for r in 0:maxr
                w = deltacounts[pd][r + 1]
                w == 0 && continue
                prefix2 = setindex(prefix, r, pd)
                acc += w * rec(dim + 1, prefix2)
            end
        end
        memo[key] = acc
        return acc
    end

    total = rec(1, z)
    return deltacounts, GenericSubtreeCount{D}(memo, total)
end

mutable struct _RecFrame{Ti<:Integer}
    dim::Int
    r::Ti
    i::Ti
end

struct RecursiveLayoutIterator{D,AxesT,IndexT,CountsT,SubtreeT}
    axes::AxesT
    indexset::IndexT
    perm::SVector{D,Int}
    refinement_caps::SVector{D,Int}
    deltacounts::CountsT
    subtree_count::SubtreeT
    total_length::Int
end

@inline function Base.getproperty(it::RecursiveLayoutIterator, s::Symbol)
    if s === :nodes
        return getfield(it, :axes)
    end
    return getfield(it, s)
end

Base.propertynames(::RecursiveLayoutIterator, private::Bool=false) =
    private ? (:axes, :nodes, :indexset, :perm, :refinement_caps, :deltacounts, :subtree_count, :total_length) :
              (:axes, :nodes, :indexset, :perm, :refinement_caps, :deltacounts, :subtree_count, :total_length)

mutable struct _RecursiveStackState{D}
    stack::Vector{_RecFrame{Int}}
    coords::MVector{D,Int}
    levels::MVector{D,Int}
    locals::MVector{D,Int}
end

Base.IteratorEltype(::Type{<:RecursiveLayoutIterator}) = Base.HasEltype()
Base.eltype(::Type{<:RecursiveLayoutIterator{D}}) where {D} = SVector{D,Int}
Base.IteratorSize(::Type{<:RecursiveLayoutIterator}) = Base.HasLength()
Base.length(it::RecursiveLayoutIterator) = Int(it.total_length)

@inline function _init_state(it::RecursiveLayoutIterator{D}) where {D}
    stack = Vector{_RecFrame{Int}}()
    sizehint!(stack, 4D)
    push!(stack, _RecFrame{Int}(1, 0, 0))
    coords = MVector{D,Int}(ntuple(_ -> 1, Val(D)))
    levels = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    locals = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    return _RecursiveStackState{D}(stack, coords, levels, locals)
end

@inline _emit(st::_RecursiveStackState{D}) where {D} = SVector{D,Int}(st.coords)

@inline function _next_entry!(it::RecursiveLayoutIterator{D}, st::_RecursiveStackState{D}) where {D}
    I = it.indexset
    caps = it.refinement_caps
    Δs = it.deltacounts
    perm = it.perm

    while !isempty(st.stack)
        f = st.stack[end]
        pd = perm[f.dim]
        prefix = _prefix_from_refinements(st.levels, perm, f.dim)
        maxr = _maxadmissible(I, caps, prefix, pd)
        if f.r > maxr
            st.levels[pd] = 0
            st.locals[pd] = 0
            pop!(st.stack)
            continue
        end

        block_len = Δs[pd][f.r + 1]
        if block_len == 0
            f.r += 1
            f.i = 0
            continue
        end

        if f.i < block_len
            st.coords[pd] = _block_start(it.axes[pd], f.r) + f.i
            st.levels[pd] = Int(f.r)
            st.locals[pd] = Int(f.i)
            f.i += 1
            if f.dim == D
                return true
            end
            push!(st.stack, _RecFrame{Int}(f.dim + 1, 0, 0))
        else
            f.r += 1
            f.i = 0
        end
    end

    return false
end

function Base.iterate(it::RecursiveLayoutIterator{D}) where {D}
    st = _init_state(it)
    _next_entry!(it, st) || return nothing
    return (_emit(st), st)
end

function Base.iterate(it::RecursiveLayoutIterator{D}, st::_RecursiveStackState{D}) where {D}
    _next_entry!(it, st) || return nothing
    return (_emit(st), st)
end

# ----------------------------------------------------------------------------
# Layout-driven traversal

"""Iterate sparse grid coordinates in a chosen layout.

This iterator emits *pseudo-recursive coordinates* `coords::SVector{D,Int}`:
for each physical dimension `d`, the coordinate is the 1-based position in the
hierarchical concatenation of refinement blocks.

Use `layout=RecursiveLayout()` for the package's native recursive order, and
`layout=SubspaceLayout()` for contiguous subspace blocks.
"""
function traverse(grid::SparseGrid; layout::AbstractLayout=RecursiveLayout())
    return traverse(layout, grid.spec)
end

"""Apply `f` to every sparse grid coordinate produced by [`traverse`](@ref)."""
function traverse(grid::SparseGrid, f; layout::AbstractLayout=RecursiveLayout())
    foreach(f, traverse(grid; layout=layout))
    return nothing
end

function traverse(::RecursiveLayout, spec::SparseGridSpec{D}) where {D}
    perm = SVector{D,Int}(ntuple(identity, Val(D)))
    I = spec.indexset
    caps = refinement_caps(I)
    deltacounts, S = _build_subtree_count(spec, perm, I)
    return RecursiveLayoutIterator(spec.axes, I, perm, caps, deltacounts, S, S.total)
end

function traverse(::SubspaceLayout, spec::SparseGridSpec{D}) where {D}
    subspaces, offsets, extents, total_length, caps = _build_subspace_blocks(spec)
    lookup = Dict{SVector{D,Int},Int}()
    sizehint!(lookup, length(subspaces))
    @inbounds for b in eachindex(subspaces)
        lookup[subspaces[b]] = b
    end
    totalsize_by_refinement = _build_totalsize_by_refinement(spec.axes, caps)
    return SubspaceLayoutIterator{D}(subspaces, offsets, extents, lookup, totalsize_by_refinement, total_length)
end
