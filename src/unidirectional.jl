"""Unidirectional sparse grid sweeps in recursive layout.

The fused kernels operate on contiguous fibers along one storage dimension while
cyclically rotating the storage-to-physical permutation.
"""

# Oriented coefficient buffers

"""Coefficient buffer with a storage-to-physical dimension permutation."""
struct OrientedCoeffs{D,T}
    data::Vector{T}
    perm::SVector{D,Int}
end

OrientedCoeffs(data::Vector{T}) where {T} = OrientedCoeffs(data, SVector{1,Int}(1))

function OrientedCoeffs{D}(data::Vector{T}) where {D,T}
    perm = SVector{D,Int}(ntuple(i -> i, D))
    return OrientedCoeffs(data, perm)
end

"""Cyclic shift of a dimension permutation: (p1,p2,...,pD) -> (pD,p1,...,p{D-1})."""
@inline function cycle_last_to_front(perm::SVector{D,Int}) where {D}
    return SVector{D,Int}(ntuple(i -> (i == 1 ? perm[D] : perm[i - 1]), D))
end

"""k-step cyclic shift (last→front) of a dimension permutation."""
@inline function cycle_last_to_front(perm::SVector{D,Int}, k::Integer) where {D}
    kk = mod(Int(k), D)
    kk == 0 && return perm
    # Right rotation by kk.
    return SVector{D,Int}(ntuple(i -> perm[mod(i - kk - 1, D) + 1], D))
end

# ---------------------------------------------------------------------------
# Line-op planning hooks

"""Return `Val(true)` if this line operator needs a cached per-refinement plan vector.

This is an extension hook. For a custom `AbstractLineOp`, you may opt into
cached per-refinement precomputation by defining:

    needs_plan(::Type{MyOp}) = Val(true)
    make_plan_shared(op, axis, rmax, ::Type{T})
    make_plan_entry(op, axis, n, r, ::Type{T}, shared)

The unidirectional engine caches `planvec` inside [`CyclicLayoutPlan`](@ref)
and passes `planvec[r+1]` into `apply_line!`.
"""
needs_plan(::Type{<:AbstractLineOp}) = Val(false)
needs_plan(::Type{<:LineTransform}) = Val(true)
needs_plan(::Type{<:LineChebyshevLegendre}) = Val(true)
needs_plan(::Type{LineHierarchize}) = Val(true)
needs_plan(::Type{LineDehierarchize}) = Val(true)
needs_plan(::Type{LineHierarchizeTranspose}) = Val(true)
needs_plan(::Type{LineDehierarchizeTranspose}) = Val(true)

@inline function _any_needs_plan(::Type{Ops}) where {Ops<:Tuple}
    Ops === Tuple{} && return false
    head = Base.tuple_type_head(Ops)
    tail = Base.tuple_type_tail(Ops)
    return (needs_plan(head) === Val(true)) || _any_needs_plan(tail)
end
needs_plan(::Type{CompositeLineOp{Ops}}) where {Ops} = Val(_any_needs_plan(Ops))

"""Build a per-refinement plan vector for a line operator.

The returned object must be indexable as `planvec[r+1]` for `r = 0..rmax`.
Concrete plan entries are built from `n = totalsize(axis, r)`.
"""
function lineplan(op::AbstractLineOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T}
    maxr = Int(rmax)
    shared = make_plan_shared(op, axis, maxr, T)
    p0 = make_plan_entry(op, axis, totalsize(axis, 0), 0, T, shared)
    plans = Vector{typeof(p0)}(undef, maxr + 1)
    plans[1] = p0
    @inbounds for r in 1:maxr
        plans[r + 1] = make_plan_entry(op, axis, totalsize(axis, r), r, T, shared)
    end
    return plans
end

@inline _lineplan_key(op::AbstractLineOp, axis::AbstractUnivariateNodes, ::Type{T}) where {T} =
    (plan_family(typeof(op)), typeof(axis), T)

@inline _lineplan_key(::LineTransform, axis::AbstractUnivariateNodes, ::Type{T}) where {T<:Number} =
    (LineTransform, typeof(axis), _realpart_type(T), T)

function _get_lineplanvec!(op_plan::Dict{Any,Any},
                           op::AbstractLineOp,
                           axis::AbstractUnivariateNodes,
                           rmax::Integer,
                           ::Type{T}) where {T}
    needs_plan(typeof(op)) === Val(false) && return nothing

    key = _lineplan_key(op, axis, T)
    if haskey(op_plan, key)
        plans = op_plan[key]
        if length(plans) < Int(rmax) + 1
            plans = lineplan(op, axis, rmax, T)
            op_plan[key] = plans
        end
        return plans
    end

    plans = lineplan(op, axis, rmax, T)
    op_plan[key] = plans
    return plans
end


# ----------------------------------------------------------------------------
# Cyclic (orientation-dependent) layout plan for unidirectional sweeps

"""Descriptor of one contiguous last-dimension fiber in recursive layout."""
struct LastDimFiber{D}
    src_offset::Int                      # 1-based offset into the recursive-layout vector
    len::Int                             # number of coefficients in this fiber
    last_refinement::Int                 # largest active refinement index in the last dimension for this fiber
end

"""Cached layout metadata for one orientation (one storage-dimension permutation)."""
struct OrientationLayout{D,Ti<:Integer}
    perm::SVector{D,Int}
    first_offsets::Vector{Ti}   # 1-based row starts for the cycled layout
    first_counts::Vector{Ti}    # #fibers contributing to each row
    maxlen::Ti                  # maximum fiber length in this orientation
    fibers::Vector{LastDimFiber{D}}
end

"""A reusable plan for cyclic unidirectional sweeps.

For each cyclic orientation, this plan caches the metadata needed to
cache-efficiently write the cycled recursive layout in a *single pass* over
last-dimension fibers.

Additionally, the plan holds:

- a cache `op_plan` of per-refinement line-op plans (FFT/DCT/etc.)
- scratch buffers (`scratch1`, `scratch2`, `scratch3`) sized to the maximum fiber length
- full-size buffers (`unidir_buf`, `x_buf`, `work_buf`, `acc_buf`) used by unidirectional
  sweeps and UpDown accumulation

The plan is typed on the coefficient element type `T` and therefore owns typed
buffers.
"""
struct CyclicLayoutPlan{D,Ti<:Integer,T<:Number}
    layouts::NTuple{D,OrientationLayout{D,Ti}}
    perm_to_orient::Dict{SVector{D,Int},Int}
    write_ptr::Vector{Ti}        # reusable scratch (length max(maxlen))
    refinement_caps::SVector{D,Int}
    scratch1::Vector{T}
    scratch2::Vector{T}
    scratch3::Vector{T}
    unidir_buf::Vector{T}
    x_buf::Vector{T}
    work_buf::Vector{T}
    acc_buf::Vector{T}
    op_plan::Dict{Any,Any}
end

# ----------------------------------------------------------------------------
# Internal: build an oriented spec (nodes + index set) in storage dimension order.

@inline _permute_tuple(t::NTuple{D,Any}, perm::SVector{D,Int}) where {D} = ntuple(i -> t[perm[i]], D)

function _permute_indexset(I::SmolyakIndexSet{D,Ti}, perm::SVector{D,Int}) where {D,Ti<:Integer}
    cap_p = SVector{D,Ti}(ntuple(i -> I.cap[perm[i]], D))
    return SmolyakIndexSet(Val(D), I.L; cap=cap_p)
end

function _permute_indexset(I::WeightedSmolyakIndexSet{D,Ti}, perm::SVector{D,Int}) where {D,Ti<:Integer}
    cap_p = SVector{D,Ti}(ntuple(i -> I.cap[perm[i]], D))
    weights_p = SVector{D,Ti}(ntuple(i -> I.weights[perm[i]], D))
    return WeightedSmolyakIndexSet(Val(D), I.L, weights_p; cap=cap_p)
end

function _permute_indexset(I::FullTensorIndexSet{D,Ti}, perm::SVector{D,Int}) where {D,Ti<:Integer}
    cap_p = SVector{D,Ti}(ntuple(i -> I.cap[perm[i]], D))
    return FullTensorIndexSet(Val(D), maximum(cap_p); cap=cap_p)
end

function oriented_spec(spec::SparseGridSpec{D}, perm::SVector{D,Int}) where {D}
    axes_p = _permute_tuple(spec.axes, perm)
    I_p = _permute_indexset(spec.indexset, perm)
    return SparseGridSpec(axes_p, I_p)
end

# ----------------------------------------------------------------------------
# Last-dimension fiber iterator (recursive layout)

"""Create the ordered list of last-dimension fibers for `spec`.

Each entry records one contiguous last-dimension fiber in recursive layout order.
"""
function each_lastdim_fiber(spec::SparseGridSpec{D}) where {D}
    perm = SVector{D,Int}(ntuple(i -> i, Val(D)))
    caps = refinement_caps(spec.indexset)
    deltacounts, subtree = _build_subtree_count(spec, perm, spec.indexset)
    axes = spec.axes

    fibers = LastDimFiber{D}[]
    levels = MVector{D,Int}(ntuple(_ -> 0, Val(D)))

    function rec(dim::Int, offset::Int)
        pd = perm[dim]
        prefix = _prefix_from_refinements(levels, perm, dim)
        maxr = _maxadmissible(spec.indexset, caps, prefix, pd)
        maxr < 0 && return

        if dim == D
            push!(fibers, LastDimFiber{D}(offset, totalsize(axes[pd], maxr), maxr))
            return
        end

        off = offset
        @inbounds for r in 0:maxr
            levels[pd] = r
            w = deltacounts[pd][r + 1]
            if w != 0
                prefix2 = setindex(prefix, r, pd)
                child_subtree = _subtree_size(subtree, dim + 1, prefix2)
                for i in 0:(w - 1)
                    rec(dim + 1, off + i * child_subtree)
                end
                off += w * child_subtree
            end
        end
        levels[pd] = 0
        return
    end

    rec(1, 1)
    return fibers
end

"""
Convenience wrapper: iterate last-dimension fibers for a `SparseGrid` in a given orientation.

`perm` maps storage dimension → physical dimension (as in `OrientedCoeffs`).
"""
function each_lastdim_fiber(grid::SparseGrid{<:SparseGridSpec{D}}, perm::SVector{D,Int}) where {D}
    spec_or = oriented_spec(grid.spec, perm)
    return each_lastdim_fiber(spec_or)
end

"""Iterate last-dimension fibers for `grid` in its default (identity) orientation."""
each_lastdim_fiber(grid::SparseGrid{<:SparseGridSpec{D}}) where {D} = each_lastdim_fiber(grid, SVector{D,Int}(ntuple(identity, D)))

### Row metadata (counts/offsets) for fused-write kernels

"""Compute row counts/offsets for the fused-write recursive layout by scanning fibers."""
function _layout_rowmeta(spec::SparseGridSpec{D}; Ti::Type{<:Integer}=Int) where {D}
    axes = spec.axes
    @inbounds for d in 1:D
        is_nested(axes[d]) || throw(ArgumentError("_layout_rowmeta requires nested 1D axis families (dim=$d, got $(typeof(axes[d])))"))
    end

    fibers = each_lastdim_fiber(spec)
    isempty(fibers) && return Ti[], Ti[], Ti(0)

    max_last_refinement = maximum(fib.last_refinement for fib in fibers)
    maxlen = maximum(fib.len for fib in fibers)
    count_by_last_refinement = zeros(Int, max_last_refinement + 1)
    for fib in fibers
        count_by_last_refinement[fib.last_refinement + 1] += 1
    end

    diff = zeros(Int, maxlen + 2)
    @inbounds for t in 0:max_last_refinement
        c = count_by_last_refinement[t + 1]
        c == 0 && continue
        len = totalsize(axes[D], t)
        diff[1] += c
        diff[len + 1] -= c
    end

    counts = Vector{Ti}(undef, maxlen)
    running = 0
    @inbounds for k in 1:maxlen
        running += diff[k]
        counts[k] = Ti(running)
    end

    offsets = Vector{Ti}(undef, maxlen)
    off = Ti(1)
    @inbounds for k in 1:maxlen
        offsets[k] = off
        off += counts[k]
    end
    return offsets, counts, Ti(maxlen)
end
# ----------------------------------------------------------------------------
# Cyclic layout plan construction

"""Build a [`CyclicLayoutPlan`](@ref) for `grid` and coefficient element type `T`."""
function CyclicLayoutPlan(grid::SparseGrid{<:SparseGridSpec{D}}, ::Type{T};
                          Ti::Type{<:Integer}=Int) where {D,T<:Number}
    I = grid.spec.indexset
    caps = refinement_caps(I)

    perms = ntuple(o -> begin
        s = o - 1
        SVector{D,Int}(ntuple(i -> mod(i - s - 1, D) + 1, D))
    end, D)
    perm_to_orient = Dict{SVector{D,Int},Int}()

    layouts = ntuple(o -> begin
        perm = perms[o]
        perm_to_orient[perm] = o
        spec_o = oriented_spec(grid.spec, perm)
        fibers = each_lastdim_fiber(spec_o)
        offsets, counts, maxlen = _layout_rowmeta(spec_o; Ti=Ti)
        OrientationLayout{D,Ti}(perm, offsets, counts, maxlen, fibers)
    end, D)

    maxmaxlen = maximum(Int(l.maxlen) for l in layouts)
    write_ptr = Vector{Ti}(undef, maxmaxlen)
    scratch1 = Vector{T}(undef, maxmaxlen)
    scratch2 = Vector{T}(undef, maxmaxlen)
    scratch3 = Vector{T}(undef, maxmaxlen)

    N = length(grid)
    unidir_buf = Vector{T}(undef, N)
    x_buf = Vector{T}(undef, N)
    work_buf = Vector{T}(undef, N)
    acc_buf = Vector{T}(undef, N)

    op_plan = Dict{Any,Any}()
    return CyclicLayoutPlan{D,Ti,T}(layouts, perm_to_orient, write_ptr, caps,
                                   scratch1, scratch2, scratch3, unidir_buf, x_buf, work_buf, acc_buf, op_plan)
end

# ----------------------------------------------------------------------------
# Unidirectional apply primitive

struct FiberChunkPlan{Ti<:Integer}
    ranges::Vector{UnitRange{Int}}
    startptrs::Matrix{Ti}   # size (P, maxlen)
end

struct FiberChunkBuffers{T<:Number,Ti<:Integer}
    bufA::Vector{Vector{T}}
    bufB::Vector{Vector{T}}
    work::Vector{Vector{T}}
    rowptr::Vector{Vector{Ti}}
end

@inline function _scatter_copy!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer}, src::AbstractVector)
    @inbounds for k in eachindex(src)
        idx = rowptr[k]
        destdata[idx] = src[k]
        rowptr[k] = idx + 1
    end
    return destdata
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::AbstractLineOp, inp::AbstractVector, outbuf::AbstractVector,
                        work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    apply_line!(outbuf, op, inp, work, axis, r, plan)
    _scatter_copy!(destdata, rowptr, outbuf)
    return destdata
end

@inline function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                ::IdentityLineOp, inp::AbstractVector, outbuf::AbstractVector,
                                work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    _scatter_copy!(destdata, rowptr, inp)
    return destdata
end

@inline function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                ::ZeroLineOp, inp::AbstractVector, outbuf::AbstractVector,
                                work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    z = zero(eltype(destdata))
    @inbounds for k in eachindex(inp)
        idx = rowptr[k]
        destdata[idx] = z
        rowptr[k] = idx + 1
    end
    return destdata
end

@inline function _run_pipeline!(ops::Tuple,
                                planvecs::Tuple,
                                seg,
                                bufA,
                                bufB,
                                work,
                                axis::AbstractUnivariateNodes,
                                r::Int,
                                ::Type{ElT}) where {ElT}
    cur = seg
    cur_is_src = true

    @inbounds for i in eachindex(ops)
        oi = ops[i]
        oi isa IdentityLineOp && continue
        pv = planvecs[i]
        plan1d = pv === nothing ? nothing : pv[r + 1]

        if oi isa ZeroLineOp
            outbuf = cur_is_src ? bufA : (cur === bufA ? bufB : bufA)
            fill!(outbuf, zero(ElT))
            cur = outbuf
            cur_is_src = false
            continue
        end

        if lineop_style(typeof(oi)) isa InPlaceOp
            if cur_is_src
                copyto!(bufA, seg)
                cur = bufA
                cur_is_src = false
            end
            tmp = cur === bufA ? bufB : bufA
            apply_line!(oi, cur, tmp, axis, r, plan1d)
        else
            if cur_is_src
                outbuf = bufA
                tmp = bufB
            else
                outbuf = cur === bufA ? bufB : bufA
                tmp = work
            end
            apply_line!(outbuf, oi, cur, tmp, axis, r, plan1d)
            cur = outbuf
            cur_is_src = false
        end
    end

    return cur, cur_is_src
end

@inline function _run_pipeline_scatter!(destdata::AbstractVector,
                                        rowptr::AbstractVector{<:Integer},
                                        ops::Tuple,
                                        planvecs::Tuple,
                                        seg,
                                        bufA,
                                        bufB,
                                        work,
                                        axis::AbstractUnivariateNodes,
                                        r::Int,
                                        ::Type{ElT}) where {ElT}
    nops = length(ops)
    nops == 0 && return _scatter_copy!(destdata, rowptr, seg)

    lastop = ops[end]
    if lastop isa IdentityLineOp || lastop isa ZeroLineOp || lineop_style(typeof(lastop)) isa OutOfPlaceOp
        if nops == 1
            cur, cur_is_src = seg, true
        else
            cur, cur_is_src = _run_pipeline!(ops[1:(end - 1)], planvecs[1:(end - 1)],
                                             seg, bufA, bufB, work,
                                             axis, r, ElT)
        end

        if cur_is_src
            outbuf = bufA
            tmp = bufB
        else
            outbuf = cur === bufA ? bufB : bufA
            tmp = work
        end

        pv = planvecs[end]
        plan1d = pv === nothing ? nothing : pv[r + 1]
        return apply_scatter!(destdata, rowptr, lastop, cur, outbuf, tmp, axis, r, plan1d)
    end

    cur, _ = _run_pipeline!(ops, planvecs, seg, bufA, bufB, work, axis, r, ElT)
    return _scatter_copy!(destdata, rowptr, cur)
end

@inline _default_fiber_workers() = max(Threads.nthreads() - 1, 0)

function _partition_fibers_by_work(fibers::Vector{<:LastDimFiber}, P::Int)
    nf = length(fibers)
    nf == 0 && return UnitRange{Int}[]
    P = clamp(P, 1, nf)
    P == 1 && return [1:nf]

    prefix = Vector{Int}(undef, nf)
    total = 0
    @inbounds for i in 1:nf
        total += fibers[i].len
        prefix[i] = total
    end

    ranges = Vector{UnitRange{Int}}(undef, P)
    start = 1
    for c in 1:(P - 1)
        target = cld(total * c, P)
        stop = searchsortedfirst(prefix, target)
        stop = max(stop, start)
        stop = min(stop, nf - (P - c))
        ranges[c] = start:stop
        start = stop + 1
    end
    ranges[P] = start:nf
    return ranges
end

function _build_fiber_chunk_plan(layout::OrientationLayout{D,Ti}, P::Int) where {D,Ti<:Integer}
    fibers = layout.fibers
    ranges = _partition_fibers_by_work(fibers, P)
    P2 = length(ranges)
    maxlen = Int(layout.maxlen)
    startptrs = Matrix{Ti}(undef, P2, maxlen)
    running = zeros(Ti, maxlen)
    diff = zeros(Int, maxlen + 1)

    @inbounds for c in 1:P2
        for k in 1:maxlen
            startptrs[c, k] = layout.first_offsets[k] + running[k]
        end

        fill!(diff, 0)
        for idx in ranges[c]
            len = fibers[idx].len
            diff[1] += 1
            len < maxlen && (diff[len + 1] -= 1)
        end

        rc = 0
        for k in 1:maxlen
            rc += diff[k]
            running[k] += Ti(rc)
        end
    end

    return FiberChunkPlan{Ti}(ranges, startptrs)
end

function _get_fiber_chunk_plan!(plan::CyclicLayoutPlan{D,Ti,T}, orient::Int, P::Int) where {D,Ti<:Integer,T<:Number}
    key = (:fiber_chunk_plan, orient, P)
    chunkplan = get(plan.op_plan, key, nothing)
    if chunkplan === nothing
        chunkplan = _build_fiber_chunk_plan(plan.layouts[orient], P)
        plan.op_plan[key] = chunkplan
    end
    return chunkplan
end

function _get_fiber_chunk_buffers!(plan::CyclicLayoutPlan{D,Ti,T}, P::Int) where {D,Ti<:Integer,T<:Number}
    key = (:fiber_chunk_buffers, P)
    bufs = get(plan.op_plan, key, nothing)
    maxlen = length(plan.write_ptr)
    if bufs === nothing || length(bufs.bufA) != P || any(length(v) != maxlen for v in bufs.bufA)
        bufs = FiberChunkBuffers{T,Ti}([Vector{T}(undef, maxlen) for _ in 1:P],
                                       [Vector{T}(undef, maxlen) for _ in 1:P],
                                       [Vector{T}(undef, maxlen) for _ in 1:P],
                                       [Vector{Ti}(undef, maxlen) for _ in 1:P])
        plan.op_plan[key] = bufs
    end
    return bufs
end

@inline function _apply_fiber_chunk!(destdata::AbstractVector,
                                     srcdata::AbstractVector,
                                     fibers::Vector{<:LastDimFiber},
                                     range::UnitRange{Int},
                                     rowptr::AbstractVector{<:Integer},
                                     ops::Tuple,
                                     planvecs::Tuple,
                                     axis_last::AbstractUnivariateNodes,
                                     bufA::AbstractVector{ElT},
                                     bufB::AbstractVector{ElT},
                                     work::AbstractVector{ElT},
                                     ::Type{ElT}) where {ElT}
    @inbounds for idx in range
        fib = fibers[idx]
        seg = @view srcdata[fib.src_offset:(fib.src_offset + fib.len - 1)]
        bufA1 = @view bufA[1:fib.len]
        bufB1 = @view bufB[1:fib.len]
        work1 = @view work[1:fib.len]
        _run_pipeline_scatter!(destdata, rowptr, ops, planvecs, seg,
                               bufA1, bufB1, work1,
                               axis_last, fib.last_refinement, ElT)
    end
    return destdata
end

"""Apply a line operator along the last storage dimension and cycle the layout.

This is the core primitive of the unidirectional sparse grid operator
infrastructure.

`op` is a 1D line operator (or a composite of them) applied to each
last-dimension fiber.

Output is written in the cycled recursive layout (last storage dim moved to the
front), and `dest.perm` is set to `cycle_last_to_front(src.perm)`.
"""
function apply_lastdim_cycled!(dest::OrientedCoeffs{D,ElT},
                               src::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}

    orient = get(plan.perm_to_orient, src.perm, 0)
    orient == 0 && throw(ArgumentError("perm=$(src.perm) is not a cyclic orientation of the provided plan"))
    layout = plan.layouts[orient]

    maxlen = Int(layout.maxlen)

    # Active last physical dimension.
    last_phys = src.perm[end]
    axis_last = grid.spec.axes[last_phys]

    # Expand composite operators into atomic ops.
    ops = lineops(op)

    # Pre-resolve plan vectors for atomic ops that need plans.
    cap_last = Int(plan.refinement_caps[last_phys])
    planvecs = map(oi -> _get_lineplanvec!(plan.op_plan, oi, axis_last, cap_last, ElT), ops)

    fibers = layout.fibers
    P = min(_default_fiber_workers(), length(fibers))
    total_work = isempty(fibers) ? 0 : sum(fib.len for fib in fibers)
    use_threaded = P > 1 && length(fibers) >= 2 * P && total_work >= 4 * maxlen

    if !use_threaded
        write_ptr = plan.write_ptr
        @inbounds copyto!(write_ptr, 1, layout.first_offsets, 1, maxlen)
        scratch1 = plan.scratch1
        scratch2 = plan.scratch2
        scratch3 = plan.scratch3
        @inbounds for fib in fibers
            seg = @view src.data[fib.src_offset:(fib.src_offset + fib.len - 1)]
            bufA = @view scratch1[1:fib.len]
            bufB = @view scratch2[1:fib.len]
            work = @view scratch3[1:fib.len]
            _run_pipeline_scatter!(dest.data, write_ptr, ops, planvecs, seg,
                                   bufA, bufB, work,
                                   axis_last, fib.last_refinement, ElT)
        end
    else
        chunkplan = _get_fiber_chunk_plan!(plan, orient, P)
        bufs = _get_fiber_chunk_buffers!(plan, length(chunkplan.ranges))
        @sync for cid in 1:length(chunkplan.ranges)
            rg = chunkplan.ranges[cid]
            isempty(rg) && continue
            rowptr = bufs.rowptr[cid]
            copyto!(rowptr, 1, @view(chunkplan.startptrs[cid, 1:maxlen]), 1, maxlen)
            bufA = bufs.bufA[cid]
            bufB = bufs.bufB[cid]
            work = bufs.work[cid]
            Threads.@spawn _apply_fiber_chunk!(dest.data, src.data, fibers, rg, rowptr,
                                               ops, planvecs, axis_last,
                                               bufA, bufB, work, ElT)
        end
    end

    dest_perm = cycle_last_to_front(src.perm)
    return OrientedCoeffs(dest.data, dest_perm)
end

"""Convenience overload: build a temporary [`CyclicLayoutPlan`](@ref) and apply."""
function apply_lastdim_cycled!(dest::OrientedCoeffs{D,ElT},
                               src::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp) where {D,ElT}
    plan = CyclicLayoutPlan(grid, ElT)
    return apply_lastdim_cycled!(dest, src, grid, op, plan)
end

# ----------------------------------------------------------------------------
# k-step cyclic rotations (last→front only)

@inline function _rank_index(it_dst::RecursiveLayoutIterator{D}, levels, locals) where {D}
    I = it_dst.indexset
    perm = it_dst.perm
    caps = it_dst.refinement_caps
    S = it_dst.subtree_count
    Δs = it_dst.deltacounts
    offset = 1
    prefix = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    @inbounds for dim in 1:D
        pd = perm[dim]
        rcur = Int(levels[pd])
        i0 = Int(locals[pd])
        prefix_sv = SVector{D,Int}(prefix)
        for r in 0:(rcur - 1)
            w = Δs[pd][r + 1]
            w == 0 && continue
            prefix_r = setindex(prefix_sv, r, pd)
            offset += w * _subtree_size(S, dim + 1, prefix_r)
        end
        prefix[pd] = rcur
        offset += i0 * _subtree_size(S, dim + 1, SVector{D,Int}(prefix))
    end
    return offset
end

function _cyclic_rotate_by!(dest::OrientedCoeffs{D,ElT},
                            src::OrientedCoeffs{D,ElT},
                            grid::SparseGrid,
                            k::Int,
                            plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    kk = mod(k, D)
    kk == 0 && (copyto!(dest.data, src.data); return OrientedCoeffs(dest.data, src.perm))

    perm_dst = cycle_last_to_front(src.perm, kk)
    orient_src = get(plan.perm_to_orient, src.perm, 0)
    orient_dst = get(plan.perm_to_orient, perm_dst, 0)
    (orient_src == 0 || orient_dst == 0) && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_by!"))

    # One-step uses the fused fiber kernel (Identity op).
    if kk == 1
        return apply_lastdim_cycled!(dest, src, grid, IdentityLineOp(), plan)
    end

    spec = grid.spec
    I = spec.indexset
    caps = refinement_caps(I)

    deltac_src, subtree_src = _build_subtree_count(spec, src.perm, I)
    deltac_dst, subtree_dst = _build_subtree_count(spec, perm_dst, I)
    it_src = RecursiveLayoutIterator(spec.axes, I, src.perm, caps, deltac_src, subtree_src, subtree_src.total)
    it_dst = RecursiveLayoutIterator(spec.axes, I, perm_dst, caps, deltac_dst, subtree_dst, subtree_dst.total)

    i = 1
    nxt = iterate(it_src)
    @inbounds while nxt !== nothing
        _, st = nxt
        j = _rank_index(it_dst, st.levels, st.locals)
        dest.data[j] = src.data[i]
        i += 1
        nxt = iterate(it_src, st)
    end

    return OrientedCoeffs(dest.data, perm_dst)
end

function _cyclic_rotate_to!(dest::Vector{ElT},
                            src::Vector{ElT},
                            grid::SparseGrid,
                            perm_src::SVector{D,Int},
                            perm_dst::SVector{D,Int},
                            plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    perm_src == perm_dst && (copyto!(dest, src); return dest)
    osrc = get(plan.perm_to_orient, perm_src, 0)
    odst = get(plan.perm_to_orient, perm_dst, 0)
    (osrc == 0 || odst == 0) && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_to!"))
    Δ = mod(odst - osrc, D)

    src_or = OrientedCoeffs(src, perm_src)
    dst_or = OrientedCoeffs(dest, perm_src)
    _cyclic_rotate_by!(dst_or, src_or, grid, Δ, plan)
    return dest
end

# ----------------------------------------------------------------------------
# Tensor (dimension-wise) unidirectional application

"""Apply `D` cyclic unidirectional sweeps of a dimension-wise operator.

The operator is provided as a [`TensorOp`](@ref)-like object mapping each
physical dimension to a line operator.

This function mutates and returns `u` (the input buffer), matching the `!` suffix convention.
"""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    # Reuse plan-owned ping-pong buffer (separate from per-line scratch).
    bufdata = plan.unidir_buf
    buf = OrientedCoeffs(bufdata, u.perm)

    src = u
    dest = buf
    steps_done = 0

    while steps_done < D
        perm = src.perm

        # Count consecutive identity dimensions in the upcoming cyclic order.
        t = 0
        @inbounds for j in 0:(D - steps_done - 1)
            pd = perm[D - j]
            lop = lineop(op, pd)
            lop isa IdentityLineOp || break
            t += 1
        end

        if t > 0
            dest = _cyclic_rotate_by!(dest, src, grid, t, plan)
            src, dest = dest, src
            steps_done += t
            continue
        end

        phys = perm[end]
        lop = lineop(op, phys)
        dest = apply_lastdim_cycled!(dest, src, grid, lop, plan)
        src, dest = dest, src
        steps_done += 1
    end

    # Final orientation matches input after D cyclic steps.
    if src.data !== u.data
        copyto!(u.data, src.data)
    end
    return u
end

"""Apply a sequential composition of full sweeps.

This overload executes each [`TensorOp`](@ref) stored in `op` sequentially on
the same coefficient vector `u`, reusing the same [`CyclicLayoutPlan`](@ref).
"""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::CompositeTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    for sweep in op.ops
        apply_unidirectional!(u, grid, sweep, plan)
    end
    return u
end

"""Broadcast a line operator to all dimensions and apply."""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    return apply_unidirectional!(u, grid, tensorize(op, Val(D)), plan)
end

"""Convenience overload: build a temporary [`CyclicLayoutPlan`](@ref) and apply."""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op) where {D,ElT}
    plan = CyclicLayoutPlan(grid, ElT)
    return apply_unidirectional!(u, grid, op, plan)
end

# ============================================================================
# Sparse-grid matrix-vector products and explicit permutation sweeps
#
# This section was previously implemented in `matvec.jl` and is now merged here.


# -----------------------------------------------------------------------------
# 1D matrix line operators

"""Per-fiber multiplication by a level-dependent 1D matrix.

`mats[ℓ+1]` must be a square matrix of size `npoints(nodes, ℓ)` in the ordering
used by the coefficient vector along that fiber.
"""
struct LineMatrixOp{MatVecT<:AbstractVector} <: AbstractLineOp
    mats::MatVecT
end

lineop_style(::Type{<:LineMatrixOp}) = OutOfPlaceOp()

"""Per-fiber evaluation by a rectangular matrix with zero padding.

`V` has size `m×n`. When applied to a fiber `x::Vector` of length `n`, the
output `y` (still length `n`, to preserve the recursive/cyclic layout contract)
is:

* `y[1:m] = V * x`
* `y[m+1:n] = 0`
"""
struct LineEvalOp{T,MatT<:AbstractMatrix{T}} <: AbstractLineOp
    V::MatT
    m::Int
end

lineop_style(::Type{<:LineEvalOp}) = OutOfPlaceOp()

"""Per-fiber multiplication by a size-parameterized diagonal family.

`f(n, T)` must return a vector of length `n`, where `n = totalsize(axis, r)` is
the 1D fiber length at refinement index `r`.

The actual per-refinement diagonals are materialized by [`lineplan`](@ref) as
leading-prefix views of the max-size diagonal, and cached inside
[`CyclicLayoutPlan`](@ref).
"""
struct LineDiagonalOp{F} <: AbstractLineOp
    f::F
end

lineop_style(::Type{<:LineDiagonalOp}) = OutOfPlaceOp()
needs_plan(::Type{<:LineDiagonalOp}) = Val(true)

"""Per-fiber multiplication by a size-parameterized banded (or dense) matrix family.

`f(n, T)` must return a square matrix of size `n`, where `n = totalsize(axis, r)` is
the 1D fiber length at refinement index `r`.

Per-refinement matrices are materialized by [`lineplan`](@ref) by taking leading
principal blocks of the max-size matrix.
"""
struct LineBandedOp{Part,F} <: AbstractLineOp
    f::F
end

LineBandedOp(f) = LineBandedOp{:full,typeof(f)}(f)
LineBandedOp(::Val{Part}, f) where {Part} = LineBandedOp{Part,typeof(f)}(f)

lineop_style(::Type{<:LineBandedOp}) = OutOfPlaceOp()
needs_plan(::Type{<:LineBandedOp}) = Val(true)

@inline function _mat_for_level(op::LineMatrixOp, ℓ::Integer)
    @inbounds return op.mats[ℓ + 1]
end

# Cache keys for line plans.
@inline _lineplan_key(op::LineDiagonalOp, ::AbstractUnivariateNodes, ::Type{T}) where {T} =
    (typeof(op), T)
@inline _lineplan_key(op::LineBandedOp{Part,F}, ::AbstractUnivariateNodes, ::Type{T}) where {Part,F,T} =
    (LineBandedOp, F, T)

function lineplan(op::LineDiagonalOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T<:Number}
    maxr = Int(rmax)
    nmax = totalsize(axis, maxr)
    dmax = op.f(nmax, T)
    length(dmax) == nmax || throw(DimensionMismatch(
        "LineDiagonalOp: f(n,T) length=$(length(dmax)) but totalsize(axis,rmax)=$nmax (note: caching ignores axis type)"))

    plans = Vector{Any}(undef, maxr + 1)
    @inbounds for r in 0:maxr
        n = totalsize(axis, r)
        plans[r + 1] = @view dmax[1:n]
    end
    return plans
end

function lineplan(op::LineBandedOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T<:Number}
    maxr = Int(rmax)
    nmax = totalsize(axis, maxr)
    Amax = op.f(nmax, T)
    size(Amax, 1) == nmax || throw(DimensionMismatch(
        "LineBandedOp: f(n,T) size=$(size(Amax)) but totalsize(axis,rmax)=$nmax (note: caching ignores axis type)"))
    size(Amax, 2) == nmax || throw(DimensionMismatch(
        "LineBandedOp: f(n,T) size=$(size(Amax)) but totalsize(axis,rmax)=$nmax (note: caching ignores axis type)"))

    is_banded = Amax isa BandedMatrices.AbstractBandedMatrix
    plans = Vector{Any}(undef, maxr + 1)
    @inbounds for r in 0:maxr
        n = totalsize(axis, r)
        if n == 0
            plans[r + 1] = is_banded ? BandedMatrices.BandedMatrix{T}(undef, (0, 0), (0, 0)) : Matrix{T}(undef, 0, 0)
            continue
        end

        S = @view Amax[1:n, 1:n]
        if is_banded
            l, u = bandwidths(S)
            plans[r + 1] = BandedMatrices._BandedMatrix(BandedMatrices.bandeddata(S), n, l, u)
        else
            plans[r + 1] = S
        end
    end
    return plans
end

"""Apply an out-of-place line operator on a single fiber.

This is the extension hook used by the unidirectional engine for custom line
operators that do not fit the built-in `LineMatrixOp` / `LineDiagonalOp` /
`LineBandedOp` / `LineEvalOp` wrappers.

The optional `plan` argument is whatever was produced by [`lineplan`](@ref) for
this operator at the corresponding refinement index (i.e. `planvec[r+1]`), or `nothing`
if the operator does not require a plan.

The `work` argument is a caller-provided scratch buffer (same length as `outbuf`/`inp`).
Thread-safe planned operators must not mutate internal scratch; use `work` instead.
"""
apply_line!(outbuf::AbstractVector, op::AbstractLineOp, inp::AbstractVector, work::AbstractVector,
            nodes::AbstractUnivariateNodes, level::Int, plan) =
    throw(ArgumentError("unsupported out-of-place line op $(typeof(op))"))

apply_line!(outbuf::AbstractVector, op::AbstractLineOp, inp::AbstractVector, work::AbstractVector,
            nodes::AbstractUnivariateNodes, level::Int) =
    apply_line!(outbuf, op, inp, work, nodes, level, nothing)

"""Apply an in-place line operator on a single fiber.

In-place line ops may still take an optional per-refinement `plan` (see [`lineplan`](@ref)).

The `work` argument is a caller-provided scratch buffer (same length as `buf`).
Thread-safe planned operators must not mutate internal scratch; use `work` instead.
"""
@inline function apply_line!(op::AbstractLineOp, buf::AbstractVector, work::AbstractVector,
                             ::AbstractUnivariateNodes, r::Int, plan)
    is_planned_inplace(typeof(op)) === Val(true) ||
        throw(ArgumentError("unsupported in-place line op $(typeof(op))"))
    plan === nothing && throw(ArgumentError("missing plan for $(typeof(op)) at refinement index=$r"))
    _apply_plan!(plan_action(typeof(op)), plan, buf, work)
    return buf
end

apply_line!(op::AbstractLineOp, buf::AbstractVector, work::AbstractVector,
            nodes::AbstractUnivariateNodes, r::Int) =
    apply_line!(op, buf, work, nodes, r, nothing)

# Built-in in-place ops
@inline apply_line!(::IdentityLineOp, buf::AbstractVector, work::AbstractVector,
                    ::AbstractUnivariateNodes, ::Int, plan) = buf

function apply_line!(outbuf::AbstractVector{T}, op::LineMatrixOp, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    A = _mat_for_level(op, level)
    n = length(inp)
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    size(A, 1) == n || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    size(A, 2) == n || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    mul!(outbuf, A, inp)
    return outbuf
end

function apply_line!(outbuf::AbstractVector{T}, op::LineBandedOp{:full}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:full} at refinement index=$level"))
    A = plan
    n = length(inp)
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    size(A, 1) == n || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    size(A, 2) == n || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    mul!(outbuf, A, inp)
    return outbuf
end

function apply_line!(outbuf::AbstractVector{T}, op::LineDiagonalOp, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineDiagonalOp at refinement index=$level"))
    d = plan
    n = length(inp)
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    length(d) == n || throw(DimensionMismatch("diag length mismatch at refinement index=$level"))
    @inbounds for k in 1:n
        outbuf[k] = d[k] * inp[k]
    end
    return outbuf
end

@inline function _banded_lower_mul!(out::AbstractVector{T}, A, x::AbstractVector{T}) where {T}
    n = length(x)
    length(out) == n || throw(DimensionMismatch("fiber length mismatch"))
    if A isa BandedMatrices.AbstractBandedMatrix
        l, _ = bandwidths(A)
        @inbounds for i in 1:n
            acc = zero(T)
            jmin = max(1, i - l)
            for j in jmin:i
                acc += A[i, j] * x[j]
            end
            out[i] = acc
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in 1:i
                acc += A[i, j] * x[j]
            end
            out[i] = acc
        end
    end
    return out
end

@inline function _banded_upper_strict_mul!(out::AbstractVector{T}, A, x::AbstractVector{T}) where {T}
    n = length(x)
    length(out) == n || throw(DimensionMismatch("fiber length mismatch"))
    if A isa BandedMatrices.AbstractBandedMatrix
        _, u = bandwidths(A)
        @inbounds for i in 1:n
            acc = zero(T)
            jmax = min(n, i + u)
            for j in (i + 1):jmax
                acc += A[i, j] * x[j]
            end
            out[i] = acc
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in (i + 1):n
                acc += A[i, j] * x[j]
            end
            out[i] = acc
        end
    end
    return out
end

function apply_line!(outbuf::AbstractVector{T}, ::LineBandedOp{:lower}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:lower} at refinement index=$level"))
    A = plan
    size(A, 1) == length(inp) || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    size(A, 2) == length(inp) || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    return _banded_lower_mul!(outbuf, A, inp)
end

function apply_line!(outbuf::AbstractVector{T}, ::LineBandedOp{:upper}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:upper} at refinement index=$level"))
    A = plan
    size(A, 1) == length(inp) || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    size(A, 2) == length(inp) || throw(DimensionMismatch("matrix size mismatch at refinement index=$level"))
    return _banded_upper_strict_mul!(outbuf, A, inp)
end

function apply_line!(outbuf::AbstractVector{T}, op::LineEvalOp{T}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    V = op.V
    m = op.m
    n = length(inp)
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    size(V, 2) == n || throw(DimensionMismatch("eval matrix column mismatch at refinement index=$level"))
    size(V, 1) == m || throw(DimensionMismatch("eval matrix row mismatch"))
    m <= n || throw(DimensionMismatch("cannot pad $m evaluation rows into fiber length $n"))
    if m == 0
        fill!(outbuf, zero(T))
    else
        mul!(view(outbuf, 1:m), V, inp)
        m < n && fill!(view(outbuf, (m + 1):n), zero(T))
    end
    return outbuf
end

@inline function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                op::LineDiagonalOp, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                                work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineDiagonalOp at refinement index=$r"))
    d = plan
    length(d) == length(inp) || throw(DimensionMismatch("diag length mismatch at refinement index=$r"))
    @inbounds for k in eachindex(inp)
        idx = rowptr[k]
        destdata[idx] = d[k] * inp[k]
        rowptr[k] = idx + 1
    end
    return destdata
end

@inline function _banded_scatter_full!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                       A, x::AbstractVector{T}) where {T}
    n = length(x)
    if A isa BandedMatrices.AbstractBandedMatrix
        l, u = bandwidths(A)
        @inbounds for i in 1:n
            acc = zero(T)
            jmin = max(1, i - l)
            jmax = min(n, i + u)
            for j in jmin:jmax
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] = acc
            rowptr[i] = idx + 1
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in 1:n
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] = acc
            rowptr[i] = idx + 1
        end
    end
    return destdata
end

@inline function _banded_scatter_lower!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                        A, x::AbstractVector{T}) where {T}
    n = length(x)
    if A isa BandedMatrices.AbstractBandedMatrix
        l, _ = bandwidths(A)
        @inbounds for i in 1:n
            acc = zero(T)
            jmin = max(1, i - l)
            for j in jmin:i
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] = acc
            rowptr[i] = idx + 1
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in 1:i
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] = acc
            rowptr[i] = idx + 1
        end
    end
    return destdata
end

@inline function _banded_scatter_upper!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                        A, x::AbstractVector{T}) where {T}
    n = length(x)
    if A isa BandedMatrices.AbstractBandedMatrix
        _, u = bandwidths(A)
        @inbounds for i in 1:n
            acc = zero(T)
            jmax = min(n, i + u)
            for j in (i + 1):jmax
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] = acc
            rowptr[i] = idx + 1
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in (i + 1):n
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] = acc
            rowptr[i] = idx + 1
        end
    end
    return destdata
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:full}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:full} at refinement index=$r"))
    A = plan
    if A isa BandedMatrices.AbstractBandedMatrix
        return _banded_scatter_full!(destdata, rowptr, A, inp)
    else
        apply_line!(outbuf, op, inp, work, axis, r, plan)
        return _scatter_copy!(destdata, rowptr, outbuf)
    end
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:lower}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:lower} at refinement index=$r"))
    return _banded_scatter_lower!(destdata, rowptr, plan, inp)
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:upper}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:upper} at refinement index=$r"))
    return _banded_scatter_upper!(destdata, rowptr, plan, inp)
end

@inline function _split_updown_matrix(A::AbstractMatrix{T}) where {T}
    L = Matrix{T}(A)
    U = Matrix{T}(A)
    n, m = size(A)
    n == m || throw(DimensionMismatch("expected square matrix, got size $(size(A))"))
    @inbounds for j in 1:n
        for i in 1:n
            if i < j
                L[i, j] = zero(T)
            elseif i > j
                U[i, j] = zero(T)
            else
                U[i, j] = zero(T)  # keep diagonal in L
            end
        end
    end
    return LowerTriangular(L), UpperTriangular(U)
end

function _split_updown_matrix(A::BandedMatrices.AbstractBandedMatrix{T}) where {T}
    n, m = size(A)
    n == m || throw(DimensionMismatch("expected square matrix, got $(n)×$(m)"))

    l, u = BandedMatrices.bandwidths(A)
    L = BandedMatrices.BandedMatrix{T}(undef, (n, n), (l, 0))
    U = BandedMatrices.BandedMatrix{T}(undef, (n, n), (0, u))
    fill!(L, zero(T))
    fill!(U, zero(T))

    @inbounds for i in 1:n
        # Lower (incl. diagonal)
        for j in max(1, i - l):i
            L[i, j] = A[i, j]
        end
        # Strict upper
        for j in (i + 1):min(n, i + u)
            U[i, j] = A[i, j]
        end
    end
    return L, U
end

"""Split an `IdentityLineOp` into L+U parts (identity + zero)."""
updown(::IdentityLineOp) = (IdentityLineOp(), ZeroLineOp())

"""Split a `LineMatrixOp` into ≤k-lower/upper parts (additive L+U).

The diagonal is assigned to the lower part, so that `L + U == A`.
"""
function updown(op::LineMatrixOp)
    isempty(op.mats) && throw(ArgumentError("cannot split an empty LineMatrixOp"))

    # Prune trivial cases: already lower/upper triangular at every level.
    all_lower = true
    all_upper = true
    @inbounds for A in op.mats
        all_lower &= istril(A)
        all_upper &= istriu(A)
    end
    if all_lower
        return op, ZeroLineOp()
    elseif all_upper
        return ZeroLineOp(), op
    end

    T = eltype(first(op.mats))
    matsL = Vector{AbstractMatrix{T}}(undef, length(op.mats))
    matsU = Vector{AbstractMatrix{T}}(undef, length(op.mats))
    @inbounds for i in eachindex(op.mats)
        L, U = _split_updown_matrix(op.mats[i])
        matsL[i] = L
        matsU[i] = U
    end
    return LineMatrixOp(matsL), LineMatrixOp(matsU)
end

"""Split a `LineDiagonalOp` into ≤k-lower/upper parts (additive L+U).

The diagonal is assigned to the lower part and the upper part is identically zero.
"""
function updown(op::LineDiagonalOp)
    return op, ZeroLineOp()
end

"""Split a `LineBandedOp` into ≤k-lower/upper parts (additive L+U)."""
function updown(op::LineBandedOp{:full})
    return LineBandedOp(Val(:lower), op.f), LineBandedOp(Val(:upper), op.f)
end
updown(op::LineBandedOp{:lower}) = (op, ZeroLineOp())
updown(op::LineBandedOp{:upper}) = (ZeroLineOp(), op)


# -----------------------------------------------------------------------------
# UpDown tensor-product operator (cyclic-only)

"""Tensor-product operator applied via an additive `L+U` split per dimension.

For each 1D line operator `A`, we form `A = L + U` where:
- `L` is (≤k)-lower-triangular,
- `U` is (≤k)-upper-triangular,
with `L + U == A`. (When a factor is already triangular, we may choose `L=0` or `U=0` to avoid branching.)

The restricted sparse grid tensor product is then evaluated by expanding the
Kronecker product into a sum of `2^m` (or `2^(m-1)` if one split is omitted), where `m` is the number of dimensions that actually require an L+U split.

This yields a sum of triangular tensor terms, each applied using the unidirectional principle.

This implementation relies **only** on cyclic layout rotations
(`CyclicLayoutPlan`) and does not construct or use any general permutation maps.

If `omit_dim != 0`, the split is omitted in that physical dimension and the full
matrix is treated as the unique non-triangular pivot permitted by Theorem 3.

If `omit_dim == 0`, the omission dimension is chosen automatically using a
simple heuristic based on `Threads.nthreads()`.
"""
struct UpDownTensorOp{D,OpsT<:NTuple{D,AbstractLineOp},LOpsT<:NTuple{D,AbstractLineOp},UOpsT<:NTuple{D,AbstractLineOp}} <: AbstractTensorOp{D}
    full::OpsT
    lower::LOpsT
    upper::UOpsT
    omit_dim::Int
    split_dims::Vector{Int}  # dims where both L and U are nontrivial (branch bits)
end

"""Construct and pre-split a per-dimension operator tuple."""
function UpDownTensorOp(ops::NTuple{D,AbstractLineOp}; omit_dim::Integer=0) where {D}
    parts = ntuple(k -> updown(ops[k]), Val(D))
    lower = ntuple(k -> parts[k][1], Val(D))
    upper = ntuple(k -> parts[k][2], Val(D))

    split_dims = Int[]
    @inbounds for d in 1:D
        (!iszero(lower[d]) && !iszero(upper[d])) && push!(split_dims, d)
    end

    return UpDownTensorOp{D,typeof(ops),typeof(lower),typeof(upper)}(ops, lower, upper, Int(omit_dim), split_dims)
end

lineop(op::UpDownTensorOp{D}, d::Integer) where {D} = op.full[d]

@inline _perm_id(::Val{D}) where {D} = SVector{D,Int}(ntuple(identity, D))

# Choose which dimension (if any) to omit from L+U splitting.
function _choose_omit_dim(op::UpDownTensorOp{D}, grid::SparseGrid) where {D}
    op.omit_dim != 0 && return op.omit_dim

    m = length(op.split_dims)
    m == 0 && return 0

    # Heuristic: if we have enough threads to cover ~half the terms, keep all splits.
    nthreads() >= (1 << max(m - 1, 0)) && return 0

    # Otherwise omit one split; choose the dimension with the longest 1D grid.
    caps = refinement_caps(grid.spec.indexset)
    maxlen_by_dim = ntuple(d -> totalsize(grid.spec.axes[d], Int(caps[d])), Val(D))

    best = op.split_dims[1]
    best_len = maxlen_by_dim[best]
    @inbounds for d in op.split_dims
        if maxlen_by_dim[d] > best_len
            best = d
            best_len = maxlen_by_dim[d]
        end
    end
    return best
end

function _cyclic_perm_with_front(plan::CyclicLayoutPlan{D}, d::Int) where {D}
    @inbounds for layout in plan.layouts
        layout.perm[1] == d && return layout.perm
    end
    throw(ArgumentError("no cyclic orientation with front=$d"))
end

"""Apply an [`UpDownTensorOp`](@ref) using cyclic-only rotations."""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::UpDownTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    idperm = _perm_id(Val(D))
    u.perm == idperm || throw(ArgumentError("UpDownTensorOp expects u.perm == identity; got $(u.perm)"))

    omit_dim = _choose_omit_dim(op, grid)

    # Base orientation: identity if no omission; otherwise rotate so that `omit_dim` is visited
    # last in a cyclic sweep (i.e. it is at the *front* of the storage permutation).
    perm0 = (omit_dim == 0) ? idperm : _cyclic_perm_with_front(plan, omit_dim)

    # Rotate the input once if needed.
    xbuf = plan.x_buf
    if perm0 == idperm
        copyto!(xbuf, u.data)
    else
        _cyclic_rotate_to!(xbuf, u.data, grid, idperm, perm0, plan)
    end
    x0 = OrientedCoeffs(xbuf, perm0)

    # Terms only branch over dimensions that have *both* L and U nontrivial.
    split_dims_eff = (omit_dim == 0) ? op.split_dims : [d for d in op.split_dims if d != omit_dim]
    nbits = length(split_dims_eff)
    nterms = UInt(1) << nbits

    # Map physical dimension -> bit position (0-based) for the branching dims.
    bitpos = fill(-1, D)
    @inbounds for (i, d) in enumerate(split_dims_eff)
        bitpos[d] = i - 1
    end

    # Accumulate in perm0 orientation.
    ybuf = plan.acc_buf
    fill!(ybuf, zero(ElT))

    work_buf = plan.work_buf

    for mask in UInt(0):(nterms - 1)
        copyto!(work_buf, x0.data)
        w = OrientedCoeffs(work_buf, perm0)

        ops1 = ntuple(d -> begin
            if d == omit_dim
                op.full[d]
            else
                L = op.lower[d]
                U = op.upper[d]
                if iszero(L)
                    U
                elseif iszero(U)
                    IdentityLineOp()
                else
                    b = bitpos[d]
                    (((mask >> b) & 0x1) == 0x1) ? U : IdentityLineOp()
                end
            end
        end, Val(D))

        ops2 = ntuple(d -> begin
            if d == omit_dim
                IdentityLineOp()
            else
                L = op.lower[d]
                U = op.upper[d]
                if iszero(L)
                    IdentityLineOp()
                elseif iszero(U)
                    L
                else
                    b = bitpos[d]
                    (((mask >> b) & 0x1) == 0x0) ? L : IdentityLineOp()
                end
            end
        end, Val(D))

        apply_unidirectional!(w, grid, TensorOp(ops1), plan)
        apply_unidirectional!(w, grid, TensorOp(ops2), plan)

        @inbounds @simd for i in eachindex(ybuf)
            ybuf[i] += w.data[i]
        end
    end

    # Rotate back once if needed.
    if perm0 == idperm
        copyto!(u.data, ybuf)
    else
        _cyclic_rotate_to!(u.data, ybuf, grid, perm0, idperm, plan)
    end

    return u
end
