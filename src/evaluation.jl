"""Evaluation of sparse grid expansions on point sets."""

# -----------------------------------------------------------------------------
# Point sets

abstract type AbstractPointSet{D,T} end

"""Return the spatial dimension `D` of the point set."""
Base.ndims(::AbstractPointSet{D}) where {D} = D
dim(::AbstractPointSet{D}) where {D} = D

"""Scattered (unstructured) points.

Points are stored as a `D×M` matrix where each column is one point.
"""
struct ScatteredPoints{D,T,AT<:AbstractMatrix{T}} <: AbstractPointSet{D,T}
    X::AT
end

ScatteredPoints(X::AbstractMatrix{T}) where {T} = ScatteredPoints{size(X, 1),T,typeof(X)}(X)
Base.length(P::ScatteredPoints) = size(P.X, 2)

"""Tensor-product structured points, represented as a tuple of 1D grids."""
struct TensorProductPoints{D,T,VT<:NTuple{D,AbstractVector{T}}} <: AbstractPointSet{D,T}
    pts::VT
end

function TensorProductPoints(pts::NTuple{D,AbstractVector}) where {D}
    Ts = ntuple(d -> eltype(pts[d]), Val(D))
    T = promote_type(Ts...)
    ptsT = ntuple(Val(D)) do d
        v = pts[d]
        eltype(v) === T ? v : T.(v)
    end
    return TensorProductPoints(ptsT)
end

Base.length(P::TensorProductPoints{D}) where {D} = prod(ntuple(d -> length(P.pts[d]), Val(D)))
npoints(P::TensorProductPoints, d::Integer) = length(P.pts[d])

# -----------------------------------------------------------------------------
# Basis helpers

@inline _normalize_basis(basis::AbstractUnivariateBasis, ::Val{D}) where {D} = ntuple(_ -> basis, Val(D))

@inline function _normalize_basis(basis::NTuple{D,<:AbstractUnivariateBasis}, ::Val{D}) where {D}
    return basis
end

@inline function _normalize_basis(basis::Tuple, ::Val{D}) where {D}
    length(basis) == D || throw(DimensionMismatch("basis tuple length mismatch"))
    return ntuple(d -> begin
        b = basis[d]
        b isa AbstractUnivariateBasis || throw(ArgumentError("basis[$d] is not an AbstractUnivariateBasis"))
        b
    end, Val(D))
end

@inline function _enclosing_level(basis::AbstractUnivariateBasis,
                                 nodes::AbstractUnivariateNodes,
                                 L0::Int,
                                 n_req::Int)
    L = L0
    while ncoeff(basis, nodes, L) < n_req
        L += 1
    end
    return L
end

# -----------------------------------------------------------------------------
# Evaluation plan + backends

abstract type AbstractEvaluationBackend end

"""Backend that evaluates tensor-product point sets via the unidirectional principle."""
struct UnidirectionalBackend{D,Ti<:Integer,ElT,GridWT,PlanWT,OpT} <: AbstractEvaluationBackend
    gridW::GridWT
    planW::PlanWT
    x_to_w::Vector{Ti}
    m::SVector{D,Int}
    nW::SVector{D,Int}
    stridesW::SVector{D,Int}
    evalV::NTuple{D,Base.RefValue{Union{Nothing,Matrix{ElT}}}}
    opW::Base.RefValue{Union{Nothing,OpT}}
end

"""Backend that evaluates by direct summation (fallback)."""
struct NaiveBackend{D,ElT,VandermondeT} <: AbstractEvaluationBackend
    V::VandermondeT
end

mutable struct _PseudoRecEvalFrame{Ti<:Integer,ElT}
    dim::Int
    r::Ti
    k::Ti
    offset::Ti
    pref::ElT
end

"""Backend that evaluates scattered points with local-support bases using a pseudorecursive traversal."""
struct PseudorecursiveBackend{D,Ti<:Integer,ElT,ItT,LevelsT} <: AbstractEvaluationBackend
    it::ItT
    stack::Vector{_PseudoRecEvalFrame{Ti,ElT}}
    levels::LevelsT
end

"""A reusable evaluation plan for a fixed grid, bases, and point set."""
struct EvaluationPlan{D,ElT,GridT,BasesT,PointsT,BackendT<:AbstractEvaluationBackend}
    grid::GridT
    bases::BasesT
    points::PointsT
    backend::BackendT
end

Base.eltype(::Type{EvaluationPlan{D,ElT}}) where {D,ElT} = ElT
Base.eltype(plan::EvaluationPlan) = eltype(typeof(plan))

"""Number of input coefficients (columns of the evaluation operator)."""
nin(plan::EvaluationPlan) = length(plan.grid)

"""Number of output values (rows of the evaluation operator)."""
nout(plan::EvaluationPlan) = length(plan.points)

"""Create an evaluation plan.

`bases` may be a single univariate basis (used for all dimensions) or an explicit `NTuple{D}`.

`backend=:auto` selects:

* `:unidirectional` for `TensorProductPoints`,
* `:pseudorecursive` for `ScatteredPoints` with all `LocalSupport()` bases,
* `:naive` otherwise.
"""
function plan_evaluate(grid::SparseGrid{<:SparseGridSpec{D}},
                       bases::Union{AbstractUnivariateBasis,NTuple{D,AbstractUnivariateBasis}},
                       points::AbstractPointSet{D},
                       ::Type{ElT}=Float64;
                       backend::Union{Symbol,AbstractEvaluationBackend}=:auto,
                       Ti::Type{<:Integer}=Int) where {D,ElT}
    basesT = _normalize_basis(bases, Val(D))

    backend_obj = if backend isa AbstractEvaluationBackend
        backend
    else
        backend_sym = backend
        if backend_sym === :auto
            if points isa TensorProductPoints{D}
                backend_sym = :unidirectional
            elseif points isa ScatteredPoints{D}
                backend_sym = all(d -> support_style(basesT[d]) isa LocalSupport, 1:D) ? :pseudorecursive : :naive
            else
                backend_sym = :naive
            end
        end

        if backend_sym === :unidirectional
            points isa TensorProductPoints{D} || throw(ArgumentError("Unidirectional backend requires TensorProductPoints"))
            P = points
            nodes = grid.spec.axes
            capsX = refinement_caps(grid.spec.indexset)

            m = SVector{D,Int}(ntuple(d -> length(P.pts[d]), Val(D)))
            capW = SVector{D,Int}(ntuple(d -> _enclosing_level(basesT[d], nodes[d], capsX[d], m[d]), Val(D)))

            IW = FullTensorIndexSet(Val(D), maximum(capW); cap=capW)
            gridW = SparseGrid(SparseGridSpec(nodes, IW))

            planW = CyclicLayoutPlan(gridW, ElT; Ti=Ti)
            x_to_w = _subsequence_index_map(traverse(grid), traverse(gridW); coord_map=identity, Ti=Ti)

            nW = SVector{D,Int}(ntuple(d -> ncoeff(basesT[d], nodes[d], capW[d]), Val(D)))
            @inbounds for d in 1:D
                nW[d] == totalsize(nodes[d], capW[d]) ||
                    throw(ArgumentError("ncoeff(basis,axis,r) must match totalsize(axis,r) for evaluation (dim=$d)"))
            end

            stridesW = let n = nW
                s = MVector{D,Int}(undef)
                s[D] = 1
                @inbounds for d in (D - 1):-1:1
                    s[d] = s[d + 1] * n[d + 1]
                end
                SVector{D,Int}(s)
            end

            evalV = ntuple(_ -> Ref{Union{Nothing,Matrix{ElT}}}(nothing), Val(D))
            OpT = TensorOp{D,NTuple{D,LineEvalOp{ElT,Matrix{ElT}}}}
            opW = Ref{Union{Nothing,OpT}}(nothing)
            UnidirectionalBackend{D,Ti,ElT,typeof(gridW),typeof(planW),OpT}(
                gridW, planW, x_to_w, m, nW, stridesW, evalV, opW)

        elseif backend_sym === :pseudorecursive
            points isa ScatteredPoints{D} || throw(ArgumentError("Pseudorecursive backend requires ScatteredPoints"))
            all(d -> support_style(basesT[d]) isa LocalSupport, 1:D) ||
                throw(ArgumentError("Pseudorecursive backend requires LocalSupport() bases in all dimensions"))
            it = traverse(grid)
            stack = Vector{_PseudoRecEvalFrame{Ti,ElT}}()
            sizehint!(stack, 4D)
            levels = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
            PseudorecursiveBackend{D,Ti,ElT,typeof(it),typeof(levels)}(it, stack, levels)

        elseif backend_sym === :naive
            nodes = grid.spec.axes
            cap = grid.spec.indexset.cap
            V = ntuple(Val(D)) do d
                if support_style(basesT[d]) isa LocalSupport
                    nothing
                else
                    n = ncoeff(basesT[d], nodes[d], cap[d])
                    xs = points isa TensorProductPoints{D} ? points.pts[d] : view(points.X, d, :)
                    vandermonde(basesT[d], xs, n; T=ElT)
                end
            end
            NaiveBackend{D,ElT,typeof(V)}(V)

        else
            throw(ArgumentError("Unknown backend: $backend_sym"))
        end
    end

    return EvaluationPlan{D,ElT,typeof(grid),typeof(basesT),typeof(points),typeof(backend_obj)}(
        grid, basesT, points, backend_obj)
end

"""Evaluate into a preallocated output vector."""
function evaluate!(y::AbstractVector,
                   plan::EvaluationPlan{D,ElT},
                   x::AbstractVector{ElT}) where {D,ElT}
    length(x) == nin(plan) || throw(DimensionMismatch("x has length $(length(x)) but expected $(nin(plan))"))
    length(y) == nout(plan) || throw(DimensionMismatch("y has length $(length(y)) but expected $(nout(plan))"))
    return _evaluate_backend!(y, plan.backend, plan, x)
end

"""Allocate-and-return evaluation."""
function evaluate(plan::EvaluationPlan{D,ElT}, x::AbstractVector{ElT}) where {D,ElT}
    y = Vector{ElT}(undef, nout(plan))
    return evaluate!(y, plan, x)
end

# -----------------------------------------------------------------------------
# Sampling a function on a sparse grid

"""Sample a function `f` on the sparse grid points.

The returned vector is ordered consistently with `traverse(grid)`.
`f` is called with an `SVector{D,T}`.
"""
function evaluate(grid::SparseGrid{<:SparseGridSpec{D}}, f, ::Type{T}=Float64) where {D,T}
    y = Vector{T}(undef, length(grid))
    return evaluate!(y, grid, f)
end

"""In-place version of [`evaluate(grid, f)`](@ref)."""
function evaluate!(y::AbstractVector{T}, grid::SparseGrid{<:SparseGridSpec{D}}, f) where {D,T}
    length(y) == length(grid) || throw(DimensionMismatch("y has length $(length(y)) but expected $(length(grid))"))

    caps = refinement_caps(grid)
    xs = ntuple(d -> points(grid.spec.axes[d], caps[d]), Val(D))
    it = traverse(grid)

    @inbounds for (i, c) in enumerate(it)
        xpt = SVector{D,T}(ntuple(d -> T(xs[d][c[d]]), Val(D)))
        y[i] = f(xpt)
    end
    return y
end

# -----------------------------------------------------------------------------
# Backend implementations

function _eval_tensorop!(backend::UnidirectionalBackend{D,Ti,ElT},
                         bases::NTuple{D,AbstractUnivariateBasis},
                         P::TensorProductPoints{D}) where {D,Ti,ElT}
    op = backend.opW[]
    op !== nothing && return op

    for d in 1:D
        backend.evalV[d][] === nothing || continue
        backend.evalV[d][] = vandermonde(bases[d], P.pts[d], backend.nW[d]; T=ElT)
    end

    ops = ntuple(d -> begin
        V = backend.evalV[d][]
        V === nothing && error("internal: eval matrix missing for dimension $d")
        LineEvalOp(V, backend.m[d])
    end, Val(D))
    backend.opW[] = TensorOp(ops)
    return backend.opW[]
end

function _evaluate_backend!(y::AbstractVector,
                            backend::UnidirectionalBackend{D,Ti,ElT},
                            plan::EvaluationPlan{D,ElT},
                            x::AbstractVector{ElT}) where {D,Ti,ElT}
    P = plan.points
    P isa TensorProductPoints{D} || throw(ArgumentError("Unidirectional backend requires TensorProductPoints"))
    backend.m == SVector{D,Int}(ntuple(d -> length(P.pts[d]), Val(D))) ||
        throw(ArgumentError("point-set size mismatch with plan"))

    w = backend.planW.work_buf
    _scatter_zero!(w, x, backend.x_to_w)

    u = OrientedCoeffs{D}(w)
    opW = _eval_tensorop!(backend, plan.bases, P)
    apply_unidirectional!(u, backend.gridW, opW, backend.planW)

    lin = LinearIndices(Tuple(backend.m))
    @inbounds for I in CartesianIndices(Tuple(backend.m))
        j = 1
        for d in 1:D
            j += (I[d] - 1) * backend.stridesW[d]
        end
        y[lin[I]] = u.data[j]
    end
    return y
end

function _evaluate_backend!(y::AbstractVector,
                            backend::NaiveBackend{D,ElT},
                            plan::EvaluationPlan{D,ElT},
                            x::AbstractVector{ElT}) where {D,ElT}
    fill!(y, zero(ElT))

    V = backend.V
    pts = plan.points
    nodes = plan.grid.spec.axes
    bases = plan.bases

    if pts isa TensorProductPoints{D}
        lengths = ntuple(d -> length(pts.pts[d]), Val(D))
        lin = LinearIndices(CartesianIndices(lengths))

        it = traverse(plan.grid)
        nxt = iterate(it)
        nxt === nothing && return y
        coords, st = nxt
        idx = 1
        while true
            c = x[idx]
            if c != zero(ElT)
                @inbounds for I in CartesianIndices(lengths)
                    acc = c
                    for d in 1:D
                        if V[d] === nothing
                            acc *= ElT(eval_local(bases[d], nodes[d], st.levels[d], st.locals[d], pts.pts[d][I[d]]))
                        else
                            acc *= V[d][I[d], coords[d]]
                        end
                    end
                    y[lin[I]] += acc
                end
            end

            nxt = iterate(it, st)
            nxt === nothing && break
            coords, st = nxt
            idx += 1
        end

    elseif pts isa ScatteredPoints{D}
        M = length(pts)

        it = traverse(plan.grid)
        nxt = iterate(it)
        nxt === nothing && return y
        coords, st = nxt
        idx = 1
        while true
            c = x[idx]
            if c != zero(ElT)
                @inbounds for j in 1:M
                    acc = c
                    for d in 1:D
                        if V[d] === nothing
                            acc *= ElT(eval_local(bases[d], nodes[d], st.levels[d], st.locals[d], pts.X[d, j]))
                        else
                            acc *= V[d][j, coords[d]]
                        end
                    end
                    y[j] += acc
                end
            end

            nxt = iterate(it, st)
            nxt === nothing && break
            coords, st = nxt
            idx += 1
        end
    else
        throw(ArgumentError("Naive backend does not support point set type $(typeof(pts))"))
    end

    return y
end

function _evaluate_backend!(y::AbstractVector,
                            backend::PseudorecursiveBackend{D,Ti,ElT},
                            plan::EvaluationPlan{D,ElT},
                            x::AbstractVector{ElT}) where {D,Ti,ElT}
    pts = plan.points
    pts isa ScatteredPoints{D} || throw(ArgumentError("Pseudorecursive backend requires ScatteredPoints"))
    return _pseudorecursive_evaluate!(y, pts, x, plan.bases, backend)
end

function _pseudorecursive_evaluate!(y::AbstractVector{ElT},
                                   pts::ScatteredPoints{D},
                                   coeffs::AbstractVector{ElT},
                                   bases::NTuple{D,AbstractUnivariateBasis},
                                   backend::PseudorecursiveBackend{D,Ti,ElT}) where {D,Ti,ElT}
    it = backend.it
    stack = backend.stack
    levels = backend.levels

    nodes = it.nodes
    I = it.indexset
    perm = it.perm
    caps = it.refinement_caps
    Δs = it.deltacounts
    S = it.subtree_count

    X = pts.X
    M = length(pts)

    @inbounds for j in 1:M
        empty!(stack)
        fill!(levels, 0)
        push!(stack, _PseudoRecEvalFrame{Ti,ElT}(1, 0, 1, 1, one(ElT)))
        acc = zero(ElT)

        while !isempty(stack)
            f = stack[end]
            pd = perm[f.dim]
            prefix = _prefix_from_refinements(levels, perm, f.dim)
            maxr = _maxadmissible(I, caps, prefix, pd)
            if f.r > maxr
                levels[pd] = 0
                pop!(stack)
                continue
            end

            block_len = Δs[pd][f.r + 1]
            if block_len == 0
                f.r += 1
                f.k = 1
                continue
            end

            prefix2 = setindex(prefix, Int(f.r), pd)
            child_subtree = _subtree_size(S, f.dim + 1, prefix2)

            act = active_local(bases[pd], nodes[pd], f.r, X[pd, j])
            nact = length(act)
            if f.k <= nact
                loc = act[f.k]
                f.k += 1
                val = eval_local(bases[pd], nodes[pd], f.r, loc, X[pd, j])
                val == 0 && continue
                newpref = f.pref * ElT(val)
                child_offset = f.offset + Ti(loc) * child_subtree
                levels[pd] = Int(f.r)
                if f.dim == D
                    acc += coeffs[child_offset] * newpref
                else
                    push!(stack, _PseudoRecEvalFrame{Ti,ElT}(f.dim + 1, 0, 1, child_offset, newpref))
                end
            else
                f.offset += Ti(block_len) * child_subtree
                f.r += 1
                f.k = 1
            end
        end

        y[j] = acc
    end
    return y
end

# -----------------------------------------------------------------------------
# Linear operator wrapper

"""Linear operator wrapper for an evaluation plan."""
struct SparseEvalOp{PlanT,ElT} <: AbstractMatrix{ElT}
    plan::PlanT
end

SparseEvalOp(plan::EvaluationPlan{D,ElT}) where {D,ElT} = SparseEvalOp{typeof(plan),ElT}(plan)

Base.size(op::SparseEvalOp) = (nout(op.plan), nin(op.plan))

function LinearAlgebra.mul!(y::AbstractVector{ElT},
                            op::SparseEvalOp{PlanT,ElT},
                            x::AbstractVector{ElT}) where {PlanT,ElT}
    return evaluate!(y, op.plan, x)
end
