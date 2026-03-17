"""Unidirectional sparse grid sweeps in recursive layout.

The fused kernels operate on contiguous fibers along one storage dimension while
cyclically rotating the storage-to-physical permutation.
"""

# Oriented coefficient buffers

"""
    struct OrientedCoeffs{D,T}

Coefficient buffer paired with a storage-to-physical dimension permutation.
"""
struct OrientedCoeffs{D,T}
    data::Vector{T}
    perm::SVector{D,Int}
end

OrientedCoeffs(data::Vector{T}) where {T} = OrientedCoeffs(data, SVector{1,Int}(1))

function OrientedCoeffs{D}(data::Vector{T}) where {D,T}
    perm = SVector{D,Int}(ntuple(i -> i, D))
    return OrientedCoeffs(data, perm)
end

"""
    cycle_last_to_front(perm)

Return the one-step cyclic rotation `(p₁, …, p_D) -> (p_D, p₁, …, p_{D-1})`.
"""
@inline function cycle_last_to_front(perm::SVector{D,Int}) where {D}
    return SVector{D,Int}(ntuple(i -> (i == 1 ? perm[D] : perm[i - 1]), D))
end

"""
    cycle_last_to_front(perm, k)

Return the `k`-step last-to-front cyclic rotation of `perm`.
"""
@inline function cycle_last_to_front(perm::SVector{D,Int}, k::Integer) where {D}
    kk = mod(Int(k), D)
    kk == 0 && return perm
    # Right rotation by kk.
    return SVector{D,Int}(ntuple(i -> perm[mod(i - kk - 1, D) + 1], D))
end

# ----------------------------------------------------------------------------
# Operator model and algebra

"""
    AbstractLineOp

Abstract supertype for one-dimensional fiber operators used by unidirectional sweeps.
"""
abstract type AbstractLineOp end

"""
    struct IdentityLineOp <: AbstractLineOp

Identity line operator.
"""
struct IdentityLineOp <: AbstractLineOp end

"""
    struct ZeroLineOp <: AbstractLineOp

Zero line operator that maps every input fiber to zeros.
"""
struct ZeroLineOp <: AbstractLineOp end

"""
    struct LineTransform{Dir} <: AbstractLineOp

Per-fiber modal transform such as an FFT or DCT.
"""
struct LineTransform{Dir} <: AbstractLineOp end

LineTransform(::Val{:forward}) = LineTransform{Val(:forward)}()
LineTransform(::Val{:inverse}) = LineTransform{Val(:inverse)}()

"""
    struct LineChebyshevLegendre{Dir} <: AbstractLineOp

Per-fiber Chebyshev–Legendre coefficient conversion.
"""
struct LineChebyshevLegendre{Dir} <: AbstractLineOp end

LineChebyshevLegendre(::Val{:forward}) = LineChebyshevLegendre{Val(:forward)}()
LineChebyshevLegendre(::Val{:inverse}) = LineChebyshevLegendre{Val(:inverse)}()

"""
    struct LineHierarchize <: AbstractLineOp

In-place hierarchization on one fiber.
"""
struct LineHierarchize <: AbstractLineOp end

"""
    struct LineDehierarchize <: AbstractLineOp

In-place dehierarchization on one fiber.
"""
struct LineDehierarchize <: AbstractLineOp end

"""
    struct LineHierarchizeTranspose <: AbstractLineOp

Transpose hierarchization on one fiber.
"""
struct LineHierarchizeTranspose <: AbstractLineOp end

"""
    struct LineDehierarchizeTranspose <: AbstractLineOp

Transpose dehierarchization on one fiber.
"""
struct LineDehierarchizeTranspose <: AbstractLineOp end

"""
    struct CompositeLineOp{Ops<:Tuple} <: AbstractLineOp

Sequential composition of several line operators.
"""
struct CompositeLineOp{Ops<:Tuple} <: AbstractLineOp
    ops::Ops
end

CompositeLineOp(ops...) = CompositeLineOp(tuple(ops...))

"""
    lineops(op)

Return the tuple of constituent line operators of `op`.
"""
lineops(op::AbstractLineOp) = (op,)
lineops(op::CompositeLineOp) = op.ops

"""
    struct LineMatrixOp{MatVecT<:AbstractVector} <: AbstractLineOp

Level-dependent dense or structured square matrix family acting on one fiber.
"""
struct LineMatrixOp{MatVecT<:AbstractVector} <: AbstractLineOp
    mats::MatVecT
end

"""
    struct LineEvalOp{T,MatT<:AbstractMatrix{T}} <: AbstractLineOp

Rectangular evaluation matrix with zero padding back to the fiber length.
"""
struct LineEvalOp{T,MatT<:AbstractMatrix{T}} <: AbstractLineOp
    V::MatT
    m::Int
end

"""
    struct LineDiagonalOp{F} <: AbstractLineOp

Size-parameterized diagonal family acting on one fiber.
"""
struct LineDiagonalOp{F} <: AbstractLineOp
    f::F
end

"""
    struct LineBandedOp{Part,F} <: AbstractLineOp

Size-parameterized banded family acting on one fiber.
"""
struct LineBandedOp{Part,F} <: AbstractLineOp
    f::F
end

LineBandedOp(f) = LineBandedOp{:full,typeof(f)}(f)
LineBandedOp(::Val{Part}, f) where {Part} = LineBandedOp{Part,typeof(f)}(f)

"""
    Base.iszero(op::AbstractLineOp)

Return `true` when `op` is algebraically zero.
"""
Base.iszero(::AbstractLineOp) = false
Base.iszero(::ZeroLineOp) = true

"""
    isidentity(op::AbstractLineOp)

Return `true` when `op` is an identity map.
"""
isidentity(::AbstractLineOp) = false
isidentity(::IdentityLineOp) = true

"""
    compose(a, b)

Compose two line or tensor operators sequentially.
"""
compose(a::AbstractLineOp, b::AbstractLineOp) = CompositeLineOp(a, b)
compose(a::CompositeLineOp, b::AbstractLineOp) = CompositeLineOp(a.ops..., b)
compose(a::AbstractLineOp, b::CompositeLineOp) = CompositeLineOp(a, b.ops...)
compose(a::CompositeLineOp, b::CompositeLineOp) = CompositeLineOp(a.ops..., b.ops...)

"""
    LineOpStyle

Trait base type describing whether a line operator is in-place or out-of-place.
"""
abstract type LineOpStyle end

"""
    struct InPlaceOp <: LineOpStyle

Trait indicating that a line operator mutates its input buffer in place.
"""
struct InPlaceOp <: LineOpStyle end

"""
    struct OutOfPlaceOp <: LineOpStyle

Trait indicating that a line operator writes to a separate output buffer.
"""
struct OutOfPlaceOp <: LineOpStyle end

"""
    lineop_style(::Type{<:AbstractLineOp})

Return the storage style trait of a line operator type.
"""
lineop_style(::Type{<:AbstractLineOp}) = OutOfPlaceOp()
lineop_style(::Type{IdentityLineOp}) = InPlaceOp()
lineop_style(::Type{ZeroLineOp}) = OutOfPlaceOp()
lineop_style(::Type{LineHierarchize}) = InPlaceOp()
lineop_style(::Type{LineDehierarchize}) = InPlaceOp()
lineop_style(::Type{LineHierarchizeTranspose}) = InPlaceOp()
lineop_style(::Type{LineDehierarchizeTranspose}) = InPlaceOp()
lineop_style(::Type{<:LineTransform}) = InPlaceOp()
lineop_style(::Type{<:LineChebyshevLegendre}) = InPlaceOp()
lineop_style(::Type{<:LineMatrixOp}) = OutOfPlaceOp()
lineop_style(::Type{<:LineEvalOp}) = OutOfPlaceOp()
lineop_style(::Type{<:LineDiagonalOp}) = OutOfPlaceOp()
lineop_style(::Type{<:LineBandedOp}) = OutOfPlaceOp()

plan_family(::Type{Op}) where {Op} = Op
plan_action(::Type{Op}) where {Op} = throw(ArgumentError("plan_action not implemented for $Op"))
is_planned_inplace(::Type{Op}) where {Op} = Val(false)

plan_family(::Type{<:LineTransform}) = LineTransform
plan_family(::Type{<:LineChebyshevLegendre}) = LineChebyshevLegendre
plan_family(::Type{LineHierarchize}) = LineHierarchize
plan_family(::Type{LineDehierarchize}) = LineHierarchize
plan_family(::Type{LineHierarchizeTranspose}) = LineHierarchize
plan_family(::Type{LineDehierarchizeTranspose}) = LineHierarchize

is_planned_inplace(::Type{<:LineTransform}) = Val(true)
is_planned_inplace(::Type{<:LineChebyshevLegendre}) = Val(true)
is_planned_inplace(::Type{LineHierarchize}) = Val(true)
is_planned_inplace(::Type{LineDehierarchize}) = Val(true)
is_planned_inplace(::Type{LineHierarchizeTranspose}) = Val(true)
is_planned_inplace(::Type{LineDehierarchizeTranspose}) = Val(true)

@inline plan_action(::Type{LineTransform{Val(:forward)}}) = Val(:mul)
@inline plan_action(::Type{LineTransform{Val(:inverse)}}) = Val(:div)
@inline plan_action(::Type{LineChebyshevLegendre{Val(:forward)}}) = Val(:mul)
@inline plan_action(::Type{LineChebyshevLegendre{Val(:inverse)}}) = Val(:div)
@inline plan_action(::Type{LineHierarchize}) = Val(:mul)
@inline plan_action(::Type{LineDehierarchize}) = Val(:div)
@inline plan_action(::Type{LineHierarchizeTranspose}) = Val(:tmul)
@inline plan_action(::Type{LineDehierarchizeTranspose}) = Val(:tdiv)

@inline function _any_outofplace(::Type{Ops}) where {Ops<:Tuple}
    Ops === Tuple{} && return false
    head = Base.tuple_type_head(Ops)
    tail = Base.tuple_type_tail(Ops)
    return (lineop_style(head) isa OutOfPlaceOp) || _any_outofplace(tail)
end

lineop_style(::Type{CompositeLineOp{Ops}}) where {Ops} =
    _any_outofplace(Ops) ? OutOfPlaceOp() : InPlaceOp()

needs_plan(::Type{<:LineDiagonalOp}) = Val(true)
needs_plan(::Type{<:LineBandedOp}) = Val(true)

function apply!(dest::AbstractVector, ::ZeroLineOp, src::AbstractVector)
    fill!(dest, zero(eltype(dest)))
    return dest
end

function apply(op::ZeroLineOp, src::AbstractVector{T}) where {T}
    dest = Vector{T}(undef, length(src))
    fill!(dest, zero(T))
    return dest
end

"""
    AbstractTensorOp{D}

Abstract supertype for operators applied by cyclic unidirectional sweeps over `D` dimensions.
"""
abstract type AbstractTensorOp{D} end

"""
    struct TensorOp{D,Ops<:Tuple} <: AbstractTensorOp{D}

One full sweep storing one line operator per physical dimension.
"""
struct TensorOp{D,Ops<:Tuple} <: AbstractTensorOp{D}
    ops::Ops
end

TensorOp(ops::Tuple) = TensorOp{length(ops),typeof(ops)}(ops)

"""
    lineop(op, d)

Return the line operator applied in physical dimension `d`.
"""
lineop(op::TensorOp{D}, d::Integer) where {D} = op.ops[d]

"""
    tensorize(op, ::Val{D})

Broadcast a line operator to all `D` physical dimensions.
"""
tensorize(op::AbstractLineOp, ::Val{D}) where {D} = TensorOp(ntuple(_ -> op, Val(D)))

TensorOp(::Val{D}, op::AbstractLineOp) where {D} = tensorize(op, Val(D))

"""
    struct CompositeTensorOp{D,Ops} <: AbstractTensorOp{D}

Sequential composition of tensor sweeps.
"""
struct CompositeTensorOp{D,Ops} <: AbstractTensorOp{D}
    ops::Ops
end

CompositeTensorOp(ops::Tuple{Vararg{TensorOp{D}}}) where {D} =
    CompositeTensorOp{D,typeof(ops)}(ops)
CompositeTensorOp(ops::AbstractVector{TensorOp{D}}) where {D} =
    CompositeTensorOp{D,typeof(ops)}(ops)

compose(A::TensorOp{D}, B::TensorOp{D}) where {D} = CompositeTensorOp((A, B))

compose(chain::CompositeTensorOp{D,<:Tuple}, op::TensorOp{D}) where {D} =
    CompositeTensorOp((chain.ops..., op))
compose(op::TensorOp{D}, chain::CompositeTensorOp{D,<:Tuple}) where {D} =
    CompositeTensorOp((op, chain.ops...))
compose(chainA::CompositeTensorOp{D,<:Tuple}, chainB::CompositeTensorOp{D,<:Tuple}) where {D} =
    CompositeTensorOp((chainA.ops..., chainB.ops...))

function compose(chain::CompositeTensorOp{D,<:AbstractVector}, op::TensorOp{D}) where {D}
    return CompositeTensorOp(vcat(chain.ops, [op]))
end
function compose(op::TensorOp{D}, chain::CompositeTensorOp{D,<:AbstractVector}) where {D}
    return CompositeTensorOp(vcat([op], chain.ops))
end
function compose(chainA::CompositeTensorOp{D,<:AbstractVector}, chainB::CompositeTensorOp{D,<:AbstractVector}) where {D}
    return CompositeTensorOp(vcat(chainA.ops, chainB.ops))
end

"""
    ForwardTransform(::Val{D})
    ForwardTransform(D)

Return the forward sparse-grid modal transform on `D` dimensions.
"""
function ForwardTransform(::Val{D}) where {D}
    A = tensorize(CompositeLineOp(LineTransform(Val(:forward)), LineHierarchize()), Val(D))
    B = tensorize(LineDehierarchize(), Val(D))
    return compose(A, B)
end
ForwardTransform(D::Integer) = ForwardTransform(Val(D))

"""
    InverseTransform(::Val{D})
    InverseTransform(D)

Return the inverse sparse-grid modal transform on `D` dimensions.
"""
function InverseTransform(::Val{D}) where {D}
    Binv = tensorize(LineHierarchize(), Val(D))
    Ainv = tensorize(CompositeLineOp(LineDehierarchize(), LineTransform(Val(:inverse))), Val(D))
    return compose(Binv, Ainv)
end
InverseTransform(D::Integer) = InverseTransform(Val(D))

"""
    updown(op)

Split a line operator into additive lower and upper factors `(L, U)`.
"""
updown(::IdentityLineOp) = (IdentityLineOp(), ZeroLineOp())

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
                U[i, j] = zero(T)
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
        for j in max(1, i - l):i
            L[i, j] = A[i, j]
        end
        for j in (i + 1):min(n, i + u)
            U[i, j] = A[i, j]
        end
    end
    return L, U
end

function updown(op::LineMatrixOp)
    isempty(op.mats) && throw(ArgumentError("cannot split an empty LineMatrixOp"))
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

updown(op::LineDiagonalOp) = (op, ZeroLineOp())
function updown(op::LineBandedOp{:full})
    return LineBandedOp(Val(:lower), op.f), LineBandedOp(Val(:upper), op.f)
end
updown(op::LineBandedOp{:lower}) = (op, ZeroLineOp())
updown(op::LineBandedOp{:upper}) = (ZeroLineOp(), op)

"""
    struct UpDownTensorOp{D,OpsT,LOpsT,UOpsT} <: AbstractTensorOp{D}

Tensor operator evaluated as a sum of triangular tensor terms from per-dimension `L + U` splits.
"""
struct UpDownTensorOp{D,OpsT<:NTuple{D,AbstractLineOp},LOpsT<:NTuple{D,AbstractLineOp},UOpsT<:NTuple{D,AbstractLineOp}} <: AbstractTensorOp{D}
    full::OpsT
    lower::LOpsT
    upper::UOpsT
    omit_dim::Int
    split_mask::UInt64
end

"""
    UpDownTensorOp(ops; omit_dim=0)

Construct an [`UpDownTensorOp`](@ref) by pre-splitting each dimension into lower and upper factors.
"""
function UpDownTensorOp(ops::NTuple{D,AbstractLineOp}; omit_dim::Integer=0) where {D}
    0 <= Int(omit_dim) <= D || throw(ArgumentError("omit_dim must satisfy 0 <= omit_dim <= $D"))
    D <= 64 || throw(ArgumentError("UpDownTensorOp currently supports D <= 64"))

    parts = ntuple(k -> updown(ops[k]), Val(D))
    lower = ntuple(k -> parts[k][1], Val(D))
    upper = ntuple(k -> parts[k][2], Val(D))

    split_mask = zero(UInt64)
    @inbounds for d in 1:D
        (!iszero(lower[d]) && !iszero(upper[d])) && (split_mask |= UInt64(1) << (d - 1))
    end

    return UpDownTensorOp{D,typeof(ops),typeof(lower),typeof(upper)}(ops, lower, upper, Int(omit_dim), split_mask)
end

lineop(op::UpDownTensorOp{D}, d::Integer) where {D} = op.full[d]

# ---------------------------------------------------------------------------
# Line-op planning hooks

"""
    needs_plan(::Type{<:AbstractLineOp})

Return `Val(true)` when a line operator family requires cached per-refinement plans.
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

"""
    lineplan(op, axis, rmax, ::Type{T})

Build the cached per-refinement plan vector for `op` on `axis` up to level `rmax`.
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

function _get_lineplanvec!(op_plan::Dict{Tuple,AbstractVector},
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

"""
    apply_line!(outbuf, op, inp, work, nodes, level, plan)
    apply_line!(op, buf, work, nodes, level, plan)

Apply one line operator to a single fiber, using `work` as caller-owned scratch.
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

@inline function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                    op::LineDiagonalOp, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                                    work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineDiagonalOp at refinement index=$r"))
    d = plan
    length(d) == length(inp) || throw(DimensionMismatch("diag length mismatch at refinement index=$r"))
    @inbounds for k in eachindex(inp)
        idx = rowptr[k]
        destdata[idx] += d[k] * inp[k]
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

@inline function _banded_scatter_full_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
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
            destdata[idx] += acc
            rowptr[i] = idx + 1
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in 1:n
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] += acc
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

@inline function _banded_scatter_lower_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
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
            destdata[idx] += acc
            rowptr[i] = idx + 1
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in 1:i
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] += acc
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

@inline function _banded_scatter_upper_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
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
            destdata[idx] += acc
            rowptr[i] = idx + 1
        end
    else
        @inbounds for i in 1:n
            acc = zero(T)
            for j in (i + 1):n
                acc += A[i, j] * x[j]
            end
            idx = rowptr[i]
            destdata[idx] += acc
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

function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::LineBandedOp{:full}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                            work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:full} at refinement index=$r"))
    A = plan
    if A isa BandedMatrices.AbstractBandedMatrix
        return _banded_scatter_full_add!(destdata, rowptr, A, inp)
    else
        apply_line!(outbuf, op, inp, work, axis, r, plan)
        return _scatter_add!(destdata, rowptr, outbuf)
    end
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:lower}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:lower} at refinement index=$r"))
    return _banded_scatter_lower!(destdata, rowptr, plan, inp)
end

function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::LineBandedOp{:lower}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                            work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:lower} at refinement index=$r"))
    return _banded_scatter_lower_add!(destdata, rowptr, plan, inp)
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:upper}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:upper} at refinement index=$r"))
    return _banded_scatter_upper!(destdata, rowptr, plan, inp)
end

function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::LineBandedOp{:upper}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                            work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:upper} at refinement index=$r"))
    return _banded_scatter_upper_add!(destdata, rowptr, plan, inp)
end

# ----------------------------------------------------------------------------
# Cyclic (orientation-dependent) layout plan for unidirectional sweeps

"""
    struct LastDimFiber{D}

Descriptor of one contiguous last-dimension fiber in recursive layout.
"""
struct LastDimFiber{D}
    src_offset::Int                      # 1-based offset into the recursive-layout vector
    len::Int                             # number of coefficients in this fiber
    last_refinement::Int                 # largest active refinement index in the last dimension for this fiber
end

"""
    struct OrientationLayout{D,Ti<:Integer}

Cached row and fiber metadata for one cyclic storage orientation.
"""
struct OrientationLayout{D,Ti<:Integer}
    perm::SVector{D,Int}
    first_offsets::Vector{Ti}   # 1-based row starts for the cycled layout
    maxlen::Ti                  # maximum fiber length in this orientation
    nfibers::Ti                 # number of last-dimension fibers in this orientation
    total_work::Ti              # total fiber length in this orientation
    fibers::Vector{LastDimFiber{D}}
end

"""
    struct FiberChunkPlan{Ti<:Integer}

Cached partition of last-dimension fibers into parallel chunks for one sweep shape.
"""
struct FiberChunkPlan{Ti<:Integer}
    ranges::Vector{UnitRange{Int}}
    startptrs::Matrix{Ti}   # size (maxlen, P)
end

"""
    struct FiberChunkBuffers{T<:Number,Ti<:Integer}

Per-chunk scratch buffers for threaded last-dimension sweeps.
"""
struct FiberChunkBuffers{T<:Number,Ti<:Integer}
    bufA::Vector{Vector{T}}
    bufB::Vector{Vector{T}}
    work::Vector{Vector{T}}
    rowptr::Matrix{Ti}
end

"""
    struct CyclicLayoutMeta{D,Ti<:Integer}

Shared orientation metadata and read-only caches for cyclic unidirectional sweeps.
"""
struct CyclicLayoutMeta{D,Ti<:Integer}
    layouts::NTuple{D,OrientationLayout{D,Ti}}
    refinement_caps::SVector{D,Int}
    lineplans::Dict{Tuple,AbstractVector}
    fiber_chunk_plans::Dict{Tuple,FiberChunkPlan{Ti}}
end

"""
    struct CyclicLayoutWorkspace{Ti<:Integer,T<:Number}

Mutable scratch buffers for one execution context of a cyclic layout plan.
"""
struct CyclicLayoutWorkspace{Ti<:Integer,T<:Number}
    write_ptr::Vector{Ti}
    scratch1::Vector{T}
    scratch2::Vector{T}
    scratch3::Vector{T}
    unidir_buf::Vector{T}
    x_buf::Vector{T}
    work_buf::Vector{T}
    acc_buf::Vector{T}
end

"""
    struct CyclicLayoutPlan{D,Ti<:Integer,T<:Number}

Reusable cyclic-layout plan with shared metadata and one mutable execution workspace.
"""
struct CyclicLayoutPlan{D,Ti<:Integer,T<:Number}
    meta::CyclicLayoutMeta{D,Ti}
    workspace::CyclicLayoutWorkspace{Ti,T}
    fiber_chunk_buffers::Dict{Int,FiberChunkBuffers{T,Ti}}
    term_workspaces::Dict{Int,Vector{CyclicLayoutWorkspace{Ti,T}}}
end

function CyclicLayoutWorkspace(::Type{Ti}, ::Type{T}, maxlen::Int, N::Int; xlen::Int=N) where {Ti<:Integer,T<:Number}
    return CyclicLayoutWorkspace{Ti,T}(Vector{Ti}(undef, maxlen),
                                       Vector{T}(undef, maxlen),
                                       Vector{T}(undef, maxlen),
                                       Vector{T}(undef, maxlen),
                                       Vector{T}(undef, N),
                                       Vector{T}(undef, xlen),
                                       Vector{T}(undef, N),
                                       Vector{T}(undef, N))
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
    shift_p = SVector{D,Ti}(ntuple(i -> I.shift[perm[i]], D))
    return WeightedSmolyakIndexSet(Val(D), I.L, weights_p; shift=shift_p, cap=cap_p)
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

"""
    each_lastdim_fiber(spec)
    each_lastdim_fiber(grid, perm)
    each_lastdim_fiber(grid)

Iterate the contiguous last-dimension fibers of a sparse-grid specification or grid.
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
    each_lastdim_fiber(grid, perm)

Iterate the last-dimension fibers of `grid` in the cyclic orientation `perm`.
"""
function each_lastdim_fiber(grid::SparseGrid{<:SparseGridSpec{D}}, perm::SVector{D,Int}) where {D}
    spec_or = oriented_spec(grid.spec, perm)
    return each_lastdim_fiber(spec_or)
end

"""
    each_lastdim_fiber(grid)

Iterate the last-dimension fibers of `grid` in the identity orientation.
"""
each_lastdim_fiber(grid::SparseGrid{<:SparseGridSpec{D}}) where {D} = each_lastdim_fiber(grid, SVector{D,Int}(ntuple(identity, D)))

### Row metadata (counts/offsets) for fused-write kernels

"""
    _layout_rowmeta(spec; Ti=Int)

Compute row offsets for fused-write recursive layout traversal.
"""
function _layout_rowmeta(spec::SparseGridSpec{D}; Ti::Type{<:Integer}=Int) where {D}
    axes = spec.axes
    @inbounds for d in 1:D
        is_nested(axes[d]) || throw(ArgumentError("_layout_rowmeta requires nested 1D axis families (dim=$d, got $(typeof(axes[d])))"))
    end

    fibers = each_lastdim_fiber(spec)
    isempty(fibers) && return Ti[], Ti(0)

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

    offsets = Vector{Ti}(undef, maxlen)
    running = 0
    off = Ti(1)
    @inbounds for k in 1:maxlen
        running += diff[k]
        offsets[k] = off
        off += Ti(running)
    end
    return offsets, Ti(maxlen)
end
# ----------------------------------------------------------------------------
# Cyclic layout plan construction

"""
    CyclicLayoutPlan(grid, ::Type{T}; Ti=Int)

Build a reusable cyclic-layout plan for `grid` and coefficient element type `T`.
"""
function CyclicLayoutPlan(grid::SparseGrid{<:SparseGridSpec{D}}, ::Type{T};
                          Ti::Type{<:Integer}=Int) where {D,T<:Number}
    I = grid.spec.indexset
    caps = refinement_caps(I)

    layouts = ntuple(o -> begin
        perm = cycle_last_to_front(_perm_id(Val(D)), o - 1)
        spec_o = oriented_spec(grid.spec, perm)
        fibers = each_lastdim_fiber(spec_o)
        offsets, maxlen = _layout_rowmeta(spec_o; Ti=Ti)
        OrientationLayout{D,Ti}(perm,
                                offsets,
                                maxlen,
                                Ti(length(fibers)),
                                Ti(sum(fib.len for fib in fibers)),
                                fibers)
    end, D)

    maxmaxlen = maximum(Int(layout.maxlen) for layout in layouts)
    workspace = CyclicLayoutWorkspace(Ti, T, maxmaxlen, length(grid))
    meta = CyclicLayoutMeta{D,Ti}(layouts,
                                  caps,
                                  Dict{Tuple,AbstractVector}(),
                                  Dict{Tuple,FiberChunkPlan{Ti}}())
    return CyclicLayoutPlan{D,Ti,T}(meta,
                                   workspace,
                                   Dict{Int,FiberChunkBuffers{T,Ti}}(),
                                   Dict{Int,Vector{CyclicLayoutWorkspace{Ti,T}}}())
end

# Unidirectional apply primitive

@inline function _scatter_copy!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer}, src::AbstractVector)
    @inbounds for k in eachindex(src)
        idx = rowptr[k]
        destdata[idx] = src[k]
        rowptr[k] = idx + 1
    end
    return destdata
end

@inline function _scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer}, src::AbstractVector)
    @inbounds for k in eachindex(src)
        idx = rowptr[k]
        destdata[idx] += src[k]
        rowptr[k] = idx + 1
    end
    return destdata
end

"""
    apply_scatter!(destdata, rowptr, op, inp, outbuf, work, axis, r, plan)

Apply one line operator and scatter the result directly into the cycled destination layout.
"""
function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::AbstractLineOp, inp::AbstractVector, outbuf::AbstractVector,
                        work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    apply_line!(outbuf, op, inp, work, axis, r, plan)
    _scatter_copy!(destdata, rowptr, outbuf)
    return destdata
end

"""
    apply_scatter_add!(destdata, rowptr, op, inp, outbuf, work, axis, r, plan)

Apply one line operator and add the scattered result into the cycled destination layout.
"""
function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::AbstractLineOp, inp::AbstractVector, outbuf::AbstractVector,
                            work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    apply_line!(outbuf, op, inp, work, axis, r, plan)
    _scatter_add!(destdata, rowptr, outbuf)
    return destdata
end

@inline function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                ::IdentityLineOp, inp::AbstractVector, outbuf::AbstractVector,
                                work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    _scatter_copy!(destdata, rowptr, inp)
    return destdata
end

@inline function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                    ::IdentityLineOp, inp::AbstractVector, outbuf::AbstractVector,
                                    work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    _scatter_add!(destdata, rowptr, inp)
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

@inline function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                                    ::ZeroLineOp, inp::AbstractVector, outbuf::AbstractVector,
                                    work::AbstractVector, axis::AbstractUnivariateNodes, r::Int, plan)
    @inbounds for k in eachindex(inp)
        rowptr[k] += 1
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

@inline function _run_pipeline_scatter_add!(destdata::AbstractVector,
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
    nops == 0 && return _scatter_add!(destdata, rowptr, seg)

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
        return apply_scatter_add!(destdata, rowptr, lastop, cur, outbuf, tmp, axis, r, plan1d)
    end

    cur, _ = _run_pipeline!(ops, planvecs, seg, bufA, bufB, work, axis, r, ElT)
    return _scatter_add!(destdata, rowptr, cur)
end

@inline _default_parallel_width() = max(Threads.threadpoolsize(:default), 1)
const _TERM_PARALLEL_WORKSPACE_BUDGET = 512 * 1024 * 1024

@inline function _orient_of_perm(perm::SVector{D,Int}) where {D}
    orient = findfirst(==(1), perm)
    orient === nothing && return 0
    @inbounds for i in 1:D
        perm[i] == mod(i - orient, D) + 1 || return 0
    end
    return orient
end

@inline _orient_for_lastdim(::Val{D}, d::Int) where {D} = mod(D - d, D) + 1

@inline function _matrix_costproxy(A, n::Int)
    nn = Float64(n)
    if A isa LinearAlgebra.Diagonal
        return nn
    elseif A isa BandedMatrices.AbstractBandedMatrix
        l, u = bandwidths(A)
        return nn * min(nn, Float64(l + u + 1))
    elseif A isa Union{LinearAlgebra.LowerTriangular,LinearAlgebra.UnitLowerTriangular,
                       LinearAlgebra.UpperTriangular,LinearAlgebra.UnitUpperTriangular}
        return 0.5 * nn * (nn + 1.0)
    else
        return nn * nn
    end
end

@inline _lineop_costproxy(::IdentityLineOp, n::Int, r::Int, plan1d) = Float64(n)
@inline _lineop_costproxy(::ZeroLineOp, n::Int, r::Int, plan1d) = Float64(n)
@inline _lineop_costproxy(::Union{LineTransform,LineChebyshevLegendre,
                                  LineHierarchize,LineDehierarchize,
                                  LineHierarchizeTranspose,LineDehierarchizeTranspose},
                          n::Int, r::Int, plan1d) = Float64(n) * log2(Float64(n) + 1.0)
@inline _lineop_costproxy(::LineDiagonalOp, n::Int, r::Int, plan1d) = Float64(n)
@inline _lineop_costproxy(op::LineEvalOp, n::Int, r::Int, plan1d) = Float64(op.m) * Float64(n)
@inline _lineop_costproxy(op::LineMatrixOp, n::Int, r::Int, plan1d) = _matrix_costproxy(_mat_for_level(op, r), n)
@inline function _lineop_costproxy(op::LineBandedOp, n::Int, r::Int, plan1d)
    plan1d === nothing && return Float64(n) * Float64(n)
    return _matrix_costproxy(plan1d, n)
end
@inline _lineop_costproxy(::AbstractLineOp, n::Int, r::Int, plan1d) = Float64(n)

function _layout_work_stats(layout::OrientationLayout, ops::Tuple, planvecs::Tuple)
    total = 0.0
    maxw = 0.0
    fibers = layout.fibers
    @inbounds for fib in fibers
        w = 0.0
        for i in eachindex(ops)
            pv = planvecs[i]
            plan1d = pv === nothing ? nothing : pv[fib.last_refinement + 1]
            w += _lineop_costproxy(ops[i], fib.len, fib.last_refinement, plan1d)
        end
        total += w
        maxw = max(maxw, w)
    end
    return total, maxw
end

@inline function _fiber_threaded_workers(layout::OrientationLayout, parallel_width::Integer,
                                         total_work::Real, max_work::Real)
    P = min(max(Int(parallel_width), 1), Int(layout.nfibers))
    P <= 1 && return 0
    return (Int(layout.nfibers) >= 2 * P && total_work >= 4 * max_work) ? P : 0
end

@inline _lineop_chunk_key(::IdentityLineOp, planvec) = (:identity,)
@inline _lineop_chunk_key(::ZeroLineOp, planvec) = (:zero,)
@inline _lineop_chunk_key(op::Union{LineTransform,LineChebyshevLegendre,
                                    LineHierarchize,LineDehierarchize,
                                    LineHierarchizeTranspose,LineDehierarchizeTranspose}, planvec) =
    (plan_family(typeof(op)),)
@inline _lineop_chunk_key(::LineDiagonalOp, planvec) = (:diagonal,)
@inline _lineop_chunk_key(op::LineEvalOp, planvec) = (:eval, op.m, size(op.V, 2))
@inline _lineop_chunk_key(op::LineMatrixOp, planvec) = (:matrix, typeof(_mat_for_level(op, length(op.mats) - 1)))
@inline function _lineop_chunk_key(op::LineBandedOp{Part,F}, planvec) where {Part,F}
    if planvec === nothing
        return (:banded, Part, F)
    end
    A = planvec[end]
    if A isa BandedMatrices.AbstractBandedMatrix
        l, u = bandwidths(A)
        return (:banded, Part, F, l, u)
    end
    return (:banded, Part, F, size(A, 1))
end
@inline _lineop_chunk_key(op::AbstractLineOp, planvec) = (typeof(op),)

function _ops_chunk_key(ops::Tuple, planvecs::Tuple)
    return ntuple(i -> _lineop_chunk_key(ops[i], planvecs[i]), length(ops))
end

function _sweep_cost_stats(layout::OrientationLayout,
                           axis::AbstractUnivariateNodes,
                           cap::Int,
                           op::AbstractLineOp,
                           lineplans::Dict{Tuple,AbstractVector},
                           ::Type{ElT},
                           parallel_width::Int) where {ElT}
    ops = lineops(op)
    planvecs = map(oi -> _get_lineplanvec!(lineplans, oi, axis, cap, ElT), ops)
    total, maxw = _layout_work_stats(layout, ops, planvecs)
    P = _fiber_threaded_workers(layout, parallel_width, total, maxw)
    threaded = P == 0 ? total : max(maxw, total / P)
    return (serial=total, threaded=threaded)
end

function _partition_ranges_by_weight(weights::AbstractVector{<:Real}, P::Int)
    n = length(weights)
    n == 0 && return UnitRange{Int}[]
    P = clamp(P, 1, n)
    P == 1 && return [1:n]

    prefix = Vector{Float64}(undef, n)
    total = 0.0
    @inbounds for i in 1:n
        total += Float64(weights[i])
        prefix[i] = total
    end

    ranges = Vector{UnitRange{Int}}(undef, P)
    start = 1
    for c in 1:(P - 1)
        target = total * c / P
        stop = searchsortedfirst(prefix, target)
        stop = max(stop, start)
        stop = min(stop, n - (P - c))
        ranges[c] = start:stop
        start = stop + 1
    end
    ranges[P] = start:n
    return ranges
end

function _build_fiber_chunk_plan(layout::OrientationLayout{D,Ti},
                                 P::Int,
                                 ops::Tuple,
                                 planvecs::Tuple) where {D,Ti<:Integer}
    fibers = layout.fibers
    weights = Vector{Float64}(undef, length(fibers))
    @inbounds for i in eachindex(fibers)
        fib = fibers[i]
        w = 0.0
        for j in eachindex(ops)
            pv = planvecs[j]
            plan1d = pv === nothing ? nothing : pv[fib.last_refinement + 1]
            w += _lineop_costproxy(ops[j], fib.len, fib.last_refinement, plan1d)
        end
        weights[i] = w
    end

    ranges = _partition_ranges_by_weight(weights, P)
    P2 = length(ranges)
    maxlen = Int(layout.maxlen)
    startptrs = Matrix{Ti}(undef, maxlen, P2)
    running = zeros(Ti, maxlen)
    diff = zeros(Int, maxlen + 1)

    @inbounds for c in 1:P2
        for k in 1:maxlen
            startptrs[k, c] = layout.first_offsets[k] + running[k]
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

function _get_fiber_chunk_plan!(plan::CyclicLayoutPlan{D,Ti,T},
                                orient::Int,
                                P::Int,
                                ops::Tuple,
                                planvecs::Tuple) where {D,Ti<:Integer,T<:Number}
    meta = plan.meta
    key = (orient, P, _ops_chunk_key(ops, planvecs))
    chunkplan = get(meta.fiber_chunk_plans, key, nothing)
    if chunkplan === nothing
        chunkplan = _build_fiber_chunk_plan(meta.layouts[orient], P, ops, planvecs)
        meta.fiber_chunk_plans[key] = chunkplan
    end
    return chunkplan
end

function _get_fiber_chunk_buffers!(plan::CyclicLayoutPlan{D,Ti,T}, P::Int) where {D,Ti<:Integer,T<:Number}
    bufs = get(plan.fiber_chunk_buffers, P, nothing)
    maxlen = length(plan.workspace.write_ptr)
    if bufs === nothing ||
       length(bufs.bufA) != P ||
       size(bufs.rowptr, 1) != maxlen ||
       size(bufs.rowptr, 2) != P ||
       any(length(v) != maxlen for v in bufs.bufA) ||
       any(length(v) != maxlen for v in bufs.bufB) ||
       any(length(v) != maxlen for v in bufs.work)
        bufs = FiberChunkBuffers{T,Ti}([Vector{T}(undef, maxlen) for _ in 1:P],
                                       [Vector{T}(undef, maxlen) for _ in 1:P],
                                       [Vector{T}(undef, maxlen) for _ in 1:P],
                                       Matrix{Ti}(undef, maxlen, P))
        plan.fiber_chunk_buffers[P] = bufs
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

@inline function _apply_fiber_chunk_add!(destdata::AbstractVector,
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
        _run_pipeline_scatter_add!(destdata, rowptr, ops, planvecs, seg,
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
function _apply_lastdim_cycled!(dest::OrientedCoeffs{D,ElT},
                                src::OrientedCoeffs{D,ElT},
                                grid::SparseGrid,
                                op::AbstractLineOp,
                                plan::CyclicLayoutPlan{D,Ti,ElT},
                                parallel_width::Int) where {D,Ti,ElT}
    parallel_width = max(Int(parallel_width), 1)

    orient = _orient_of_perm(src.perm)
    orient == 0 && throw(ArgumentError("perm=$(src.perm) is not a cyclic orientation of the provided plan"))
    layout = plan.meta.layouts[orient]
    workspace = plan.workspace
    maxlen = Int(layout.maxlen)

    last_phys = src.perm[end]
    axis_last = grid.spec.axes[last_phys]
    ops = lineops(op)
    cap_last = Int(plan.meta.refinement_caps[last_phys])
    planvecs = map(oi -> _get_lineplanvec!(plan.meta.lineplans, oi, axis_last, cap_last, ElT), ops)

    fibers = layout.fibers
    P = 0
    if parallel_width > 1 && Int(layout.nfibers) > 1
        total, maxw = _layout_work_stats(layout, ops, planvecs)
        P = _fiber_threaded_workers(layout, parallel_width, total, maxw)
    end

    if P == 0
        write_ptr = workspace.write_ptr
        @inbounds copyto!(write_ptr, 1, layout.first_offsets, 1, maxlen)
        scratch1 = workspace.scratch1
        scratch2 = workspace.scratch2
        scratch3 = workspace.scratch3
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
        chunkplan = _get_fiber_chunk_plan!(plan, orient, P, ops, planvecs)
        bufs = _get_fiber_chunk_buffers!(plan, length(chunkplan.ranges))
        destdata = dest.data
        srcdata = src.data
        @sync for cid in 1:length(chunkplan.ranges)
            rg = chunkplan.ranges[cid]
            isempty(rg) && continue
            rowptr = @view bufs.rowptr[:, cid]
            copyto!(rowptr, @view chunkplan.startptrs[:, cid])
            bufA = bufs.bufA[cid]
            bufB = bufs.bufB[cid]
            work = bufs.work[cid]
            Threads.@spawn :default _apply_fiber_chunk!($destdata, $srcdata, $fibers, $rg, $rowptr,
                                                        $ops, $planvecs, $axis_last,
                                                        $bufA, $bufB, $work, $ElT)
        end
    end

    return OrientedCoeffs(dest.data, cycle_last_to_front(src.perm))
end

function _apply_lastdim_cycled_add!(destdata::Vector{ElT},
                                    src::OrientedCoeffs{D,ElT},
                                    grid::SparseGrid,
                                    op::AbstractLineOp,
                                    plan::CyclicLayoutPlan{D,Ti,ElT},
                                    parallel_width::Int) where {D,Ti,ElT}
    parallel_width = max(Int(parallel_width), 1)

    orient = _orient_of_perm(src.perm)
    orient == 0 && throw(ArgumentError("perm=$(src.perm) is not a cyclic orientation of the provided plan"))
    layout = plan.meta.layouts[orient]
    workspace = plan.workspace
    maxlen = Int(layout.maxlen)

    last_phys = src.perm[end]
    axis_last = grid.spec.axes[last_phys]
    ops = lineops(op)
    cap_last = Int(plan.meta.refinement_caps[last_phys])
    planvecs = map(oi -> _get_lineplanvec!(plan.meta.lineplans, oi, axis_last, cap_last, ElT), ops)

    fibers = layout.fibers
    P = 0
    if parallel_width > 1 && Int(layout.nfibers) > 1
        total, maxw = _layout_work_stats(layout, ops, planvecs)
        P = _fiber_threaded_workers(layout, parallel_width, total, maxw)
    end

    if P == 0
        write_ptr = workspace.write_ptr
        @inbounds copyto!(write_ptr, 1, layout.first_offsets, 1, maxlen)
        scratch1 = workspace.scratch1
        scratch2 = workspace.scratch2
        scratch3 = workspace.scratch3
        @inbounds for fib in fibers
            seg = @view src.data[fib.src_offset:(fib.src_offset + fib.len - 1)]
            bufA = @view scratch1[1:fib.len]
            bufB = @view scratch2[1:fib.len]
            work = @view scratch3[1:fib.len]
            _run_pipeline_scatter_add!(destdata, write_ptr, ops, planvecs, seg,
                                       bufA, bufB, work,
                                       axis_last, fib.last_refinement, ElT)
        end
    else
        chunkplan = _get_fiber_chunk_plan!(plan, orient, P, ops, planvecs)
        bufs = _get_fiber_chunk_buffers!(plan, length(chunkplan.ranges))
        srcdata = src.data
        @sync for cid in 1:length(chunkplan.ranges)
            rg = chunkplan.ranges[cid]
            isempty(rg) && continue
            rowptr = @view bufs.rowptr[:, cid]
            copyto!(rowptr, @view chunkplan.startptrs[:, cid])
            bufA = bufs.bufA[cid]
            bufB = bufs.bufB[cid]
            work = bufs.work[cid]
            Threads.@spawn :default _apply_fiber_chunk_add!($destdata, $srcdata, $fibers, $rg, $rowptr,
                                                            $ops, $planvecs, $axis_last,
                                                            $bufA, $bufB, $work, $ElT)
        end
    end

    return OrientedCoeffs(destdata, cycle_last_to_front(src.perm))
end

"""
    apply_lastdim_cycled!(dest, src, grid, op, plan; parallel_width=_default_parallel_width())
    apply_lastdim_cycled!(dest, src, grid, op; kwargs...)

Apply `op` along the last storage dimension and write the result in the next cyclic orientation.
"""
function apply_lastdim_cycled!(dest::OrientedCoeffs{D,ElT},
                               src::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT};
                               parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return _apply_lastdim_cycled!(dest, src, grid, op, plan, parallel_width)
end

"""Convenience overload: build a temporary [`CyclicLayoutPlan`](@ref) and apply."""
function apply_lastdim_cycled!(dest::OrientedCoeffs{D,ElT},
                               src::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp; kwargs...) where {D,ElT}
    plan = CyclicLayoutPlan(grid, ElT)
    return apply_lastdim_cycled!(dest, src, grid, op, plan; kwargs...)
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
                            plan::CyclicLayoutPlan{D,Ti,ElT},
                            parallel_width::Int) where {D,Ti,ElT}
    kk = mod(k, D)
    kk == 0 && (copyto!(dest.data, src.data); return OrientedCoeffs(dest.data, src.perm))

    perm_dst = cycle_last_to_front(src.perm, kk)
    orient_src = _orient_of_perm(src.perm)
    orient_dst = _orient_of_perm(perm_dst)
    (orient_src == 0 || orient_dst == 0) && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_by!"))

    if kk == 1
        return _apply_lastdim_cycled!(dest, src, grid, IdentityLineOp(), plan, parallel_width)
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

function _cyclic_rotate_by_add!(destdata::Vector{ElT},
                                src::OrientedCoeffs{D,ElT},
                                grid::SparseGrid,
                                k::Int,
                                plan::CyclicLayoutPlan{D,Ti,ElT},
                                parallel_width::Int) where {D,Ti,ElT}
    kk = mod(k, D)
    if kk == 0
        @inbounds @simd for i in eachindex(src.data)
            destdata[i] += src.data[i]
        end
        return OrientedCoeffs(destdata, src.perm)
    end

    perm_dst = cycle_last_to_front(src.perm, kk)
    orient_src = _orient_of_perm(src.perm)
    orient_dst = _orient_of_perm(perm_dst)
    (orient_src == 0 || orient_dst == 0) && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_by_add!"))

    if kk == 1
        return _apply_lastdim_cycled_add!(destdata, src, grid, IdentityLineOp(), plan, parallel_width)
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
        destdata[j] += src.data[i]
        i += 1
        nxt = iterate(it_src, st)
    end

    return OrientedCoeffs(destdata, perm_dst)
end

function _cyclic_rotate_by!(dest::OrientedCoeffs{D,ElT},
                            src::OrientedCoeffs{D,ElT},
                            grid::SparseGrid,
                            k::Int,
                            plan::CyclicLayoutPlan{D,Ti,ElT};
                            parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return _cyclic_rotate_by!(dest, src, grid, k, plan, parallel_width)
end

function _cyclic_rotate_to!(dest::Vector{ElT},
                            src::Vector{ElT},
                            grid::SparseGrid,
                            perm_src::SVector{D,Int},
                            perm_dst::SVector{D,Int},
                            plan::CyclicLayoutPlan{D,Ti,ElT},
                            parallel_width::Int) where {D,Ti,ElT}
    perm_src == perm_dst && (copyto!(dest, src); return dest)
    osrc = _orient_of_perm(perm_src)
    odst = _orient_of_perm(perm_dst)
    (osrc == 0 || odst == 0) && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_to!"))

    src_or = OrientedCoeffs(src, perm_src)
    dst_or = OrientedCoeffs(dest, perm_src)
    _cyclic_rotate_by!(dst_or, src_or, grid, mod(odst - osrc, D), plan, parallel_width)
    return dest
end

function _cyclic_rotate_to!(dest::Vector{ElT},
                            src::Vector{ElT},
                            grid::SparseGrid,
                            perm_src::SVector{D,Int},
                            perm_dst::SVector{D,Int},
                            plan::CyclicLayoutPlan{D,Ti,ElT};
                            parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return _cyclic_rotate_to!(dest, src, grid, perm_src, perm_dst, plan, parallel_width)
end

@inline _perm_id(::Val{D}) where {D} = SVector{D,Int}(ntuple(identity, D))

@inline function _cyclic_perm_with_front(::Val{D}, d::Int) where {D}
    1 <= d <= D || throw(ArgumentError("front dimension must satisfy 1 <= d <= $D"))
    return cycle_last_to_front(_perm_id(Val(D)), mod(1 - d, D))
end

# Choose execution mode and omission dimension for UpDown tensor sweeps.

@inline function _rotation_cost(srcperm::SVector{D,Int}, k::Int, identity_stats, N::Int) where {D}
    kk = mod(k, D)
    if kk == 0
        return (serial=0.0, threaded=0.0)
    elseif kk == 1
        return identity_stats[srcperm[end]]
    else
        c = Float64(N)
        return (serial=c, threaded=c)
    end
end

@inline function _term_parallel_workers(nterms::Int, parallel_width::Int, bytes_per_worker::Int)
    return min(nterms,
               parallel_width,
               max(1, fld(_TERM_PARALLEL_WORKSPACE_BUDGET, max(bytes_per_worker, 1))))
end

function _collect_term_masks(split_mask::UInt64, ::Val{D}) where {D}
    nbits = count_ones(split_mask)
    nbits >= Sys.WORD_SIZE - 1 && throw(ArgumentError("too many split terms for UpDownTensorOp: nsplit=$nbits"))

    nterms = 1 << nbits
    term_masks = Vector{UInt64}(undef, nterms)
    @inbounds for tidx in 0:(nterms - 1)
        term_mask = zero(UInt64)
        bit = 0
        for d in 1:D
            dbit = UInt64(1) << (d - 1)
            (split_mask & dbit) == 0 && continue
            ((tidx >> bit) & 0x1) == 0x1 && (term_mask |= dbit)
            bit += 1
        end
        term_masks[tidx + 1] = term_mask
    end
    return term_masks
end

function _estimate_updown_term_cost(op::UpDownTensorOp{D},
                                    omit_dim::Int,
                                    term_mask::UInt64,
                                    perm0::SVector{D,Int},
                                    full_stats,
                                    lower_stats,
                                    upper_stats,
                                    identity_stats,
                                    N::Int) where {D}
    serial = 0.0
    threaded = 0.0
    perm = perm0
    steps_done = 0

    while steps_done < D
        t = 0
        @inbounds for j in 0:(D - steps_done - 1)
            pd = perm[D - j]
            _updown_phase1_lineop(op, pd, omit_dim, term_mask) isa IdentityLineOp || break
            t += 1
        end

        if t > 0
            s = _rotation_cost(perm, t, identity_stats, N)
            serial += s.serial
            threaded += s.threaded
            perm = cycle_last_to_front(perm, t)
            steps_done += t
            continue
        end

        phys = perm[end]
        s = if phys == omit_dim
            full_stats[phys]
        elseif iszero(op.lower[phys])
            upper_stats[phys]
        elseif iszero(op.upper[phys])
            identity_stats[phys]
        elseif ((term_mask >> (phys - 1)) & 0x1) == 0x1
            upper_stats[phys]
        else
            identity_stats[phys]
        end
        serial += s.serial
        threaded += s.threaded
        perm = cycle_last_to_front(perm)
        steps_done += 1
    end

    perm = perm0
    steps_done = 0
    while steps_done < D
        t = 0
        @inbounds for j in 0:(D - steps_done - 1)
            pd = perm[D - j]
            _updown_phase2_lineop(op, pd, omit_dim, term_mask) isa IdentityLineOp || break
            t += 1
        end

        if t > 0
            s = _rotation_cost(perm, t, identity_stats, N)
            serial += s.serial
            threaded += s.threaded
            perm = cycle_last_to_front(perm, t)
            steps_done += t
            continue
        end

        phys = perm[end]
        s = if phys == omit_dim
            identity_stats[phys]
        elseif iszero(op.lower[phys])
            identity_stats[phys]
        elseif iszero(op.upper[phys])
            lower_stats[phys]
        elseif ((term_mask >> (phys - 1)) & 0x1) == 0x0
            lower_stats[phys]
        else
            identity_stats[phys]
        end
        serial += s.serial
        threaded += s.threaded
        perm = cycle_last_to_front(perm)
        steps_done += 1
    end

    return serial, threaded
end

@inline function _updown_phase1_lineop(op::UpDownTensorOp{D}, d::Int, omit_dim::Int, term_mask::UInt64) where {D}
    d == omit_dim && return op.full[d]
    L = op.lower[d]
    U = op.upper[d]
    iszero(L) && return U
    iszero(U) && return IdentityLineOp()
    return ((term_mask >> (d - 1)) & 0x1) == 0x1 ? U : IdentityLineOp()
end

@inline function _updown_phase2_lineop(op::UpDownTensorOp{D}, d::Int, omit_dim::Int, term_mask::UInt64) where {D}
    d == omit_dim && return IdentityLineOp()
    L = op.lower[d]
    U = op.upper[d]
    iszero(L) && return IdentityLineOp()
    iszero(U) && return L
    return ((term_mask >> (d - 1)) & 0x1) == 0x0 ? L : IdentityLineOp()
end

function _apply_updown_term_mask_add!(acc::Vector{ElT},
                                      xbuf::Vector{ElT},
                                      perm0::SVector{D,Int},
                                      grid::SparseGrid,
                                      op::UpDownTensorOp{D},
                                      plan::CyclicLayoutPlan{D,Ti,ElT},
                                      omit_dim::Int,
                                      term_mask::UInt64,
                                      parallel_width::Int) where {D,Ti,ElT}
    parallel_width = max(Int(parallel_width), 1)
    bufA = plan.workspace.work_buf
    bufB = plan.workspace.unidir_buf
    src = OrientedCoeffs(xbuf, perm0)
    steps_done = 0

    while steps_done < D
        perm = src.perm
        t = 0
        @inbounds for j in 0:(D - steps_done - 1)
            pd = perm[D - j]
            _updown_phase1_lineop(op, pd, omit_dim, term_mask) isa IdentityLineOp || break
            t += 1
        end

        if t > 0
            destbuf = src.data === bufA ? bufB : bufA
            src = _cyclic_rotate_by!(OrientedCoeffs(destbuf, src.perm), src, grid, t, plan, parallel_width)
            steps_done += t
            continue
        end

        phys = perm[end]
        destbuf = src.data === bufA ? bufB : bufA
        src = _apply_lastdim_cycled!(OrientedCoeffs(destbuf, src.perm), src, grid,
                                     _updown_phase1_lineop(op, phys, omit_dim, term_mask),
                                     plan, parallel_width)
        steps_done += 1
    end

    steps_done = 0
    while steps_done < D
        perm = src.perm
        t = 0
        @inbounds for j in 0:(D - steps_done - 1)
            pd = perm[D - j]
            _updown_phase2_lineop(op, pd, omit_dim, term_mask) isa IdentityLineOp || break
            t += 1
        end

        if t > 0
            if steps_done + t == D
                _cyclic_rotate_by_add!(acc, src, grid, t, plan, parallel_width)
                return acc
            end
            destbuf = src.data === bufA ? bufB : bufA
            src = _cyclic_rotate_by!(OrientedCoeffs(destbuf, src.perm), src, grid, t, plan, parallel_width)
            steps_done += t
            continue
        end

        phys = perm[end]
        lop = _updown_phase2_lineop(op, phys, omit_dim, term_mask)
        if steps_done + 1 == D
            _apply_lastdim_cycled_add!(acc, src, grid, lop, plan, parallel_width)
            return acc
        end
        destbuf = src.data === bufA ? bufB : bufA
        src = _apply_lastdim_cycled!(OrientedCoeffs(destbuf, src.perm), src, grid, lop, plan, parallel_width)
        steps_done += 1
    end

    return acc
end

function _apply_updown_fiber!(ybuf::Vector{ElT},
                              xbuf::Vector{ElT},
                              perm0::SVector{D,Int},
                              grid::SparseGrid,
                              op::UpDownTensorOp{D},
                              plan::CyclicLayoutPlan{D,Ti,ElT},
                              omit_dim::Int,
                              term_masks::Vector{UInt64},
                              parallel_width::Int) where {D,Ti,ElT}
    @inbounds for term_mask in term_masks
        _apply_updown_term_mask_add!(ybuf, xbuf, perm0, grid, op, plan, omit_dim, term_mask, parallel_width)
    end
    return ybuf
end

function _apply_updown_term!(ybuf::Vector{ElT},
                             xbuf::Vector{ElT},
                             perm0::SVector{D,Int},
                             grid::SparseGrid,
                             op::UpDownTensorOp{D},
                             plan::CyclicLayoutPlan{D,Ti,ElT},
                             omit_dim::Int,
                             term_masks::Vector{UInt64},
                             term_ranges::Vector{UnitRange{Int}}) where {D,Ti,ElT}
    term_workers = length(term_ranges)
    workspace = plan.workspace
    N = length(workspace.work_buf)
    maxlen = length(workspace.write_ptr)
    workers = get(plan.term_workspaces, term_workers, nothing)
    if workers === nothing ||
       length(workers) != term_workers ||
       any(length(ws.write_ptr) != maxlen ||
           length(ws.scratch1) != maxlen ||
           length(ws.scratch2) != maxlen ||
           length(ws.scratch3) != maxlen ||
           length(ws.unidir_buf) != N ||
           length(ws.work_buf) != N ||
           length(ws.acc_buf) != N ||
           length(ws.x_buf) != 0 for ws in workers)
        workers = [CyclicLayoutWorkspace(Ti, ElT, maxlen, N; xlen=0) for _ in 1:term_workers]
        plan.term_workspaces[term_workers] = workers
    end

    @sync for wid in 1:term_workers
        rg = term_ranges[wid]
        isempty(rg) && continue
        local_workspace = workers[wid]
        local_plan = CyclicLayoutPlan{D,Ti,ElT}(plan.meta, local_workspace, plan.fiber_chunk_buffers, plan.term_workspaces)
        local_acc = local_workspace.acc_buf
        fill!(local_acc, zero(ElT))
        Threads.@spawn :default begin
            grid1 = $grid
            op1 = $op
            perm1 = $perm0
            omit1 = $omit_dim
            rg1 = $rg
            xdata = $xbuf
            term_masks1 = $term_masks
            local_plan1 = $local_plan
            local_acc1 = $local_acc
            for tidx in rg1
                _apply_updown_term_mask_add!(local_acc1, xdata, perm1, grid1, op1, local_plan1, omit1, term_masks1[tidx], 1)
            end
        end
    end

    @inbounds for wid in 1:term_workers
        local_acc = workers[wid].acc_buf
        for i in eachindex(ybuf)
            ybuf[i] += local_acc[i]
        end
    end
    return ybuf
end

# ----------------------------------------------------------------------------
# Tensor (dimension-wise) unidirectional application

"""
    apply_unidirectional!(u, grid, op, plan; parallel_width=_default_parallel_width())
    apply_unidirectional!(u, grid, op; kwargs...)

Apply a line, tensor, composite, or UpDown operator to `u` using cyclic unidirectional sweeps.
"""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT},
                               parallel_width::Int) where {D,Ti,ElT}
    parallel_width = max(Int(parallel_width), 1)
    buf = OrientedCoeffs(plan.workspace.unidir_buf, u.perm)
    src = u
    dest = buf
    steps_done = 0

    while steps_done < D
        perm = src.perm
        t = 0
        @inbounds for j in 0:(D - steps_done - 1)
            pd = perm[D - j]
            lop = lineop(op, pd)
            lop isa IdentityLineOp || break
            t += 1
        end

        if t > 0
            dest = _cyclic_rotate_by!(dest, src, grid, t, plan, parallel_width)
            src, dest = dest, src
            steps_done += t
            continue
        end

        phys = perm[end]
        lop = lineop(op, phys)
        dest = _apply_lastdim_cycled!(dest, src, grid, lop, plan, parallel_width)
        src, dest = dest, src
        steps_done += 1
    end

    src.data === u.data || copyto!(u.data, src.data)
    return u
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT};
                               parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return apply_unidirectional!(u, grid, op, plan, parallel_width)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::CompositeTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT},
                               parallel_width::Int) where {D,Ti,ElT}
    for sweep in op.ops
        apply_unidirectional!(u, grid, sweep, plan, parallel_width)
    end
    return u
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::CompositeTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT};
                               parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return apply_unidirectional!(u, grid, op, plan, parallel_width)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT},
                               parallel_width::Int) where {D,Ti,ElT}
    return apply_unidirectional!(u, grid, tensorize(op, Val(D)), plan, parallel_width)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT};
                               parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return apply_unidirectional!(u, grid, op, plan, parallel_width)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::UpDownTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT},
                               parallel::Symbol,
                               parallel_width::Int) where {D,Ti,ElT}
    parallel === :auto || parallel === :fiber || parallel === :term ||
        throw(ArgumentError("parallel must be :auto, :fiber, or :term; got $parallel"))

    parallel_width = max(Int(parallel_width), 1)
    idperm = _perm_id(Val(D))
    u.perm == idperm || throw(ArgumentError("UpDownTensorOp expects u.perm == identity; got $(u.perm)"))

    N = length(plan.workspace.work_buf)
    maxlen = length(plan.workspace.write_ptr)
    bytes_per_worker = max(sizeof(Ti) * maxlen + sizeof(ElT) * (3 * maxlen + 3 * N), 1)

    identity_stats = ntuple(d -> begin
        axis = grid.spec.axes[d]
        cap = Int(plan.meta.refinement_caps[d])
        layout = plan.meta.layouts[_orient_for_lastdim(Val(D), d)]
        _sweep_cost_stats(layout, axis, cap, IdentityLineOp(), plan.meta.lineplans, ElT, parallel_width)
    end, Val(D))
    full_stats = ntuple(d -> begin
        axis = grid.spec.axes[d]
        cap = Int(plan.meta.refinement_caps[d])
        layout = plan.meta.layouts[_orient_for_lastdim(Val(D), d)]
        _sweep_cost_stats(layout, axis, cap, op.full[d], plan.meta.lineplans, ElT, parallel_width)
    end, Val(D))
    lower_stats = ntuple(d -> begin
        axis = grid.spec.axes[d]
        cap = Int(plan.meta.refinement_caps[d])
        layout = plan.meta.layouts[_orient_for_lastdim(Val(D), d)]
        _sweep_cost_stats(layout, axis, cap, op.lower[d], plan.meta.lineplans, ElT, parallel_width)
    end, Val(D))
    upper_stats = ntuple(d -> begin
        axis = grid.spec.axes[d]
        cap = Int(plan.meta.refinement_caps[d])
        layout = plan.meta.layouts[_orient_for_lastdim(Val(D), d)]
        _sweep_cost_stats(layout, axis, cap, op.upper[d], plan.meta.lineplans, ElT, parallel_width)
    end, Val(D))

    candidate_omits = op.omit_dim != 0 ? Int[op.omit_dim] : Int[0]
    if op.omit_dim == 0
        @inbounds for d in 1:D
            ((op.split_mask >> (d - 1)) & 0x1) == 0x1 && push!(candidate_omits, d)
        end
    end

    best_time = Inf
    best_mode = :fiber
    best_omit = 0
    best_perm0 = idperm
    best_term_masks = UInt64[]
    best_term_workers = 1
    best_term_costs = Float64[]

    for omit_dim in candidate_omits
        perm0 = omit_dim == 0 ? idperm : _cyclic_perm_with_front(Val(D), omit_dim)
        split_mask = omit_dim == 0 ? op.split_mask : (op.split_mask & ~(UInt64(1) << (omit_dim - 1)))
        nbits = count_ones(split_mask)
        if nbits >= Sys.WORD_SIZE - 1
            op.omit_dim == omit_dim && throw(ArgumentError("too many split terms for UpDownTensorOp: nsplit=$nbits"))
            continue
        end
        term_masks = _collect_term_masks(split_mask, Val(D))
        nterms = length(term_masks)
        term_workers = _term_parallel_workers(nterms, parallel_width, bytes_per_worker)

        setup = _rotation_cost(idperm, mod(_orient_of_perm(perm0) - 1, D), identity_stats, N)
        teardown = _rotation_cost(perm0, mod(1 - _orient_of_perm(perm0), D), identity_stats, N)
        fiber_time = setup.threaded + teardown.threaded
        term_costs = Vector{Float64}(undef, nterms)
        @inbounds for i in eachindex(term_masks)
            serial_cost, fiber_cost = _estimate_updown_term_cost(op, omit_dim, term_masks[i], perm0,
                                                                 full_stats, lower_stats, upper_stats,
                                                                 identity_stats, N)
            term_costs[i] = serial_cost
            fiber_time += fiber_cost
        end

        term_time = Inf
        if parallel === :term || (parallel === :auto && term_workers > 1 && nterms > 1)
            if term_workers > 1
                term_ranges = _partition_ranges_by_weight(term_costs, term_workers)
                term_time = 0.0
                @inbounds for rg in term_ranges
                    load = 0.0
                    for i in rg
                        load += term_costs[i]
                    end
                    term_time = max(term_time, load)
                end
                term_time += Float64(N * term_workers)
            else
                term_time = sum(term_costs)
            end
            term_time += setup.threaded + teardown.threaded
        end

        cand_mode = parallel === :term ? :term : :fiber
        cand_time = parallel === :term ? term_time : fiber_time
        if parallel === :auto && term_time < fiber_time
            cand_mode = :term
            cand_time = term_time
        end

        if cand_time < best_time
            best_time = cand_time
            best_mode = cand_mode
            best_omit = omit_dim
            best_perm0 = perm0
            best_term_masks = term_masks
            best_term_workers = term_workers
            best_term_costs = term_costs
        end
    end

    isfinite(best_time) || throw(ArgumentError("failed to choose an UpDownTensorOp execution plan"))

    xbuf = plan.workspace.x_buf
    if best_perm0 == idperm
        copyto!(xbuf, u.data)
    else
        _cyclic_rotate_to!(xbuf, u.data, grid, idperm, best_perm0, plan, parallel_width)
    end

    ybuf = plan.workspace.acc_buf
    fill!(ybuf, zero(ElT))

    if best_mode === :term && best_term_workers > 1
        term_ranges = _partition_ranges_by_weight(best_term_costs, best_term_workers)
        _apply_updown_term!(ybuf, xbuf, best_perm0, grid, op, plan, best_omit, best_term_masks, term_ranges)
    elseif best_mode === :term
        _apply_updown_fiber!(ybuf, xbuf, best_perm0, grid, op, plan, best_omit, best_term_masks, 1)
    else
        _apply_updown_fiber!(ybuf, xbuf, best_perm0, grid, op, plan, best_omit, best_term_masks, parallel_width)
    end

    if best_perm0 == idperm
        copyto!(u.data, ybuf)
    else
        _cyclic_rotate_to!(u.data, ybuf, grid, best_perm0, idperm, plan, parallel_width)
    end
    return u
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::UpDownTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT};
                               parallel::Symbol=:auto,
                               parallel_width::Int=_default_parallel_width()) where {D,Ti,ElT}
    return apply_unidirectional!(u, grid, op, plan, parallel, parallel_width)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op; kwargs...) where {D,ElT}
    plan = CyclicLayoutPlan(grid, ElT)
    return apply_unidirectional!(u, grid, op, plan; kwargs...)
end
