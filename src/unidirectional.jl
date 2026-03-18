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

Per-fiber Chebyshev-Legendre coefficient conversion.
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

The family function `f(n, T)` must return a vector-like container of diagonal
entries of length `n`. Smaller refinement levels reuse prefix views `d[1:m]`,
so the family must be prefix-compatible across `n`.
"""
struct LineDiagonalOp{F} <: AbstractLineOp
    f::F
end

"""
    struct LineBandedOp{Part,F} <: AbstractLineOp

Size-parameterized banded family acting on one fiber.

The family function `f(n, T)` must return a square
`BandedMatrices.AbstractBandedMatrix` of size `n × n`; the recommended concrete
return type is `BandedMatrix`. Smaller refinement levels reuse leading
principal banded views, so the family must be prefix-compatible across `n`.

When a banded family is algebraically lower- or upper-triangular, prefer
`LineBandedOp(Val(:lower), f)` or `LineBandedOp(Val(:upper), f)` so
[`UpDownTensorOp`](@ref) can avoid splitting that dimension.
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

const _UPDOWN_ZERO = UInt8(0)
const _UPDOWN_IDENTITY = UInt8(1)
const _UPDOWN_LOWER_ONLY = UInt8(2)
const _UPDOWN_UPPER_ONLY = UInt8(3)
const _UPDOWN_SPLIT = UInt8(4)

function _split_updown_matrix(A::AbstractMatrix{T}) where {T}
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
    l, u = bandwidths(A)
    L = BandedMatrix{T}(undef, (n, n), (l, 0))
    U = BandedMatrix{T}(undef, (n, n), (0, u))
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

"""
    updown(op)

Split a line operator into additive lower and upper factors `(L, U)`.
"""
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

updown(::IdentityLineOp) = (IdentityLineOp(), ZeroLineOp())
updown(::ZeroLineOp) = (ZeroLineOp(), ZeroLineOp())
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
    mode::NTuple{D,UInt8}
    omit_dim::Int
    split_mask::UInt64
    has_zero::Bool
end

"""
    UpDownTensorOp(ops; omit_dim=1)

Construct an [`UpDownTensorOp`](@ref) by pre-splitting each dimension into lower and upper factors.

`omit_dim == 0` keeps all split dimensions in the term expansion. `1 <= omit_dim <= D`
omits that physical dimension from the split and instead applies the unsplit line operator in
that dimension.
"""
function UpDownTensorOp(ops::NTuple{D,AbstractLineOp}; omit_dim::Integer=1) where {D}
    0 <= Int(omit_dim) <= D || throw(ArgumentError("omit_dim must satisfy 0 <= omit_dim <= $D"))
    D <= 64 || throw(ArgumentError("UpDownTensorOp currently supports D <= 64"))

    parts = ntuple(k -> updown(ops[k]), Val(D))
    lower = ntuple(k -> parts[k][1], Val(D))
    upper = ntuple(k -> parts[k][2], Val(D))

    split_mask = zero(UInt64)
    has_zero = false
    mode = ntuple(d -> begin
        fd = ops[d]
        L = lower[d]
        U = upper[d]
        if iszero(fd)
            has_zero = true
            _UPDOWN_ZERO
        elseif isidentity(fd)
            _UPDOWN_IDENTITY
        elseif !iszero(L) && !iszero(U)
            split_mask |= UInt64(1) << (d - 1)
            _UPDOWN_SPLIT
        elseif iszero(L) && iszero(U)
            has_zero = true
            _UPDOWN_ZERO
        elseif iszero(U)
            _UPDOWN_LOWER_ONLY
        else
            _UPDOWN_UPPER_ONLY
        end
    end, Val(D))

    return UpDownTensorOp{D,typeof(ops),typeof(lower),typeof(upper)}(ops, lower, upper, mode, Int(omit_dim), split_mask, has_zero)
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

"""
    lineplan(op::LineDiagonalOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T<:Number}

Build a cached vector of diagonal-data prefixes for `LineDiagonalOp`. The
family function `f(n, T)` must return a vector-like container of length `n`,
and smaller levels reuse prefix views `d[1:m]`.
"""
function lineplan(op::LineDiagonalOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T<:Number}
    maxr = Int(rmax)
    nmax = totalsize(axis, maxr)
    dmax = op.f(nmax, T)
    length(dmax) == nmax || throw(DimensionMismatch(
        "LineDiagonalOp: f(n,T) length=$(length(dmax)) but totalsize(axis,rmax)=$nmax (note: caching ignores axis type)"))

    p0 = @view dmax[1:totalsize(axis, 0)]
    plans = Vector{typeof(p0)}(undef, maxr + 1)
    plans[1] = p0
    @inbounds for r in 1:maxr
        n = totalsize(axis, r)
        plans[r + 1] = @view dmax[1:n]
    end
    return plans
end

"""
    lineplan(op::LineBandedOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T<:Number}

Build a cached vector of banded principal-prefix views for `LineBandedOp`. The
family function `f(n, T)` must return a square
`BandedMatrices.AbstractBandedMatrix` of size `n × n`, and smaller levels reuse
leading principal banded views.
"""
function lineplan(op::LineBandedOp, axis::AbstractUnivariateNodes, rmax::Integer, ::Type{T}) where {T<:Number}
    maxr = Int(rmax)
    nmax = totalsize(axis, maxr)
    Amax = op.f(nmax, T)
    Amax isa BandedMatrices.AbstractBandedMatrix || throw(ArgumentError("LineBandedOp: f(n,T) must return a BandedMatrices.AbstractBandedMatrix (recommended: BandedMatrix), got $(typeof(A))"))
    size(Amax, 1) == nmax || throw(DimensionMismatch(
        "LineBandedOp: f(n,T) size=$(size(Amax)) but totalsize(axis,rmax)=$nmax (note: caching ignores axis type)"))
    size(Amax, 2) == nmax || throw(DimensionMismatch(
        "LineBandedOp: f(n,T) size=$(size(Amax)) but totalsize(axis,rmax)=$nmax (note: caching ignores axis type)"))

    n0 = totalsize(axis, 0)
    S0 = @view Amax[1:n0, 1:n0]
    l0, u0 = bandwidths(S0)
    p0 = BandedMatrices._BandedMatrix(BandedMatrices.bandeddata(S0), n0, l0, u0)

    plans = Vector{typeof(p0)}(undef, maxr + 1)
    plans[1] = p0
    @inbounds for r in 1:maxr
        n = totalsize(axis, r)
        S = @view Amax[1:n, 1:n]
        l, u = bandwidths(S)
        plans[r + 1] = BandedMatrices._BandedMatrix(BandedMatrices.bandeddata(S), n, l, u)
    end
    return plans
end

@inline _lineplan_key(op::AbstractLineOp, axis::AbstractUnivariateNodes, ::Type{T}) where {T} =
    (plan_family(typeof(op)), typeof(axis), T)

@inline _lineplan_key(::LineTransform, axis::AbstractUnivariateNodes, ::Type{T}) where {T<:Number} =
    (LineTransform, typeof(axis), _realpart_type(T), T)

@inline _lineplan_key(op::LineDiagonalOp, ::AbstractUnivariateNodes, ::Type{T}) where {T} =
    (typeof(op), T)

@inline _lineplan_key(op::LineBandedOp{Part,F}, ::AbstractUnivariateNodes, ::Type{T}) where {Part,F,T} =
    (LineBandedOp, F, T)

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

@inline apply_line!(::IdentityLineOp, buf::AbstractVector, work::AbstractVector,
                    ::AbstractUnivariateNodes, ::Int, plan) = buf

function apply_line!(outbuf::AbstractVector{T}, ::ZeroLineOp, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    length(outbuf) == inp || throw(DimensionMismatch("fiber length mismatch"))
    fill!(outbuf, zero(T))
    return outbuf
end

function apply_line!(outbuf::AbstractVector{T}, op::LineMatrixOp, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    A = op.mats[level + 1]
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

function apply_line!(outbuf::AbstractVector{T}, op::LineBandedOp{:full}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:full} at refinement index=$level"))
    A = plan
    n = length(inp)
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    size(A, 1) == n || throw(DimensionMismatch("LineBandedOp{:full} size mismatch at refinement index=$level"))
    size(A, 2) == n || throw(DimensionMismatch("LineBandedOp{:full} size mismatch at refinement index=$level"))
    mul!(outbuf, A, inp)
    return outbuf
end

function apply_line!(outbuf::AbstractVector{T}, ::LineBandedOp{:lower}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:lower} at refinement index=$level"))
    A = plan
    n = length(inp)
    size(A, 1) == n || throw(DimensionMismatch("LineBandedOp{:lower} size mismatch at refinement index=$level"))
    size(A, 2) == n || throw(DimensionMismatch("LineBandedOp{:lower} size mismatch at refinement index=$level"))
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    l, _ = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmin = max(1, i - l)
        for j in jmin:i
            acc += A[i, j] * inp[j]
        end
        outbuf[i] = acc
    end
    return outbuf
end

function apply_line!(outbuf::AbstractVector{T}, ::LineBandedOp{:upper}, inp::AbstractVector{T}, work::AbstractVector{T},
                     ::AbstractUnivariateNodes, level::Int, plan) where {T}
    plan === nothing && throw(ArgumentError("missing plan for LineBandedOp{:upper} at refinement index=$level"))
    A = plan
    n = length(inp)
    size(A, 1) == n || throw(DimensionMismatch("LineBandedOp{:upper} size mismatch at refinement index=$level"))
    size(A, 2) == n || throw(DimensionMismatch("LineBandedOp{:upper} size mismatch at refinement index=$level"))
    length(outbuf) == n || throw(DimensionMismatch("fiber length mismatch"))
    _, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmax = min(n, i + u)
        for j in (i + 1):jmax
            acc += A[i, j] * inp[j]
        end
        outbuf[i] = acc
    end
    return outbuf
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

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:full}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, A::BandedMatrices.AbstractBandedMatrix) where {T}
    n = length(inp)
    l, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmin = max(1, i - l)
        jmax = min(n, i + u)
        for j in jmin:jmax
            acc += A[i, j] * inp[j]
        end
        idx = rowptr[i]
        destdata[idx] = acc
        rowptr[i] = idx + 1
    end
    return destdata
end

function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::LineBandedOp{:full}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                            work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, A::BandedMatrices.AbstractBandedMatrix) where {T}
    n = length(inp)
    l, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmin = max(1, i - l)
        jmax = min(n, i + u)
        for j in jmin:jmax
            acc += A[i, j] * inp[j]
        end
        idx = rowptr[i]
        destdata[idx] += acc
        rowptr[i] = idx + 1
    end
    return destdata
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:lower}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, A::BandedMatrices.AbstractBandedMatrix) where {T}
    n = length(inp)
    l, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmin = max(1, i - l)
        for j in jmin:i
            acc += A[i, j] * inp[j]
        end
        idx = rowptr[i]
        destdata[idx] = acc
        rowptr[i] = idx + 1
    end
    return destdata
end

function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::LineBandedOp{:lower}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                            work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, A::BandedMatrices.AbstractBandedMatrix) where {T}
    n = length(inp)
    l, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmin = max(1, i - l)
        for j in jmin:i
            acc += A[i, j] * inp[j]
        end
        idx = rowptr[i]
        destdata[idx] += acc
        rowptr[i] = idx + 1
    end
    return destdata
end

function apply_scatter!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                        op::LineBandedOp{:upper}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                        work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, A::BandedMatrices.AbstractBandedMatrix) where {T}
    n = length(inp)
    _, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmax = min(n, i + u)
        for j in (i + 1):jmax
            acc += A[i, j] * inp[j]
        end
        idx = rowptr[i]
        destdata[idx] = acc
        rowptr[i] = idx + 1
    end
    return destdata
end

function apply_scatter_add!(destdata::AbstractVector, rowptr::AbstractVector{<:Integer},
                            op::LineBandedOp{:upper}, inp::AbstractVector{T}, outbuf::AbstractVector{T},
                            work::AbstractVector{T}, axis::AbstractUnivariateNodes, r::Int, A::BandedMatrices.AbstractBandedMatrix) where {T}
    n = length(inp)
    _, u = bandwidths(A)
    @inbounds for i in 1:n
        acc = zero(T)
        jmax = min(n, i + u)
        for j in (i + 1):jmax
            acc += A[i, j] * inp[j]
        end
        idx = rowptr[i]
        destdata[idx] += acc
        rowptr[i] = idx + 1
    end
    return destdata
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
    first_offsets::Vector{Ti}   # 1-based row starts for the cycled layout
    maxlen::Ti                  # maximum fiber length in this orientation
    nfibers::Ti                 # number of last-dimension fibers in this orientation
    fibers::Vector{LastDimFiber{D}}
end

"""
    struct FiberChunkPlan{Ti<:Integer}

Cached overdecomposition of last-dimension fibers into contiguous queue chunks.
"""
struct FiberChunkPlan{Ti<:Integer}
    ranges::Vector{UnitRange{Int}}
    startptrs::Matrix{Ti}   # size (maxlen, nchunks)
end

"""
    struct FiberWorkerBuffers{T<:Number,Ti<:Integer}

Per-worker scratch buffers for threaded last-dimension sweeps.
"""
struct FiberWorkerBuffers{T<:Number,Ti<:Integer}
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
    fiber_queue_plans::NTuple{D,FiberChunkPlan{Ti}}
    pool::Int
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
    rot_buf::Vector{T}
    work_buf::Vector{T}
    acc_buf::Vector{T}
end

"""
    struct CyclicLayoutPlan{D,Ti<:Integer,T<:Number}

Reusable cyclic-layout plan with shared metadata and mutable worker scratch.
"""
struct CyclicLayoutPlan{D,Ti<:Integer,T<:Number}
    meta::CyclicLayoutMeta{D,Ti}
    workspace::CyclicLayoutWorkspace{Ti,T}
    fiber_workers::FiberWorkerBuffers{T,Ti}
    term_workspaces::Vector{CyclicLayoutWorkspace{Ti,T}}
end

function CyclicLayoutWorkspace(::Type{Ti}, ::Type{T}, maxlen::Int, N::Int; xlen::Int=N) where {Ti<:Integer,T<:Number}
    return CyclicLayoutWorkspace{Ti,T}(Vector{Ti}(undef, maxlen),
                                       Vector{T}(undef, maxlen),
                                       Vector{T}(undef, maxlen),
                                       Vector{T}(undef, maxlen),
                                       Vector{T}(undef, N),
                                       Vector{T}(undef, xlen),
                                       Vector{T}(undef, N),
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

# ----------------------------------------------------------------------------
# Cyclic layout plan construction

"""
    CyclicLayoutPlan(grid, ::Type{T}; Ti=Int)

Build a reusable cyclic-layout plan for `grid` and coefficient element type `T`.
The plan captures the current `:default` thread-pool size and precomputes one
layout-only fiber queue per cyclic orientation.
"""
function CyclicLayoutPlan(grid::SparseGrid{<:SparseGridSpec{D}}, ::Type{T};
                          Ti::Type{<:Integer}=Int) where {D,T<:Number}
    I = grid.spec.indexset
    caps = refinement_caps(I)

    idperm = SVector{D,Int}(ntuple(identity, Val(D)))
    layouts = ntuple(o -> begin
        perm = cycle_last_to_front(idperm, o - 1)
        spec_o = oriented_spec(grid.spec, perm)
        axes_o = spec_o.axes
        @inbounds for d in 1:D
            is_nested(axes_o[d]) || throw(ArgumentError("CyclicLayoutPlan requires nested 1D axis families (dim=$d, got $(typeof(axes_o[d])))"))
        end

        fibers = each_lastdim_fiber(spec_o)
        if isempty(fibers)
            offsets = Ti[]
            maxlen = Ti(0)
        else
            max_last_refinement = maximum(fib.last_refinement for fib in fibers)
            maxlen_int = maximum(fib.len for fib in fibers)
            count_by_last_refinement = zeros(Int, max_last_refinement + 1)
            @inbounds for fib in fibers
                count_by_last_refinement[fib.last_refinement + 1] += 1
            end

            diff = zeros(Int, maxlen_int + 2)
            @inbounds for t in 0:max_last_refinement
                c = count_by_last_refinement[t + 1]
                c == 0 && continue
                len = totalsize(axes_o[D], t)
                diff[1] += c
                diff[len + 1] -= c
            end

            offsets = Vector{Ti}(undef, maxlen_int)
            running = 0
            off = Ti(1)
            @inbounds for k in 1:maxlen_int
                running += diff[k]
                offsets[k] = off
                off += Ti(running)
            end
            maxlen = Ti(maxlen_int)
        end
        OrientationLayout{D,Ti}(offsets,
                                maxlen,
                                Ti(length(fibers)),
                                fibers)
    end, D)

    pool = max(Threads.threadpoolsize(:default), 1)
    fiber_queue_plans = ntuple(o -> _build_fiber_queue_plan(layouts[o], pool), D)
    maxmaxlen = maximum(Int(layout.maxlen) for layout in layouts)
    N = length(grid)
    workspace = CyclicLayoutWorkspace(Ti, T, maxmaxlen, N)
    fiber_workers = FiberWorkerBuffers{T,Ti}([Vector{T}(undef, maxmaxlen) for _ in 1:pool],
                                             [Vector{T}(undef, maxmaxlen) for _ in 1:pool],
                                             [Vector{T}(undef, maxmaxlen) for _ in 1:pool],
                                             Matrix{Ti}(undef, maxmaxlen, pool))
    term_workspaces = [CyclicLayoutWorkspace(Ti, T, maxmaxlen, N; xlen=0) for _ in 1:pool]
    meta = CyclicLayoutMeta{D,Ti}(layouts,
                                  caps,
                                  Dict{Tuple,AbstractVector}(),
                                  fiber_queue_plans,
                                  pool)
    return CyclicLayoutPlan{D,Ti,T}(meta, workspace, fiber_workers, term_workspaces)
end

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

@inline function _orient_of_perm(perm::SVector{D,Int}) where {D}
    orient = findfirst(==(1), perm)
    orient === nothing && return 0
    @inbounds for i in 1:D
        perm[i] == mod(i - orient, D) + 1 || return 0
    end
    return orient
end

function _build_fiber_queue_plan(layout::OrientationLayout{D,Ti}, pool::Int) where {D,Ti<:Integer}
    nfibers = Int(layout.nfibers)
    maxlen = Int(layout.maxlen)
    nfibers == 0 && return FiberChunkPlan{Ti}(UnitRange{Int}[], Matrix{Ti}(undef, maxlen, 0))

    nchunks = min(nfibers, max(1, 4 * max(pool, 1)))
    ranges = Vector{UnitRange{Int}}(undef, nchunks)
    base = fld(nfibers, nchunks)
    rem = mod(nfibers, nchunks)
    start = 1
    @inbounds for c in 1:nchunks
        len = base + (c <= rem ? 1 : 0)
        stop = start + len - 1
        ranges[c] = start:stop
        start = stop + 1
    end

    startptrs = Matrix{Ti}(undef, maxlen, nchunks)
    running = zeros(Ti, maxlen)
    diff = zeros(Int, maxlen + 1)
    fibers = layout.fibers
    @inbounds for c in 1:nchunks
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
                                nworkers::Int) where {D,Ti,ElT}
    nworkers = clamp(Int(nworkers), 1, plan.meta.pool)

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
    chunkplan = plan.meta.fiber_queue_plans[orient]
    worker_count = min(nworkers, length(chunkplan.ranges))

    if worker_count <= 1 || Int(layout.nfibers) <= 1
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
        bufs = plan.fiber_workers
        nextchunk = Threads.Atomic{Int}(1)
        destdata = dest.data
        srcdata = src.data
        ranges = chunkplan.ranges
        startptrs = chunkplan.startptrs
        nchunks = length(ranges)
        @sync for wid in 1:worker_count
            rowptr = @view bufs.rowptr[:, wid]
            bufA = bufs.bufA[wid]
            bufB = bufs.bufB[wid]
            work = bufs.work[wid]
            Threads.@spawn :default begin
                destdata1 = $destdata
                srcdata1 = $srcdata
                fibers1 = $fibers
                ranges1 = $ranges
                startptrs1 = $startptrs
                nextchunk1 = $nextchunk
                nchunks1 = $nchunks
                rowptr1 = $rowptr
                ops1 = $ops
                planvecs1 = $planvecs
                axis_last1 = $axis_last
                bufA1 = $bufA
                bufB1 = $bufB
                work1 = $work
                ElT1 = $ElT
                while true
                    cid = Threads.atomic_add!(nextchunk1, 1)
                    cid > nchunks1 && break
                    copyto!(rowptr1, @view startptrs1[:, cid])
                    _apply_fiber_chunk!(destdata1, srcdata1, fibers1, ranges1[cid], rowptr1,
                                        ops1, planvecs1, axis_last1,
                                        bufA1, bufB1, work1, ElT1)
                end
            end
        end
    end

    return OrientedCoeffs(dest.data, cycle_last_to_front(src.perm))
end

function _apply_lastdim_cycled_add!(destdata::Vector{ElT},
                                    src::OrientedCoeffs{D,ElT},
                                    grid::SparseGrid,
                                    op::AbstractLineOp,
                                    plan::CyclicLayoutPlan{D,Ti,ElT},
                                    nworkers::Int) where {D,Ti,ElT}
    nworkers = clamp(Int(nworkers), 1, plan.meta.pool)

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
    chunkplan = plan.meta.fiber_queue_plans[orient]
    worker_count = min(nworkers, length(chunkplan.ranges))

    if worker_count <= 1 || Int(layout.nfibers) <= 1
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
        bufs = plan.fiber_workers
        nextchunk = Threads.Atomic{Int}(1)
        srcdata = src.data
        ranges = chunkplan.ranges
        startptrs = chunkplan.startptrs
        nchunks = length(ranges)
        @sync for wid in 1:worker_count
            rowptr = @view bufs.rowptr[:, wid]
            bufA = bufs.bufA[wid]
            bufB = bufs.bufB[wid]
            work = bufs.work[wid]
            Threads.@spawn :default begin
                destdata1 = $destdata
                srcdata1 = $srcdata
                fibers1 = $fibers
                ranges1 = $ranges
                startptrs1 = $startptrs
                nextchunk1 = $nextchunk
                nchunks1 = $nchunks
                rowptr1 = $rowptr
                ops1 = $ops
                planvecs1 = $planvecs
                axis_last1 = $axis_last
                bufA1 = $bufA
                bufB1 = $bufB
                work1 = $work
                ElT1 = $ElT
                while true
                    cid = Threads.atomic_add!(nextchunk1, 1)
                    cid > nchunks1 && break
                    copyto!(rowptr1, @view startptrs1[:, cid])
                    _apply_fiber_chunk_add!(destdata1, srcdata1, fibers1, ranges1[cid], rowptr1,
                                            ops1, planvecs1, axis_last1,
                                            bufA1, bufB1, work1, ElT1)
                end
            end
        end
    end

    return OrientedCoeffs(destdata, cycle_last_to_front(src.perm))
end

"""
    apply_lastdim_cycled!(dest, src, grid, op, plan)
    apply_lastdim_cycled!(dest, src, grid, op)

Apply `op` along the last storage dimension and write the result in the next cyclic orientation.
"""
function apply_lastdim_cycled!(dest::OrientedCoeffs{D,ElT},
                               src::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    return _apply_lastdim_cycled!(dest, src, grid, op, plan, plan.meta.pool)
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
# k-step cyclic rotations (last-to-front only)

"""
    _cyclic_rotate_by!(dest, src, grid, k, plan, nworkers)

Rotate `src` forward by `k` cyclic orientations and write the result into `dest`.

For `k > 1`, the rotation is evaluated as repeated identity last-dimension
sweeps so that it reuses the same queueing and scratch model as ordinary tensor
application.
"""
function _cyclic_rotate_by!(dest::OrientedCoeffs{D,ElT},
                            src::OrientedCoeffs{D,ElT},
                            grid::SparseGrid,
                            k::Int,
                            plan::CyclicLayoutPlan{D,Ti,ElT},
                            nworkers::Int) where {D,Ti,ElT}
    kk = mod(k, D)
    kk == 0 && (copyto!(dest.data, src.data); return OrientedCoeffs(dest.data, src.perm))
    src.data === dest.data && throw(ArgumentError("rotation requires distinct src and dest buffers when k != 0"))

    orient_src = _orient_of_perm(src.perm)
    orient_src == 0 && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_by!"))

    kk == 1 && return _apply_lastdim_cycled!(dest, src, grid, IdentityLineOp(), plan, nworkers)

    workspace = plan.workspace
    tmp = nothing
    @inbounds for buf in (workspace.unidir_buf, workspace.work_buf, workspace.rot_buf)
        if buf !== src.data && buf !== dest.data
            tmp = buf
            break
        end
    end
    tmp === nothing && throw(ArgumentError("rotation requires one temporary buffer distinct from src and dest"))

    current = src
    nextbuf = isodd(kk) ? dest.data : tmp
    @inbounds for _ in 1:kk
        current = _apply_lastdim_cycled!(OrientedCoeffs(nextbuf, current.perm),
                                         current,
                                         grid,
                                         IdentityLineOp(),
                                         plan,
                                         nworkers)
        nextbuf = nextbuf === dest.data ? tmp : dest.data
    end
    return current
end

"""
    _cyclic_rotate_by_add!(destdata, src, grid, k, plan, nworkers)

Rotate `src` forward by `k` cyclic orientations and add the result into
`destdata`.
"""
function _cyclic_rotate_by_add!(destdata::Vector{ElT},
                                src::OrientedCoeffs{D,ElT},
                                grid::SparseGrid,
                                k::Int,
                                plan::CyclicLayoutPlan{D,Ti,ElT},
                                nworkers::Int) where {D,Ti,ElT}
    kk = mod(k, D)
    if kk == 0
        @inbounds @simd for i in eachindex(src.data)
            destdata[i] += src.data[i]
        end
        return OrientedCoeffs(destdata, src.perm)
    end

    orient_src = _orient_of_perm(src.perm)
    orient_src == 0 && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_by_add!"))

    kk == 1 && return _apply_lastdim_cycled_add!(destdata, src, grid, IdentityLineOp(), plan, nworkers)

    workspace = plan.workspace
    tmpA = nothing
    tmpB = nothing
    @inbounds for buf in (workspace.unidir_buf, workspace.work_buf, workspace.rot_buf)
        if buf !== src.data && buf !== destdata
            if tmpA === nothing
                tmpA = buf
            elseif buf !== tmpA
                tmpB = buf
                break
            end
        end
    end
    (tmpA === nothing || tmpB === nothing) &&
        throw(ArgumentError("rotation-add requires two temporary buffers distinct from src and dest"))

    current = src
    nextbuf = tmpA
    @inbounds for _ in 1:(kk - 1)
        current = _apply_lastdim_cycled!(OrientedCoeffs(nextbuf, current.perm),
                                         current,
                                         grid,
                                         IdentityLineOp(),
                                         plan,
                                         nworkers)
        nextbuf = nextbuf === tmpA ? tmpB : tmpA
    end
    return _apply_lastdim_cycled_add!(destdata, current, grid, IdentityLineOp(), plan, nworkers)
end

"""
    _cyclic_rotate_to!(dest, src, grid, perm_src, perm_dst, plan, nworkers)

Rotate `src` from `perm_src` to `perm_dst` and write the result into `dest`.
Both permutations must be cyclic orientations of the identity storage order.
"""
function _cyclic_rotate_to!(dest::Vector{ElT},
                            src::Vector{ElT},
                            grid::SparseGrid,
                            perm_src::SVector{D,Int},
                            perm_dst::SVector{D,Int},
                            plan::CyclicLayoutPlan{D,Ti,ElT},
                            nworkers::Int) where {D,Ti,ElT}
    perm_src == perm_dst && (copyto!(dest, src); return dest)
    src === dest && throw(ArgumentError("rotation requires distinct src and dest buffers when perm_src != perm_dst"))

    osrc = _orient_of_perm(perm_src)
    odst = _orient_of_perm(perm_dst)
    (osrc == 0 || odst == 0) && throw(ArgumentError("non-cyclic perm in _cyclic_rotate_to!"))

    src_or = OrientedCoeffs(src, perm_src)
    dst_or = OrientedCoeffs(dest, perm_src)
    _cyclic_rotate_by!(dst_or, src_or, grid, mod(odst - osrc, D), plan, nworkers)
    return dest
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
            ((split_mask >> (d - 1)) & 0x1) == 0x0 && continue
            ((tidx >> bit) & 0x1) == 0x1 && (term_mask |= dbit)
            bit += 1
        end
        term_masks[tidx + 1] = term_mask
    end
    return term_masks
end

@inline function _updown_phase1_lineop(op::UpDownTensorOp{D}, d::Int, omit_dim::Int, term_mask::UInt64) where {D}
    d == omit_dim && return op.full[d]
    mode = op.mode[d]
    mode == _UPDOWN_UPPER_ONLY && return op.upper[d]
    mode == _UPDOWN_SPLIT && return ((term_mask >> (d - 1)) & 0x1) == 0x1 ? op.upper[d] : IdentityLineOp()
    mode == _UPDOWN_ZERO && return ZeroLineOp()
    return IdentityLineOp()
end

@inline function _updown_phase2_lineop(op::UpDownTensorOp{D}, d::Int, omit_dim::Int, term_mask::UInt64) where {D}
    d == omit_dim && return IdentityLineOp()
    mode = op.mode[d]
    mode == _UPDOWN_LOWER_ONLY && return op.lower[d]
    mode == _UPDOWN_SPLIT && return ((term_mask >> (d - 1)) & 0x1) == 0x0 ? op.lower[d] : IdentityLineOp()
    return IdentityLineOp()
end

function _apply_updown_term_mask_add!(acc::Vector{ElT},
                                      xbuf::Vector{ElT},
                                      perm0::SVector{D,Int},
                                      grid::SparseGrid,
                                      op::UpDownTensorOp{D},
                                      plan::CyclicLayoutPlan{D,Ti,ElT},
                                      omit_dim::Int,
                                      term_mask::UInt64,
                                      nworkers::Int) where {D,Ti,ElT}
    nworkers = max(Int(nworkers), 1)
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
            src = _cyclic_rotate_by!(OrientedCoeffs(destbuf, src.perm), src, grid, t, plan, nworkers)
            steps_done += t
            continue
        end

        phys = perm[end]
        destbuf = src.data === bufA ? bufB : bufA
        src = _apply_lastdim_cycled!(OrientedCoeffs(destbuf, src.perm), src, grid,
                                     _updown_phase1_lineop(op, phys, omit_dim, term_mask),
                                     plan, nworkers)
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
                _cyclic_rotate_by_add!(acc, src, grid, t, plan, nworkers)
                return acc
            end
            destbuf = src.data === bufA ? bufB : bufA
            src = _cyclic_rotate_by!(OrientedCoeffs(destbuf, src.perm), src, grid, t, plan, nworkers)
            steps_done += t
            continue
        end

        phys = perm[end]
        lop = _updown_phase2_lineop(op, phys, omit_dim, term_mask)
        if steps_done + 1 == D
            _apply_lastdim_cycled_add!(acc, src, grid, lop, plan, nworkers)
            return acc
        end
        destbuf = src.data === bufA ? bufB : bufA
        src = _apply_lastdim_cycled!(OrientedCoeffs(destbuf, src.perm), src, grid, lop, plan, nworkers)
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
                              pool::Int) where {D,Ti,ElT}
    fill!(ybuf, zero(ElT))
    @inbounds for term_mask in term_masks
        _apply_updown_term_mask_add!(ybuf, xbuf, perm0, grid, op, plan, omit_dim, term_mask, pool)
    end
    return ybuf
end

"""
    _apply_updown_term!(ybuf, xbuf, perm0, grid, op, plan, omit_dim, term_masks, pool)

Apply the UpDown split terms with a dynamic term queue and a striped reduction of
worker-local accumulators.
"""
function _apply_updown_term!(ybuf::Vector{ElT},
                             xbuf::Vector{ElT},
                             perm0::SVector{D,Int},
                             grid::SparseGrid,
                             op::UpDownTensorOp{D},
                             plan::CyclicLayoutPlan{D,Ti,ElT},
                             omit_dim::Int,
                             term_masks::Vector{UInt64},
                             pool::Int) where {D,Ti,ElT}
    term_workers = min(max(Int(pool), 1), length(term_masks))
    term_workers == 1 && begin
        fill!(ybuf, zero(ElT))
        @inbounds for term_mask in term_masks
            _apply_updown_term_mask_add!(ybuf, xbuf, perm0, grid, op, plan, omit_dim, term_mask, 1)
        end
        return ybuf
    end

    caps = plan.meta.refinement_caps
    @inbounds for d in 1:D
        mode = op.mode[d]
        mode == _UPDOWN_IDENTITY && continue
        mode == _UPDOWN_ZERO && continue
        axis = grid.spec.axes[d]
        cap = Int(caps[d])
        if d == omit_dim
            for oi in lineops(op.full[d])
                _get_lineplanvec!(plan.meta.lineplans, oi, axis, cap, ElT)
            end
        elseif mode == _UPDOWN_SPLIT
            for oi in lineops(op.lower[d])
                _get_lineplanvec!(plan.meta.lineplans, oi, axis, cap, ElT)
            end
            for oi in lineops(op.upper[d])
                _get_lineplanvec!(plan.meta.lineplans, oi, axis, cap, ElT)
            end
        elseif mode == _UPDOWN_LOWER_ONLY
            for oi in lineops(op.lower[d])
                _get_lineplanvec!(plan.meta.lineplans, oi, axis, cap, ElT)
            end
        elseif mode == _UPDOWN_UPPER_ONLY
            for oi in lineops(op.upper[d])
                _get_lineplanvec!(plan.meta.lineplans, oi, axis, cap, ElT)
            end
        end
    end

    workers = plan.term_workspaces
    nterms = length(term_masks)
    nextidx = Threads.Atomic{Int}(1)
    @sync for wid in 1:term_workers
        local_workspace = workers[wid]
        local_plan = CyclicLayoutPlan{D,Ti,ElT}(plan.meta, local_workspace, plan.fiber_workers, plan.term_workspaces)
        local_acc = local_workspace.acc_buf
        fill!(local_acc, zero(ElT))
        Threads.@spawn :default begin
            grid1 = $grid
            op1 = $op
            perm1 = $perm0
            omit1 = $omit_dim
            xdata = $xbuf
            term_masks1 = $term_masks
            nterms1 = $nterms
            nextidx1 = $nextidx
            local_plan1 = $local_plan
            local_acc1 = $local_acc
            while true
                tidx = Threads.atomic_add!(nextidx1, 1)
                tidx > nterms1 && break
                _apply_updown_term_mask_add!(local_acc1, xdata, perm1, grid1, op1, local_plan1, omit1, term_masks1[tidx], 1)
            end
        end
    end

    N = length(ybuf)
    stripe = cld(N, term_workers)
    @sync for rid in 1:term_workers
        lo = (rid - 1) * stripe + 1
        lo > N && break
        hi = min(N, rid * stripe)
        Threads.@spawn :default begin
            ybuf1 = $ybuf
            workers1 = $workers
            term_workers1 = $term_workers
            lo1 = $lo
            hi1 = $hi
            base_acc = workers1[1].acc_buf
            @inbounds @simd for i in lo1:hi1
                ybuf1[i] = base_acc[i]
            end
            @inbounds for wid in 2:term_workers1
                local_acc = workers1[wid].acc_buf
                @simd for i in lo1:hi1
                    ybuf1[i] += local_acc[i]
                end
            end
        end
    end
    return ybuf
end

# ----------------------------------------------------------------------------
# Tensor (dimension-wise) unidirectional application

"""
    _apply_tensor_sweep!(u, grid, op, plan, nworkers)

Apply one [`TensorOp`](@ref) sweep to `u` using cyclic rotations and
last-dimension kernels.

`nworkers` is an internal worker budget used to suppress nested threading when a
larger outer parallel region already exists.
"""
function _apply_tensor_sweep!(u::OrientedCoeffs{D,ElT},
                              grid::SparseGrid,
                              op::TensorOp{D},
                              plan::CyclicLayoutPlan{D,Ti,ElT},
                              nworkers::Int) where {D,Ti,ElT}
    nworkers = clamp(Int(nworkers), 1, plan.meta.pool)
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
            dest = _cyclic_rotate_by!(dest, src, grid, t, plan, nworkers)
            src, dest = dest, src
            steps_done += t
            continue
        end

        phys = perm[end]
        lop = lineop(op, phys)
        dest = _apply_lastdim_cycled!(dest, src, grid, lop, plan, nworkers)
        src, dest = dest, src
        steps_done += 1
    end

    src.data === u.data || copyto!(u.data, src.data)
    return u
end

"""
    _apply_tensor_chain!(u, grid, op, plan, nworkers)

Apply each tensor sweep stored in a [`CompositeTensorOp`](@ref) sequentially.
"""
function _apply_tensor_chain!(u::OrientedCoeffs{D,ElT},
                              grid::SparseGrid,
                              op::CompositeTensorOp{D},
                              plan::CyclicLayoutPlan{D,Ti,ElT},
                              nworkers::Int) where {D,Ti,ElT}
    nworkers = clamp(Int(nworkers), 1, plan.meta.pool)
    for sweep in op.ops
        _apply_tensor_sweep!(u, grid, sweep, plan, nworkers)
    end
    return u
end

"""
    _apply_updown!(u, grid, op, plan)

Apply an [`UpDownTensorOp`](@ref), selecting fiber-level or term-level
parallelism from the effective number of split terms.
"""
function _apply_updown!(u::OrientedCoeffs{D,ElT},
                        grid::SparseGrid,
                        op::UpDownTensorOp{D},
                        plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    pool = plan.meta.pool
    idperm = SVector{D,Int}(ntuple(identity, Val(D)))
    u.perm == idperm || throw(ArgumentError("UpDownTensorOp expects u.perm == identity; got $(u.perm)"))
    op.has_zero && return (fill!(u.data, zero(ElT)); u)

    omit_eff = if op.omit_dim == 0
        0
    elseif ((op.split_mask >> (op.omit_dim - 1)) & 0x1) == 0x1
        op.omit_dim
    else
        0
    end
    perm0 = omit_eff == 0 ? idperm : cycle_last_to_front(idperm, mod(1 - omit_eff, D))
    split_mask = omit_eff == 0 ? op.split_mask : (op.split_mask & ~(UInt64(1) << (omit_eff - 1)))
    term_masks = _collect_term_masks(split_mask, Val(D))
    nterms = length(term_masks)
    use_term = pool > 1 && 2 * nterms >= pool

    xdata = if perm0 == idperm
        u.data
    else
        xbuf = plan.workspace.x_buf
        _cyclic_rotate_to!(xbuf, u.data, grid, idperm, perm0, plan, pool)
        xbuf
    end

    ybuf = plan.workspace.acc_buf
    if use_term
        _apply_updown_term!(ybuf, xdata, perm0, grid, op, plan, omit_eff, term_masks, pool)
    else
        _apply_updown_fiber!(ybuf, xdata, perm0, grid, op, plan, omit_eff, term_masks, pool)
    end

    if perm0 == idperm
        copyto!(u.data, ybuf)
    else
        _cyclic_rotate_to!(u.data, ybuf, grid, perm0, idperm, plan, pool)
    end
    return u
end

"""
    apply_unidirectional!(u, grid, op, plan)
    apply_unidirectional!(u, grid, op)

Apply a line, tensor, composite, or UpDown operator to `u` using cyclic
unidirectional sweeps.

For [`UpDownTensorOp`](@ref), the backend is chosen automatically from the
number of effective split terms: if `Threads.threadpoolsize(:default) > 1` and
`2*nterms >= pool`, term-level parallelism is used; otherwise fiber-level
parallelism is used.
"""
function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::TensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    return _apply_tensor_sweep!(u, grid, op, plan, plan.meta.pool)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::CompositeTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    return _apply_tensor_chain!(u, grid, op, plan, plan.meta.pool)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::AbstractLineOp,
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    return _apply_tensor_sweep!(u, grid, tensorize(op, Val(D)), plan, plan.meta.pool)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op::UpDownTensorOp{D},
                               plan::CyclicLayoutPlan{D,Ti,ElT}) where {D,Ti,ElT}
    return _apply_updown!(u, grid, op, plan)
end

function apply_unidirectional!(u::OrientedCoeffs{D,ElT},
                               grid::SparseGrid,
                               op) where {D,ElT}
    plan = CyclicLayoutPlan(grid, ElT)
    return apply_unidirectional!(u, grid, op, plan)
end
