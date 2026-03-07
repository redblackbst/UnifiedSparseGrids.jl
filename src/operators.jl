"""Operators for unidirectional sparse grid sweeps.

`AbstractLineOp` acts on one contiguous 1D fiber.
`TensorOp{D}` represents one full sweep over all physical dimensions.
`CompositeTensorOp{D}` is a sequence of sweeps.
"""

# Line (1D fiber) operators

abstract type AbstractLineOp end

"""Identity line operator."""
struct IdentityLineOp <: AbstractLineOp end

"""Zero line operator (maps any input fiber to zeros)."""
struct ZeroLineOp <: AbstractLineOp end

"""Per-fiber modal transform (FFT/DCT/etc.)."""
struct LineTransform{Dir} <: AbstractLineOp end

LineTransform(::Val{:forward}) = LineTransform{Val(:forward)}()
LineTransform(::Val{:inverse}) = LineTransform{Val(:inverse)}()

"""Per-fiber Chebyshev (T) ↔ Legendre (P) coefficient conversion.

- `:forward` applies Chebyshev → Legendre.
- `:inverse` applies Legendre → Chebyshev.

This operator acts on coefficient vectors of length `n` (degrees `0:(n-1)`)."""
struct LineChebyshevLegendre{Dir} <: AbstractLineOp end

LineChebyshevLegendre(::Val{:forward}) = LineChebyshevLegendre{Val(:forward)}()
LineChebyshevLegendre(::Val{:inverse}) = LineChebyshevLegendre{Val(:inverse)}()

"""1D hierarchization on modal coefficients (triangular, in-place)."""
struct LineHierarchize <: AbstractLineOp end

"""1D dehierarchization on modal coefficients (triangular, in-place)."""
struct LineDehierarchize <: AbstractLineOp end

"""Transpose hierarchization on modal coefficients (triangular, in-place).

This represents the action of ``Y^T`` for hierarchical transforms.
"""
struct LineHierarchizeTranspose <: AbstractLineOp end

"""Transpose dehierarchization on modal coefficients (triangular, in-place).

This represents the action of ``Y^{-T}`` for hierarchical transforms.
"""
struct LineDehierarchizeTranspose <: AbstractLineOp end

"""Sequential composition of multiple line operators."""
struct CompositeLineOp{Ops<:Tuple} <: AbstractLineOp
    ops::Ops
end

CompositeLineOp(ops...) = CompositeLineOp(tuple(ops...))

lineops(op::AbstractLineOp) = (op,)
lineops(op::CompositeLineOp) = op.ops

# --- lightweight traits ---

"""Return `true` if this line operator is algebraically zero.

Used to prune UpDown tensor expansions.
"""
iszero(::AbstractLineOp) = false
iszero(::ZeroLineOp) = true

"""Return `true` if this line operator is an identity map.
"""
isidentity(::AbstractLineOp) = false
isidentity(::IdentityLineOp) = true

compose(a::AbstractLineOp, b::AbstractLineOp) = CompositeLineOp(a, b)
compose(a::CompositeLineOp, b::AbstractLineOp) = CompositeLineOp(a.ops..., b)
compose(a::AbstractLineOp, b::CompositeLineOp) = CompositeLineOp(a, b.ops...)
compose(a::CompositeLineOp, b::CompositeLineOp) = CompositeLineOp(a.ops..., b.ops...)

# In-place vs out-of-place line-op trait.
abstract type LineOpStyle end
struct InPlaceOp    <: LineOpStyle end
struct OutOfPlaceOp <: LineOpStyle end

lineop_style(::Type{<:AbstractLineOp}) = OutOfPlaceOp()
lineop_style(::Type{IdentityLineOp}) = InPlaceOp()
lineop_style(::Type{ZeroLineOp}) = OutOfPlaceOp()

plan_family(::Type{Op}) where {Op} = Op
plan_action(::Type{Op}) where {Op} = throw(ArgumentError("plan_action not implemented for $Op"))
is_planned_inplace(::Type{Op}) where {Op} = Val(false)

# ZeroLineOp: map any input fiber to zeros.
function apply!(dest::AbstractVector, ::ZeroLineOp, src::AbstractVector)
    fill!(dest, zero(eltype(dest)))
    return dest
end

function apply(op::ZeroLineOp, src::AbstractVector{T}) where {T}
    dest = Vector{T}(undef, length(src))
    fill!(dest, zero(T))
    return dest
end
lineop_style(::Type{LineHierarchize}) = InPlaceOp()
lineop_style(::Type{LineDehierarchize}) = InPlaceOp()
lineop_style(::Type{LineHierarchizeTranspose}) = InPlaceOp()
lineop_style(::Type{LineDehierarchizeTranspose}) = InPlaceOp()
lineop_style(::Type{<:LineTransform}) = InPlaceOp()
lineop_style(::Type{<:LineChebyshevLegendre}) = InPlaceOp()

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

# Tensor (sweep) operators

abstract type AbstractTensorOp{D} end

"""One sweep: one `AbstractLineOp` per physical dimension."""
struct TensorOp{D,Ops<:Tuple} <: AbstractTensorOp{D}
    ops::Ops
end

TensorOp(ops::Tuple) = TensorOp{length(ops),typeof(ops)}(ops)

lineop(op::TensorOp{D}, d::Integer) where {D} = op.ops[d]

tensorize(op::AbstractLineOp, ::Val{D}) where {D} = TensorOp(ntuple(_ -> op, Val(D)))

TensorOp(::Val{D}, op::AbstractLineOp) where {D} = tensorize(op, Val(D))

"""Sequence of sweeps (sequential composition of `TensorOp{D}`)."""
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

"""Forward sparse grid modal transform (nodal → standard modal)."""
function ForwardTransform(::Val{D}) where {D}
    A = tensorize(CompositeLineOp(LineTransform(Val(:forward)), LineHierarchize()), Val(D))
    B = tensorize(LineDehierarchize(), Val(D))
    return compose(A, B)
end

"""Inverse sparse grid modal transform (standard modal → nodal)."""
function InverseTransform(::Val{D}) where {D}
    Binv = tensorize(LineHierarchize(), Val(D))
    Ainv = tensorize(CompositeLineOp(LineDehierarchize(), LineTransform(Val(:inverse))), Val(D))
    return compose(Binv, Ainv)
end

ForwardTransform(D::Integer) = ForwardTransform(Val(D))
InverseTransform(D::Integer) = InverseTransform(Val(D))
