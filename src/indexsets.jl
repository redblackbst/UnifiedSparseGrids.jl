abstract type AbstractIndexSet{D} end

"""Spatial dimension of the index set."""
dim(::AbstractIndexSet{D}) where {D} = D

"""Return true if the refinement multi-index `r` is contained in the index set."""
contains(I::AbstractIndexSet{D}, r::SVector{D,<:Integer}) where {D} =
    error("contains not implemented for $(typeof(I))")

contains(I::AbstractIndexSet{D}, r::NTuple{D,<:Integer}) where {D} =
    contains(I, SVector{D,Int}(r))

"""Return the per-dimension refinement caps of an index set."""
refinement_caps(I::AbstractIndexSet) =
    error("refinement_caps not implemented for $(typeof(I))")

# --- helpers -----------------------------------------------------------------

@inline _cap_default(::Val{D}, L::Ti) where {D,Ti<:Integer} =
    SVector{D,Ti}(ntuple(_ -> L, D))

@inline function _cap_convert(::Val{D}, ::Type{Ti}, cap) where {D,Ti<:Integer}
    cap isa SVector{D,Ti} && return cap
    cap isa Integer && return SVector{D,Ti}(ntuple(_ -> Ti(cap), D))
    cap isa NTuple{D,<:Integer} && return SVector{D,Ti}(cap)
    cap isa AbstractVector{<:Integer} || throw(ArgumentError("cap must be a length-$D integer container"))
    length(cap) == D || throw(ArgumentError("cap must have length $D, got $(length(cap))"))
    return SVector{D,Ti}(Tuple(cap))
end

@inline function _check_nonneg(r)
    any(<(0), r) && return true
    return false
end

# --- index sets --------------------------------------------------------------

raw"""Classical Smolyak index set in refinement-index coordinates.

Multi-indices satisfy

```math
r \in \mathbb{N}_0^d,\qquad \sum_{j=1}^d r_j \le L,\qquad r \le \mathrm{cap}.
```

Mapping to the classical 1-based formulation:
$r_{\mathrm{old}} = r + 1$ and $q_{\mathrm{old}} = L + 1$ yield
$\|r_{\mathrm{old}}\|_1 \le q_{\mathrm{old}} + d - 1$.
"""
struct SmolyakIndexSet{D,Ti<:Integer} <: AbstractIndexSet{D}
    L::Ti
    cap::SVector{D,Ti}

    function SmolyakIndexSet(::Val{D}, L::Ti; cap=nothing) where {D,Ti<:Integer}
        L < 0 && throw(ArgumentError("L must be ≥ 0, got $L"))
        cap_s = cap === nothing ? _cap_default(Val(D), L) : _cap_convert(Val(D), Ti, cap)
        return new{D,Ti}(L, cap_s)
    end
end

SmolyakIndexSet(d::Integer, L::Integer; kwargs...) = SmolyakIndexSet(Val(d), L; kwargs...)

function contains(I::SmolyakIndexSet{D}, r::SVector{D,<:Integer}) where {D}
    _check_nonneg(r) && return false
    @inbounds for j in 1:D
        r[j] > I.cap[j] && return false
    end
    return sum(r) <= I.L
end

refinement_caps(I::SmolyakIndexSet{D}) where {D} = SVector{D,Int}(Tuple(I.cap))

raw"""Weighted Smolyak index set in refinement-index coordinates.

Multi-indices satisfy

```math
r \in \mathbb{N}_0^d,\qquad \sum_{j=1}^d \theta_j r_j \le L,\qquad r \le \mathrm{cap},
```

with positive integer weights `\theta_j = weights[j]`.
"""
struct WeightedSmolyakIndexSet{D,Ti<:Integer} <: AbstractIndexSet{D}
    L::Ti
    weights::SVector{D,Ti}
    cap::SVector{D,Ti}

    function WeightedSmolyakIndexSet(::Val{D}, L::Ti, weights; cap=nothing) where {D,Ti<:Integer}
        L < 0 && throw(ArgumentError("L must be ≥ 0, got $L"))
        weights_s = _cap_convert(Val(D), Ti, weights)
        any(w -> w <= 0, weights_s) && throw(ArgumentError("weights must be positive, got $weights_s"))
        cap_default = SVector{D,Ti}(ntuple(j -> fld(L, weights_s[j]), D))
        cap_s = cap === nothing ? cap_default : _cap_convert(Val(D), Ti, cap)
        any(ci -> ci < 0, cap_s) && throw(ArgumentError("cap entries must be ≥ 0, got $cap_s"))
        return new{D,Ti}(L, weights_s, cap_s)
    end
end

WeightedSmolyakIndexSet(d::Integer, L::Integer, weights; kwargs...) =
    WeightedSmolyakIndexSet(Val(d), L, weights; kwargs...)

function contains(I::WeightedSmolyakIndexSet{D}, r::SVector{D,<:Integer}) where {D}
    _check_nonneg(r) && return false
    acc = zero(Int)
    @inbounds for j in 1:D
        rj = Int(r[j])
        rj > I.cap[j] && return false
        acc += Int(I.weights[j]) * rj
        acc > I.L && return false
    end
    return true
end

refinement_caps(I::WeightedSmolyakIndexSet{D}) where {D} = SVector{D,Int}(Tuple(I.cap))

raw"""Full tensor (box) index set in refinement-index coordinates.

Multi-indices satisfy

```math
r \in \mathbb{N}_0^d,\qquad 0 \le r_j \le \mathrm{cap}_j.
```

The approximation meaning depends on the chosen axis families.
"""
struct FullTensorIndexSet{D,Ti<:Integer} <: AbstractIndexSet{D}
    cap::SVector{D,Ti}
    function FullTensorIndexSet(::Val{D}, L::Ti; cap=nothing) where {D,Ti<:Integer}
        L < 0 && throw(ArgumentError("L must be ≥ 0, got $L"))
        cap_s = cap === nothing ? SVector{D,Ti}(ntuple(_ -> L, D)) : _cap_convert(Val(D), Ti, cap)
        any(ci -> ci < 0, cap_s) && throw(ArgumentError("cap entries must be ≥ 0, got $cap_s"))
        return new{D,Ti}(cap_s)
    end
end

FullTensorIndexSet(d::Integer, L::Integer; kwargs...) = FullTensorIndexSet(Val(d), L; kwargs...)

function contains(I::FullTensorIndexSet{D}, r::SVector{D,<:Integer}) where {D}
    _check_nonneg(r) && return false
    @inbounds for j in 1:D
        r[j] > I.cap[j] && return false
    end
    return true
end

refinement_caps(I::FullTensorIndexSet{D}) where {D} = SVector{D,Int}(Tuple(I.cap))

# Prefer Julia's standard membership API: `r ∈ I`.
Base.in(r::SVector{D,<:Integer}, I::AbstractIndexSet{D}) where {D} = contains(I, r)
Base.in(r::NTuple{D,<:Integer}, I::AbstractIndexSet{D}) where {D} = contains(I, r)
