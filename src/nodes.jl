# Rule / axis-family constructor.

abstract type AbstractAxisFamily end
abstract type AbstractUnivariateNodes <: AbstractAxisFamily end
abstract type AbstractNestedNodes <: AbstractUnivariateNodes end
abstract type AbstractNonNestedNodes <: AbstractUnivariateNodes end

const AbstractNestedAxisFamily = AbstractNestedNodes
const AbstractNonNestedAxisFamily = AbstractNonNestedNodes

# -----------------------------------------------------------------------------
# Endpoint mode (whether a nested node family includes boundary points)

abstract type AbstractEndpointMode end

"""Two-bit endpoint mask.

Bits indicate which endpoints are **kept** (bit = 1) at level 0.

* bit 0 (0b01): right endpoint
* bit 1 (0b10): left endpoint

So the four cases are:

* `EndpointMask{0b00}`: no endpoints
* `EndpointMask{0b01}`: right only
* `EndpointMask{0b10}`: left only
* `EndpointMask{0b11}`: both
"""
struct EndpointMask{M} <: AbstractEndpointMode end

const NoEndpoints    = EndpointMask{0x0}
const RightEndpoint  = EndpointMask{0x1}
const LeftEndpoint   = EndpointMask{0x2}
const BothEndpoints  = EndpointMask{0x3}

@inline has_left(::Type{EndpointMask{M}}) where {M} = (M & 0x2) != 0
@inline has_right(::Type{EndpointMask{M}}) where {M} = (M & 0x1) != 0

@inline has_left(::EndpointMask{M}) where {M} = has_left(EndpointMask{M})
@inline has_right(::EndpointMask{M}) where {M} = has_right(EndpointMask{M})

@inline n_endpoints(::Type{EndpointMask{M}}) where {M} =
    (has_left(EndpointMask{M}) ? 1 : 0) + (has_right(EndpointMask{M}) ? 1 : 0)
@inline n_endpoints(::EndpointMask{M}) where {M} = n_endpoints(EndpointMask{M})

"""Number of endpoints implied by the node rule."""
@inline n_endpoints(n::AbstractUnivariateNodes) =
    throw(ArgumentError("n_endpoints is not defined for $(typeof(n))"))

@inline _endpoint_mask_value(endpoints::Bool) = endpoints ? 0x3 : 0x0
@inline _endpoint_mask_value(endpoints::Symbol) =
    endpoints === :none  ? 0x0 :
    endpoints === :right ? 0x1 :
    endpoints === :left  ? 0x2 :
    endpoints === :both  ? 0x3 :
    throw(ArgumentError("endpoints must be one of :none/:left/:right/:both, got endpoints=$endpoints"))
@inline _endpoint_mask_value(endpoints::Integer) = Int(endpoints)
@inline _endpoint_mask_value(::Val{M}) where {M} = Int(M)

@inline function _endpoint_mode(endpoints)
    M = _endpoint_mask_value(endpoints)
    (0 <= M <= 3) || throw(ArgumentError("endpoint mask must satisfy 0 ≤ mask ≤ 3, got mask=$M"))
    return EndpointMask{M}
end

"""Return the underlying measure (A) associated with a node rule."""
measure(n::AbstractUnivariateNodes) = error("measure not implemented for $(typeof(n))")

"""Return the node/storage order associated with the rule (e.g. `LevelOrder`)."""
nodeorder(n::AbstractUnivariateNodes) = error("nodeorder not implemented for $(typeof(n))")

"""Whether the sequence of node sets is nested across increasing levels."""
is_nested(::AbstractNestedNodes) = true
is_nested(::AbstractNonNestedNodes) = false

"""Total number of entries in the refinement rule `X_r`."""
totalsize(axis::AbstractAxisFamily, r::Integer) = error("totalsize not implemented for $(typeof(axis))")

"""Number of entries added by the increment block `ΔX_r`."""
blocksize(axis::AbstractAxisFamily, r::Integer) = error("blocksize not implemented for $(typeof(axis))")

"""Number of points in the refinement rule `X_r` (legacy alias for [`totalsize`](@ref))."""
npoints(axis::AbstractAxisFamily, r::Integer) = totalsize(axis, r)

"""Return the refinement index `r` such that `totalsize(axis, r) == n`."""
refinement_index(axis::AbstractAxisFamily, n::Integer) =
    error("refinement_index not implemented for $(typeof(axis))")

"""Coordinates of points for the refinement rule `X_r`."""
points(axis::AbstractAxisFamily, r::Integer) = error("points not implemented for $(typeof(axis))")

raw"""Coordinates of the *new* points added at refinement index `r` (i.e. points in $\Delta X_r$)."""
newpoints(axis::AbstractAxisFamily, r::Integer) = error("newpoints not implemented for $(typeof(axis))")

# Nodal axes satisfy the generic size interface via their point counts.
totalsize(n::AbstractUnivariateNodes, r::Integer) = npoints(n, r)

# --- generic Gaussian-type rules (usually non-nested) --------------------------

struct GaussNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNonNestedNodes
    measure::M
    order::O
end
GaussNodes() = GaussNodes(LegendreMeasure(), NaturalOrder())
GaussNodes(measure::M) where {M<:AbstractMeasure} = GaussNodes(measure, NaturalOrder())
measure(n::GaussNodes) = n.measure
nodeorder(n::GaussNodes) = n.order

struct GaussLobattoNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNonNestedNodes
    measure::M
    order::O
end
GaussLobattoNodes() = GaussLobattoNodes(LegendreMeasure(), NaturalOrder())
GaussLobattoNodes(measure::M) where {M<:AbstractMeasure} = GaussLobattoNodes(measure, NaturalOrder())
measure(n::GaussLobattoNodes) = n.measure
nodeorder(n::GaussLobattoNodes) = n.order

# --- nested rules --------------------------------------------------------------

"""Gauss–Patterson nested rules (Legendre by default)."""
struct PattersonNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNestedNodes
    measure::M
    order::O
end
PattersonNodes() = PattersonNodes(LegendreMeasure(), NaturalOrder())
PattersonNodes(measure::M) where {M<:AbstractMeasure} = PattersonNodes(measure, NaturalOrder())
measure(n::PattersonNodes) = n.measure
nodeorder(n::PattersonNodes) = n.order

"""Weighted Leja sequences (nested, typically 1 point per refinement step)."""
struct LejaNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNestedNodes
    measure::M
    order::O
end
LejaNodes(measure::M) where {M<:AbstractMeasure} = LejaNodes(measure, NaturalOrder())
measure(n::LejaNodes) = n.measure
nodeorder(n::LejaNodes) = n.order

"""Pseudo-Gaussian nested sequences (as in Leja/pseudo-Gauss constructions)."""
struct PseudoGaussNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNestedNodes
    measure::M
    order::O
end
PseudoGaussNodes(measure::M) where {M<:AbstractMeasure} = PseudoGaussNodes(measure, NaturalOrder())
measure(n::PseudoGaussNodes) = n.measure
nodeorder(n::PseudoGaussNodes) = n.order

"""Chebyshev–Gauss–Lobatto nodes (a.k.a. Clenshaw–Curtis nodes) with dyadic nesting."""
struct ChebyshevGaussLobattoNodes{O<:AbstractNodeOrder,EM<:AbstractEndpointMode} <: AbstractNestedNodes
    order::O
end

function ChebyshevGaussLobattoNodes(order::O=LevelOrder(); endpoints=:both) where {O<:AbstractNodeOrder}
    EM = _endpoint_mode(endpoints)
    return ChebyshevGaussLobattoNodes{O,EM}(order)
end
measure(::ChebyshevGaussLobattoNodes) = ChebyshevTMeasure()
nodeorder(n::ChebyshevGaussLobattoNodes) = n.order

@inline n_endpoints(::ChebyshevGaussLobattoNodes{<:Any,EM}) where {EM<:AbstractEndpointMode} = n_endpoints(EM)

const ClenshawCurtisNodes = ChebyshevGaussLobattoNodes

"""Equispaced nodes on [0,1) (periodic when used with FourierMeasure)."""
struct FourierEquispacedNodes{O<:AbstractNodeOrder} <: AbstractNestedNodes
    order::O
end
FourierEquispacedNodes() = FourierEquispacedNodes(LevelOrder())
measure(::FourierEquispacedNodes) = FourierMeasure()
nodeorder(n::FourierEquispacedNodes) = n.order

@inline n_endpoints(::FourierEquispacedNodes) = 1

const EquispacedNodes = FourierEquispacedNodes

"""Dyadic nested nodes on the unit interval [0,1] with endpoints at level 0.

This is the standard dyadic refinement of a closed interval:

* level 0: `{0, 1}`
* level 1 adds `{1/2}`
* level 2 adds `{1/4, 3/4}`
* …

So the total number of points is `N = 2^ℓ + 1`.
"""
struct DyadicNodes{O<:AbstractNodeOrder,EM<:AbstractEndpointMode} <: AbstractNestedNodes
    order::O
end

function DyadicNodes(order::O=LevelOrder(); endpoints=:both) where {O<:AbstractNodeOrder}
    M = _endpoint_mask_value(endpoints)
    (M == 0x0 || M == 0x3) ||
        throw(ArgumentError("DyadicNodes currently supports only endpoints=:none or :both, got mask=$M"))
    EM = EndpointMask{M}
    return DyadicNodes{O,EM}(order)
end
measure(::DyadicNodes) = UnitIntervalMeasure()
nodeorder(n::DyadicNodes) = n.order

@inline n_endpoints(::DyadicNodes{<:Any,EM}) where {EM<:AbstractEndpointMode} = n_endpoints(EM)

# --- pass-1 node generators (CGL/Fourier/Dyadic) ------------------------------

@inline _check_level(ℓ::Integer) = (ℓ < 0) && throw(ArgumentError("level ℓ must satisfy ℓ ≥ 0, got $ℓ"))

# Chebyshev–Gauss–Lobatto (CGL / Clenshaw–Curtis): N = 2^ℓ intervals.

npoints(::ChebyshevGaussLobattoNodes{<:Any,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode} =
    (_check_level(ℓ); ((1 << ℓ) - 1) + n_endpoints(EM))

function newpoints(::ChebyshevGaussLobattoNodes{<:NaturalOrder,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode}
    _check_level(ℓ)
    # Natural Chebyshev index order j = 0..N (x_j = cos(jπ/N)), not coordinate-sorted.
    if ℓ == 0
        xs = Float64[]
        has_right(EM) && push!(xs, 1.0)
        has_left(EM) && push!(xs, -1.0)
        return xs
    elseif ℓ == 1
        return [0.0]
    end
    N = 1 << ℓ
    ks = collect(1:2:(N - 1))
    # For k increasing, cos(kπ/N) decreases from near 1 toward -1.
    return cos.(pi .* ks ./ N)
end

function newpoints(::ChebyshevGaussLobattoNodes{<:LevelOrder,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode}
    _check_level(ℓ)
    if ℓ == 0
        xs = Float64[]
        has_right(EM) && push!(xs, 1.0)
        has_left(EM) && push!(xs, -1.0)
        return xs
    elseif ℓ == 1
        return [0.0]
    end
    N = 1 << ℓ
    # IMPORTANT:
    # We deliberately order the new (odd-index) points at each Δ-level in
    # **natural odd-index order** (k = 1,3,5,...) rather than bit-reversal order.
    #
    # This makes the LevelOrder storage *compatible with a permutation-free*
    # radix-2 DCT-I recursion:
    #   - even-index samples form a contiguous prefix (recursed in LevelOrder)
    #   - odd-index samples form a contiguous suffix (in NaturalOrder)
    #
    # See `ChebyshevPlan` in `src/transforms.jl` for the corresponding DCT-I plan.
    ks = collect(1:2:(N - 1))
    return cos.(pi .* ks ./ N)
end

"""Return Chebyshev indices `k=0..N` in the package's CGL `LevelOrder`.

This returns a vector `idx` of length `N+1` where `idx[pos] = k` gives the
natural Chebyshev index of the point stored at position `pos` in
`ChebyshevGaussLobattoNodes(LevelOrder())` for level `ℓ`.

This order is a recursive parity split:

* the first `N/2+1` entries are the even indices `2k` in the LevelOrder of level `ℓ-1`
* the last  `N/2` entries are the odd indices `1,3,5,...,N-1` in increasing order

This property is relied upon by the permutation-free DCT-I implementation.
"""
function _cgl_levelorder_indices(ℓ::Integer)
    _check_level(ℓ)
    if ℓ == 0
        return [0, 1]
    end
    N = 1 << ℓ
    if ℓ == 1
        return [0, N, N >>> 1]
    end
    # Recursive parity split: even subtree (scaled indices) then odd indices.
    even = 2 .* _cgl_levelorder_indices(ℓ - 1)
    odd  = collect(1:2:(N - 1))
    return vcat(even, odd)
end

function points(n::ChebyshevGaussLobattoNodes{<:AbstractNodeOrder,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode}
    _check_level(ℓ)
    if nodeorder(n) isa NaturalOrder
        N = 1 << ℓ
        k0 = has_right(EM) ? 0 : 1
        k1 = has_left(EM) ? N : (N - 1)
        k0 > k1 && return Float64[]
        return cos.(pi .* collect(k0:k1) ./ N)
    end

    xs = Float64[]
    for j in 0:ℓ
        append!(xs, newpoints(n, j))
    end
    return xs
end

# Fourier equispaced: N = 2^ℓ points in [0,1).

npoints(::FourierEquispacedNodes, ℓ::Integer) = (_check_level(ℓ); 1 << ℓ)

function newpoints(::FourierEquispacedNodes{<:NaturalOrder}, ℓ::Integer)
    _check_level(ℓ)
    if ℓ == 0
        return [0.0]
    end
    m = 1 << (ℓ - 1)
    denom = 1 << ℓ
    return (2 .* (0:(m - 1)) .+ 1) ./ denom
end

function newpoints(::FourierEquispacedNodes{<:LevelOrder}, ℓ::Integer)
    _check_level(ℓ)
    if ℓ == 0
        return [0.0]
    end
    m = 1 << (ℓ - 1)
    denom = 1 << ℓ
    perm = bitrevperm(m)
    return (2 .* perm .+ 1) ./ denom
end

function points(n::FourierEquispacedNodes, ℓ::Integer)
    _check_level(ℓ)
    N = 1 << ℓ
    if nodeorder(n) isa NaturalOrder
        return collect(0:(N - 1)) ./ N
    else
        perm = bitrevperm(N)
        return perm ./ N
    end
end

# Dyadic nodes on [0,1]: N = 2^ℓ intervals.

npoints(::DyadicNodes{<:Any,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode} =
    (_check_level(ℓ); ((1 << ℓ) - 1) + n_endpoints(EM))

function newpoints(::DyadicNodes{<:NaturalOrder,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode}
    _check_level(ℓ)
    if ℓ == 0
        xs = Float64[]
        has_left(EM) && push!(xs, 0.0)
        has_right(EM) && push!(xs, 1.0)
        return xs
    end
    m = 1 << (ℓ - 1)
    denom = 1 << ℓ
    # New points are the odd dyadics (2k+1)/2^ℓ.
    return (2 .* (0:(m - 1)) .+ 1) ./ denom
end

function newpoints(::DyadicNodes{<:LevelOrder,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode}
    _check_level(ℓ)
    if ℓ == 0
        xs = Float64[]
        has_left(EM) && push!(xs, 0.0)
        has_right(EM) && push!(xs, 1.0)
        return xs
    end
    m = 1 << (ℓ - 1)
    denom = 1 << ℓ
    # New points are the odd dyadics (2k+1)/2^ℓ, in increasing order.
    return (2 .* (0:(m - 1)) .+ 1) ./ denom
end

function points(n::DyadicNodes{<:Any,EM}, ℓ::Integer) where {EM<:AbstractEndpointMode}
    _check_level(ℓ)
    N = 1 << ℓ
    if nodeorder(n) isa NaturalOrder
        xs = Float64[]
        has_left(EM) && push!(xs, 0.0)
        if N > 1
            append!(xs, collect(1:(N - 1)) ./ N)
        end
        has_right(EM) && push!(xs, 1.0)
        return xs
    else
        # Hierarchical concatenation of Δ-levels.
        xs = Float64[]
        for j in 0:ℓ
            append!(xs, newpoints(n, j))
        end
        return xs
    end
end

"""The number of new nodes introduced at refinement index `r` (legacy alias for [`blocksize`](@ref))."""
@inline function delta_count(n::AbstractUnivariateNodes, r::Integer)
    r == 0 && return npoints(n, 0)
    return npoints(n, r) - npoints(n, r - 1)
end

blocksize(n::AbstractUnivariateNodes, r::Integer) = delta_count(n, r)

@inline function _refinement_index_pow2(m::Integer, name::AbstractString)
    m > 0 || throw(ArgumentError("$name must be positive, got m=$m"))
    ispow2(m) || throw(ArgumentError("$name must be a power of two, got m=$m"))
    return trailing_zeros(Int(m))
end

function refinement_index(axis::ChebyshevGaussLobattoNodes{<:Any,EM}, n::Integer) where {EM<:AbstractEndpointMode}
    n >= 0 || throw(ArgumentError("n must be nonnegative, got n=$n"))
    m = Int(n) + 1 - n_endpoints(EM)
    return _refinement_index_pow2(m, "n + 1 - n_endpoints")
end

function refinement_index(::FourierEquispacedNodes, n::Integer)
    n > 0 || throw(ArgumentError("n must be positive, got n=$n"))
    return _refinement_index_pow2(Int(n), "n")
end

function refinement_index(axis::DyadicNodes{<:Any,EM}, n::Integer) where {EM<:AbstractEndpointMode}
    n >= 0 || throw(ArgumentError("n must be nonnegative, got n=$n"))
    m = Int(n) + 1 - n_endpoints(EM)
    return _refinement_index_pow2(m, "n + 1 - n_endpoints")
end
