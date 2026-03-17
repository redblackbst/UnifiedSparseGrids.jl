"""Axis families, measures, node orderings, and one-dimensional node sets."""

# -----------------------------------------------------------------------------
# Measures

"""
    AbstractMeasure

Abstract supertype for reference measures associated with univariate nodes and
bases.
"""
abstract type AbstractMeasure end

"""
    struct LegendreMeasure <: AbstractMeasure

Uniform measure on `[-1, 1]` associated with Legendre polynomials.
"""
struct LegendreMeasure <: AbstractMeasure end

"""
    struct ChebyshevTMeasure <: AbstractMeasure

Chebyshev first-kind measure on `[-1, 1]`.
"""
struct ChebyshevTMeasure <: AbstractMeasure end

"""
    struct ChebyshevUMeasure <: AbstractMeasure

Chebyshev second-kind measure on `[-1, 1]`.
"""
struct ChebyshevUMeasure <: AbstractMeasure end

"""
    struct JacobiMeasure{T<:Real} <: AbstractMeasure

Jacobi measure with parameters `alpha = α` and `beta = β`, both greater than `-1`.
"""
struct JacobiMeasure{T<:Real} <: AbstractMeasure
    α::T
    β::T
end

"""
    struct LaguerreMeasure{T<:Real} <: AbstractMeasure

Laguerre measure with parameter `alpha = α > -1` on `[0, Inf)`.
"""
struct LaguerreMeasure{T<:Real} <: AbstractMeasure
    α::T
end

"""
    struct HermiteMeasure <: AbstractMeasure

Hermite measure on `(-Inf, Inf)`.
"""
struct HermiteMeasure <: AbstractMeasure end

"""
    struct FourierMeasure <: AbstractMeasure

Periodic measure on `[0, 1)` used by Fourier-based node families.
"""
struct FourierMeasure <: AbstractMeasure end

"""
    struct UnitIntervalMeasure <: AbstractMeasure

Uniform measure on `[0, 1]`.
"""
struct UnitIntervalMeasure <: AbstractMeasure end

"""
    domain(measure::AbstractMeasure)

Return the reference domain of `measure`.
"""
domain(::LegendreMeasure) = (-1.0, 1.0)
domain(::ChebyshevTMeasure) = (-1.0, 1.0)
domain(::ChebyshevUMeasure) = (-1.0, 1.0)
domain(::JacobiMeasure) = (-1.0, 1.0)
domain(::LaguerreMeasure) = (0.0, Inf)
domain(::HermiteMeasure) = (-Inf, Inf)
domain(::FourierMeasure) = (0.0, 1.0)
domain(::UnitIntervalMeasure) = (0.0, 1.0)

# -----------------------------------------------------------------------------
# Node orders

"""
    AbstractNodeOrder

Abstract supertype for one-dimensional storage orders.
"""
abstract type AbstractNodeOrder end

"""
    struct NaturalOrder <: AbstractNodeOrder

Natural one-dimensional order, typically increasing coordinate order.
"""
struct NaturalOrder <: AbstractNodeOrder end

"""
    struct LevelOrder <: AbstractNodeOrder

Hierarchical one-dimensional order obtained by recursively splitting even indices
before odd indices. On radix-2 grids this agrees with bit reversal.
"""
struct LevelOrder <: AbstractNodeOrder end

"""
    bitrev(j, m) -> Int

Reverse the lowest `m` bits of the nonnegative integer `j`.
"""
function bitrev(j::Integer, m::Integer)::Int
    j >= 0 || throw(ArgumentError("bitrev requires j >= 0, got $j"))
    m >= 0 || throw(ArgumentError("bitrev requires m >= 0, got $m"))
    x = UInt(j)
    r = UInt(0)
    for _ in 1:m
        r = (r << 1) | (x & UInt(1))
        x >>= 1
    end
    return Int(r)
end

"""
    bitrevperm(N) -> Vector{Int}

Return the radix-2 bit-reversal permutation of length `N`.
"""
function bitrevperm(N::Integer)::Vector{Int}
    N > 0 || throw(ArgumentError("bitrevperm requires N > 0, got $N"))
    (N & (N - 1)) == 0 || throw(ArgumentError("bitrevperm requires N to be a power of two, got $N"))
    m = trailing_zeros(UInt(N))
    p = Vector{Int}(undef, N)
    for i in 0:(N - 1)
        p[i + 1] = bitrev(i, m)
    end
    return p
end

# -----------------------------------------------------------------------------
# Axis families and node sets

"""
    AbstractAxisFamily

Abstract supertype for one-dimensional refinement families.
"""
abstract type AbstractAxisFamily end

"""
    AbstractUnivariateNodes <: AbstractAxisFamily

Abstract supertype for one-dimensional node families.
"""
abstract type AbstractUnivariateNodes <: AbstractAxisFamily end

"""
    AbstractNestedNodes <: AbstractUnivariateNodes

Abstract supertype for nested one-dimensional node families.
"""
abstract type AbstractNestedNodes <: AbstractUnivariateNodes end

"""
    AbstractNonNestedNodes <: AbstractUnivariateNodes

Abstract supertype for non-nested one-dimensional node families.
"""
abstract type AbstractNonNestedNodes <: AbstractUnivariateNodes end

"""
    AbstractNestedAxisFamily

Alias for [`AbstractNestedNodes`](@ref).
"""
const AbstractNestedAxisFamily = AbstractNestedNodes

"""
    AbstractNonNestedAxisFamily

Alias for [`AbstractNonNestedNodes`](@ref).
"""
const AbstractNonNestedAxisFamily = AbstractNonNestedNodes

abstract type AbstractEndpointMode end

"""
    struct EndpointMask{M} <: AbstractEndpointMode

Two-bit endpoint mask used by nested node families. Bit `0` keeps the right
endpoint and bit `1` keeps the left endpoint at level `0`.
"""
struct EndpointMask{M} <: AbstractEndpointMode end

const NoEndpoints   = EndpointMask{0x0}
const RightEndpoint = EndpointMask{0x1}
const LeftEndpoint  = EndpointMask{0x2}
const BothEndpoints = EndpointMask{0x3}

@inline has_left(::Type{EndpointMask{M}}) where {M} = (M & 0x2) != 0
@inline has_right(::Type{EndpointMask{M}}) where {M} = (M & 0x1) != 0
@inline has_left(::EndpointMask{M}) where {M} = has_left(EndpointMask{M})
@inline has_right(::EndpointMask{M}) where {M} = has_right(EndpointMask{M})

"""
    n_endpoints(::Type{<:EndpointMask})
    n_endpoints(::EndpointMask)
    n_endpoints(axis::AbstractUnivariateNodes)

Return the number of endpoints included at level `0`.
"""
@inline n_endpoints(::Type{EndpointMask{M}}) where {M} =
    (has_left(EndpointMask{M}) ? 1 : 0) + (has_right(EndpointMask{M}) ? 1 : 0)
@inline n_endpoints(::EndpointMask{M}) where {M} = n_endpoints(EndpointMask{M})
@inline n_endpoints(axis::AbstractUnivariateNodes) =
    throw(ArgumentError("n_endpoints is not defined for $(typeof(axis))"))

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
    mask = _endpoint_mask_value(endpoints)
    (0 <= mask <= 3) || throw(ArgumentError("endpoint mask must satisfy 0 <= mask <= 3, got mask=$mask"))
    return EndpointMask{mask}
end

"""
    measure(axis::AbstractUnivariateNodes)

Return the reference measure associated with `axis`.
"""
measure(axis::AbstractUnivariateNodes) = error("measure not implemented for $(typeof(axis))")

"""
    nodeorder(axis::AbstractUnivariateNodes)

Return the one-dimensional storage order associated with `axis`.
"""
nodeorder(axis::AbstractUnivariateNodes) = error("nodeorder not implemented for $(typeof(axis))")

"""
    is_nested(axis::AbstractUnivariateNodes) -> Bool

Return `true` when `axis` is nested across refinement levels.
"""
is_nested(::AbstractNestedNodes) = true
is_nested(::AbstractNonNestedNodes) = false

"""
    totalsize(axis::AbstractAxisFamily, r::Integer)

Return the total number of entries in the refinement family `X_r`.
"""
totalsize(axis::AbstractAxisFamily, r::Integer) = error("totalsize not implemented for $(typeof(axis))")

"""
    blocksize(axis::AbstractAxisFamily, r::Integer)

Return the number of entries in the increment block added at refinement level `r`.
"""
blocksize(axis::AbstractAxisFamily, r::Integer) = error("blocksize not implemented for $(typeof(axis))")

"""
    npoints(axis::AbstractAxisFamily, r::Integer)

Alias for [`totalsize`](@ref) on node-based axis families.
"""
npoints(axis::AbstractAxisFamily, r::Integer) = totalsize(axis, r)

"""
    refinement_index(axis::AbstractAxisFamily, n::Integer)

Return the refinement index `r` such that `totalsize(axis, r) == n`.
"""
refinement_index(axis::AbstractAxisFamily, n::Integer) =
    error("refinement_index not implemented for $(typeof(axis))")

"""
    points(axis::AbstractAxisFamily, r::Integer)

Return all points in the refinement family `X_r`.
"""
points(axis::AbstractAxisFamily, r::Integer) = error("points not implemented for $(typeof(axis))")

"""
    newpoints(axis::AbstractAxisFamily, r::Integer)

Return the points first introduced at refinement level `r`.
"""
newpoints(axis::AbstractAxisFamily, r::Integer) = error("newpoints not implemented for $(typeof(axis))")

# Nodal axes satisfy the generic size interface via their point counts.
totalsize(axis::AbstractUnivariateNodes, r::Integer) = npoints(axis, r)

"""
    struct GaussNodes{M,O} <: AbstractNonNestedNodes

Gaussian node family with measure `M` and storage order `O`.
"""
struct GaussNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNonNestedNodes
    measure::M
    order::O
end
GaussNodes() = GaussNodes(LegendreMeasure(), NaturalOrder())
GaussNodes(measure::M) where {M<:AbstractMeasure} = GaussNodes(measure, NaturalOrder())
measure(axis::GaussNodes) = axis.measure
nodeorder(axis::GaussNodes) = axis.order

"""
    struct GaussLobattoNodes{M,O} <: AbstractNonNestedNodes

Gauss-Lobatto node family with measure `M` and storage order `O`.
"""
struct GaussLobattoNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNonNestedNodes
    measure::M
    order::O
end
GaussLobattoNodes() = GaussLobattoNodes(LegendreMeasure(), NaturalOrder())
GaussLobattoNodes(measure::M) where {M<:AbstractMeasure} = GaussLobattoNodes(measure, NaturalOrder())
measure(axis::GaussLobattoNodes) = axis.measure
nodeorder(axis::GaussLobattoNodes) = axis.order

"""
    struct PattersonNodes{M,O} <: AbstractNestedNodes

Gauss-Patterson node family with measure `M` and storage order `O`.
"""
struct PattersonNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNestedNodes
    measure::M
    order::O
end
PattersonNodes() = PattersonNodes(LegendreMeasure(), NaturalOrder())
PattersonNodes(measure::M) where {M<:AbstractMeasure} = PattersonNodes(measure, NaturalOrder())
measure(axis::PattersonNodes) = axis.measure
nodeorder(axis::PattersonNodes) = axis.order

"""
    struct LejaNodes{M,O} <: AbstractNestedNodes

Weighted Leja node family with measure `M` and storage order `O`.
"""
struct LejaNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNestedNodes
    measure::M
    order::O
end
LejaNodes(measure::M) where {M<:AbstractMeasure} = LejaNodes(measure, NaturalOrder())
measure(axis::LejaNodes) = axis.measure
nodeorder(axis::LejaNodes) = axis.order

"""
    struct PseudoGaussNodes{M,O} <: AbstractNestedNodes

Pseudo-Gaussian nested node family with measure `M` and storage order `O`.
"""
struct PseudoGaussNodes{M<:AbstractMeasure,O<:AbstractNodeOrder} <: AbstractNestedNodes
    measure::M
    order::O
end
PseudoGaussNodes(measure::M) where {M<:AbstractMeasure} = PseudoGaussNodes(measure, NaturalOrder())
measure(axis::PseudoGaussNodes) = axis.measure
nodeorder(axis::PseudoGaussNodes) = axis.order

"""
    struct ChebyshevGaussLobattoNodes{O,EM} <: AbstractNestedNodes

Nested Chebyshev-Gauss-Lobatto node family with storage order `O` and endpoint
mask `EM`.
"""
struct ChebyshevGaussLobattoNodes{O<:AbstractNodeOrder,EM<:AbstractEndpointMode} <: AbstractNestedNodes
    order::O
end

function ChebyshevGaussLobattoNodes(order::O=LevelOrder(); endpoints=:both) where {O<:AbstractNodeOrder}
    endpoint_mode = _endpoint_mode(endpoints)
    return ChebyshevGaussLobattoNodes{O,endpoint_mode}(order)
end
measure(::ChebyshevGaussLobattoNodes) = ChebyshevTMeasure()
nodeorder(axis::ChebyshevGaussLobattoNodes) = axis.order
@inline n_endpoints(::ChebyshevGaussLobattoNodes{<:Any,EM}) where {EM<:AbstractEndpointMode} = n_endpoints(EM)

"""
    ClenshawCurtisNodes

Alias for [`ChebyshevGaussLobattoNodes`](@ref).
"""
const ClenshawCurtisNodes = ChebyshevGaussLobattoNodes

"""
    struct FourierEquispacedNodes{O} <: AbstractNestedNodes

Nested equispaced node family on `[0, 1)` with storage order `O`.
"""
struct FourierEquispacedNodes{O<:AbstractNodeOrder} <: AbstractNestedNodes
    order::O
end
FourierEquispacedNodes() = FourierEquispacedNodes(LevelOrder())
measure(::FourierEquispacedNodes) = FourierMeasure()
nodeorder(axis::FourierEquispacedNodes) = axis.order
@inline n_endpoints(::FourierEquispacedNodes) = 1

"""
    EquispacedNodes

Alias for [`FourierEquispacedNodes`](@ref).
"""
const EquispacedNodes = FourierEquispacedNodes

"""
    struct DyadicNodes{O,EM} <: AbstractNestedNodes

Nested dyadic node family on `[0, 1]` with storage order `O` and endpoint mask
`EM`. With both endpoints present, the total number of points is `2^level + 1`
at refinement level `level`.
"""
struct DyadicNodes{O<:AbstractNodeOrder,EM<:AbstractEndpointMode} <: AbstractNestedNodes
    order::O
end

function DyadicNodes(order::O=LevelOrder(); endpoints=:both) where {O<:AbstractNodeOrder}
    mask = _endpoint_mask_value(endpoints)
    (mask == 0x0 || mask == 0x3) ||
        throw(ArgumentError("DyadicNodes currently supports only endpoints=:none or :both, got mask=$mask"))
    endpoint_mode = EndpointMask{mask}
    return DyadicNodes{O,endpoint_mode}(order)
end
measure(::DyadicNodes) = UnitIntervalMeasure()
nodeorder(axis::DyadicNodes) = axis.order
@inline n_endpoints(::DyadicNodes{<:Any,EM}) where {EM<:AbstractEndpointMode} = n_endpoints(EM)

@inline _check_level(level::Integer) = (level < 0) &&
    throw(ArgumentError("level must satisfy level >= 0, got level=$level"))

# Chebyshev-Gauss-Lobatto / Clenshaw-Curtis: N = 2^level intervals.

npoints(::ChebyshevGaussLobattoNodes{<:Any,EM}, level::Integer) where {EM<:AbstractEndpointMode} =
    (_check_level(level); ((1 << level) - 1) + n_endpoints(EM))

function newpoints(::ChebyshevGaussLobattoNodes{<:NaturalOrder,EM}, level::Integer) where {EM<:AbstractEndpointMode}
    _check_level(level)
    if level == 0
        xs = Float64[]
        has_right(EM) && push!(xs, 1.0)
        has_left(EM) && push!(xs, -1.0)
        return xs
    elseif level == 1
        return [0.0]
    end
    N = 1 << level
    ks = collect(1:2:(N - 1))
    return cos.(pi .* ks ./ N)
end

function newpoints(::ChebyshevGaussLobattoNodes{<:LevelOrder,EM}, level::Integer) where {EM<:AbstractEndpointMode}
    _check_level(level)
    if level == 0
        xs = Float64[]
        has_right(EM) && push!(xs, 1.0)
        has_left(EM) && push!(xs, -1.0)
        return xs
    elseif level == 1
        return [0.0]
    end
    N = 1 << level
    ks = collect(1:2:(N - 1))
    return cos.(pi .* ks ./ N)
end

"""
    _cgl_levelorder_indices(level::Integer) -> Vector{Int}

Return the natural Chebyshev indices stored at each position of the package
`LevelOrder` for `ChebyshevGaussLobattoNodes`.
"""
function _cgl_levelorder_indices(level::Integer)
    _check_level(level)
    if level == 0
        return [0, 1]
    end
    N = 1 << level
    if level == 1
        return [0, N, N >>> 1]
    end
    even = 2 .* _cgl_levelorder_indices(level - 1)
    odd = collect(1:2:(N - 1))
    return vcat(even, odd)
end

function points(axis::ChebyshevGaussLobattoNodes{<:AbstractNodeOrder,EM}, level::Integer) where {EM<:AbstractEndpointMode}
    _check_level(level)
    if nodeorder(axis) isa NaturalOrder
        N = 1 << level
        k0 = has_right(EM) ? 0 : 1
        k1 = has_left(EM) ? N : (N - 1)
        k0 > k1 && return Float64[]
        return cos.(pi .* collect(k0:k1) ./ N)
    end

    xs = Float64[]
    for j in 0:level
        append!(xs, newpoints(axis, j))
    end
    return xs
end

# Fourier equispaced: N = 2^level points in [0, 1).

npoints(::FourierEquispacedNodes, level::Integer) = (_check_level(level); 1 << level)

function newpoints(::FourierEquispacedNodes{<:NaturalOrder}, level::Integer)
    _check_level(level)
    if level == 0
        return [0.0]
    end
    m = 1 << (level - 1)
    denom = 1 << level
    return (2 .* (0:(m - 1)) .+ 1) ./ denom
end

function newpoints(::FourierEquispacedNodes{<:LevelOrder}, level::Integer)
    _check_level(level)
    if level == 0
        return [0.0]
    end
    m = 1 << (level - 1)
    denom = 1 << level
    perm = bitrevperm(m)
    return (2 .* perm .+ 1) ./ denom
end

function points(axis::FourierEquispacedNodes, level::Integer)
    _check_level(level)
    N = 1 << level
    if nodeorder(axis) isa NaturalOrder
        return collect(0:(N - 1)) ./ N
    else
        perm = bitrevperm(N)
        return perm ./ N
    end
end

# Dyadic nodes on [0, 1]: N = 2^level intervals.

npoints(::DyadicNodes{<:Any,EM}, level::Integer) where {EM<:AbstractEndpointMode} =
    (_check_level(level); ((1 << level) - 1) + n_endpoints(EM))

function newpoints(::DyadicNodes{<:NaturalOrder,EM}, level::Integer) where {EM<:AbstractEndpointMode}
    _check_level(level)
    if level == 0
        xs = Float64[]
        has_left(EM) && push!(xs, 0.0)
        has_right(EM) && push!(xs, 1.0)
        return xs
    end
    m = 1 << (level - 1)
    denom = 1 << level
    return (2 .* (0:(m - 1)) .+ 1) ./ denom
end

function newpoints(::DyadicNodes{<:LevelOrder,EM}, level::Integer) where {EM<:AbstractEndpointMode}
    _check_level(level)
    if level == 0
        xs = Float64[]
        has_left(EM) && push!(xs, 0.0)
        has_right(EM) && push!(xs, 1.0)
        return xs
    end
    m = 1 << (level - 1)
    denom = 1 << level
    return (2 .* (0:(m - 1)) .+ 1) ./ denom
end

function points(axis::DyadicNodes{<:Any,EM}, level::Integer) where {EM<:AbstractEndpointMode}
    _check_level(level)
    N = 1 << level
    if nodeorder(axis) isa NaturalOrder
        xs = Float64[]
        has_left(EM) && push!(xs, 0.0)
        if N > 1
            append!(xs, collect(1:(N - 1)) ./ N)
        end
        has_right(EM) && push!(xs, 1.0)
        return xs
    else
        xs = Float64[]
        for j in 0:level
            append!(xs, newpoints(axis, j))
        end
        return xs
    end
end

"""
    delta_count(axis::AbstractUnivariateNodes, r::Integer)

Return the number of new points added at refinement level `r`.
"""
@inline function delta_count(axis::AbstractUnivariateNodes, r::Integer)
    r == 0 && return npoints(axis, 0)
    return npoints(axis, r) - npoints(axis, r - 1)
end

blocksize(axis::AbstractUnivariateNodes, r::Integer) = delta_count(axis, r)

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
