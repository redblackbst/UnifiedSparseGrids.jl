"""
Polynomial / weight family.

A measure encodes the *reference* domain and weight function for an
orthogonal family (Legendre/Chebyshev/Jacobi/…), or a non-polynomial
measure used for node sequences (e.g. Fourier on a periodic interval).

Pass-1 scope: measures are primarily type tags used for dispatch.
"""
abstract type AbstractMeasure end

"""Uniform weight on [-1,1] (Legendre family)."""
struct LegendreMeasure <: AbstractMeasure end

"""Chebyshev of the first kind (T_n), weight w(x) = (1-x^2)^(-1/2) on [-1,1]."""
struct ChebyshevTMeasure <: AbstractMeasure end

"""Chebyshev of the second kind (U_n), weight w(x) = (1-x^2)^(+1/2) on [-1,1]."""
struct ChebyshevUMeasure <: AbstractMeasure end

"""Jacobi measure with parameters α, β (α,β > -1) on [-1,1]."""
struct JacobiMeasure{T<:Real} <: AbstractMeasure
    α::T
    β::T
end

"""Laguerre measure with parameter α (α > -1) on [0,∞)."""
struct LaguerreMeasure{T<:Real} <: AbstractMeasure
    α::T
end

"""Hermite measure on (-∞,∞)."""
struct HermiteMeasure <: AbstractMeasure end

"""Periodic (Fourier) measure on [0,1)."""
struct FourierMeasure <: AbstractMeasure end

"""Uniform measure on the unit interval [0,1]."""
struct UnitIntervalMeasure <: AbstractMeasure end

"""Reference domain of a measure as a tuple (a,b) or (a,b; periodic)."""
domain(::LegendreMeasure) = (-1.0, 1.0)
domain(::ChebyshevTMeasure) = (-1.0, 1.0)
domain(::ChebyshevUMeasure) = (-1.0, 1.0)
domain(::JacobiMeasure) = (-1.0, 1.0)
domain(::LaguerreMeasure) = (0.0, Inf)
domain(::HermiteMeasure) = (-Inf, Inf)
domain(::FourierMeasure) = (0.0, 1.0)
domain(::UnitIntervalMeasure) = (0.0, 1.0)
