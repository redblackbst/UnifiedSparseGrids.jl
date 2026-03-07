raw"""Gradinaru (2007) Section 4: time-dependent Schrödinger equation (TDSE) on sparse grids.

This script implements the spatial sparse grid Fourier collocation method combined with
Strang splitting in time, following the discretization in Section 4 of:

    V. Gradinaru (2007), *Fourier transform on sparse grids: Code design and the
    time dependent Schrödinger equation*.

We consider the periodic TDSE on ``\Omega = [0,2\pi]^d``

```math
i\,\varepsilon\,\partial_t u = -\frac{\varepsilon^2}{2}\,\Delta u + V(x)\,u,
```

with the harmonic potential

```math
V(x) = \frac12 \sum_{s=1}^d (x_s - \pi)^2,
```

and a Gaussian initial condition (Gradinaru's example).

The Strang step is applied in the equivalent form (paper Eq. (11)):

```math
u \leftarrow e^{-i\,\tau V/(2\varepsilon)}\, F^{-1}\, e^{-i\,\tau (\varepsilon\,|\omega|^2/2)}\, F\, e^{-i\,\tau V/(2\varepsilon)}\,u,
```

where ``F`` / ``F^{-1}`` are the sparse grid Fourier transforms implemented by
`ForwardTransform(d)` / `InverseTransform(d)`.

Run:

    julia --project=. examples/gradinaru_2007_tdse.jl --d=2 --n=10 --eps=0.1 --tau=0.01 --T=1.0
"""

using UnifiedSparseGrids
using StaticArrays
using LinearAlgebra

@inline function freq_from_nat(k::Int, N::Int)
    N == 1 && return 0
    half = N >>> 1
    return (k <= half) ? k : (k - N)
end

function _parse_kv_args(args)
    kv = Dict{String,String}()
    for a in args
        startswith(a, "--") || continue
        s = a[3:end]
        i = findfirst(==( '=' ), s)
        i === nothing && continue
        kv[s[1:i-1]] = s[i+1:end]
    end
    return kv
end

function run_tdse(; d::Int=2,
                  n::Int=10,
                  ε::Float64=0.1,
                  τ::Float64=0.01,
                  Tfinal::Float64=1.0,
                  print_every::Int=10,
                  renorm::Bool=false)
    axes = ntuple(_ -> FourierEquispacedNodes(LevelOrder()), d)
    I = SmolyakIndexSet(Val(d), n)
    grid = SparseGrid(SparseGridSpec(axes, I))

    D = dim(grid)
    rmax = refinement_caps(grid)

    # Potential and initial data are defined on x ∈ [0,2π]^d.
    V_of_t(t::SVector{Dt,Float64}) where {Dt} = begin
        s = 0.0
        @inbounds for j in 1:Dt
            xj = 2pi * t[j]
            s += (xj - pi)^2
        end
        0.5 * s
    end
    g_of_t(t::SVector{Dt,Float64}) where {Dt} = begin
        pref = (2 / (pi * ε))^(Dt / 4)
        x1 = 2pi * t[1]
        s = (x1 - (3pi / 2))^2
        @inbounds for j in 2:Dt
            xj = 2pi * t[j]
            s += (xj - pi)^2
        end
        pref * exp(-s / ε)
    end

    V = evaluate(grid, V_of_t, Float64)
    g = evaluate(grid, g_of_t, Float64)

    u = ComplexF64.(g)
    ucoeffs = OrientedCoeffs{D}(u)
    plan = CyclicLayoutPlan(grid, ComplexF64)

    # Diagonal phases for Strang splitting.
    phaseV = cis.(-(τ / (2ε)) .* V)

    # IMPORTANT:
    # Our sparse grid FFT stores modal coefficients in standard FFTW *natural order*.
    # The physical (signed) frequency ω depends on the active 1D resolution N=2^ℓ.
    # On sparse grids, each coefficient belongs to some tensor subspace W_ℓ, so we must
    # interpret each coordinate component with the corresponding local N.

    # totalsize_by_refinement[d][r+1] = totalsize(axes[d], r) for r=0..rmax[d].
    totalsize_by_refinement = ntuple(Val(D)) do j
        R = rmax[j]
        v = Vector{Int}(undef, R + 1)
        @inbounds for r in 0:R
            v[r + 1] = totalsize(axes[j], r)
        end
        v
    end

    ω2 = Vector{Float64}(undef, length(grid))
    it = traverse(grid)
    @inbounds for (i, c) in enumerate(it)
        acc = 0.0
        for j in 1:D
            coord = Int(c[j])
            npts = totalsize_by_refinement[j]
            # Smallest refinement index r such that coord ≤ totalsize(axes[j], r).
            ℓidx = searchsortedfirst(npts, coord)
            N = npts[ℓidx]
            k = coord - 1
            ω = freq_from_nat(k, N)
            acc += ω * ω
        end
        ω2[i] = acc
    end

    phaseK = cis.(-((τ * ε / 2) .* ω2))

    nsteps = max(0, round(Int, Tfinal / τ))

    function report(step::Int, uvec::Vector{ComplexF64})
        n = length(uvec)

        # Work in sparse Fourier coefficients aω (orthonormal basis ei ω·x):
        # ‖u‖²_{L2}  = (2π)^d * sum(|a_ω|^2) and
        # kinetic energy = (ε^2/2) * (2π)^d * sum(|ω|^2 * |a_ω|^2).
        a = copy(uvec)
        ac = OrientedCoeffs{D}(a)
        apply_unidirectional!(ac, grid, ForwardTransform(D), plan)

        a2 = sum(abs2, a)
        kin = (ε^2 / 2) * sum(@. ω2 * abs2(a))

        # Potential energy via collocation values (cheap but not a high-order quadrature).
        pot = sum(@. V * abs2(uvec)) / n

        println("step=$step, t=$(step * τ):  ||a||₂≈$(sqrt(a2)),  E≈$(kin + pot)")
        return nothing
    end

    println("TDSE on sparse grid: d=$D, n=$n, #dofs=$(length(grid)), eps=$ε, tau=$τ, steps=$nsteps")
    report(0, u)

    for m in 1:nsteps
        @. u *= phaseV

        apply_unidirectional!(ucoeffs, grid, ForwardTransform(D), plan)

        @. u *= phaseK

        apply_unidirectional!(ucoeffs, grid, InverseTransform(D), plan)

        @. u *= phaseV

        if renorm
            a = copy(u)
            ac = OrientedCoeffs{D}(a)
            apply_unidirectional!(ac, grid, ForwardTransform(D), plan)
            u ./= sqrt(sum(abs2, a))
        end

        (print_every > 0 && (m % print_every == 0)) && report(m, u)
    end

    return u
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = _parse_kv_args(ARGS)
    d = parse(Int, get(kv, "d", "2"))
    n = parse(Int, get(kv, "n", "10"))
    ε = parse(Float64, get(kv, "eps", "0.1"))
    τ = parse(Float64, get(kv, "tau", "0.01"))
    Tfinal = parse(Float64, get(kv, "T", "1.0"))
    print_every = parse(Int, get(kv, "print_every", "10"))
    renorm = parse(Int, get(kv, "renorm", "0")) != 0
    run_tdse(; d, n, ε, τ, Tfinal, print_every, renorm)
end
