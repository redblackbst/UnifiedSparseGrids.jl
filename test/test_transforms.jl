
using Test
using Random
using StaticArrays
using LinearAlgebra
using UnifiedSparseGrids

function eval_chebT_series(a::AbstractVector, x::Real)
    N = length(a) - 1
    N == 0 && return a[1]
    Tkm1 = one(x)
    Tk = x
    s = a[1] * Tkm1 + a[2] * Tk
    @inbounds for k in 1:(N - 1)
        Tkp1 = 2x * Tk - Tkm1
        s += a[k + 2] * Tkp1
        Tkm1, Tk = Tk, Tkp1
    end
    return s
end

function eval_legendre_series(b::AbstractVector, x::Real)
    N = length(b) - 1
    N == 0 && return b[1]
    Pnm1 = one(x)
    Pn = x
    s = b[1] * Pnm1 + b[2] * Pn
    @inbounds for n in 1:(N - 1)
        Pnp1 = ((2n + 1) * x * Pn - n * Pnm1) / (n + 1)
        s += b[n + 2] * Pnp1
        Pnm1, Pn = Pn, Pnp1
    end
    return s
end

function _anisotropic_cap(::Val{D}, L::Int) where {D}
    return SVector{D,Int}(ntuple(d -> max(L - (d - 1), 0), Val(D)))
end

function _fft_forward_matrix_from_size(n::Int, ::Type{T}=ComplexF64) where {T<:Complex}
    Tr = real(T)
    F = Matrix{T}(undef, n, n)
    @inbounds for k in 0:(n - 1)
        for j in 0:(n - 1)
            F[k + 1, j + 1] = T(cispi(-2 * Tr(j * k) / Tr(n))) / n
        end
    end
    perm = bitrevperm(n)
    return F[:, perm .+ 1]
end

@testset "Fourier line-transform roundtrip" begin
    rng = MersenneTwister(0)
    r = 4
    axis = FourierEquispacedNodes(LevelOrder())
    n = totalsize(axis, r)
    plan = UnifiedSparseGrids.lineplan(LineTransform(Val(:forward)), axis, r, ComplexF64)[r + 1]
    x0 = randn(rng, ComplexF64, n)
    x = copy(x0)
    work = similar(x)
    apply_line!(LineTransform(Val(:forward)), x, work, axis, r, plan)
    apply_line!(LineTransform(Val(:inverse)), x, work, axis, r, plan)
    @test isapprox(x, x0; rtol=1e-12, atol=1e-12)
end

@testset "Chebyshev-Gauss-Lobatto line-transform roundtrip" begin
    rng = MersenneTwister(0)
    r = 4
    axis = ChebyshevGaussLobattoNodes(LevelOrder())
    n = totalsize(axis, r)
    plan = UnifiedSparseGrids.lineplan(LineTransform(Val(:forward)), axis, r, Float64)[r + 1]
    f0 = randn(rng, Float64, n)
    f = copy(f0)
    work = similar(f)
    apply_line!(LineTransform(Val(:forward)), f, work, axis, r, plan)
    apply_line!(LineTransform(Val(:inverse)), f, work, axis, r, plan)
    @test isapprox(f, f0; rtol=1e-12, atol=1e-12)
end

@testset "Chebyshev-Legendre coefficient contract" begin
    rng = MersenneTwister(0)
    for r in 0:4
        n = totalsize(ChebyshevGaussLobattoNodes(LevelOrder()), r)
        P = UnifiedSparseGrids.ChebyshevLegendrePlan(n, Float64)
        a = randn(rng, n)
        b = copy(a)
        lmul!(P, b)
        a2 = copy(b)
        ldiv!(P, a2)
        @test isapprox(a2, a; rtol=1e-12, atol=1e-12)
        xs = rand(rng, 10) .* 2 .- 1
        for x in xs
            @test isapprox(eval_chebT_series(a, x), eval_legendre_series(b, x); rtol=1e-11, atol=1e-11)
        end
    end
end

@testset "Transform plan sharing and sparse grid roundtrip" begin
    axes = (
        ChebyshevGaussLobattoNodes(LevelOrder()),
        ChebyshevGaussLobattoNodes(LevelOrder()),
        FourierEquispacedNodes(LevelOrder()),
    )
    I = SmolyakIndexSet(Val(3), 4; cap=SVector(2, 4, 3))
    grid = SparseGrid(SparseGridSpec(axes, I))
    plan = CyclicLayoutPlan(grid, Float64)

    pf = UnifiedSparseGrids._get_lineplanvec!(plan.meta.lineplans, LineTransform(Val(:forward)), axes[1], 4, Float64)
    pi = UnifiedSparseGrids._get_lineplanvec!(plan.meta.lineplans, LineTransform(Val(:inverse)), axes[2], 4, Float64)
    @test pf === pi

    pcf = UnifiedSparseGrids._get_lineplanvec!(plan.meta.lineplans, LineChebyshevLegendre(Val(:forward)), axes[1], 4, Float64)
    pci = UnifiedSparseGrids._get_lineplanvec!(plan.meta.lineplans, LineChebyshevLegendre(Val(:inverse)), axes[2], 4, Float64)
    @test pcf === pci

    rng = MersenneTwister(1)
    for (nodes_ctor, T) in ((() -> ChebyshevGaussLobattoNodes(LevelOrder()), Float64), (() -> FourierEquispacedNodes(LevelOrder()), ComplexF64))
        D = 2
        L = 3
        cap = _anisotropic_cap(Val(D), L)
        nodes = ntuple(_ -> nodes_ctor(), Val(D))
        gridT = SparseGrid(SparseGridSpec(nodes, SmolyakIndexSet(Val(D), L; cap=cap)))
        planT = CyclicLayoutPlan(gridT, T)
        x0 = randn(rng, T, length(gridT))
        u = OrientedCoeffs{D}(copy(x0))
        apply_unidirectional!(u, gridT, ForwardTransform(Val(D)), planT)
        apply_unidirectional!(u, gridT, InverseTransform(Val(D)), planT)
        @test isapprox(u.data, x0; rtol=1e-10, atol=1e-10)
    end
end

@testset "Shared-plan concurrency smoke" begin
    using Base.Threads

    r = 6
    axisC = ChebyshevGaussLobattoNodes(LevelOrder())
    planC = UnifiedSparseGrids.lineplan(LineTransform(Val(:forward)), axisC, r, Float64)[r + 1]
    nC = totalsize(axisC, r)

    axisF = FourierEquispacedNodes(LevelOrder())
    planF = UnifiedSparseGrids.lineplan(LineTransform(Val(:forward)), axisF, r, ComplexF64)[r + 1]
    nF = totalsize(axisF, r)

    ntasks = max(2, Threads.nthreads())
    tasks = map(1:ntasks) do tid
        Threads.@spawn begin
            rng = MersenneTwister(tid)
            xC0 = randn(rng, Float64, nC)
            xC = copy(xC0)
            workC = similar(xC)
            apply_line!(LineTransform(Val(:forward)), xC, workC, axisC, r, planC)
            apply_line!(LineTransform(Val(:inverse)), xC, workC, axisC, r, planC)

            xF0 = randn(rng, ComplexF64, nF)
            xF = copy(xF0)
            workF = similar(xF)
            apply_line!(LineTransform(Val(:forward)), xF, workF, axisF, r, planF)
            apply_line!(LineTransform(Val(:inverse)), xF, workF, axisF, r, planF)

            return isapprox(xC, xC0; rtol=1e-12, atol=1e-12) && isapprox(xF, xF0; rtol=1e-12, atol=1e-12)
        end
    end
    @test all(fetch.(tasks))
end

@testset "Small dense/operator parity case" begin
    rng = MersenneTwister(2)
    D = 2
    r = 3
    nodes = ntuple(_ -> FourierEquispacedNodes(LevelOrder()), Val(D))
    grid = SparseGrid(SparseGridSpec(nodes, FullTensorIndexSet(Val(D), r)))
    x = randn(rng, ComplexF64, length(grid))
    plan = CyclicLayoutPlan(grid, ComplexF64)

    u_fast = OrientedCoeffs{D}(copy(x))
    apply_unidirectional!(u_fast, grid, ForwardTransform(Val(D)), plan)

    mats = [ _fft_forward_matrix_from_size(totalsize(nodes[1], rr), ComplexF64) for rr in 0:r ]
    op_ref = UpDownTensorOp(ntuple(_ -> LineMatrixOp(mats), Val(D)); omit_dim=1)
    u_ref = OrientedCoeffs{D}(copy(x))
    apply_unidirectional!(u_ref, grid, op_ref, plan)

    @test isapprox(u_fast.data, u_ref.data; rtol=1e-12, atol=1e-12)
end
