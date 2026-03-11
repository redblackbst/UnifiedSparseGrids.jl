
using Test
using Random
using LinearAlgebra
using StaticArrays
using UnifiedSparseGrids

function restricted_kron_matvec(coords::Vector{SVector{D,Int}}, A::NTuple{D,AbstractMatrix}, x::AbstractVector) where {D}
    n = length(coords)
    y = similar(x, n)
    fill!(y, zero(eltype(y)))
    @inbounds for i in 1:n
        ci = coords[i]
        acc = zero(eltype(y))
        for j in 1:n
            cj = coords[j]
            w = one(eltype(y))
            for d in 1:D
                w *= A[d][ci[d], cj[d]]
            end
            acc += w * x[j]
        end
        y[i] = acc
    end
    return y
end

function restricted_kron_matrix(coords::Vector{SVector{D,Int}}, A::NTuple{D,AbstractMatrix}) where {D}
    n = length(coords)
    M = Matrix{eltype(A[1])}(undef, n, n)
    @inbounds for i in 1:n
        ci = coords[i]
        for j in 1:n
            cj = coords[j]
            w = one(eltype(A[1]))
            for d in 1:D
                w *= A[d][ci[d], cj[d]]
            end
            M[i, j] = w
        end
    end
    return M
end

function nested_dense_mats(rng::AbstractRNG, axis::AbstractAxisFamily, rmax::Int, ::Type{T}) where {T}
    mats = Vector{Matrix{T}}(undef, rmax + 1)
    n0 = totalsize(axis, 0)
    mats[1] = randn(rng, T, n0, n0)
    for r in 1:rmax
        n = totalsize(axis, r)
        M = randn(rng, T, n, n)
        nprev = size(mats[r], 1)
        M[1:nprev, 1:nprev] .= mats[r]
        mats[r + 1] = M
    end
    return mats
end

diag2_cgl(n::Integer, ::Type{T}) where {T<:Number} = fill(T(2), n)
diag_ones3(n::Integer, ::Type{T}) where {T<:Number} = ones(T, n)

@inline function _make_triangular!(M::AbstractMatrix{T}, kind::Symbol) where {T}
    n, m = size(M)
    n == m || throw(DimensionMismatch("expected square matrix"))
    if kind === :lower
        @inbounds for j in 1:n, i in 1:(j - 1)
            M[i, j] = zero(T)
        end
    else
        @inbounds for j in 1:n, i in (j + 1):n
            M[i, j] = zero(T)
        end
    end
    return M
end

function nested_triangular_mats(rng::AbstractRNG, axis::AbstractAxisFamily, rmax::Int, ::Type{T}; kind::Symbol) where {T}
    mats = Vector{Matrix{T}}(undef, rmax + 1)
    n0 = totalsize(axis, 0)
    M0 = randn(rng, T, n0, n0)
    _make_triangular!(M0, kind)
    mats[1] = M0
    for r in 1:rmax
        n = totalsize(axis, r)
        M = randn(rng, T, n, n)
        _make_triangular!(M, kind)
        nprev = size(mats[r], 1)
        M[1:nprev, 1:nprev] .= mats[r]
        mats[r + 1] = M
    end
    return mats
end

function nested_invertible_triangular_mats(rng::AbstractRNG, axis::AbstractAxisFamily, rmax::Int, ::Type{T}; kind::Symbol, offdiag_scale::Real=0.1) where {T}
    mats = Vector{Matrix{T}}(undef, rmax + 1)
    n0 = totalsize(axis, 0)
    M0 = offdiag_scale .* randn(rng, T, n0, n0)
    _make_triangular!(M0, kind)
    @inbounds for i in 1:n0
        M0[i, i] = one(T) + offdiag_scale * randn(rng, T)
    end
    mats[1] = M0
    for r in 1:rmax
        n = totalsize(axis, r)
        M = offdiag_scale .* randn(rng, T, n, n)
        _make_triangular!(M, kind)
        nprev = size(mats[r], 1)
        M[1:nprev, 1:nprev] .= mats[r]
        if nprev < n
            @inbounds for i in (nprev + 1):n
                M[i, i] = one(T) + offdiag_scale * randn(rng, T)
            end
        end
        mats[r + 1] = M
    end
    return mats
end

function _fourier_dft_mat(N::Int)
    ω = exp(-2π * im / N)
    [ω^(k * n) for k in 0:(N - 1), n in 0:(N - 1)]
end

function _fourier_idft_mat(N::Int)
    ω = exp(2π * im / N)
    [ω^(k * n) / N for n in 0:(N - 1), k in 0:(N - 1)]
end

@testset "Restricted-Kronecker matvec parity" begin
    rng = MersenneTwister(0)
    for (axis1d, T, D, L) in ((DyadicNodes(NaturalOrder()), Float64, 3, 3), (FourierEquispacedNodes(NaturalOrder()), ComplexF64, 2, 3))
        axes = ntuple(_ -> axis1d, D)
        IS = SmolyakIndexSet(Val(D), L; cap=SVector{D,Int}(ntuple(_ -> L, D)))
        grid = SparseGrid(SparseGridSpec(axes, IS))
        rmax = refinement_caps(IS)
        mats = ntuple(d -> nested_dense_mats(rng, axes[d], rmax[d], T), Val(D))
        op = UpDownTensorOp(ntuple(d -> LineMatrixOp(mats[d]), Val(D)))
        plan = CyclicLayoutPlan(grid, T)
        x = randn(rng, T, length(grid))
        u = OrientedCoeffs{D}(copy(x))
        apply_unidirectional!(u, grid, op, plan)
        coords = collect(SVector{D,Int}, traverse(grid))
        Amax = ntuple(d -> mats[d][rmax[d] + 1], Val(D))
        @test isapprox(u.data, restricted_kron_matvec(coords, Amax, x); rtol=1e-10, atol=1e-10)
    end
end

@testset "Triangular restriction commutation" begin
    rng = MersenneTwister(1)
    for D in (2, 3)
        L = 3
        axes = ntuple(_ -> DyadicNodes(NaturalOrder()), D)
        IS = SmolyakIndexSet(Val(D), L; cap=SVector{D,Int}(ntuple(_ -> L, D)))
        grid = SparseGrid(SparseGridSpec(axes, IS))
        rmax = refinement_caps(IS)
        coords = collect(SVector{D,Int}, traverse(grid))
        plan = CyclicLayoutPlan(grid, Float64)
        x = randn(rng, Float64, length(grid))

        matsA = ntuple(d -> nested_triangular_mats(rng, axes[d], rmax[d], Float64; kind=:lower), Val(D))
        matsB = ntuple(d -> nested_dense_mats(rng, axes[d], rmax[d], Float64), Val(D))
        opA = TensorOp(ntuple(d -> LineMatrixOp(matsA[d]), Val(D)))
        opB = UpDownTensorOp(ntuple(d -> LineMatrixOp(matsB[d]), Val(D)); omit_dim=1)
        u = OrientedCoeffs{D}(copy(x))
        apply_unidirectional!(u, grid, opB, plan)
        apply_unidirectional!(u, grid, opA, plan)

        Amax = ntuple(d -> matsA[d][rmax[d] + 1], Val(D))
        Bmax = ntuple(d -> matsB[d][rmax[d] + 1], Val(D))
        ABmax = ntuple(d -> Amax[d] * Bmax[d], Val(D))
        @test isapprox(u.data, restricted_kron_matvec(coords, ABmax, x); rtol=1e-10, atol=1e-10)
    end
end

@testset "Triangular inversion commutation" begin
    rng = MersenneTwister(2)
    for kind in (:lower, :upper)
        D = 2
        L = 3
        axes = ntuple(_ -> DyadicNodes(NaturalOrder()), D)
        I = SmolyakIndexSet(Val(D), L; cap=SVector{D,Int}(ntuple(_ -> L, D)))
        grid = SparseGrid(SparseGridSpec(axes, I))
        rmax = refinement_caps(I)
        coords = collect(SVector{D,Int}, traverse(grid))
        matsA = ntuple(d -> nested_invertible_triangular_mats(rng, axes[d], rmax[d], Float64; kind=kind), Val(D))
        matsAinv = ntuple(d -> begin
            invs = Vector{Matrix{Float64}}(undef, rmax[d] + 1)
            Amax = matsA[d][rmax[d] + 1]
            Amax_inv = Matrix(inv(kind === :lower ? LowerTriangular(Amax) : UpperTriangular(Amax)))
            for rr in 0:rmax[d]
                n = totalsize(axes[d], rr)
                invs[rr + 1] = Matrix(@view Amax_inv[1:n, 1:n])
            end
            invs
        end, Val(D))
        opA = TensorOp(ntuple(d -> LineMatrixOp(matsA[d]), Val(D)))
        opAinv = TensorOp(ntuple(d -> LineMatrixOp(matsAinv[d]), Val(D)))
        plan = CyclicLayoutPlan(grid, Float64)
        x = randn(rng, Float64, length(grid))
        u = OrientedCoeffs{D}(copy(x))
        apply_unidirectional!(u, grid, opA, plan)
        apply_unidirectional!(u, grid, opAinv, plan)
        @test isapprox(u.data, x; rtol=1e-10, atol=1e-10)

        Amax = ntuple(d -> matsA[d][rmax[d] + 1], Val(D))
        Ainvmax = ntuple(d -> matsAinv[d][rmax[d] + 1], Val(D))
        Ahat = restricted_kron_matrix(coords, Amax)
        Ainvhat = restricted_kron_matrix(coords, Ainvmax)
        Idref = Matrix{Float64}(LinearAlgebra.I, length(grid), length(grid))
        @test isapprox(Ainvhat * Ahat, Idref; rtol=1e-10, atol=1e-10)
    end
end

@testset "UpDown split and omit bookkeeping" begin
    rng = MersenneTwister(3)
    D = 3
    L = 4
    axes = ntuple(_ -> DyadicNodes(NaturalOrder()), D)
    I = SmolyakIndexSet(Val(D), L; cap=SVector{D,Int}(ntuple(_ -> L, D)))
    grid = SparseGrid(SparseGridSpec(axes, I))
    rmax = refinement_caps(I)
    mats = ntuple(d -> nested_dense_mats(rng, axes[d], rmax[d], Float64), Val(D))
    ops = ntuple(d -> LineMatrixOp(mats[d]), Val(D))
    op = UpDownTensorOp(ops; omit_dim=0)
    plan = CyclicLayoutPlan(grid, Float64)
    x = randn(rng, Float64, length(grid))
    u_auto = OrientedCoeffs{D}(copy(x))
    apply_unidirectional!(u_auto, grid, op, plan)

    omit_ref = begin
        nterms_full = 1 << D
        if Threads.nthreads() >= (nterms_full >>> 1)
            0
        else
            best = 1
            bestlen = 0
            for k in 1:D
                lenk = totalsize(axes[k], rmax[k])
                if lenk > bestlen
                    best, bestlen = k, lenk
                end
            end
            best
        end
    end
    u_ref = OrientedCoeffs{D}(copy(x))
    apply_unidirectional!(u_ref, grid, UpDownTensorOp(ops; omit_dim=omit_ref), plan)
    @test isapprox(u_auto.data, u_ref.data; rtol=1e-12, atol=1e-12)

    lop1 = LineMatrixOp([LowerTriangular([1.0 0.0; 2.0 3.0])])
    lop2 = LineMatrixOp([[1.0 2.0; 3.0 4.0]])
    lop3 = LineDiagonalOp(diag_ones3)
    ud = UpDownTensorOp((lop1, lop2, lop3))
    @test ud.split_dims == [2]
end

@testset "Scheduling parity" begin
    D = 3
    q = 3
    axes = ntuple(_ -> ChebyshevGaussLobattoNodes(LevelOrder()), D)
    grid = SparseGrid(SparseGridSpec(axes, SmolyakIndexSet(Val(D), q; cap=SVector(ntuple(_ -> q, Val(D))))))
    plan = CyclicLayoutPlan(grid, Float64)
    op = TensorOp((IdentityLineOp(), IdentityLineOp(), LineDiagonalOp(diag2_cgl)))
    x = randn(length(grid))
    u_fast = OrientedCoeffs{D}(copy(x))
    apply_unidirectional!(u_fast, grid, op, plan)

    function naive_apply!(u::OrientedCoeffs{D,ElT}, grid, op::TensorOp{D}, plan) where {D,ElT}
        buf = Vector{ElT}(undef, length(u.data))
        dest = OrientedCoeffs(buf, u.perm)
        src = u
        for _ in 1:D
            dest = apply_lastdim_cycled!(dest, src, grid, lineop(op, src.perm[end]), plan)
            src, dest = dest, src
        end
        if src.data !== u.data
            copyto!(u.data, src.data)
        end
        return u
    end

    u_naive = OrientedCoeffs{D}(copy(x))
    naive_apply!(u_naive, grid, op, plan)
    @test u_fast.perm == u_naive.perm
    @test isapprox(u_fast.data, u_naive.data; rtol=1e-12, atol=1e-12)
end

@testset "Full vs genuine sparse grid transform/operator sanity" begin
    rng = MersenneTwister(4)
    D = 2
    L = 3
    axes = ntuple(_ -> FourierEquispacedNodes(NaturalOrder()), D)
    matsF = ntuple(d -> [ _fourier_dft_mat(totalsize(axes[d], r)) for r in 0:L ], Val(D))
    matsFinv = ntuple(d -> [ _fourier_idft_mat(totalsize(axes[d], r)) for r in 0:L ], Val(D))
    opF = UpDownTensorOp(ntuple(d -> LineMatrixOp(matsF[d]), Val(D)))
    opFinv = UpDownTensorOp(ntuple(d -> LineMatrixOp(matsFinv[d]), Val(D)))

    grid_full = SparseGrid(SparseGridSpec(axes, SmolyakIndexSet(Val(D), D * L; cap=SVector{D,Int}(ntuple(_ -> L, D)))) )
    plan_full = CyclicLayoutPlan(grid_full, ComplexF64)
    c = randn(rng, ComplexF64, length(grid_full))
    v = OrientedCoeffs{D}(copy(c))
    apply_unidirectional!(v, grid_full, opFinv, plan_full)
    apply_unidirectional!(v, grid_full, opF, plan_full)
    @test isapprox(v.data, c; rtol=1e-10, atol=1e-10)

    grid_sparse = SparseGrid(SparseGridSpec(axes, SmolyakIndexSet(Val(D), L; cap=SVector{D,Int}(ntuple(_ -> L, D)))))
    plan_sparse = CyclicLayoutPlan(grid_sparse, ComplexF64)
    c2 = randn(rng, ComplexF64, length(grid_sparse))
    v2 = OrientedCoeffs{D}(copy(c2))
    apply_unidirectional!(v2, grid_sparse, opFinv, plan_sparse)
    apply_unidirectional!(v2, grid_sparse, opF, plan_sparse)
    @test !isapprox(v2.data, c2; rtol=1e-10, atol=1e-10)
end
