
using Test
using LinearAlgebra
using UnifiedSparseGrids

function _line_matrix(op, axis, r::Int, ::Type{T}) where {T}
    plan = UnifiedSparseGrids.lineplan(op, axis, r, T)[r + 1]
    n = totalsize(axis, r)
    M = Matrix{T}(undef, n, n)
    for j in 1:n
        v = zeros(T, n)
        v[j] = one(T)
        work = similar(v)
        apply_line!(op, v, work, axis, r, plan)
        M[:, j] .= v
    end
    return M
end

@testset "Hierarchy roundtrip" begin
    for axis in (ChebyshevGaussLobattoNodes(LevelOrder()), FourierEquispacedNodes(LevelOrder()))
        r = 3
        x = randn(totalsize(axis, r))
        H = UnifiedSparseGrids.lineplan(LineHierarchize(), axis, r, Float64)[r + 1]
        y = copy(x)
        work = similar(y)
        apply_line!(LineHierarchize(), y, work, axis, r, H)
        apply_line!(LineDehierarchize(), y, work, axis, r, H)
        @test isapprox(y, x; atol=1e-12, rtol=0)
    end
end

@testset "Transpose hierarchy contract" begin
    for axis in (ChebyshevGaussLobattoNodes(LevelOrder()), FourierEquispacedNodes(LevelOrder()), DyadicNodes(LevelOrder()))
        r = axis isa FourierEquispacedNodes ? 3 : 2
        H  = _line_matrix(LineHierarchize(), axis, r, Float64)
        D  = _line_matrix(LineDehierarchize(), axis, r, Float64)
        HT = _line_matrix(LineHierarchizeTranspose(), axis, r, Float64)
        DT = _line_matrix(LineDehierarchizeTranspose(), axis, r, Float64)
        Iref = Matrix{Float64}(I, size(H, 1), size(H, 2))
        @test isapprox(H * D, Iref; atol=1e-12, rtol=1e-12)
        @test isapprox(D * H, Iref; atol=1e-12, rtol=1e-12)
        @test isapprox(HT, transpose(H); atol=1e-12, rtol=1e-12)
        @test isapprox(DT, transpose(D); atol=1e-12, rtol=1e-12)
    end
end

@testset "Local-support basis semantics" begin
    @test support_style(ChebyshevTBasis()) isa UnifiedSparseGrids.GlobalSupport
    @test support_style(HatBasis()) isa UnifiedSparseGrids.LocalSupport
    @test support_style(PiecewisePolynomialBasis()) isa UnifiedSparseGrids.LocalSupport

    b = HatBasis()
    nodes_nat = DyadicNodes(NaturalOrder())
    nodes_lvl = DyadicNodes(LevelOrder())
    @test active_local(b, nodes_nat, 1, 0.25) == (0,)
    @test eval_local(b, nodes_nat, 1, 0, 0.5) ≈ 1.0
    @test eval_local(b, nodes_nat, 1, 0, 0.25) ≈ 0.5
    @test active_local(b, nodes_nat, 1, -0.1) == ()
    @test active_local(b, nodes_lvl, 3, 0.3) == (1,)
    @test eval_local(b, nodes_nat, 3, 1, 0.3) ≈ eval_local(b, nodes_lvl, 3, 1, 0.3)

    bp = PiecewisePolynomialBasis(pmax=4)
    @test active_local(bp, nodes_nat, 2, 0.25) == (0,)
    @test eval_local(bp, nodes_nat, 2, 0, 0.25) ≈ 1.0
end

@testset "Endpoint-sensitive boundary functions" begin
    b = HatBasis()
    nodes_with = DyadicNodes(NaturalOrder(); endpoints=:both)
    nodes_none = DyadicNodes(NaturalOrder(); endpoints=:none)

    @test active_local(b, nodes_with, 0, 0.25) == (0, 1)
    @test eval_local(b, nodes_with, 0, 0, 0.25) ≈ 0.75
    @test eval_local(b, nodes_with, 0, 1, 0.25) ≈ 0.25
    @test active_local(b, nodes_none, 0, 0.25) == ()
    @test_throws ArgumentError eval_local(b, nodes_none, 0, 0, 0.25)

    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    V = vandermonde(HatBasis(), x, 3)
    @test size(V) == (length(x), 3)
    @test V[3, 1] ≈ 1.0
    @test V[2, 1] ≈ 0.5
    @test V[2, 2] ≈ 1.0
end

@testset "Dirichlet basis kernel sanity" begin
    c = randn(18)
    out = zeros(Float64, 16)
    legendre_dirichlet_rhs!(out, c)
    M = zeros(Float64, 16, 18)
    for k in 0:15
        M[k + 1, k + 1] += 2 / (2k + 1)
        M[k + 1, k + 3] -= 2 / (2k + 5)
    end
    @test isapprox(out, M * c; rtol=1e-14, atol=1e-14)

    u = randn(12)
    cdir = zeros(Float64, 14)
    dirichlet_to_legendre!(cdir, u)
    u2 = zeros(Float64, 12)
    legendre_to_dirichlet!(u2, cdir)
    @test isapprox(u2, u; rtol=1e-14, atol=1e-14)
end
