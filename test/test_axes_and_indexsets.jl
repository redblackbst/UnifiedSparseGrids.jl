
using Test
using StaticArrays
using UnifiedSparseGrids

function _prefix_nested(axis, rmax)
    for r in 1:rmax
        pts_prev = points(axis, r - 1)
        pts = points(axis, r)
        @test pts[1:length(pts_prev)] ≈ pts_prev atol=1e-14 rtol=1e-14
    end
end

@testset "Axis-family size contract" begin
    axes = (
        DyadicNodes(LevelOrder(); endpoints=:both),
        ChebyshevGaussLobattoNodes(LevelOrder(); endpoints=:both),
        FourierEquispacedNodes(LevelOrder()),
    )
    for axis in axes
        @test axis isa AbstractAxisFamily
        running = 0
        for r in 0:4
            running += blocksize(axis, r)
            @test totalsize(axis, r) == running
            @test npoints(axis, r) == totalsize(axis, r)
            @test delta_count(axis, r) == blocksize(axis, r)
            @test refinement_index(axis, totalsize(axis, r)) == r
        end
    end
end

@testset "Endpoint and storage-order contract" begin
    dy_both = DyadicNodes(LevelOrder(); endpoints=:both)
    dy_none = DyadicNodes(LevelOrder(); endpoints=:none)
    cgl_both = ChebyshevGaussLobattoNodes(LevelOrder(); endpoints=:both)
    cgl_none = ChebyshevGaussLobattoNodes(LevelOrder(); endpoints=:none)

    @test totalsize(dy_both, 0) == 2
    @test blocksize(dy_both, 0) == 2
    @test totalsize(dy_none, 0) == 0
    @test blocksize(dy_none, 0) == 0
    @test points(dy_none, 2) == [0.5, 0.25, 0.75]

    @test totalsize(cgl_both, 0) == 2
    @test totalsize(cgl_none, 0) == 0
    @test points(cgl_none, 2) ≈ [0.0, 0.7071067811865476, -0.7071067811865475] atol=1e-14 rtol=0

    _prefix_nested(dy_both, 4)
    _prefix_nested(cgl_both, 4)
end

@testset "Fourier storage-order sanity" begin
    axis = FourierEquispacedNodes(LevelOrder())
    @test totalsize(axis, 0) == 1
    @test totalsize(axis, 1) == 2
    @test totalsize(axis, 2) == 4
    @test totalsize(axis, 3) == 8
    @test blocksize(axis, 0) == 1
    @test blocksize(axis, 1) == 1
    @test blocksize(axis, 2) == 2
    @test blocksize(axis, 3) == 4
    @test points(axis, 3) == [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    _prefix_nested(axis, 5)
end

@testset "Index-set membership and refinement caps" begin
    I = SmolyakIndexSet(Val(3), 4; cap=SVector(2, 5, 3))
    @test refinement_caps(I) == SVector(2, 5, 3)
    @test SVector(0, 0, 0) in I
    @test SVector(1, 2, 1) in I
    @test !(SVector(2, 2, 2) in I)
    @test !(SVector(3, 0, 0) in I)

    J = FullTensorIndexSet(Val(3), 3; cap=SVector(2, 1, 3))
    @test refinement_caps(J) == SVector(2, 1, 3)
    @test SVector(2, 1, 3) in J
    @test !(SVector(2, 2, 3) in J)
    @test !(SVector(0, 0, 4) in J)
end

@testset "Weighted Smolyak anisotropy sanity" begin
    I = WeightedSmolyakIndexSet(Val(3), 8, SVector(2, 1, 1); cap=SVector(4, 3, 2))
    @test refinement_caps(I) == SVector(4, 3, 2)
    @test SVector(0, 0, 0) in I
    @test SVector(1, 2, 2) in I
    @test SVector(2, 2, 1) in I
    @test !(SVector(3, 2, 1) in I)
    @test !(SVector(5, 0, 0) in I)

    J = WeightedSmolyakIndexSet(Val(4), 8, SVector(1, 2, 2, 2); shift=SVector(0, 1, 1, 1))
    @test refinement_caps(J) == SVector(8, 5, 5, 5)
    @test SVector(0, 1, 1, 1) in J
    @test SVector(2, 1, 1, 1) in J
    @test SVector(2, 2, 1, 1) in J
    @test !(SVector(3, 3, 2, 1) in J)

    perm = SVector(2, 3, 4, 1)
    Jp = UnifiedSparseGrids._permute_indexset(J, perm)
    r = SVector(2, 3, 1, 1)
    rp = SVector(r[perm[1]], r[perm[2]], r[perm[3]], r[perm[4]])
    @test contains(J, r) == contains(Jp, rp)
end

