
using Test
using StaticArrays
using UnifiedSparseGrids

function _bruteforce_expected_length(axes::NTuple{D,AbstractAxisFamily}, I::AbstractIndexSet{D}) where {D}
    caps = refinement_caps(I)
    ranges = ntuple(d -> 0:caps[d], D)
    total = 0
    for t in Iterators.product(ranges...)
        r = SVector{D,Int}(t)
        if r in I
            prod = 1
            @inbounds for d in 1:D
                prod *= blocksize(axes[d], r[d])
            end
            total += prod
        end
    end
    return total
end

@inline _pos_from_refinement_local(axis::AbstractAxisFamily, r::Int, loc::Int) =
    (r == 0 ? 1 : totalsize(axis, r - 1) + 1) + (loc - 1)

function _refinement_local_from_pos(axis::AbstractAxisFamily, pos::Int, rmax::Int)
    start = 1
    @inbounds for r in 0:rmax
        cnt = blocksize(axis, r)
        stop = start + cnt - 1
        if start <= pos <= stop
            return r, pos - start + 1
        end
        start = stop + 1
    end
    throw(ArgumentError("pos=$pos exceeds totalsize(axis, rmax)=$(totalsize(axis, rmax))"))
end

function _decode_coords(axes::NTuple{D,AbstractAxisFamily}, caps::SVector{D,Int}, coords::SVector{D,Int}) where {D}
    rs = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    locs = MVector{D,Int}(ntuple(_ -> 0, Val(D)))
    @inbounds for d in 1:D
        r, loc = _refinement_local_from_pos(axes[d], coords[d], caps[d])
        rs[d] = r
        locs[d] = loc
    end
    return SVector{D,Int}(rs), SVector{D,Int}(locs)
end

function scan_rowmeta(grid::SparseGrid{<:SparseGridSpec{D}}, perm::SVector{D,Int}) where {D}
    fibers = collect(each_lastdim_fiber(grid, perm))
    maxlen = maximum(f.len for f in fibers)
    hist = zeros(Int, maxlen)
    for f in fibers
        hist[f.len] += 1
    end
    counts = zeros(Int, maxlen)
    running = 0
    for k in maxlen:-1:1
        running += hist[k]
        counts[k] = running
    end
    offsets = zeros(Int, maxlen)
    off = 1
    for k in 1:maxlen
        offsets[k] = off
        off += counts[k]
    end
    return offsets, counts, maxlen, off - 1
end

function _check_recursive_entries!(coords, axes::NTuple{D,AbstractAxisFamily}, I::AbstractIndexSet{D}) where {D}
    caps = refinement_caps(I)
    for c in coords
        r, loc = _decode_coords(axes, caps, c)
        @test r in I
        for d in 1:D
            @test 1 <= loc[d] <= blocksize(axes[d], r[d])
        end
    end
end

function _check_subspace_bijection(axes::NTuple{D,AbstractAxisFamily}, I::AbstractIndexSet{D}) where {D}
    grid = SparseGrid(SparseGridSpec(axes, I))
    it_sub = traverse(grid; layout=SubspaceLayout())
    coords = collect(traverse(grid; layout=RecursiveLayout()))
    idxs = [UnifiedSparseGrids._subspace_linear_index(it_sub, c) for c in coords]
    @test sort(idxs) == collect(1:length(grid))
    @test length(idxs) == length(grid)
    @test recursive_to_subspace(grid, subspace_to_recursive(grid, randn(length(grid)))) isa Vector
end

@testset "Recursive traversal invariants" begin
    axis = DyadicNodes(LevelOrder())
    I = SmolyakIndexSet(2, 2)
    grid = SparseGrid(SparseGridSpec(axis, I))
    got = collect(traverse(grid; layout=RecursiveLayout()))

    expected_pairs = Tuple{SVector{2,Int},SVector{2,Int}}[
        (SVector(0, 0), SVector(1, 1)),
        (SVector(0, 0), SVector(1, 2)),
        (SVector(0, 1), SVector(1, 1)),
        (SVector(0, 2), SVector(1, 1)),
        (SVector(0, 2), SVector(1, 2)),
        (SVector(0, 0), SVector(2, 1)),
        (SVector(0, 0), SVector(2, 2)),
        (SVector(0, 1), SVector(2, 1)),
        (SVector(0, 2), SVector(2, 1)),
        (SVector(0, 2), SVector(2, 2)),
        (SVector(1, 0), SVector(1, 1)),
        (SVector(1, 0), SVector(1, 2)),
        (SVector(1, 1), SVector(1, 1)),
        (SVector(2, 0), SVector(1, 1)),
        (SVector(2, 0), SVector(1, 2)),
        (SVector(2, 0), SVector(2, 1)),
        (SVector(2, 0), SVector(2, 2)),
    ]
    expected = SVector{2,Int}[
        SVector(_pos_from_refinement_local(axis, p[1][1], p[2][1]), _pos_from_refinement_local(axis, p[1][2], p[2][2]))
        for p in expected_pairs
    ]
    @test got == expected

    cases = (
        ((DyadicNodes(LevelOrder()), DyadicNodes(LevelOrder())), SmolyakIndexSet(2, 3; cap=SVector(3, 1))),
        ((DyadicNodes(LevelOrder()), DyadicNodes(LevelOrder()), DyadicNodes(LevelOrder())), SmolyakIndexSet(3, 3; cap=SVector(3, 1, 2))),
        ((DyadicNodes(LevelOrder()), ChebyshevGaussLobattoNodes(LevelOrder()), FourierEquispacedNodes(LevelOrder())), SmolyakIndexSet(3, 3; cap=SVector(2, 3, 1))),
        ((DyadicNodes(LevelOrder()), DyadicNodes(LevelOrder()), DyadicNodes(LevelOrder())), WeightedSmolyakIndexSet(3, 8, SVector(2, 1, 1); cap=SVector(3, 2, 1))),
    )
    for (axes, J) in cases
        gridJ = SparseGrid(SparseGridSpec(axes, J))
        coords = collect(traverse(gridJ; layout=RecursiveLayout()))
        _check_recursive_entries!(coords, axes, J)
        @test length(coords) == _bruteforce_expected_length(axes, J)
    end
end

@testset "Subspace layout bijection and roundtrip" begin
    _check_subspace_bijection((DyadicNodes(), DyadicNodes()), SmolyakIndexSet(2, 2))
    _check_subspace_bijection((DyadicNodes(), DyadicNodes()), WeightedSmolyakIndexSet(2, 8, SVector(2, 1); cap=SVector(3, 3)))
    _check_subspace_bijection((DyadicNodes(), DyadicNodes()), FullTensorIndexSet(2, 3))
    _check_subspace_bijection((DyadicNodes(), ChebyshevGaussLobattoNodes(), FourierEquispacedNodes()), SmolyakIndexSet(3, 3; cap=SVector(2, 3, 1)))
end

@testset "Cyclic rowmeta vs scan" begin
    cases = (
        ((ChebyshevGaussLobattoNodes(LevelOrder()), ChebyshevGaussLobattoNodes(LevelOrder()), ChebyshevGaussLobattoNodes(LevelOrder())),
         SmolyakIndexSet(Val(3), 4; cap=SVector(2, 5, 3))),
        ((ChebyshevGaussLobattoNodes(LevelOrder()), DyadicNodes(LevelOrder()), FourierEquispacedNodes(LevelOrder()), ChebyshevGaussLobattoNodes(LevelOrder())),
         SmolyakIndexSet(Val(4), 4; cap=SVector(3, 4, 2, 5))),
    )
    for (axes, I) in cases
        grid = SparseGrid(SparseGridSpec(axes, I))
        plan = CyclicLayoutPlan(grid, Float64)
        for layout in plan.layouts
            off_ref, cnt_ref, maxlen_ref, n_ref = scan_rowmeta(grid, layout.perm)
            @test Int(layout.maxlen) == maxlen_ref
            @test n_ref == length(grid)
            @test layout.first_offsets == off_ref
            @test layout.first_counts == cnt_ref
        end
    end
end

@testset "Plan sharing across dimensions" begin
    axes = (
        ChebyshevGaussLobattoNodes(LevelOrder()),
        ChebyshevGaussLobattoNodes(LevelOrder()),
        FourierEquispacedNodes(LevelOrder()),
    )
    I = SmolyakIndexSet(Val(3), 4; cap=SVector(2, 4, 3))
    grid = SparseGrid(SparseGridSpec(axes, I))
    plan = CyclicLayoutPlan(grid, Float64)
    opf = LineTransform(Val(:forward))

    plans_cgl_1 = UnifiedSparseGrids._get_lineplanvec!(plan.op_plan, opf, axes[1], 4, Float64)
    plans_cgl_2 = UnifiedSparseGrids._get_lineplanvec!(plan.op_plan, opf, axes[2], 4, Float64)
    @test plans_cgl_1 === plans_cgl_2

    plans_fft = UnifiedSparseGrids._get_lineplanvec!(plan.op_plan, opf, axes[3], 3, Float64)
    @test length(plans_fft) == 4
end

@testset "Heterogeneous-axis sparse grid layout sanity" begin
    axes = (
        ChebyshevGaussLobattoNodes(LevelOrder()),
        DyadicNodes(LevelOrder()),
        FourierEquispacedNodes(LevelOrder()),
    )
    I = SmolyakIndexSet(Val(3), 4; cap=SVector(3, 4, 2))
    grid = SparseGrid(SparseGridSpec(axes, I))

    perms = (SVector(1, 2, 3), SVector(3, 1, 2), SVector(2, 3, 1))
    for perm in perms
        fibers = collect(each_lastdim_fiber(grid, perm))
        @test sum(f.len for f in fibers) == length(grid)
        off = 1
        for fib in fibers
            @test fib.src_offset == off
            off += fib.len
        end
        @test off == length(grid) + 1
    end

    bad_grid = SparseGrid(SparseGridSpec((DyadicNodes(), GaussNodes()), SmolyakIndexSet(2, 1)))
    @test_throws ArgumentError traverse(bad_grid; layout=SubspaceLayout())
end
