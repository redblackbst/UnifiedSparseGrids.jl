
@testset "Plots extension smoke" begin
    ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
    try
        import Plots
    catch
        @test_broken false
        return
    end

    grid = SparseGrid(SparseGridSpec(DyadicNodes(), SmolyakIndexSet(2, 2)))
    p = plot_subspace_layout(grid; show_points=false, annotate_offsets=false)
    @test p isa Plots.Plot

    q = plot_combination_technique(2, 3; annotate_weights=false)
    @test q isa Plots.Plot
end
