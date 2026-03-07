using Documenter
using UnifiedSparseGrids

DocMeta.setdocmeta!(UnifiedSparseGrids, :DocTestSetup, :(using UnifiedSparseGrids); recursive=true)

const REPO = Documenter.Remotes.GitHub("redblackbst", "UnifiedSparseGrids.jl")

makedocs(
    modules = [UnifiedSparseGrids],
    sitename = "UnifiedSparseGrids.jl",
    repo = REPO,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://redblackbst.github.io/UnifiedSparseGrids.jl/stable/",
        edit_link = "main",
        mathengine = Documenter.MathJax3(Dict(
            :tex => Dict(
                "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                "tags" => "ams",
                "packages" => ["base", "ams", "autoload"],
            ),
        )),
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Layouts and iteration" => joinpath("manual", "layouts.md"),
            "Unidirectional principle" => joinpath("manual", "unidirectional_principle.md"),
            "Evaluation" => joinpath("manual", "evaluation.md"),
            "Galerkin tools" => joinpath("manual", "galerkin.md"),
        ],
        "Tutorials" => [
            "Time-dependent Schrödinger (Gradinaru 2007)" => joinpath("examples", "gradinaru_2007_tdse.md"),
            "Sparse spectral Galerkin elliptic (Shen–Yu 2010)" => joinpath("examples", "shen_yu_2010_sec4.md"),
            "Real Helmholtz in hat basis (Balder–Zenger 1996)" => joinpath("examples", "balder_zenger_1996_helmholtz.md"),
        ],
        "API" => [
            "Public" => joinpath("api", "public.md"),
            "Internals" => joinpath("api", "internals.md"),
        ],
        "Advanced topics" => [
            "Conventions" => joinpath("advance", "conventions.md"),
            "Development" => joinpath("advance", "development.md"),
        ],
    ],
)

deploydocs(
    repo = "github.com/redblackbst/UnifiedSparseGrids.jl.git",
    devbranch = "main",
)
