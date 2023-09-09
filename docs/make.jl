using BoxOptNewton
using Documenter

DocMeta.setdocmeta!(BoxOptNewton, :DocTestSetup, :(using BoxOptNewton); recursive=true)

makedocs(;
    modules=[BoxOptNewton],
    authors="chriselrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/BoxOptNewton.jl/blob/{commit}{path}#{line}",
    sitename="BoxOptNewton.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
