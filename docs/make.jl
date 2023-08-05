using AbstractDifferentiation
using Documenter

DocMeta.setdocmeta!(AbstractDifferentiation, :DocTestSetup, :(using AbstractDifferentiation); recursive=true)

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/JuliaDiff/AbstractDifferentiation.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)

open(joinpath(generated_path, "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules=[AbstractDifferentiation],
    authors="Mohamed Tarek <mohamed82008@gmail.com> and contributors",
    repo="https://github.com/JuliaDiff/AbstractDifferentiation.jl/blob/{commit}{path}#{line}",
    sitename="AbstractDifferentiation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaDiff.github.io/AbstractDifferentiation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaDiff/AbstractDifferentiation.jl",
    devbranch="master",
)
