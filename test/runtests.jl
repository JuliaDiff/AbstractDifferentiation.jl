using AbstractDifferentiation
using Documenter
using Test

@testset verbose = true "AbstractDifferentiation.jl" begin
    doctest(AbstractDifferentiation)
    include("test_utils.jl")
    include("defaults.jl")
    include("forwarddiff.jl")
    include("reversediff.jl")
    include("finitedifferences.jl")
    include("tracker.jl")
    include("ruleconfig.jl")
end
