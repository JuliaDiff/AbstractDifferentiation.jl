using AbstractDifferentiation
using Test

@testset "AbstractDifferentiation.jl" begin
    include("test_utils.jl")
    include("defaults.jl")
    include("forwarddiff.jl")
    include("reversediff.jl")
    include("finitedifferences.jl")
    include("tracker.jl")
    include("ruleconfig.jl")
    include("enzyme.jl")
end
