using AbstractDifferentiation
using Test

@testset "AbstractDifferentiation.jl" begin
    include("test_utils.jl")
    include("defaults.jl")
    include("forwarddiff.jl")
    include("reversediff.jl")
    include("finitedifferences.jl")
    include("tracker.jl")
    @static if VERSION >= v"1.6"
        include("ruleconfig.jl")
    end
end
