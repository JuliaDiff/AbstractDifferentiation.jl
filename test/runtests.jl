using AbstractDifferentiation
using Test

@testset "AbstractDifferentiation.jl" begin
    include("test_utils.jl")
    include("defaults.jl")
    include("forwarddiff.jl")
    include("finitedifferences.jl")
end
