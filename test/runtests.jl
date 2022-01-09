using AbstractDifferentiation
using Test

@testset "AbstractDifferentiation.jl" begin
    include("defaults.jl")
    include("forwarddiff.jl")
end
