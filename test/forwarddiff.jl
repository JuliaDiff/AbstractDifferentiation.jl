using AbstractDifferentiation
using Test
using ForwardDiff

include("test_utils.jl")

@testset "ForwardDiffBackend" begin
    forwarddiff_backend = AD.ForwardDiffBackend()
    @testset "Derivative" begin
        test_derivatives(forwarddiff_backend)
    end
    @testset "Gradient" begin
        test_gradients(forwarddiff_backend)
    end
    @testset "Jacobian" begin
        test_jacobians(forwarddiff_backend)
    end
    @testset "Hessian" begin
        test_hessians(forwarddiff_backend)
    end
    @testset "jvp" begin
        test_jvp(forwarddiff_backend; vaugmented=true)
    end
    @testset "j′vp" begin
        test_j′vp(forwarddiff_backend)
    end
    @testset "Lazy Derivative" begin
        test_lazy_derivatives(forwarddiff_backend)
    end
    @testset "Lazy Gradient" begin
        test_lazy_gradients(forwarddiff_backend)
    end
    @testset "Lazy Jacobian" begin
        test_lazy_jacobians(forwarddiff_backend; vaugmented=true)
    end
    @testset "Lazy Hessian" begin
        test_lazy_hessians(forwarddiff_backend)
    end
end
