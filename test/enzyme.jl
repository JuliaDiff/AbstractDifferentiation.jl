import AbstractDifferentiation as AD
using Test
using Enzyme

backends = [
    "EnzymeForwardBackend" => AD.EnzymeForwardBackend(),
    "EnzymeReverseBackend" => AD.EnzymeReverseBackend(),
]

@testset "$name" for (name, backend) in backends
    if name == "EnzymeForwardBackend"
        @test backend isa AD.AbstractForwardMode
    else
        @test backend isa AD.AbstractReverseMode
    end

    @testset "Derivative" begin
        test_derivatives(backend; multiple_inputs=false)
    end
    @testset "Gradient" begin
        test_gradients(backend; multiple_inputs=false)
    end
    @testset "Jacobian" begin
        test_jacobians(backend; multiple_inputs=false)
    end
    # @testset "Hessian" begin
    #     test_hessians(backend, multiple_inputs = false)
    # end
    @testset "jvp" begin
        test_jvp(backend; multiple_inputs=false, vaugmented=true)
    end
    @testset "j′vp" begin
        test_j′vp(backend; multiple_inputs=false)
    end
    @testset "Lazy Derivative" begin
        test_lazy_derivatives(backend; multiple_inputs=false)
    end
    @testset "Lazy Gradient" begin
        test_lazy_gradients(backend; multiple_inputs=false)
    end
    @testset "Lazy Jacobian" begin
        test_lazy_jacobians(backend; multiple_inputs=false, vaugmented=true)
    end
    # @testset "Lazy Hessian" begin
    #     test_lazy_hessians(backend, multiple_inputs = false)
    # end
end
