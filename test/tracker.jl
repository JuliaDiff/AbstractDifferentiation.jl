using AbstractDifferentiation
using Test
using Tracker

@testset "TrackerBackend" begin
    backends = [@inferred(AD.TrackerBackend())]
    @testset for backend in backends
        @testset "errors when nested" begin
            @test_throws ArgumentError AD.second_lowest(backend)
            @test_throws ArgumentError AD.hessian(backend, sum, randn(3))
        end
        @testset "Derivative" begin
            test_derivatives(backend)
        end
        @testset "Gradient" begin
            test_gradients(backend)
        end
        @testset "Jacobian" begin
            test_jacobians(backend)
        end
        @testset "jvp" begin
            test_jvp(backend)
        end
        @testset "j′vp" begin
            test_j′vp(backend)
        end
        @testset "Lazy Derivative" begin
            test_lazy_derivatives(backend)
        end
        @testset "Lazy Gradient" begin
            test_lazy_gradients(backend)
        end
        @testset "Lazy Jacobian" begin
            test_lazy_jacobians(backend)
        end
    end
end
