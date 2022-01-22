using AbstractDifferentiation
using Test
using ForwardDiff

@testset "ForwardDiffBackend" begin
    backends = [
        @inferred(AD.ForwardDiffBackend())
        @inferred(AD.ForwardDiffBackend(; chunksize=Val{1}()))
    ]
    @testset for backend in backends
        @testset "Derivative" begin
            test_derivatives(backend)
        end
        @testset "Gradient" begin
            test_gradients(backend)
        end
        @testset "Jacobian" begin
            test_jacobians(backend)
        end
        @testset "Hessian" begin
            test_hessians(backend)
        end
        @testset "jvp" begin
            test_jvp(backend; vaugmented=true)
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
            test_lazy_jacobians(backend; vaugmented=true)
        end
        @testset "Lazy Hessian" begin
            test_lazy_hessians(backend)
        end
    end
end
