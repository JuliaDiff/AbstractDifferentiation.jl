using AbstractDifferentiation
using Test
using FiniteDifferences

@testset "FiniteDifferencesBackend" begin
    method = FiniteDifferences.central_fdm(6, 1)
    # `central_fdm(5, 1)` is not type-inferrable, so only check inferrability
    # with user-specified method
    backends = [
        AD.FiniteDifferencesBackend(), @inferred(AD.FiniteDifferencesBackend(method))
    ]
    @test backends[2].method === method

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
        @testset "Second derivative" begin
            test_second_derivatives(backend)
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
