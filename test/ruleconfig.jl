using AbstractDifferentiation
using Test
using Zygote

@testset "ReverseRuleConfigBackend(ZygoteRuleConfig())" begin
    backends = [@inferred(AD.ZygoteBackend())]
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
