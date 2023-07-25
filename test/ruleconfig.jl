using AbstractDifferentiation
using ChainRulesCore
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

    # issue #69
    @testset "Zygote context" begin
        ad = AD.ZygoteBackend()

        # example in #69: context is not mutated
        @test ad.ruleconfig.context.cache === nothing
        @test AD.derivative(ad, exp, 1.0) === (exp(1.0),)
        @test ad.ruleconfig.context.cache === nothing
        @test AD.derivative(ad, exp, 1.0) === (exp(1.0),)
        @test ad.ruleconfig.context.cache === nothing

        # Jacobian computation still works
        # https://github.com/JuliaDiff/AbstractDifferentiation.jl/pull/70#issuecomment-1449481724
        function f(x, a)
           r = Ref(x)
           r[] = r[] + r[]
           r[] = r[] * a
           r[]
        end
        @test AD.jacobian(ad, f, [1, 2, 3], 3) == ([6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0], [2.0, 4.0, 6.0])
    end

    # issue #57
    @testset "primal computation in rrule" begin
        function myfunc(x)
            @info "This should not be logged if I have an rrule"
            x
        end
        ChainRulesCore.rrule(::typeof(myfunc), x) = (x, (y -> (NoTangent(), y)))

        @test_logs Zygote.gradient(myfunc, 1) # nothing is logged
        @test_logs AD.derivative(AD.ZygoteBackend(), myfunc, 1) # nothing is logged
    end
end
