using AbstractDifferentiation
using Test
using FiniteDifferences, ForwardDiff, Zygote

const AD = AbstractDifferentiation
const FDM = FiniteDifferences

## FiniteDifferences
struct FDMBackend1{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend1() = FDMBackend1(central_fdm(5, 1))
const fdm_backend1 = FDMBackend1()
# Minimal interface
function AD.jacobian(ab::FDMBackend1, f, xs...)
    return FDM.jacobian(ab.alg, f, xs...)
end

struct FDMBackend2{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend2() = FDMBackend2(central_fdm(5, 1))
const fdm_backend2 = FDMBackend2()
AD.@primitive function pushforward_function(ab::FDMBackend2, f, xs...)
    return function (vs)
        ws = FDM.jvp(ab.alg, f, tuple.(xs, vs)...)
        return length(xs) == 1 ? (ws,) : ws
    end
end

struct FDMBackend3{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend3() = FDMBackend3(central_fdm(5, 1))
const fdm_backend3 = FDMBackend3()
AD.@primitive function value_and_pullback_function(ab::FDMBackend3, f, xs...)
    value = f(xs...)
    function fd3_pullback(vs)
        # Supports only single output
        _vs = vs isa AbstractVector ? vs : only(vs)
        return FDM.j′vp(ab.alg, f, _vs, xs...)
    end
    return value, fd3_pullback
end
##

## ForwardDiff
struct ForwardDiffBackend1 <: AD.AbstractForwardMode end
const forwarddiff_backend1 = ForwardDiffBackend1()
function AD.jacobian(ab::ForwardDiffBackend1, f, xs)
    if xs isa Number
        return (ForwardDiff.derivative(f, xs),)
    elseif xs isa AbstractArray
        out = f(xs)
        if out isa Number
            return (adjoint(ForwardDiff.gradient(f, xs)),)
        else
            return (ForwardDiff.jacobian(f, xs),)
        end
    elseif xs isa Tuple
        error(typeof(xs))
    else
        error(typeof(xs))
    end
end
AD.primal_value(::ForwardDiffBackend1, ::Any, f, xs) = ForwardDiff.value.(f(xs...))

struct ForwardDiffBackend2 <: AD.AbstractForwardMode end
const forwarddiff_backend2 = ForwardDiffBackend2()
AD.@primitive function pushforward_function(ab::ForwardDiffBackend2, f, xs...)
    # jvp = f'(x)*v, i.e., differentiate f(x + h*v) wrt h at 0
    return function (vs)
        if xs isa Tuple
            @assert length(xs) <= 2
            if length(xs) == 1
                (ForwardDiff.derivative(h -> f(xs[1] + h * vs[1]), 0),)
            else
                ForwardDiff.derivative(h -> f(xs[1] + h * vs[1], xs[2] + h * vs[2]), 0)
            end
        else
            ForwardDiff.derivative(h -> f(xs + h * vs), 0)
        end
    end
end
AD.primal_value(::ForwardDiffBackend2, ::Any, f, xs) = ForwardDiff.value.(f(xs...))
##

## Zygote
struct ZygoteBackend1 <: AD.AbstractReverseMode end
const zygote_backend1 = ZygoteBackend1()
AD.@primitive function value_and_pullback_function(ab::ZygoteBackend1, f, xs...)
    # Supports only single output
    value, back = Zygote.pullback(f, xs...)
    function zygote_pullback(vs)
        _vs = vs isa AbstractVector ? vs : only(vs)
        return back(_vs)
    end
    return value, zygote_pullback
end

@testset "defaults" begin
    @testset "Utils" begin
        test_higher_order_backend(
            fdm_backend1, fdm_backend2, fdm_backend3, zygote_backend1, forwarddiff_backend2
        )
    end
    @testset "FiniteDifferences" begin
        @testset "Derivative" begin
            test_derivatives(fdm_backend1)
            test_derivatives(fdm_backend2)
            test_derivatives(fdm_backend3)
        end
        @testset "Gradient" begin
            test_gradients(fdm_backend1)
            test_gradients(fdm_backend2)
            test_gradients(fdm_backend3)
        end
        @testset "Jacobian" begin
            test_jacobians(fdm_backend1)
            test_jacobians(fdm_backend2)
            test_jacobians(fdm_backend3)
        end
        @testset "Hessian" begin
            test_hessians(fdm_backend1)
            test_hessians(fdm_backend2)
            test_hessians(fdm_backend3)
        end
        @testset "jvp" begin
            test_jvp(fdm_backend1; test_types=false)
            test_jvp(fdm_backend2; vaugmented=true)
            test_jvp(fdm_backend3)
        end
        @testset "j′vp" begin
            test_j′vp(fdm_backend1)
            test_j′vp(fdm_backend2)
            test_j′vp(fdm_backend3)
        end
        @testset "Lazy Derivative" begin
            test_lazy_derivatives(fdm_backend1)
            test_lazy_derivatives(fdm_backend2)
            test_lazy_derivatives(fdm_backend3)
        end
        @testset "Lazy Gradient" begin
            test_lazy_gradients(fdm_backend1)
            test_lazy_gradients(fdm_backend2)
            test_lazy_gradients(fdm_backend3)
        end
        @testset "Lazy Jacobian" begin
            test_lazy_jacobians(fdm_backend1)
            test_lazy_jacobians(fdm_backend2; vaugmented=true)
            test_lazy_jacobians(fdm_backend3)
        end
        @testset "Lazy Hessian" begin
            test_lazy_hessians(fdm_backend1)
            test_lazy_hessians(fdm_backend2)
            test_lazy_hessians(fdm_backend3)
        end
    end
    @testset "ForwardDiff" begin
        @testset "Derivative" begin
            test_derivatives(forwarddiff_backend1; multiple_inputs=false)
            test_derivatives(forwarddiff_backend2)
        end
        @testset "Gradient" begin
            test_gradients(forwarddiff_backend1; multiple_inputs=false)
            test_gradients(forwarddiff_backend2)
        end
        @testset "Jacobian" begin
            test_jacobians(forwarddiff_backend1; multiple_inputs=false)
            test_jacobians(forwarddiff_backend2)
        end
        @testset "Hessian" begin
            test_hessians(forwarddiff_backend1; multiple_inputs=false)
            test_hessians(forwarddiff_backend2)
        end
        @testset "jvp" begin
            test_jvp(forwarddiff_backend1; multiple_inputs=false)
            test_jvp(forwarddiff_backend2; vaugmented=true)
        end
        @testset "j′vp" begin
            test_j′vp(forwarddiff_backend1; multiple_inputs=false)
            test_j′vp(forwarddiff_backend2)
        end
        @testset "Lazy Derivative" begin
            test_lazy_derivatives(forwarddiff_backend1; multiple_inputs=false)
            test_lazy_derivatives(forwarddiff_backend2)
        end
        @testset "Lazy Gradient" begin
            test_lazy_gradients(forwarddiff_backend1; multiple_inputs=false)
            test_lazy_gradients(forwarddiff_backend2)
        end
        @testset "Lazy Jacobian" begin
            test_lazy_jacobians(forwarddiff_backend1; multiple_inputs=false)
            test_lazy_jacobians(forwarddiff_backend2; vaugmented=true)
        end
        @testset "Lazy Hessian" begin
            test_lazy_hessians(forwarddiff_backend1; multiple_inputs=false)
            test_lazy_hessians(forwarddiff_backend2)
        end
    end
    @testset "Zygote" begin
        @testset "Derivative" begin
            test_derivatives(zygote_backend1)
        end
        @testset "Gradient" begin
            test_gradients(zygote_backend1)
        end
        @testset "Jacobian" begin
            test_jacobians(zygote_backend1)
        end
        @testset "Hessian" begin
            # Zygote over Zygote problems
            backends = AD.HigherOrderBackend((forwarddiff_backend2, zygote_backend1))
            test_hessians(backends)
            backends = AD.HigherOrderBackend((zygote_backend1, forwarddiff_backend1))
            test_hessians(backends)
            # fails:
            # backends = AD.HigherOrderBackend((zygote_backend1,forwarddiff_backend2))
            # test_hessians(backends)
        end
        @testset "jvp" begin
            test_jvp(zygote_backend1)
        end
        @testset "j′vp" begin
            test_j′vp(zygote_backend1)
        end
        @testset "Lazy Derivative" begin
            test_lazy_derivatives(zygote_backend1)
        end
        @testset "Lazy Gradient" begin
            test_lazy_gradients(zygote_backend1)
        end
        @testset "Lazy Jacobian" begin
            test_lazy_jacobians(zygote_backend1)
        end
        @testset "Lazy Hessian" begin
            # Zygote over Zygote problems
            backends = AD.HigherOrderBackend((forwarddiff_backend2, zygote_backend1))
            test_lazy_hessians(backends)
            backends = AD.HigherOrderBackend((zygote_backend1, forwarddiff_backend1))
            test_lazy_hessians(backends)
        end
    end
end
