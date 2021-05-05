using AbstractDifferentiation
using Test, FiniteDifferences, LinearAlgebra
using Random
Random.seed!(1234)

const FDM = FiniteDifferences

struct FDMBackend1{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend1() = FDMBackend1(central_fdm(5, 1))
const fdm_backend1 = FDMBackend1()
# Minimal interface
AD.@primitive function jacobian(ab::FDMBackend1, f, xs...)
    return jacobian(ab.alg, f, xs...)
end

struct FDMBackend2{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend2() = FDMBackend2(central_fdm(5, 1))
const fdm_backend2 = FDMBackend2()
AD.@primitive function pushforward_function(ab::FDMBackend2, f, xs...)
    return (vs) -> jvp(ab.alg, f, tuple.(xs, vs)...)
end

struct FDMBackend3{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend3() = FDMBackend3(central_fdm(5, 1))
const fdm_backend3 = FDMBackend3()
AD.@primitive function pullback_function(ab::FDMBackend3, f, xs...)
    return function (vs)
        # Supports only single output
        if vs isa AbstractVector
            return j′vp(ab.alg, f, vs, xs...)
        else
            @assert length(vs) == 1
            return j′vp(ab.alg, f, vs[1], xs...)

        end
    end
end

fder(x, y) = exp(y) * x + y * log(x)
dfderdx(x, y) = exp(y) + y * 1/x
dfderdy(x, y) = exp(y) * x + log(x)

fgrad(x, y) = prod(x) + sum(y ./ (1:length(y)))
dfgraddx(x, y) = prod(x)./x
dfgraddy(x, y) = one(eltype(y)) ./ (1:length(y))
dfgraddxdx(x, y) = prod(x)./(x*x') - Diagonal(diag(prod(x)./(x*x')))

function fjac(x, y)
    x + Bidiagonal(-ones(length(y)) * 3, ones(length(y) - 1) / 2, :U) * y
end
dfjacdx(x, y) = I(length(x))
dfjacdy(x, y) = Bidiagonal(-ones(length(y)) * 3, ones(length(y) - 1) / 2, :U)

# Jvp
jxvp(x,y,v) = dfjacdx(x,y)*v
jyvp(x,y,v) = dfjacdy(x,y)*v

# vJp
vJxp(x,y,v) = dfjacdx(x,y)'*v
vJyp(x,y,v) = dfjacdy(x,y)'*v

const xscalar = rand()
const yscalar = rand()

const xvec = rand(5)
const yvec = rand(5)

# to check if vectors get mutated
xvec2 = deepcopy(xvec)
yvec2 = deepcopy(yvec)

function test_fdm_derivatives(fdm_backend)
    der1 = AD.derivative(fdm_backend, fder, xscalar, yscalar)
    der2 = (
        fdm_backend.alg(x -> fder(x, yscalar), xscalar),
        fdm_backend.alg(y -> fder(xscalar, y), yscalar),
    )
    @test norm.(der1 .- der2) == (0, 0)
    valscalar, der3 = AD.value_and_derivative(fdm_backend, fder, xscalar, yscalar)
    @test valscalar == fder(xscalar, yscalar)
    @test der3 .- der1 == (0, 0)
    der_exact = (dfderdx(xscalar,yscalar), dfderdy(xscalar,yscalar))
    @test minimum(isapprox.(der_exact, der1, rtol=1e-10))
end

function test_fdm_gradients(fdm_backend)
    grad1 = AD.gradient(fdm_backend, fgrad, xvec, yvec)
    grad2 = FDM.grad(fdm_backend.alg, fgrad, xvec, yvec)
    @test norm.(grad1 .- grad2) == (0, 0)
    valscalar, grad3 = AD.value_and_gradient(fdm_backend, fgrad, xvec, yvec)
    @test valscalar == fgrad(xvec, yvec)
    @test norm.(grad3 .- grad1) == (0, 0)
    grad_exact = (dfgraddx(xvec,yvec), dfgraddy(xvec,yvec))
    @test minimum(isapprox.(grad_exact, grad1, rtol=1e-10))
    @test xvec == xvec2
    @test yvec == yvec2
end

function test_fdm_jacobians(fdm_backend)
    jac1 = AD.jacobian(fdm_backend, fjac, xvec, yvec)
    jac2 = FDM.jacobian(fdm_backend.alg, fjac, xvec, yvec)
    @test norm.(jac1 .- jac2) == (0, 0)
    valvec, jac3 = AD.value_and_jacobian(fdm_backend, fjac, xvec, yvec)
    @test valvec == fjac(xvec, yvec)
    @test norm.(jac3 .- jac1) == (0, 0)
    grad_exact = (dfjacdx(xvec, yvec), dfjacdy(xvec, yvec))
    @test minimum(isapprox.(grad_exact, jac1, rtol=1e-10))
    @test xvec == xvec2
    @test yvec == yvec2
end

function test_fdm_hessians(fdm_backend)
    fhess = x -> fgrad(x, yvec)
    hess1 = AD.hessian(fdm_backend, fhess, xvec)
    hess2 = FDM.jacobian(
        fdm_backend.alg,
        (x) -> begin
            FDM.grad(
                fdm_backend.alg,
                fhess,
                x,
            )
        end,
        xvec,
    )
    @test norm.(hess1 .- hess2) == (0,)
    valscalar, hess3 = AD.value_and_hessian(fdm_backend, fhess, xvec)
    @test valscalar == fgrad(xvec, yvec)
    @test norm.(hess3 .- hess1) == (0,)
    valscalar, grad, hess4 = AD.value_gradient_and_hessian(fdm_backend, fhess, xvec)
    @test valscalar == fgrad(xvec, yvec)
    @test norm.(grad .- AD.gradient(fdm_backend, fhess, xvec)) == (0,)
    @test norm.(hess4 .- hess1) == (0,)
    @test dfgraddxdx(xvec,yvec) ≈ hess1[1] atol=1e-10
    @test xvec == xvec2
    @test yvec == yvec2
    fhess2 = x-> dfgraddx(x, yvec)
    hess5 = AD.jacobian(fdm_backend, fhess2, xvec)
    @test minimum(isapprox.(hess5, hess1, atol=1e-10))
end

function test_fdm_jvp(fdm_backend)
    v = (rand(length(xvec)), rand(length(yvec)))
    pf1 = AD.pushforward_function(fdm_backend, fjac, xvec, yvec)(v)
    pf2 = (
        FDM.jvp(fdm_backend.alg, x -> fjac(x, yvec), (xvec, v[1])),
        FDM.jvp(fdm_backend.alg, y -> fjac(xvec, y), (yvec, v[2])),
    )
    @test norm.(pf1 .- pf2) == (0, 0)
    valvec, pf3 = AD.value_and_pushforward_function(fdm_backend, fjac, xvec, yvec)(v)
    @test valvec == fjac(xvec, yvec)
    @test norm.(pf3 .- pf1) == (0, 0)
end

function test_fdm_j′vp(fdm_backend)
    w = rand(length(fjac(xvec, yvec)))
    pb1 = AD.pullback_function(fdm_backend, fjac, xvec, yvec)(w)
    pb2 = FDM.j′vp(fdm_backend.alg, fjac, w, xvec, yvec)
    @test all(norm.(pb1 .- pb2) .<= (1e-10, 1e-10))
    valvec, pb3 = AD.value_and_pullback_function(fdm_backend, fjac, xvec, yvec)(w)
    @test valvec == fjac(xvec, yvec)
    @test norm.(pb3 .- pb1) == (0, 0)
    @test minimum(isapprox.(pb1, (vJxp(xvec,yvec,w), vJyp(xvec,yvec,w)), atol=1e-10))
    @test xvec == xvec2
    @test yvec == yvec2
end

@testset "AbstractDifferentiation.jl" begin
    @testset "FiniteDifferences" begin
        @testset "Derivative" begin
            test_fdm_derivatives(fdm_backend1)
            test_fdm_derivatives(fdm_backend2)
            test_fdm_derivatives(fdm_backend3)
        end
        @testset "Gradient" begin
            test_fdm_gradients(fdm_backend1)
            test_fdm_gradients(fdm_backend2)
            test_fdm_gradients(fdm_backend3)
        end
        @testset "Jacobian" begin
            test_fdm_jacobians(fdm_backend1)
            test_fdm_jacobians(fdm_backend2)
            test_fdm_jacobians(fdm_backend3)
        end
        @testset "Hessian" begin
            # Works but super slow
            test_fdm_hessians(fdm_backend1)
            test_fdm_hessians(fdm_backend2)
            test_fdm_hessians(fdm_backend3)
        end
        @testset "jvp" begin
            test_fdm_jvp(fdm_backend1)
            # Errors
            test_fdm_jvp(fdm_backend2)
            test_fdm_jvp(fdm_backend3)
        end
        @testset "j′vp" begin
            test_fdm_j′vp(fdm_backend1)
            test_fdm_j′vp(fdm_backend2)
            test_fdm_j′vp(fdm_backend3)
        end
    end
end
