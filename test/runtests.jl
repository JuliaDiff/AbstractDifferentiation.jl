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
    return function (vs)
        jvp(ab.alg, f, tuple.(xs, vs)...)
    end
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

    if fdm_backend isa FDMBackend2 # augmented version of v
        identity_like = AD.identity_matrix_like(v)
        vaug = map(identity_like) do identity_like_i
            identity_like_i .* v
        end

        pf1 = map(v->AD.pushforward_function(fdm_backend, fjac, xvec, yvec)(v), vaug)
        ((valvec1, pf3x), (valvec2, pf3y)) = map(v->AD.value_and_pushforward_function(fdm_backend, fjac, xvec, yvec)(v), vaug)
    else
        pf1 = AD.pushforward_function(fdm_backend, fjac, xvec, yvec)(v)
        valvec, pf3 = AD.value_and_pushforward_function(fdm_backend, fjac, xvec, yvec)(v)
        ((valvec1, pf3x), (valvec2, pf3y)) = (valvec, pf3[1]), (valvec, pf3[2])
    end
    pf2 = (
        FDM.jvp(fdm_backend.alg, x -> fjac(x, yvec), (xvec, v[1])),
        FDM.jvp(fdm_backend.alg, y -> fjac(xvec, y), (yvec, v[2])),
    )
    @test norm.(pf1 .- pf2) == (0, 0)

    @test valvec1 == fjac(xvec, yvec)
    @test valvec2 == fjac(xvec, yvec)
    @test norm.((pf3x,pf3y) .- pf1) == (0, 0)
    @test minimum(isapprox.(pf1, (jxvp(xvec,yvec,v[1]), jyvp(xvec,yvec,v[2])), atol=1e-10))
    @test xvec == xvec2
    @test yvec == yvec2
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

function test_fdm_lazy_derivatives(fdm_backend)
    # single input function
    der1 = AD.derivative(fdm_backend, x->fder(x, yscalar), xscalar)
    der2 = (
        fdm_backend.alg(x -> fder(x, yscalar), xscalar),
        fdm_backend.alg(y -> fder(xscalar, y), yscalar),
    )

    lazyder = AD.LazyDerivative(fdm_backend, x->fder(x, yscalar), xscalar)

    # multiplication with scalar
    @test der1[1]*yscalar == der2[1]*yscalar
    @test lazyder*yscalar == der1.*yscalar
    @test lazyder*yscalar isa Tuple

    @test yscalar*der1[1] == yscalar*der2[1]
    @test yscalar*lazyder == yscalar.*der1
    @test yscalar*lazyder isa Tuple

    # multiplication with array
    @test der1[1]*yvec == der2[1]*yvec
    @test lazyder*yvec == (der1.*yvec,)
    @test lazyder*yvec isa Tuple

    @test yvec*der1[1] == yvec*der2[1]
    @test yvec*lazyder == (yvec.*der1,)
    @test yvec*lazyder isa Tuple

    # multiplication with tuple
    @test lazyder*(yscalar,) == lazyder*yscalar
    @test lazyder*(yvec,) == lazyder*yvec

    @test (yscalar,)*lazyder == yscalar*lazyder
    @test (yvec,)*lazyder == yvec*lazyder

    # two input function
    der1 = AD.derivative(fdm_backend, fder, xscalar, yscalar)
    der2 = (
        fdm_backend.alg(x -> fder(x, yscalar), xscalar),
        fdm_backend.alg(y -> fder(xscalar, y), yscalar),
    )

    lazyder = AD.LazyDerivative(fdm_backend, fder, (xscalar, yscalar))

    # multiplication with scalar
    @test der1.*yscalar == der2.*yscalar
    @test lazyder*yscalar == der1.*yscalar
    @test lazyder*yscalar isa Tuple

    @test yscalar.*der1 == yscalar.*der2
    @test yscalar*lazyder == yscalar.*der1
    @test yscalar*lazyder isa Tuple

    # multiplication with array
    @test (der1[1]*yvec, der1[2]*yvec) == (der2[1]*yvec, der2[2]*yvec)
    @test lazyder*yvec == (der1[1]*yvec, der1[2]*yvec)
    @test lazyder*yvec isa Tuple

    @test (yvec*der1[1], yvec*der1[2]) == (yvec*der2[1], yvec*der2[2])
    @test yvec*lazyder == (yvec*der1[1], yvec*der1[2])
    @test lazyder*yvec isa Tuple

    # multiplication with tuple
    @test lazyder*(yscalar,) == lazyder*yscalar
    @test lazyder*(yvec,) == lazyder*yvec

    @test (yscalar,)*lazyder == yscalar*lazyder
    @test (yvec,)*lazyder == yvec*lazyder
end

function test_fdm_lazy_gradients(fdm_backend)
    # single input function
    grad1 = AD.gradient(fdm_backend, x->fgrad(x, yvec), xvec)
    grad2 = FDM.grad(fdm_backend.alg, x->fgrad(x, yvec), xvec)
    lazygrad = AD.LazyGradient(fdm_backend, x->fgrad(x, yvec), xvec)

    # multiplication with scalar
    @test norm.(grad1.*yscalar .- grad2.*yscalar) == (0,)
    @test norm.(lazygrad*yscalar .- grad1.*yscalar) == (0,)
    @test lazygrad*yscalar isa Tuple

    @test norm.(yscalar.*grad1 .- yscalar.*grad2) == (0,)
    @test norm.(yscalar*lazygrad .- yscalar.*grad1) == (0,)
    @test yscalar*lazygrad isa Tuple

    # multiplication with tuple
    @test lazygrad*(yscalar,) == lazygrad*yscalar
    @test (yscalar,)*lazygrad == yscalar*lazygrad

    # two input function
    grad1 = AD.gradient(fdm_backend, fgrad, xvec, yvec)
    grad2 = FDM.grad(fdm_backend.alg, fgrad, xvec, yvec)
    lazygrad = AD.LazyGradient(fdm_backend, fgrad, (xvec, yvec))

    # multiplication with scalar
    @test norm.(grad1.*yscalar .- grad2.*yscalar) == (0,0)
    @test norm.(lazygrad*yscalar .- grad1.*yscalar) == (0,0)
    @test lazygrad*yscalar isa Tuple

    @test norm.(yscalar.*grad1 .- yscalar.*grad2) == (0,0)
    @test norm.(yscalar*lazygrad .- yscalar.*grad1) == (0,0)
    @test yscalar*lazygrad isa Tuple

    # multiplication with tuple
    @test lazygrad*(yscalar,) == lazygrad*yscalar
    @test (yscalar,)*lazygrad == yscalar*lazygrad
end

function test_fdm_lazy_jacobians(fdm_backend)
    # single input function
    jac1 = AD.jacobian(fdm_backend, x->fjac(x, yvec), xvec)
    jac2 = FDM.jacobian(fdm_backend.alg, x->fjac(x, yvec), xvec)
    lazyjac = AD.LazyJacobian(fdm_backend, x->fjac(x, yvec), xvec)

    # multiplication with scalar
    @test norm.(jac1.*yscalar .- jac2.*yscalar) == (0,)
    @test norm.(lazyjac*yscalar .- jac1.*yscalar) == (0,)
    @test lazyjac*yscalar isa Tuple

    @test norm.(yscalar.*jac1 .- yscalar.*jac2) == (0,)
    @test norm.(yscalar*lazyjac .- yscalar.*jac1) == (0,)
    @test yscalar*lazyjac isa Tuple

    w = adjoint(rand(length(fjac(xvec, yvec))))
    v = (rand(length(xvec)),rand(length(xvec)))

    # vjp
    pb1 = FDM.j′vp(fdm_backend.alg, x -> fjac(x, yvec), w, xvec)
    res = w*lazyjac
    @test minimum(isapprox.(pb1, res, atol=1e-10))
    @test res isa Tuple

    # jvp
    pf1 = (FDM.jvp(fdm_backend.alg, x -> fjac(x, yvec), (xvec, v[1])),)
    res = lazyjac*v[1]
    @test minimum(isapprox.(pf1, res, atol=1e-10))
    @test res isa Tuple

    # two input function
    jac1 = AD.jacobian(fdm_backend, fjac, xvec, yvec)
    jac2 = FDM.jacobian(fdm_backend.alg, fjac, xvec, yvec)
    lazyjac = AD.LazyJacobian(fdm_backend, fjac, (xvec, yvec))

    # multiplication with scalar
    @test norm.(jac1.*yscalar .- jac2.*yscalar) == (0,0)
    @test norm.(lazyjac*yscalar .- jac1.*yscalar) == (0,0)
    @test lazyjac*yscalar isa Tuple

    @test norm.(yscalar.*jac1 .- yscalar.*jac2) == (0,0)
    @test norm.(yscalar*lazyjac .- yscalar.*jac1) == (0,0)
    @test yscalar*lazyjac isa Tuple

    # vjp
    pb1 = FDM.j′vp(fdm_backend.alg, fjac, w, xvec, yvec)
    res = w*lazyjac
    @test minimum(isapprox.(pb1, res, atol=1e-10))
    @test res isa Tuple

    # jvp
    pf1 = (
        FDM.jvp(fdm_backend.alg, x -> fjac(x, yvec), (xvec, v[1])),
        FDM.jvp(fdm_backend.alg, y -> fjac(xvec, y), (yvec, v[2])),
    )

    if fdm_backend isa FDMBackend2 # augmented version of v
        identity_like = AD.identity_matrix_like(v)
        vaug = map(identity_like) do identity_like_i
            identity_like_i .* v
        end

        res = map(v->(lazyjac*v)[1], vaug)
    else
        res = lazyjac*v
    end

    @test minimum(isapprox.(pf1, res, atol=1e-10))
    @test res isa Tuple
end

function test_fdm_lazy_hessians(fdm_backend)
    # single input function
    fhess = x -> fgrad(x, yvec)
    hess1 = (dfgraddxdx(xvec,yvec),)
    lazyhess = AD.LazyHessian(fdm_backend, x->fgrad(x, yvec), xvec)

    # multiplication with scalar
    @test minimum(isapprox.(lazyhess*yscalar, hess1.*yscalar, atol=1e-10))
    @test lazyhess*yscalar isa Tuple

    # multiplication with scalar
    @test minimum(isapprox.(yscalar*lazyhess, yscalar.*hess1, atol=1e-10))
    @test yscalar*lazyhess isa Tuple

    w = adjoint(rand(length(xvec)))
    v = rand(length(xvec))

    # Hvp
    Hv = map(h->h*v, hess1)
    res = lazyhess*v
    @test minimum(isapprox.(Hv, res, atol=1e-10))
    @test res isa Tuple

    # H′vp
    wH = map(h->h'*adjoint(w), hess1)
    res = w*lazyhess
    @test minimum(isapprox.(wH, res, atol=1e-10))
    @test res isa Tuple
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
            test_fdm_jvp(fdm_backend2)
            test_fdm_jvp(fdm_backend3)
        end
        @testset "j′vp" begin
            test_fdm_j′vp(fdm_backend1)
            test_fdm_j′vp(fdm_backend2)
            test_fdm_j′vp(fdm_backend3)
        end
        @testset "Lazy Derivative" begin
            test_fdm_lazy_derivatives(fdm_backend1)
            test_fdm_lazy_derivatives(fdm_backend2)
            test_fdm_lazy_derivatives(fdm_backend3)
        end
        @testset "Lazy Gradient" begin
            test_fdm_lazy_gradients(fdm_backend1)
            test_fdm_lazy_gradients(fdm_backend2)
            test_fdm_lazy_gradients(fdm_backend3)
        end
        @testset "Lazy Jacobian" begin
            test_fdm_lazy_jacobians(fdm_backend1)
            test_fdm_lazy_jacobians(fdm_backend2)
            test_fdm_lazy_jacobians(fdm_backend3)
        end
        @testset "Lazy Hessian" begin
            test_fdm_lazy_hessians(fdm_backend1)
            test_fdm_lazy_hessians(fdm_backend2)
            test_fdm_lazy_hessians(fdm_backend3)
        end
    end
end
