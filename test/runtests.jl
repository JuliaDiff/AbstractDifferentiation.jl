using AbstractDifferentiation
using Test, FiniteDifferences, LinearAlgebra
using ForwardDiff
using Zygote
using Random
Random.seed!(1234)

const FDM = FiniteDifferences

## FiniteDifferences
struct FDMBackend1{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend1() = FDMBackend1(central_fdm(5, 1))
const fdm_backend1 = FDMBackend1()
# Minimal interface
AD.@primitive function jacobian(ab::FDMBackend1, f, xs...)
    return FDM.jacobian(ab.alg, f, xs...)
end

struct FDMBackend2{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend2() = FDMBackend2(central_fdm(5, 1))
const fdm_backend2 = FDMBackend2()
AD.@primitive function pushforward_function(ab::FDMBackend2, f, xs...)
    return function (vs)
        FDM.jvp(ab.alg, f, tuple.(xs, vs)...)
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
            return FDM.j′vp(ab.alg, f, vs, xs...)
        else
            @assert length(vs) == 1
            return FDM.j′vp(ab.alg, f, vs[1], xs...)

        end
    end
end
##


## ForwardDiff
struct ForwardDiffBackend1 <: AD.AbstractForwardMode end
const forwarddiff_backend1 = ForwardDiffBackend1()
AD.@primitive function jacobian(ab::ForwardDiffBackend1, f, xs)
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
AD.primalvalue(::ForwardDiffBackend1, ::Any, f, xs) = ForwardDiff.value.(f(xs...))

struct ForwardDiffBackend2 <: AD.AbstractForwardMode end
const forwarddiff_backend2 = ForwardDiffBackend2()
AD.@primitive function pushforward_function(ab::ForwardDiffBackend2, f, xs...)
    # jvp = f'(x)*v, i.e., differentiate f(x + h*v) wrt h at 0
    return function (vs)
        if xs isa Tuple
            @assert length(xs) <= 2
            if length(xs) == 1
                (ForwardDiff.derivative(h->f(xs[1]+h*vs[1]),0),)
            else
                ForwardDiff.derivative(h->f(xs[1]+h*vs[1], xs[2]+h*vs[2]),0)
            end
        else
            ForwardDiff.derivative(h->f(xs+h*vs),0)
        end
    end
end
AD.primalvalue(::ForwardDiffBackend2, ::Any, f, xs) = ForwardDiff.value.(f(xs...))
##

## Zygote
struct ZygoteBackend1 <: AD.AbstractReverseMode end
const zygote_backend1 = ZygoteBackend1()
AD.@primitive function pullback_function(ab::ZygoteBackend1, f, xs...)
    return function (vs)
        # Supports only single output
        _, back = Zygote.pullback(f, xs...)
        if vs isa AbstractVector
            back(vs)
        else
            @assert length(vs) == 1
            back(vs[1])
        end
    end
end
##

struct ZygoteBackend <: AD.AbstractReverseMode end
AD.@primitive function pullback_function(ab::ZygoteBackend, f, xs...)
    return function (vs)
        # Supports only single output and tuple as input
        _, back = Zygote.pullback(f, xs...)
        back(vs[1])
    end
end

fder(x, y) = exp(y) * x + y * log(x)
dfderdx(x, y) = exp(y) + y * 1/x
dfderdy(x, y) = exp(y) * x + log(x)

fgrad(x, y) = prod(x) + sum(y ./ (1:length(y)))
dfgraddx(x, y) = prod(x)./x
dfgraddy(x, y) = one(eltype(y)) ./ (1:length(y))
dfgraddxdx(x, y) = prod(x)./(x*x') - Diagonal(diag(prod(x)./(x*x')))
dfgraddydy(x, y) = zeros(length(y),length(y))

function fjac(x, y)
    x + -3*y + [y[2:end];zero(y[end])]/2# Bidiagonal(-ones(length(y)) * 3, ones(length(y) - 1) / 2, :U) * y
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


function test_derivatives(backend; multiple_inputs=true)
    # test with respect to analytical solution
    der_exact = (dfderdx(xscalar,yscalar), dfderdy(xscalar,yscalar))
    if multiple_inputs
        der1 = AD.derivative(backend, fder, xscalar, yscalar)
        @test minimum(isapprox.(der_exact, der1, rtol=1e-10))
        valscalar, der2 = AD.value_and_derivative(backend, fder, xscalar, yscalar)
        @test valscalar == fder(xscalar, yscalar)
        @test der2 .- der1 == (0, 0)
    end
    # test if single input (no tuple works)
    valscalara, dera = AD.value_and_derivative(backend, x -> fder(x, yscalar), xscalar)
    valscalarb, derb = AD.value_and_derivative(backend, y -> fder(xscalar, y), yscalar)
    @test fder(xscalar, yscalar) == valscalara
    @test fder(xscalar, yscalar) == valscalarb
    @test isapprox(dera[1], der_exact[1], rtol=1e-10)
    @test isapprox(derb[1], der_exact[2], rtol=1e-10)
end

function test_gradients(backend; multiple_inputs=true)
    # test with respect to analytical solution
    grad_exact = (dfgraddx(xvec,yvec), dfgraddy(xvec,yvec))
    if multiple_inputs
        grad1 = AD.gradient(backend, fgrad, xvec, yvec)
        @test minimum(isapprox.(grad_exact, grad1, rtol=1e-10))
        valscalar, grad2 = AD.value_and_gradient(backend, fgrad, xvec, yvec)
        @test valscalar == fgrad(xvec, yvec)
        @test norm.(grad2 .- grad1) == (0, 0)
    end
    # test if single input (no tuple works)
    valscalara, grada = AD.value_and_gradient(backend, x -> fgrad(x, yvec), xvec)
    valscalarb, gradb = AD.value_and_gradient(backend, y -> fgrad(xvec, y), yvec)
    @test fgrad(xvec, yvec) == valscalara
    @test fgrad(xvec, yvec) == valscalarb
    @test isapprox(grada[1], grad_exact[1], rtol=1e-10)
    @test isapprox(gradb[1], grad_exact[2], rtol=1e-10)
    @test xvec == xvec2
    @test yvec == yvec2
end

function test_jacobians(backend; multiple_inputs=true)
    # test with respect to analytical solution
    jac_exact = (dfjacdx(xvec, yvec), dfjacdy(xvec, yvec))
    if multiple_inputs
        jac1 = AD.jacobian(backend, fjac, xvec, yvec)
        @test minimum(isapprox.(jac_exact, jac1, rtol=1e-10))
        valvec, jac2 = AD.value_and_jacobian(backend, fjac, xvec, yvec)
        @test valvec == fjac(xvec, yvec)
        @test norm.(jac2 .- jac1) == (0, 0)
    end
    
    # test if single input (no tuple works)
    valveca, jaca = AD.value_and_jacobian(backend, x -> fjac(x, yvec), xvec)
    valvecb, jacb = AD.value_and_jacobian(backend, y -> fjac(xvec, y), yvec)
    @test fjac(xvec, yvec) == valveca
    @test fjac(xvec, yvec) == valvecb
    @test isapprox(jaca[1], jac_exact[1], rtol=1e-10)
    @test isapprox(jacb[1], jac_exact[2], rtol=1e-10)
    @test xvec == xvec2
    @test yvec == yvec2
end

function test_hessians(backend; multiple_inputs=false)
    if multiple_inputs
        # ... but 
        error("multiple_inputs=true is not supported.")
    else
        # explicit test that AbstractDifferentiation throws an error
        # don't support tuple of Hessians
        @test_throws AssertionError H1 = AD.hessian(backend, fgrad, xvec, yvec)
    end
   
    # @test dfgraddxdx(xvec,yvec) ≈ H1[1] atol=1e-10
    # @test dfgraddydy(xvec,yvec) ≈ H1[2] atol=1e-10

    # test if single input (no tuple works)
    fhess = x -> fgrad(x, yvec)
    hess1 = AD.hessian(backend, fhess, xvec)
    # test with respect to analytical solution
    @test dfgraddxdx(xvec,yvec) ≈ hess1[1] atol=1e-10
    
    valscalar, hess2 = AD.value_and_hessian(backend, fhess, xvec)
    @test valscalar == fgrad(xvec, yvec)
    @test norm.(hess2 .- hess1) == (0,)
    valscalar, grad, hess3 = AD.value_gradient_and_hessian(backend, fhess, xvec)
    @test valscalar == fgrad(xvec, yvec)
    @test norm.(grad .- AD.gradient(backend, fhess, xvec)) == (0,)
    @test norm.(hess3 .- hess1) == (0,)
    
    @test xvec == xvec2
    @test yvec == yvec2
    fhess2 = x-> dfgraddx(x, yvec)
    hess4 = AD.jacobian(backend, fhess2, xvec)
    @test minimum(isapprox.(hess4, hess1, atol=1e-10))
end

function test_jvp(backend; multiple_inputs=true)
    v = (rand(length(xvec)), rand(length(yvec)))

    if multiple_inputs
        if backend isa Union{FDMBackend2,ForwardDiffBackend2} # augmented version of v
            identity_like = AD.identity_matrix_like(v)
            vaug = map(identity_like) do identity_like_i
                identity_like_i .* v
            end

            pf1 = map(v->AD.pushforward_function(backend, fjac, xvec, yvec)(v), vaug)
            ((valvec1, pf2x), (valvec2, pf2y)) = map(v->AD.value_and_pushforward_function(backend, fjac, xvec, yvec)(v), vaug)
        else
            pf1 = AD.pushforward_function(backend, fjac, xvec, yvec)(v)
            valvec, pf2 = AD.value_and_pushforward_function(backend, fjac, xvec, yvec)(v) 
            ((valvec1, pf2x), (valvec2, pf2y)) = (valvec, pf2[1]), (valvec, pf2[2])
        end
       
        @test valvec1 == fjac(xvec, yvec)
        @test valvec2 == fjac(xvec, yvec)
        @test norm.((pf2x,pf2y) .- pf1) == (0, 0)
        # test with respect to analytical solution
        @test minimum(isapprox.(pf1, (jxvp(xvec,yvec,v[1]), jyvp(xvec,yvec,v[2])), atol=1e-10))
        @test xvec == xvec2
        @test yvec == yvec2
    end

    valvec1, pf1 = AD.value_and_pushforward_function(backend, x -> fjac(x, yvec), xvec)(v[1])
    valvec2, pf2 = AD.value_and_pushforward_function(backend, y -> fjac(xvec, y), yvec)(v[2])

    if backend isa Union{FDMBackend2}
      pf1 = (pf1,)  
      pf2 = (pf2,)
    end
    @test valvec1 == fjac(xvec, yvec)
    @test valvec2 == fjac(xvec, yvec)
    @test minimum(isapprox.((pf1[1],pf2[1]), (jxvp(xvec,yvec,v[1]), jyvp(xvec,yvec,v[2])), atol=1e-10))
end

function test_j′vp(backend; multiple_inputs=true)
    # test with respect to analytical solution
    w = rand(length(fjac(xvec, yvec)))
    if multiple_inputs
        pb1 = AD.pullback_function(backend, fjac, xvec, yvec)(w)
        valvec, pb2 = AD.value_and_pullback_function(backend, fjac, xvec, yvec)(w)
        @test valvec == fjac(xvec, yvec)
        @test norm.(pb2 .- pb1) == (0, 0)
        @test minimum(isapprox.(pb1, (vJxp(xvec,yvec,w), vJyp(xvec,yvec,w)), atol=1e-10))
        @test xvec == xvec2
        @test yvec == yvec2
    end

    valvec1, pb1 = AD.value_and_pullback_function(backend, x -> fjac(x, yvec), xvec)(w)
    valvec2, pb2 = AD.value_and_pullback_function(backend, y -> fjac(xvec, y), yvec)(w)
    @test valvec1 == fjac(xvec, yvec)
    @test valvec2 == fjac(xvec, yvec)
    @test minimum(isapprox.((pb1[1],pb2[1]), (vJxp(xvec,yvec,w), vJyp(xvec,yvec,w)), atol=1e-10))
end

function test_lazy_derivatives(backend; multiple_inputs=true)
    # single input function
    der1 = AD.derivative(backend, x->fder(x, yscalar), xscalar)
    lazyder = AD.LazyDerivative(backend, x->fder(x, yscalar), xscalar)

    # multiplication with scalar
    @test lazyder*yscalar == der1.*yscalar
    @test lazyder*yscalar isa Tuple

    @test yscalar*lazyder == yscalar.*der1 
    @test yscalar*lazyder isa Tuple

    # multiplication with array
    @test lazyder*yvec == (der1.*yvec,)
    @test lazyder*yvec isa Tuple

    @test yvec*lazyder == (yvec.*der1,)
    @test yvec*lazyder isa Tuple

    # multiplication with tuple
    @test lazyder*(yscalar,) == lazyder*yscalar
    @test lazyder*(yvec,) == lazyder*yvec

    @test (yscalar,)*lazyder == yscalar*lazyder
    @test (yvec,)*lazyder == yvec*lazyder

    # two input function
    if multiple_inputs
        der1 = AD.derivative(backend, fder, xscalar, yscalar)
        lazyder = AD.LazyDerivative(backend, fder, (xscalar, yscalar))

        # multiplication with scalar
        @test lazyder*yscalar == der1.*yscalar
        @test lazyder*yscalar isa Tuple

        @test yscalar*lazyder == yscalar.*der1
        @test yscalar*lazyder isa Tuple

        # multiplication with array
        @test lazyder*yvec == (der1[1]*yvec, der1[2]*yvec)
        @test lazyder*yvec isa Tuple

        @test yvec*lazyder == (yvec*der1[1], yvec*der1[2])
        @test lazyder*yvec isa Tuple

        # multiplication with tuple
        @test_throws AssertionError lazyder*(yscalar,)
        @test_throws AssertionError lazyder*(yvec,)

        @test_throws AssertionError (yscalar,)*lazyder 
        @test_throws AssertionError (yvec,)*lazyder
    end
end

function test_lazy_gradients(backend; multiple_inputs=true)
    # single input function
    grad1 = AD.gradient(backend, x->fgrad(x, yvec), xvec)
    lazygrad = AD.LazyGradient(backend, x->fgrad(x, yvec), xvec)

    # multiplication with scalar
    @test norm.(lazygrad*yscalar .- grad1.*yscalar) == (0,)
    @test lazygrad*yscalar isa Tuple

    @test norm.(yscalar*lazygrad .- yscalar.*grad1) == (0,)
    @test yscalar*lazygrad isa Tuple

    # multiplication with tuple
    @test lazygrad*(yscalar,) == lazygrad*yscalar
    @test (yscalar,)*lazygrad == yscalar*lazygrad

    # two input function
    if multiple_inputs
        grad1 = AD.gradient(backend, fgrad, xvec, yvec)
        lazygrad = AD.LazyGradient(backend, fgrad, (xvec, yvec))

        # multiplication with scalar
        @test norm.(lazygrad*yscalar .- grad1.*yscalar) == (0,0)
        @test lazygrad*yscalar isa Tuple

        @test norm.(yscalar*lazygrad .- yscalar.*grad1) == (0,0)
        @test yscalar*lazygrad isa Tuple

        # multiplication with tuple
        @test_throws AssertionError lazygrad*(yscalar,) == lazygrad*yscalar
        @test_throws AssertionError (yscalar,)*lazygrad == yscalar*lazygrad
    end
end

function test_lazy_jacobians(backend; multiple_inputs=true)
    # single input function
    jac1 = AD.jacobian(backend, x->fjac(x, yvec), xvec)
    lazyjac = AD.LazyJacobian(backend, x->fjac(x, yvec), xvec)

    # multiplication with scalar
    @test norm.(lazyjac*yscalar .- jac1.*yscalar) == (0,)
    @test lazyjac*yscalar isa Tuple

    @test norm.(yscalar*lazyjac .- yscalar.*jac1) == (0,)
    @test yscalar*lazyjac isa Tuple

    w = rand(length(fjac(xvec, yvec)))
    v = (rand(length(xvec)),rand(length(xvec)))

    # vjp
    pb1 = (vJxp(xvec,yvec,w),)
    res = w'*lazyjac
    @test minimum(isapprox.(pb1, res, atol=1e-10))
    @test res isa Tuple

    # jvp
    pf1 = (jxvp(xvec,yvec,v[1]),)
    res = lazyjac*v[1]
    @test minimum(isapprox.(pf1, res, atol=1e-10))
    @test res isa Tuple

    # two input function
    if multiple_inputs
        jac1 = AD.jacobian(backend, fjac, xvec, yvec)
        lazyjac = AD.LazyJacobian(backend, fjac, (xvec, yvec))

        # multiplication with scalar
        @test norm.(lazyjac*yscalar .- jac1.*yscalar) == (0,0)
        @test lazyjac*yscalar isa Tuple

        @test norm.(yscalar*lazyjac .- yscalar.*jac1) == (0,0)
        @test yscalar*lazyjac isa Tuple

        # vjp
        pb1 = (vJxp(xvec,yvec,w), vJyp(xvec,yvec,w))
        res = w'lazyjac
        @test minimum(isapprox.(pb1, res, atol=1e-10))
        @test res isa Tuple

        # jvp
        pf1 = (jxvp(xvec,yvec,v[1]), jyvp(xvec,yvec,v[2]))

        if backend isa Union{FDMBackend2,ForwardDiffBackend2} # augmented version of v
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
end

function test_lazy_hessians(backend; multiple_inputs=true)
    # fdm_backend not used here yet..
    # single input function
    fhess = x -> fgrad(x, yvec)
    hess1 = (dfgraddxdx(xvec,yvec),)
    lazyhess = AD.LazyHessian(backend, fhess, xvec)

    # multiplication with scalar
    @test minimum(isapprox.(lazyhess*yscalar, hess1.*yscalar, atol=1e-10))
    @test lazyhess*yscalar isa Tuple

    # multiplication with scalar
    @test minimum(isapprox.(yscalar*lazyhess, yscalar.*hess1, atol=1e-10))
    @test yscalar*lazyhess isa Tuple

    w = rand(length(xvec))
    v = rand(length(xvec))

    # Hvp
    Hv = map(h->h*v, hess1)
    res = lazyhess*v
    @test minimum(isapprox.(Hv, res, atol=1e-10))
    @test res isa Tuple

    # H′vp
    wH = map(h->h'*w, hess1)
    res = w'*lazyhess
    @test minimum(isapprox.(wH, res, atol=1e-10))
    @test res isa Tuple
end

@testset "AbstractDifferentiation.jl" begin
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
            test_jvp(fdm_backend1)
            test_jvp(fdm_backend2)
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
            test_lazy_jacobians(fdm_backend2)
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
            test_jvp(forwarddiff_backend2)
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
            test_lazy_jacobians(forwarddiff_backend2)
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
            test_hessians(zygote_backend1)
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
            test_lazy_hessians(zygote_backend1)
        end
    end
end

