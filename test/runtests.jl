using AbstractDifferentiation
using Test, FiniteDifferences, LinearAlgebra

const FDM = FiniteDifferences
struct FDMBackend{A} <: AD.AbstractFiniteDifference
    alg::A
end
FDMBackend() = FDMBackend(central_fdm(5, 1))
const fdm_backend = FDMBackend()

# Minimal interface
function AD.jacobian(ab::FDMBackend, f, xs...)
    return jacobian(ab.alg, f, xs...)
end

fder(x, y) = exp(y) * x + y * log(x)
fgrad(x, y) = prod(x) + sum(y ./ (1:length(y)))
function fjac(x, y)
    x + Bidiagonal(-ones(length(y)) * 3, ones(length(y) - 1) / 2, :U) * y
end

const xscalar = rand()
const yscalar = rand()

const xvec = rand(5)
const yvec = rand(5)

@testset "AbstractDifferentiation.jl" begin
    @testset "FiniteDifferences" begin
        @testset "Derivative" begin
            der1 = AD.derivative(fdm_backend, fder, xscalar, yscalar)
            der2 = (
                fdm_backend.alg(x -> fder(x, yscalar), xscalar),
                fdm_backend.alg(y -> fder(xscalar, y), yscalar),
            )
            @test norm.(der1 .- der2) == (0, 0)
            valscalar, der3 = AD.value_and_derivative(fdm_backend, fder, xscalar, yscalar)
            @test valscalar == fder(xscalar, yscalar)
            @test der3 .- der1 == (0, 0)
        end
        @testset "Gradient" begin
            grad1 = AD.gradient(fdm_backend, fgrad, xvec, yvec)
            grad2 = FDM.grad(fdm_backend.alg, fgrad, xvec, yvec)
            @test norm.(grad1 .- grad2) == (0, 0)
            valscalar, grad3 = AD.value_and_gradient(fdm_backend, fgrad, xvec, yvec)
            @test valscalar == fgrad(xvec, yvec)
            @test norm.(grad3 .- grad1) == (0, 0)
        end
        @testset "Jacobian" begin
            jac1 = AD.jacobian(fdm_backend, fjac, xvec, yvec)
            jac2 = FDM.jacobian(fdm_backend.alg, fjac, xvec, yvec)
            @test norm.(jac1 .- jac2) == (0, 0)
            valvec, jac3 = AD.value_and_jacobian(fdm_backend, fjac, xvec, yvec)
            @test valvec == fjac(xvec, yvec)
            @test norm.(jac3 .- jac1) == (0, 0)
        end
        @testset "jvp" begin
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
        @testset "j′vp" begin
            w = rand(length(fjac(xvec, yvec)))
            pb1 = AD.pullback_function(fdm_backend, fjac, xvec, yvec)(w)
            pb2 = FDM.j′vp(fdm_backend.alg, fjac, w, xvec, yvec)
            @test all(norm.(pb1 .- pb2) .<= (1e-10, 1e-10))
            valvec, pb3 = AD.value_and_pullback_function(fdm_backend, fjac, xvec, yvec)(w)
            @test valvec == fjac(xvec, yvec)
            @test norm.(pb3 .- pb1) == (0, 0)
        end
    end
end
