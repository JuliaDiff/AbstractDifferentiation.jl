module AbstractDifferentiation

using LinearAlgebra

export AD

const AD = AbstractDifferentiation

abstract type AbstractBackend end
abstract type AbstractFiniteDifference <: AbstractBackend end
abstract type AbstractForwardMode <: AbstractBackend end
abstract type AbstractReverseMode <: AbstractBackend end

struct HigherOrderBackend{B} <: AbstractBackend
    backends::B
end
reduceorder(b::AbstractBackend) = b
function reduceorder(b::HigherOrderBackend)
    return HigherOrderBackend(reverse(Base.tail(reverse(b.backends))))
end
lowest(b::AbstractBackend) = b
lowest(b::HigherOrderBackend) = b.backends[end]
secondlowest(b::HigherOrderBackend) = lowest(reduceorder(b))

# If the primal value is in y, extract it.
# Otherwise, re-compute it, e.g. in finite diff.
primalvalue(::AbstractFiniteDifference, ::Any, f, xs) = f(xs...)
primalvalue(::AbstractBackend, ys, ::Any, ::Any) = primalvalue(ys)
primalvalue(x::Tuple) = map(primalvalue, x)
primalvalue(x) = x

function derivative(ab::AbstractBackend, f, xs::Number...)
    return getindex.(jacobian(lowest(ab), f, xs...), 1)
end
function gradient(ab::AbstractBackend, f, xs...)
    return adjoint.(jacobian(lowest(ab), f, xs...))
end
function jacobian(ab::AbstractBackend, f, xs...) end
function hessian(ab::AbstractBackend, f, xs...)
    return jacobian(secondlowest(ab), (xs...,) -> begin
        gradient(lowest(ab), f, xs...)
    end, xs...)
end

function value_and_derivative(ab::AbstractBackend, f, xs::Number...)
    value, jacs = value_and_jacobian(lowest(ab), f, xs...)
    return value[1], getindex.(jacs, 1)
end
function value_and_gradient(ab::AbstractBackend, f, xs...)
    value, jacs = value_and_jacobian(lowest(ab), f, xs...)
    return value, adjoint.(jacs)
end
function value_and_jacobian(ab::AbstractBackend, f, xs...)
    local value
    primalcalled = false
    jacs = jacobian(lowest(ab), (_xs...,) -> begin
        v = f(_xs...)
        if !primalcalled
            value = primalvalue(ab, v, f, xs)
            primalcalled = true
        end
        return v
    end, xs...)
    return value, jacs
end
function value_and_hessian(ab::AbstractBackend, f, xs...)
    local value
    primalcalled = false
    hess = jacobian(secondlowest(ab), (_xs...,) -> begin
        v, g = value_and_gradient(lowest(ab), f, _xs...)
        if !primalcalled
            value = primalvalue(ab, v, f, xs)
            primalcalled = true
        end
        return g
    end, xs...)
    return value, hess
end
function value_and_hessian(ab::HigherOrderBackend, f, xs...)
    local value
    primalcalled = false
    hess = jacobian(secondlowest(ab), (_xs...,) -> begin
        v, g = value_and_gradient(lowest(ab), f, _xs...)
        if !primalcalled
            value = primalvalue(ab, v, f, xs)
            primalcalled = true
        end
        return g
    end, xs...)
    return value, hess
end
function value_gradient_and_hessian(ab::AbstractBackend, f, xs...)
    local value
    primalcalled = false
    grads, hess = value_and_jacobian(secondlowest(ab), (_xs...,) -> begin
        v, g = value_and_gradient(lowest(ab), f, _xs...)
        if !primalcalled
            value = primalvalue(secondlowest(ab), v, f, xs)
            primalcalled = true
        end
        return g
    end, xs...)
    return value, grads, hess
end
function value_gradient_and_hessian(ab::HigherOrderBackend, f, xs...)
    local value
    primalcalled = false
    grads, hess = value_and_jacobian(secondlowest(ab), (_xs...,) -> begin
        v, g = value_and_gradient(lowest(ab), f, _xs...)
        if !primalcalled
            value = primalvalue(secondlowest(ab), v, f, xs)
            primalcalled = true
        end
        return g
    end, xs...)
    return value, grads, hess
end

function pushforward_function(
    ab::AbstractBackend,
    f,
    xs::Union{Number, AbstractArray{<:Number}}...,
)
    return (ds) -> begin
        return jacobian(lowest(ab), (xds...,) -> begin
            if ds isa Tuple
                @assert length(xs) == length(ds)
                newxs = xs .+ ds .* xds
                return f(newxs...)
            else
                @assert length(xs) == length(xds) == 1
                newx = xs[1] + ds * xds[1]
                return f(newx)
            end
        end, _zero.(xs, ds)...)
    end
end
function value_and_pushforward_function(
    ab::AbstractBackend,
    f,
    xs::Union{Number, AbstractArray{<:Number}}...,
)
    return (ds) -> begin
        @assert ds isa Tuple && length(ds) == length(xs)
        return value_and_jacobian(lowest(ab), (xds...,) -> begin
            if ds isa Tuple
                @assert length(xs) == length(ds)
                newxs = xs .+ ds .* xds
                return f(newxs...)
            else
                @assert length(xs) == length(xds) == 1
                newx = xs[1] + ds * xds[1]
                return f(newx)
            end
        end, _zero.(xs, ds)...)
    end
end

_zero(::Number, d::Number) = zero(d)
_zero(::Number, d::AbstractVector) = zero(d)
_zero(::AbstractVector, d::AbstractVector) = zero(eltype(d))
_zero(::AbstractVector, d::AbstractMatrix) = zero(similar(d, size(d, 2)))
_zero(::AbstractMatrix, d::AbstractMatrix) = zero(d)
_zero(::Any, d::Any) = zero(d)

function pullback_function(ab::AbstractBackend, f, xs...)
    return (ws) -> begin
        jacs = jacobian(lowest(ab), (xs...,) -> begin
            vs = f(xs...)
            if ws isa Tuple
                @assert length(vs) == length(ws)
                return sum(zip(vs, ws)) do v, w
                    if w isa Union{AbstractMatrix, UniformScaling} && v isa AbstractVector
                        return w' * v
                    else
                        # for arbitrary arrays
                        return dot(w, v)
                    end
                end
            else
                w, v = ws, vs
                if w isa Union{AbstractMatrix, UniformScaling} && v isa AbstractVector
                    return w' * v
                else
                    # for arbitrary arrays
                    return dot(w, v)
                end
            end
        end, xs...)
        return adjoint.(jacs)
    end
end
function value_and_pullback_function(
    ab::AbstractBackend,
    f,
    xs...,
)
    return (ws) -> begin
        local value
        primalcalled = false
        jacs = jacobian(lowest(ab), (_xs...,) -> begin
            vs = f(_xs...)
            if !primalcalled
                value = primalvalue(lowest(ab), vs, f, xs)
                primalcalled = true
            end
            if ws isa Tuple
                @assert length(vs) == length(ws)
                return sum(zip(vs, ws)) do v, w
                    if w isa Union{AbstractMatrix, UniformScaling} && v isa AbstractVector
                        return w' * v
                    else
                        # for arbitrary arrays
                        return dot(w, v)
                    end
                end
            else
                w, v = ws, vs
                if w isa Union{AbstractMatrix, UniformScaling} && v isa AbstractVector
                    return w' * v
                else
                    # for arbitrary arrays
                    return dot(w, v)
                end
            end
        end, xs...)
        return value, adjoint.(jacs)
    end
end

struct LazyDerivative{B, F, X}
    backend::B
    f::F
    xs::X
end
function Base.:*(d::LazyDerivative, y)
    return derivative(d.ab, d.f, d.xs...) * y
end
function Base.:*(y, d::LazyDerivative)
    return y * derivative(d.ab, d.f, d.xs...)
end

struct LazyGradient{B, F, X}
    backend::B
    f::F
    xs::X
end
Base.:*(d::LazyGradient, y) = gradient(d.ab, d.f, d.xs...) * y
Base.:*(y, d::LazyGradient) = y * gradient(d.ab, d.f, d.xs...)

struct LazyJacobian{B, F, X}
    backend::B
    f::F
    xs::X
end
function Base.:*(d::LazyJacobian, ys)
    return pushforward_function(d.ab, d.f, d.xs...)(ys)
end
function Base.:*(ys, d::LazyJacobian)
    if ys isa Tuple
        ya = adjoint.(ys)
    else
        ya = adjoint(ys)
    end
    return pullback_function(d.ab, d.f, d.xs...)(ya)
end

struct LazyHessian{B, F, X}
    backend::B
    f::F
    xs::X
end
function Base.:*(d::LazyHessian, ys)
    return pushforward_function(
        secondlowest(d.ab),
        (xs...,) -> gradient(lowest(d.ab), d.f, xs...),
        d.xs...,
    )(ys)
end
function Base.:*(ys, d::LazyHessian)
    if ys isa Tuple
        ya = adjoint.(ys)
    else
        ya = adjoint(ys)
    end
    return pullback_function(
        secondlowest(d.ab),
        (xs...,) -> gradient(lowest(d.ab), d.f, xs...),
        d.xs...,
    )(ya)
end

function lazyderivative(ab::AbstractBackend, f, xs::Number...)
    return LazyDerivative(ab, f, xs)
end
function lazygradient(ab::AbstractBackend, f, xs...)
    return LazyGradient(ab, f, xs)
end
function lazyhessian(ab::AbstractBackend, f, xs...)
    return LazyHessian(ab, f, xs)
end
function lazyjacobian(ab::AbstractBackend, f, xs...)
    return LazyJacobian(ab, f, xs)
end

struct D{B, F}
    backend::B
    f::F
end
D(b::AbstractBackend, d::D) = H(HigherOrderBackend((b, d.b)), d.f)
D(d::D) = H(HigherOrderBackend((d.backend, d.backend)), d.f)
function (d::D)(xs...; lazy = true)
    if lazy
        return lazyjacobian(d.ab, d.f, xs...)
    else
        return jacobian(d.ab, d.f, xs...)
    end
end

struct H{B, F}
    backend::B
    f::F
end
function (h::H)(xs...; lazy = true)
    if lazy
        return lazyhessian(h.ab, h.f, xs...)
    else
        return hessian(h.ab, h.f, xs...)
    end
end

end