module AbstractDifferentiation

using LinearAlgebra, ExprTools

abstract type AbstractBackend end
abstract type AbstractFiniteDifference <: AbstractBackend end
abstract type AbstractForwardMode <: AbstractBackend end
abstract type AbstractReverseMode <: AbstractBackend end

struct HigherOrderBackend{B} <: AbstractBackend
    backends::B
end
reduce_order(b::AbstractBackend) = b
function reduce_order(b::HigherOrderBackend)
    if length(b.backends)==1 
        return lowest(b) # prevent zero tuple and subsequent error when reducing over HigherOrderBackend
    else
        return HigherOrderBackend(reverse(Base.tail(reverse(b.backends))))
    end
end
lowest(b::AbstractBackend) = b
lowest(b::HigherOrderBackend) = b.backends[end]
second_lowest(b::AbstractBackend) = b
second_lowest(b::HigherOrderBackend) = lowest(reduce_order(b))

# If the primal value is in y, extract it.
# Otherwise, re-compute it, e.g. in finite diff.
primal_value(::AbstractFiniteDifference, ::Any, f, xs) = f(xs...)
primal_value(::AbstractBackend, ys, ::Any, ::Any) = primal_value(ys)
primal_value(x::Tuple) = map(primal_value, x)
primal_value(x) = x

function derivative(ab::AbstractBackend, f, xs::Number...)
    der = getindex.(jacobian(lowest(ab), f, xs...), 1)
    if der isa Tuple
        return der
    else
        return (der,)
    end
end

function gradient(ab::AbstractBackend, f, xs...)
    return reshape.(adjoint.(jacobian(lowest(ab), f, xs...)),size.(xs))
end
function jacobian(ab::AbstractBackend, f, xs...) end
function jacobian(ab::HigherOrderBackend, f, xs...) 
    jacobian(lowest(ab), f, xs...)
end

function hessian(ab::AbstractBackend, f, x)
    if x isa Tuple
        # only support computation of Hessian for functions with single input argument
        x = only(x)
    end
    return jacobian(second_lowest(ab), x -> begin
        gradient(lowest(ab), f, x)[1] # gradient returns a tuple
    end, x)
end

function value_and_derivative(ab::AbstractBackend, f, xs::Number...)
    value, jacs = value_and_jacobian(lowest(ab), f, xs...)
    return value[1], getindex.(jacs, 1)
end
function value_and_gradient(ab::AbstractBackend, f, xs...)
    value, jacs = value_and_jacobian(lowest(ab), f, xs...)
    return value, reshape.(adjoint.(jacs),size.(xs))
end
function value_and_jacobian(ab::AbstractBackend, f, xs...)
    local value
    primalcalled = false
    if lowest(ab) isa AbstractFiniteDifference
        value = primal_value(ab, nothing, f, xs)
        primalcalled = true
    end
    jacs = jacobian(lowest(ab), (_xs...,) -> begin
        v = f(_xs...)
        if !primalcalled
            value = primal_value(ab, v, f, xs)
            primalcalled = true
        end
        return v
    end, xs...)

    return value, jacs
end
function value_and_hessian(ab::AbstractBackend, f, x)
    if x isa Tuple
        # only support computation of Hessian for functions with single input argument
        x = only(x)
    end

    local value
    primalcalled = false
    if ab isa AbstractFiniteDifference
        value = primal_value(ab, nothing, f, (x,))
        primalcalled = true
    end
    hess = jacobian(second_lowest(ab), _x -> begin
        v, g = value_and_gradient(lowest(ab), f, _x)
        if !primalcalled
            value = primal_value(ab, v, f, (x,))
            primalcalled = true
        end
        return g[1] # gradient returns a tuple
    end, x)
    return value, hess
end
function value_and_hessian(ab::HigherOrderBackend, f, x)
    if x isa Tuple
        # only support computation of Hessian for functions with single input argument
        x = only(x)
    end
    local value
    primalcalled = false
    hess = jacobian(second_lowest(ab), (_x,) -> begin
        v, g = value_and_gradient(lowest(ab), f, _x)
        if !primalcalled
            value = primal_value(ab, v, f, (x,))
            primalcalled = true
        end
        return g[1]  # gradient returns a tuple
    end, x)
    return value, hess
end
function value_gradient_and_hessian(ab::AbstractBackend, f, x)
    if x isa Tuple
        # only support computation of Hessian for functions with single input argument
        x = only(x)
    end
    local value
    primalcalled = false
    grads, hess = value_and_jacobian(second_lowest(ab), _x -> begin
        v, g = value_and_gradient(lowest(ab), f, _x)
        if !primalcalled
            value = primal_value(second_lowest(ab), v, f, (x,))
            primalcalled = true
        end
        return g[1] # gradient returns a tuple
    end, x)
    return value, (grads,), hess
end
function value_gradient_and_hessian(ab::HigherOrderBackend, f, x)
    if x isa Tuple
        # only support computation of Hessian for functions with single input argument
        x = only(x)
    end
    local value
    primalcalled = false
    grads, hess = value_and_jacobian(second_lowest(ab), _x -> begin
        v, g = value_and_gradient(lowest(ab), f, _x)
        if !primalcalled
            value = primal_value(second_lowest(ab), v, f, (x,))
            primalcalled = true
        end
        return g[1] # gradient returns a tuple
    end, x)
    return value, (grads,), hess
end

function pushforward_function(
    ab::AbstractBackend,
    f,
    xs...,
)
    return (ds) -> begin
        return jacobian(lowest(ab), (xds...,) -> begin
            if ds isa Tuple
                @assert length(xs) == length(ds)
                newxs = xs .+ ds .* xds
                return f(newxs...)
            else
                newx = only(xs) + ds * only(xds)
                return f(newx)
            end
        end, _zero.(xs, ds)...)
    end
end
function value_and_pushforward_function(
    ab::AbstractBackend,
    f,
    xs...,
)
    return (ds) -> begin
        if !(ds isa Tuple)
            ds = (ds,)    
        end
        @assert length(ds) == length(xs)
        local value
        primalcalled = false
        if ab isa AbstractFiniteDifference
            value = primal_value(ab, nothing, f, xs)
            primalcalled = true
        end
        pf = pushforward_function(lowest(ab), (_xs...,) -> begin
            vs = f(_xs...)
            if !primalcalled
                value = primal_value(lowest(ab), vs, f, xs)
                primalcalled = true
            end
            return vs
        end, xs...)(ds)
        
        return value, pf
    end
end

_zero(::Number, d::Number) = zero(d)
_zero(::Number, d::AbstractVector) = zero(d)
_zero(::AbstractVector, d::AbstractVector) = zero(eltype(d))
_zero(::AbstractVector, d::AbstractMatrix) = zero(similar(d, size(d, 2)))
_zero(::AbstractMatrix, d::AbstractMatrix) = zero(d)
_zero(::Any, d::Any) = zero(d)

@inline _dot(x, y) = dot(x, y)
@inline function _dot(x::AbstractVector, y::UniformScaling)
    return @inbounds dot(only(x), y.λ)
end
@inline function _dot(x::AbstractVector, y::AbstractMatrix)
    @assert size(y, 2) == 1
    return dot(x, y)
end

function pullback_function(ab::AbstractBackend, f, xs...)
    _, pbf = value_and_pullback_function(ab, f, xs...)
    return pbf
end
function value_and_pullback_function(
    ab::AbstractBackend,
    f,
    xs...,
)
    value = f(xs...)
    pbf = ws -> begin
        return gradient(lowest(ab), (_xs...,) -> begin
            vs = f(_xs...)
            if ws isa Tuple
                @assert length(vs) == length(ws)
                return sum(Base.splat(_dot), zip(ws, vs))
            else
                return _dot(vs, ws)
            end
        end, xs...)
    end
    return value, pbf
end

struct LazyDerivative{B, F, X}
    backend::B
    f::F
    xs::X
end

function Base.:*(d::LazyDerivative, y)
    return derivative(d.backend, d.f, d.xs...) * y
end

function Base.:*(y, d::LazyDerivative)
    return y * derivative(d.backend, d.f, d.xs...)
end

function Base.:*(d::LazyDerivative, y::Union{Number,Tuple})
    if y isa Tuple && d.xs isa Tuple
        @assert length(y) == length(d.xs) 
    end
    return derivative(d.backend, d.f, d.xs...) .* y
end

function Base.:*(y::Union{Number,Tuple}, d::LazyDerivative)
    if y isa Tuple && d.xs isa Tuple
        @assert length(y) == length(d.xs) 
    end
    return y .* derivative(d.backend, d.f, d.xs...)
end

function Base.:*(d::LazyDerivative, y::AbstractArray)
    return map((d)-> d*y, derivative(d.backend, d.f, d.xs...))
end

function Base.:*(y::AbstractArray, d::LazyDerivative)
    return map((d)-> y*d, derivative(d.backend, d.f, d.xs...))
end


struct LazyGradient{B, F, X}
    backend::B
    f::F
    xs::X
end
Base.:*(d::LazyGradient, y) = gradient(d.backend, d.f, d.xs...) * y
Base.:*(y, d::LazyGradient) = y * gradient(d.backend, d.f, d.xs...)

function Base.:*(d::LazyGradient, y::Union{Number,Tuple})
    if y isa Tuple && d.xs isa Tuple
        @assert length(y) == length(d.xs) 
    end
    if d.xs isa Tuple
        return gradient(d.backend, d.f, d.xs...) .* y
    else
        return gradient(d.backend, d.f, d.xs) .* y
    end
end

function Base.:*(y::Union{Number,Tuple}, d::LazyGradient)
    if y isa Tuple && d.xs isa Tuple
        @assert length(y) == length(d.xs) 
    end
    if d.xs isa Tuple
        return y .* gradient(d.backend, d.f, d.xs...)
    else
        return y .* gradient(d.backend, d.f, d.xs)
    end
end


struct LazyJacobian{B, F, X}
    backend::B
    f::F
    xs::X
end

function Base.:*(d::LazyJacobian, ys)
    if !(ys isa Tuple)
        ys = (ys, )
    end
    if d.xs isa Tuple
        vjp = pushforward_function(d.backend, d.f, d.xs...)(ys)
    else
        vjp = pushforward_function(d.backend, d.f, d.xs)(ys)
    end
    if vjp isa Tuple
        return vjp
    else
        return (vjp,)
    end
end

function Base.:*(ys, d::LazyJacobian)
    if ys isa Tuple
        ya = adjoint.(ys)
    else
        ya = adjoint(ys)
    end
    if d.xs isa Tuple
        return pullback_function(d.backend, d.f, d.xs...)(ya)
    else
        return pullback_function(d.backend, d.f, d.xs)(ya)
    end
end

function Base.:*(d::LazyJacobian, ys::Number)
    if d.xs isa Tuple
        return jacobian(d.backend, d.f, d.xs...) .* ys
    else
        return jacobian(d.backend, d.f, d.xs) .* ys
    end
end

function Base.:*(ys::Number, d::LazyJacobian)
    if d.xs isa Tuple
        return jacobian(d.backend, d.f, d.xs...) .* ys
    else
        return jacobian(d.backend, d.f, d.xs) .* ys
    end
end


struct LazyHessian{B, F, X}
    backend::B
    f::F
    xs::X
end

function Base.:*(d::LazyHessian, ys)
    if !(ys isa Tuple)
        ys = (ys, )
    end

    if d.xs isa Tuple
        res =  pushforward_function(
            second_lowest(d.backend),
            (xs...,) -> gradient(lowest(d.backend), d.f, xs...)[1], d.xs...,)(ys)  # [1] because gradient returns a tuple
    else
        res =  pushforward_function(
            second_lowest(d.backend),
            (xs,) -> gradient(lowest(d.backend), d.f, xs)[1],d.xs,)(ys)  # gradient returns a tuple
    end
    if res isa Tuple
        return res
    else
        return (res,)
    end
end

function Base.:*(ys, d::LazyHessian)
    if ys isa Tuple
        ya = adjoint.(ys)
    else
        ya = adjoint(ys)
    end
    if d.xs isa Tuple
        return pullback_function(
            second_lowest(d.backend),
            (xs...,) -> gradient(lowest(d.backend), d.f, xs...),
            d.xs...,
            )(ya)
    else
        return pullback_function(
            second_lowest(d.backend),
            (xs,) -> gradient(lowest(d.backend), d.f, xs)[1],
            d.xs,
            )(ya)
    end
end

function Base.:*(d::LazyHessian, ys::Number)
    if d.xs isa Tuple
        return hessian(d.backend, d.f, d.xs...).*ys
    else
        return hessian(d.backend, d.f, d.xs).*ys
    end
end

function Base.:*(ys::Number, d::LazyHessian)
    if d.xs isa Tuple
        return ys.*hessian(d.backend, d.f, d.xs...)
    else
        return ys.*hessian(d.backend, d.f, d.xs)
    end
end


function lazy_derivative(ab::AbstractBackend, f, xs::Number...)
    return LazyDerivative(ab, f, xs)
end
function lazy_gradient(ab::AbstractBackend, f, xs...)
    return LazyGradient(ab, f, xs)
end
function lazy_hessian(ab::AbstractBackend, f, xs...)
    return LazyHessian(ab, f, xs)
end
function lazy_jacobian(ab::AbstractBackend, f, xs...)
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
        return lazy_jacobian(d.ab, d.f, xs...)
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
        return lazy_hessian(h.ab, h.f, xs...)
    else
        return hessian(h.ab, h.f, xs...)
    end
end

macro primitive(expr)
    fdef = ExprTools.splitdef(expr)
    name = fdef[:name]
    if name == :pushforward_function
        return define_pushforward_function_and_friends(fdef) |> esc
    elseif name == :value_and_pullback_function
        return define_value_and_pullback_function_and_friends(fdef) |> esc
    elseif name == :jacobian
        return define_jacobian_and_friends(fdef) |> esc
    elseif name == :primal_value
        return define_primal_value(fdef) |> esc
    else
        throw("Unsupported AD primitive.")
    end
end

function define_pushforward_function_and_friends(fdef)
    fdef[:name] = :($(AbstractDifferentiation).pushforward_function)
    args = fdef[:args]
    funcs = quote
        $(ExprTools.combinedef(fdef))
        function $(AbstractDifferentiation).jacobian($(args...),)
            identity_like = $(identity_matrix_like)($(args[3:end]...),)
            pff = $(pushforward_function)($(args...),)
            if eltype(identity_like) <: Tuple{Vararg{Union{AbstractMatrix, Number}}}
                return map(identity_like) do identity_like_i
                    return mapreduce(hcat, $(_eachcol).(identity_like_i)...) do (cols...)
                        pff(cols)
                    end
                end
            elseif eltype(identity_like) <: AbstractMatrix
                # needed for the computation of the Hessian and Jacobian
                ret = hcat.(mapslices(identity_like[1], dims=1) do cols
                    # cols loop over basis states   
                    pf = pff((cols,))
                    if typeof(pf) <: AbstractVector
                        # to make the hcat. work / get correct matrix-like, non-flat output dimension
                        return (pf, )
                    else
                        return pf
                    end
                end ...)
                return ret isa Tuple ? ret : (ret,)

            else
                return pff(identity_like)
            end
        end
    end
    return funcs
end

function define_value_and_pullback_function_and_friends(fdef)
    fdef[:name] = :($(AbstractDifferentiation).value_and_pullback_function)
    args = fdef[:args]
    funcs = quote
        $(ExprTools.combinedef(fdef))
        function $(AbstractDifferentiation).jacobian($(args...),)
            value, pbf = $(value_and_pullback_function)($(args...),)
            identity_like = $(identity_matrix_like)(value)
            if eltype(identity_like) <: Tuple{Vararg{AbstractMatrix}}
                return map(identity_like) do identity_like_i
                    return mapreduce(vcat, $(_eachcol).(identity_like_i)...) do (cols...)
                        pbf(cols)'
                    end
                end
            elseif eltype(identity_like) <: AbstractMatrix
                # needed for Hessian computation:
                # value is a (grad,). Then, identity_like is a (matrix,).
                # cols loops over columns of the matrix  
                return vcat.(mapslices(identity_like[1], dims=1) do cols
                    adjoint.(pbf((cols,)))
                end ...)
            else
                return adjoint.(pbf(identity_like))
            end
        end
    end
    return funcs
end

_eachcol(a::Number) = (a,)
_eachcol(a) = eachcol(a)

function define_jacobian_and_friends(fdef)
    fdef[:name] = :($(AbstractDifferentiation).jacobian)
    return ExprTools.combinedef(fdef)
end

function define_primal_value(fdef)
    fdef[:name] = :($(AbstractDifferentiation).primal_value)
    return ExprTools.combinedef(fdef)
end

function identity_matrix_like(x)
    throw("The function `identity_matrix_like` is not defined for the type $(typeof(x)).")
end
function identity_matrix_like(x::AbstractVector)
    return (Matrix{eltype(x)}(I, length(x), length(x)),)
end
function identity_matrix_like(x::Number)
    return (one(x),)
end
identity_matrix_like(x::Tuple) = identity_matrix_like(x...)
@generated function identity_matrix_like(x...)
    expr = :(())
    for i in 1:length(x)
        push!(expr.args, :(()))
        for j in 1:i-1
            push!(expr.args[i].args, :((zero_matrix_like(x[$j])[1])))
        end
        push!(expr.args[i].args, :((identity_matrix_like(x[$i]))[1]))
        for j in i+1:length(x)
            push!(expr.args[i].args, :(zero_matrix_like(x[$j])[1]))
        end
    end
    return expr
end

zero_matrix_like(x::Tuple) = zero_matrix_like(x...)
zero_matrix_like(x...) = map(zero_matrix_like, x)
zero_matrix_like(x::AbstractVector) = (zero(similar(x, length(x), length(x))),)
zero_matrix_like(x::Number) = (zero(x),)
function zero_matrix_like(x)
    throw("The function `zero_matrix_like` is not defined for the type $(typeof(x)).")
end

@inline asarray(x) = [x]
@inline asarray(x::AbstractArray) = x

include("backends.jl")

# TODO: Replace with proper version
const EXTENSIONS_SUPPORTED = isdefined(Base, :get_extension)
if !EXTENSIONS_SUPPORTED
   using Requires: @require
   include("../ext/AbstractDifferentiationChainRulesCoreExt.jl")
end
@static if !EXTENSIONS_SUPPORTED
    function __init__()
        @require DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5" begin
            @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("../ext/AbstractDifferentiationForwardDiffExt.jl")
            @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("../ext/AbstractDifferentiationReverseDiffExt.jl")
        end
        @require FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000" include("../ext/AbstractDifferentiationFiniteDifferencesExt.jl")
        @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/AbstractDifferentiationTrackerExt.jl")
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("../ext/AbstractDifferentiationZygoteExt.jl")
    end
end

end
