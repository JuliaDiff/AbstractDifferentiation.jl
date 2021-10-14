using .ForwardDiff: ForwardDiff, DiffResults, StaticArrays

struct ForwardDiffBackend <: AbstractBackend end

@primitive function pushforward_function(ba::ForwardDiffBackend, f, xs...)
    return function pushforward(vs)
        return ForwardDiff.derivative(h -> f(step_toward.(xs, vs, h)...), 0)
    end
end
function pushforward_function(::ForwardDiffBackend, f, x)
    return function pushforward(v)
        if v isa Tuple
            @assert length(v) == 1
            return (ForwardDiff.derivative(h -> f(step_toward(x, v[1], h)), 0),)
        else
            return (ForwardDiff.derivative(h -> f(step_toward(x, v, h)), 0),)
        end
    end
end

primal_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# these implementations are more efficient than the fallbacks

gradient(::ForwardDiffBackend, f, x::AbstractArray) = (ForwardDiff.gradient(f, x),)

function jacobian(ba::ForwardDiffBackend, f, x::AbstractArray)
    return value_and_jacobian(ba, f, x)[2]
end
jacobian(::ForwardDiffBackend, f, x::Number) = (ForwardDiff.derivative(f, x),)

hessian(::ForwardDiffBackend, f, x::AbstractArray) = (ForwardDiff.hessian(f, x),)

function value_and_gradient(::ForwardDiffBackend, f, x::AbstractArray)
    result = ForwardDiff.gradient!(DiffResults.GradientResult(x), f, x)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function value_and_jacobian(::ForwardDiffBackend, f, xs::AbstractArray)
    y = f(xs)
    if y isa Number
        return y, (adjoint(ForwardDiff.gradient(f, xs)),)
    else
        return y, (ForwardDiff.jacobian(f, xs),)
    end
end
function value_and_jacobian(::ForwardDiffBackend, f, x::Number)
    result = ForwardDiff.derivative!(DiffResults.DiffResult(x, x), f, x)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end
function value_and_jacobian(::ForwardDiffBackend, f, xs::Number...)
    xs_vec = StaticArrays.SVector(xs...)
    result = ForwardDiff.gradient!(DiffResults.GradientResult(xs_vec), xs -> f(xs...), xs_vec)
    return DiffResults.value(result), Tuple(DiffResults.derivative(result))
end

function value_and_hessian(::ForwardDiffBackend, f, x)
    result = ForwardDiff.hessian!(DiffResults.HessianResult(x), f, x)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

function value_gradient_and_hessian(::ForwardDiffBackend, f, x)
    result = ForwardDiff.hessian!(DiffResults.HessianResult(x), f, x)
    return (
        DiffResults.value(result),
        (DiffResults.gradient(result),),
        (DiffResults.hessian(result),),
    )
end

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v
