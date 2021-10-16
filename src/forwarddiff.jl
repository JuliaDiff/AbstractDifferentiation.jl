using .ForwardDiff: ForwardDiff, DiffResults, StaticArrays

struct ForwardDiffBackend <: AbstractBackend end

@primitive function pushforward_function(ba::ForwardDiffBackend, f, xs...)
    return function pushforward(vs)
        if length(xs) == 1
            v = vs isa Tuple ? only(vs) : vs
            return (ForwardDiff.derivative(h -> f(step_toward(xs[1], v, h)), 0),)
        else
            return ForwardDiff.derivative(h -> f(step_toward.(xs, vs, h)...), 0)
        end
    end
end

primal_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# these implementations are more efficient than the fallbacks

gradient(::ForwardDiffBackend, f, x::AbstractArray) = (ForwardDiff.gradient(f, x),)

function jacobian(ba::ForwardDiffBackend, f, x::AbstractArray)
    return (ForwardDiff.jacobian(asarray ∘ f, x),)
end
jacobian(::ForwardDiffBackend, f, x::Number) = (ForwardDiff.derivative(f, x),)

hessian(::ForwardDiffBackend, f, x::AbstractArray) = (ForwardDiff.hessian(f, x),)

function value_and_gradient(::ForwardDiffBackend, f, x::AbstractArray)
    result = ForwardDiff.gradient!(DiffResults.GradientResult(x), f, x)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function value_and_hessian(::ForwardDiffBackend, f, x)
    result = ForwardDiff.hessian!(DiffResults.HessianResult(x), f, x)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v

@inline asarray(x) = [x]
@inline asarray(x::AbstractArray) = x
