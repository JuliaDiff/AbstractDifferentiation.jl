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
    y = f(x)
    if y isa Number
        return (ForwardDiff.jacobian(Base.vect ∘ f, x),)
    else
        return (ForwardDiff.jacobian(f, x),)
    end
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
