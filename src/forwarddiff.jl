using .ForwardDiff: ForwardDiff, DiffResults, StaticArrays

struct ForwardDiffBackend{CS} <: AbstractBackend
    ForwardDiffBackend{CS}() where {CS} = new{CS}()
end
ForwardDiffBackend(; chunksize=nothing) = ForwardDiffBackend{getchunksize(chunksize)}()

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

function gradient(ba::ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.GradientConfig(f, x, chunk(ba, x))
    return (ForwardDiff.gradient(f, x, cfg),)
end

function jacobian(ba::ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.JacobianConfig(asarray ∘ f, x, chunk(ba, x))
    return (ForwardDiff.jacobian(asarray ∘ f, x, cfg),)
end
jacobian(::ForwardDiffBackend, f, x::Number) = (ForwardDiff.derivative(f, x),)

function hessian(ba::ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.HessianConfig(f, x, chunk(ba, x))
    return (ForwardDiff.hessian(f, x, cfg),)
end

function value_and_gradient(ba::ForwardDiffBackend, f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    cfg = ForwardDiff.GradientConfig(f, x, chunk(ba, x))
    ForwardDiff.gradient!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function value_and_hessian(ba::ForwardDiffBackend, f, x)
    result = DiffResults.HessianResult(x)
    cfg = ForwardDiff.HessianConfig(f, result, x, chunk(ba, x))
    ForwardDiff.hessian!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v

@inline asarray(x) = [x]
@inline asarray(x::AbstractArray) = x

getchunksize(_) = Nothing
getchunksize(::Val{N}) where {N} = N

chunk(::ForwardDiffBackend{Nothing}, x) = ForwardDiff.Chunk(x)
chunk(::ForwardDiffBackend{N}, _) where {N} = ForwardDiff.Chunk{N}()
