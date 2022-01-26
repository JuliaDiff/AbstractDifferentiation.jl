using .ForwardDiff: ForwardDiff, DiffResults, StaticArrays

"""
    ForwardDiffBackend{CS}

AD backend that uses forward mode with ForwardDiff.jl.

The type parameter `CS` denotes the chunk size of the differentiation algorithm. If it is
`Nothing`, then ForwardiffDiff uses a heuristic to set the chunk size based on the input.

See also: [ForwardDiff.jl: Configuring Chunk Size](https://juliadiff.org/ForwardDiff.jl/dev/user/advanced/#Configuring-Chunk-Size)
"""
struct ForwardDiffBackend{CS} <: AbstractForwardMode end

"""
    ForwardDiffBackend(; chunksize::Union{Val,Nothing}=nothing)

Create an AD backend that uses forward mode with ForwardDiff.jl.

If the `chunksize` of the differentiation algorithm is set to `nothing` (the default), then
ForwarddDiff uses a heuristic to set the chunk size based on the input. Alternatively, if
`chunksize=Val{N}()`, then the chunk size is set to `N`.

See also: [ForwardDiff.jl: Configuring Chunk Size](https://juliadiff.org/ForwardDiff.jl/dev/user/advanced/#Configuring-Chunk-Size)
"""
function ForwardDiffBackend(; chunksize::Union{Val,Nothing}=nothing)
    return ForwardDiffBackend{getchunksize(chunksize)}()
end

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

getchunksize(::Nothing) = Nothing
getchunksize(::Val{N}) where {N} = N

chunk(::ForwardDiffBackend{Nothing}, x) = ForwardDiff.Chunk(x)
chunk(::ForwardDiffBackend{N}, _) where {N} = ForwardDiff.Chunk{N}()
