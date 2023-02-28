module AbstractDifferentiationForwardDiffExt

using AbstractDifferentiation: AbstractDifferentiation, asarray, EXTENSIONS_SUPPORTED, ForwardDiffBackend
if EXTENSIONS_SUPPORTED
    using DiffResults: DiffResults
    using ForwardDiff: ForwardDiff
else
    using ..DiffResults: DiffResults
    using ..ForwardDiff: ForwardDiff
end

const AD = AbstractDifferentiation

"""
    ForwardDiffBackend(; chunksize::Union{Val,Nothing}=nothing)

Create an AD backend that uses forward mode with ForwardDiff.jl.

If the `chunksize` of the differentiation algorithm is set to `nothing` (the default), then
ForwarddDiff uses a heuristic to set the chunk size based on the input. Alternatively, if
`chunksize=Val{N}()`, then the chunk size is set to `N`.

See also: [ForwardDiff.jl: Configuring Chunk Size](https://juliadiff.org/ForwardDiff.jl/dev/user/advanced/#Configuring-Chunk-Size)
"""
function AD.ForwardDiffBackend(; chunksize::Union{Val,Nothing}=nothing)
    return ForwardDiffBackend{getchunksize(chunksize)}()
end

AD.@primitive function pushforward_function(ba::ForwardDiffBackend, f, xs...)
    return function pushforward(vs)
        if length(xs) == 1
            v = vs isa Tuple ? only(vs) : vs
            return (ForwardDiff.derivative(h -> f(step_toward(xs[1], v, h)), 0),)
        else
            return ForwardDiff.derivative(h -> f(step_toward.(xs, vs, h)...), 0)
        end
    end
end

AD.primal_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
AD.primal_value(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)

# these implementations are more efficient than the fallbacks

function AD.gradient(ba::ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.GradientConfig(f, x, chunk(ba, x))
    return (ForwardDiff.gradient(f, x, cfg),)
end

function AD.jacobian(ba::ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.JacobianConfig(asarray ∘ f, x, chunk(ba, x))
    return (ForwardDiff.jacobian(asarray ∘ f, x, cfg),)
end
AD.jacobian(::ForwardDiffBackend, f, x::Number) = (ForwardDiff.derivative(f, x),)

function AD.hessian(ba::ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.HessianConfig(f, x, chunk(ba, x))
    return (ForwardDiff.hessian(f, x, cfg),)
end

function AD.value_and_gradient(ba::ForwardDiffBackend, f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    cfg = ForwardDiff.GradientConfig(f, x, chunk(ba, x))
    ForwardDiff.gradient!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_hessian(ba::ForwardDiffBackend, f, x)
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

end # module
