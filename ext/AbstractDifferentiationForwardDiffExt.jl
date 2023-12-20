module AbstractDifferentiationForwardDiffExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using DiffResults: DiffResults
    using ForwardDiff: ForwardDiff
else
    import ..AbstractDifferentiation as AD
    using ..DiffResults: DiffResults
    using ..ForwardDiff: ForwardDiff
end

"""
    ForwardDiffBackend(; chunksize::Union{Val,Nothing}=nothing)

Create an AD backend that uses forward mode with ForwardDiff.jl.

If the `chunksize` of the differentiation algorithm is set to `nothing` (the default), then
ForwarddDiff uses a heuristic to set the chunk size based on the input. Alternatively, if
`chunksize=Val{N}()`, then the chunk size is set to `N`.

See also: [ForwardDiff.jl: Configuring Chunk Size](https://juliadiff.org/ForwardDiff.jl/dev/user/advanced/#Configuring-Chunk-Size)
"""
function AD.ForwardDiffBackend(; chunksize::Union{Val,Nothing}=nothing)
    return AD.ForwardDiffBackend{getchunksize(chunksize)}()
end

AD.@primitive function pushforward_function(ba::AD.ForwardDiffBackend, f, xs...)
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

function AD.derivative(::AD.ForwardDiffBackend, f, x::Real)
    return (ForwardDiff.derivative(f, x),)
end

function AD.gradient(ba::AD.ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.GradientConfig(f, x, chunk(ba, x))
    return (ForwardDiff.gradient(f, x, cfg),)
end

function AD.jacobian(ba::AD.ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.JacobianConfig(AD.asarray ∘ f, x, chunk(ba, x))
    return (ForwardDiff.jacobian(AD.asarray ∘ f, x, cfg),)
end
AD.jacobian(::AD.ForwardDiffBackend, f, x::Real) = (ForwardDiff.derivative(f, x),)

function AD.hessian(ba::AD.ForwardDiffBackend, f, x::AbstractArray)
    cfg = ForwardDiff.HessianConfig(f, x, chunk(ba, x))
    return (ForwardDiff.hessian(f, x, cfg),)
end

function AD.value_and_derivative(::AD.ForwardDiffBackend, f, x::Real)
    T = typeof(ForwardDiff.Tag(f, typeof(x)))
    ydual = f(ForwardDiff.Dual{T}(x, one(x)))
    return ForwardDiff.value(T, ydual), (ForwardDiff.extract_derivative(T, ydual),)
end

function AD.value_and_gradient(ba::AD.ForwardDiffBackend, f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    cfg = ForwardDiff.GradientConfig(f, x, chunk(ba, x))
    ForwardDiff.gradient!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_second_derivative(ba::AD.ForwardDiffBackend, f, x::Real)
    T = typeof(ForwardDiff.Tag(f, typeof(x)))
    ydual, ddual = AD.value_and_derivative(ba, f, ForwardDiff.Dual{T}(x, one(x)))
    return ForwardDiff.value(T, ydual), (ForwardDiff.extract_derivative(T, ddual[1]),)
end

function AD.value_and_hessian(ba::AD.ForwardDiffBackend, f, x)
    result = DiffResults.HessianResult(x)
    cfg = ForwardDiff.HessianConfig(f, result, x, chunk(ba, x))
    ForwardDiff.hessian!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

function AD.value_and_derivatives(ba::AD.ForwardDiffBackend, f, x::Real)
    T = typeof(ForwardDiff.Tag(f, typeof(x)))
    ydual, ddual = AD.value_and_derivative(ba, f, ForwardDiff.Dual{T}(x, one(x)))
    return ForwardDiff.value(T, ydual),
    (ForwardDiff.value(T, ddual[1]),),
    (ForwardDiff.extract_derivative(T, ddual[1]),)
end

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v

getchunksize(::Nothing) = Nothing
getchunksize(::Val{N}) where {N} = N

chunk(::AD.ForwardDiffBackend{Nothing}, x) = ForwardDiff.Chunk(x)
chunk(::AD.ForwardDiffBackend{N}, _) where {N} = ForwardDiff.Chunk{N}()

end # module
