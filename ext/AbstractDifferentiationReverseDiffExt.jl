module AbstractDifferentiationReverseDiffExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using DiffResults: DiffResults
    using ReverseDiff: ReverseDiff
else
    using ..DiffResults: DiffResults
    using ..ReverseDiff: ReverseDiff
end

AD.primal_value(x::ReverseDiff.TrackedReal) = ReverseDiff.value(x)
AD.primal_value(x::AbstractArray{<:ReverseDiff.TrackedReal}) = ReverseDiff.value.(x)
AD.primal_value(x::ReverseDiff.TrackedArray) = ReverseDiff.value(x)

AD.@primitive function jacobian(::AD.ReverseDiffBackend, f, xs...)
    xs_arr = map(AD.asarray, xs)
    tape = ReverseDiff.JacobianTape(xs_arr) do (xs_arr...)
        xs_new = map(xs, xs_arr) do x, x_arr
            return x isa Number ? only(x_arr) : x_arr
        end
        return AD.asarray(f(xs_new...))
    end
    results = ReverseDiff.jacobian!(tape, xs_arr)
    return map(xs, results) do x, result
        return x isa Number ? vec(result) : result
    end
end
function AD.jacobian(::AD.ReverseDiffBackend, f, xs::AbstractArray...)
    return ReverseDiff.jacobian(AD.asarray ∘ f, xs)
end

function AD.derivative(::AD.ReverseDiffBackend, f, xs::Number...)
    tape = ReverseDiff.InstructionTape()
    xs_tracked = ReverseDiff.TrackedReal.(xs, zero.(xs), Ref(tape))
    y_tracked = f(xs_tracked...)
    ReverseDiff.seed!(y_tracked)
    ReverseDiff.reverse_pass!(tape)
    return ReverseDiff.deriv.(xs_tracked)
end

function AD.gradient(::AD.ReverseDiffBackend, f, xs::AbstractArray...)
    return ReverseDiff.gradient(f, xs)
end

function AD.hessian(::AD.ReverseDiffBackend, f, x::AbstractArray)
    return (ReverseDiff.hessian(f, x),)
end

function AD.value_and_gradient(::AD.ReverseDiffBackend, f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    cfg = ReverseDiff.GradientConfig(x)
    ReverseDiff.gradient!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_hessian(::AD.ReverseDiffBackend, f, x)
    result = DiffResults.HessianResult(x)
    cfg = ReverseDiff.HessianConfig(result, x)
    ReverseDiff.hessian!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

end # module
