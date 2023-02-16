module AbstractDifferentiationReverseDiffExt

using AbstractDifferentiation: AbstractDifferentiation, asarray, EXTENSIONS_SUPPORTED, ReverseDiffBackend
if EXTENSIONS_SUPPORTED
    using ReverseDiff: ReverseDiff, DiffResults
else
    using ..ReverseDiff: ReverseDiff, DiffResults
end
if VERSION < v"1.4.0-DEV.142"
    using Compat: only
end

const AD = AbstractDifferentiation

AD.primal_value(x::ReverseDiff.TrackedReal) = ReverseDiff.value(x)
AD.primal_value(x::AbstractArray{<:ReverseDiff.TrackedReal}) = ReverseDiff.value.(x)
AD.primal_value(x::ReverseDiff.TrackedArray) = ReverseDiff.value(x)

AD.@primitive function jacobian(ba::ReverseDiffBackend, f, xs...)
    xs_arr = map(asarray, xs)
    tape = ReverseDiff.JacobianTape(xs_arr) do (xs_arr...)
        xs_new = map(xs, xs_arr) do x, x_arr
            return x isa Number ? only(x_arr) : x_arr
        end
        return asarray(f(xs_new...))
    end
    results = ReverseDiff.jacobian!(tape, xs_arr)
    return map(xs, results) do x, result
        return x isa Number ? vec(result) : result
    end
end
function AD.jacobian(ba::ReverseDiffBackend, f, xs::AbstractArray...)
    return ReverseDiff.jacobian(asarray ∘ f, xs)
end

function AD.derivative(ba::ReverseDiffBackend, f, xs::Number...)
    tape = ReverseDiff.InstructionTape()
    xs_tracked = ReverseDiff.TrackedReal.(xs, zero.(xs), Ref(tape))
    y_tracked = f(xs_tracked...)
    ReverseDiff.seed!(y_tracked)
    ReverseDiff.reverse_pass!(tape)
    return ReverseDiff.deriv.(xs_tracked)
end

function AD.gradient(ba::ReverseDiffBackend, f, xs::AbstractArray...)
    return ReverseDiff.gradient(f, xs)
end

function AD.hessian(ba::ReverseDiffBackend, f, x::AbstractArray)
    return (ReverseDiff.hessian(f, x),)
end

function AD.value_and_gradient(ba::ReverseDiffBackend, f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    cfg = ReverseDiff.GradientConfig(x)
    ReverseDiff.gradient!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_hessian(ba::ReverseDiffBackend, f, x)
    result = DiffResults.HessianResult(x)
    cfg = ReverseDiff.HessianConfig(result, x)
    ReverseDiff.hessian!(result, f, x, cfg)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

end # module
