using .ReverseDiff: ReverseDiff

primal_value(x::ReverseDiff.TrackedReal) = ReverseDiff.value(x)

struct ReverseDiffBackend <: AbstractBackend end

@primitive function jacobian(ba::ReverseDiffBackend, f, xs...)
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
