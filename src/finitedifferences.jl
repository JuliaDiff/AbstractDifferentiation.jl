using .FiniteDifferences: FiniteDifferences

"""
    FiniteDifferencesBackend{M}

AD backend that uses forward mode with FiniteDifferences.jl.

The type parameter `M` is the type of the method used to perform finite differences.
"""
struct FiniteDifferencesBackend{M} <: AbstractFiniteDifference
    method::M
end

"""
    FiniteDifferencesBackend(method=FiniteDifferences.central_fdm(5, 1))

Create an AD backend that uses forward mode with FiniteDifferences.jl.
"""
FiniteDifferencesBackend() = FiniteDifferencesBackend(FiniteDifferences.central_fdm(5, 1))

@primitive function jacobian(ba::FiniteDifferencesBackend, f, xs...)
    return FiniteDifferences.jacobian(ba.method, f, xs...)
end

derivative(ba::FiniteDifferencesBackend, f, x::Number) = ba.method(f, x)

function pushforward_function(ba::FiniteDifferencesBackend, f, xs...)
    return function pushforward(vs)
        ws = FiniteDifferences.jvp(ba.method, f, tuple.(xs, vs)...)
        return length(xs) == 1 ? (ws,) : ws
    end
end

function pullback_function(ba::FiniteDifferencesBackend, f, xs...)
    function pullback(vs)
        return FiniteDifferences.j′vp(ba.method, f, vs, xs...)
    end
end
