module FiniteDifferencesExt

using AbstractDifferentiation: AbstractDifferentiation, EXTENSIONS_SUPPORTED, FiniteDifferencesBackend
if EXTENSIONS_SUPPORTED
    using FiniteDifferences: FiniteDifferences
else
    using ..FiniteDifferences: FiniteDifferences
end

const AD = AbstractDifferentiation

"""
    FiniteDifferencesBackend(method=FiniteDifferences.central_fdm(5, 1))

Create an AD backend that uses forward mode with FiniteDifferences.jl.
"""
AD.FiniteDifferencesBackend() = FiniteDifferencesBackend(FiniteDifferences.central_fdm(5, 1))

AD.@primitive function jacobian(ba::FiniteDifferencesBackend, f, xs...)
    return FiniteDifferences.jacobian(ba.method, f, xs...)
end

function AD.pushforward_function(ba::FiniteDifferencesBackend, f, xs...)
    return function pushforward(vs)
        ws = FiniteDifferences.jvp(ba.method, f, tuple.(xs, vs)...)
        return length(xs) == 1 ? (ws,) : ws
    end
end

function AD.pullback_function(ba::FiniteDifferencesBackend, f, xs...)
    function pullback(vs)
        return FiniteDifferences.j′vp(ba.method, f, vs, xs...)
    end
end

end # module
