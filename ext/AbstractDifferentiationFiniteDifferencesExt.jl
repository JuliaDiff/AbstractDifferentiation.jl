module AbstractDifferentiationFiniteDifferencesExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using FiniteDifferences: FiniteDifferences
else
    import ..AbstractDifferentiation as AD
    using ..FiniteDifferences: FiniteDifferences
end

"""
    FiniteDifferencesBackend(method=FiniteDifferences.central_fdm(5, 1))

Create an AD backend that uses forward mode with FiniteDifferences.jl.
"""
AD.FiniteDifferencesBackend() = AD.FiniteDifferencesBackend(FiniteDifferences.central_fdm(5, 1))

AD.@primitive function jacobian(ba::AD.FiniteDifferencesBackend, f, xs...)
    return FiniteDifferences.jacobian(ba.method, f, xs...)
end

function AD.gradient(ba::AD.FiniteDifferencesBackend, f, xs...)
    return FiniteDifferences.grad(ba.method, f, xs...)
end

function AD.pushforward_function(ba::AD.FiniteDifferencesBackend, f, xs...)
    return function pushforward(vs)
        ws = FiniteDifferences.jvp(ba.method, f, tuple.(xs, vs)...)
        return length(xs) == 1 ? (ws,) : ws
    end
end

function AD.pullback_function(ba::AD.FiniteDifferencesBackend, f, xs...)
    function pullback(vs)
        return FiniteDifferences.j′vp(ba.method, f, vs, xs...)
    end
end

# Ensure consistency with `value_and_pullback` function
function AD.value_and_pullback_function(ba::AD.FiniteDifferencesBackend, f, xs...)
    value = f(xs...)
    function value_and_pullback(vs)
        return value, FiniteDifferences.j′vp(ba.method, f, vs, xs...)
    end
end

end # module
