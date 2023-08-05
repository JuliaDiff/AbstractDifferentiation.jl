module AbstractDifferentiationEnzymeExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using Enzyme: Enzyme
else
    import ..AbstractDifferentiation as AD
    using ..Enzyme: Enzyme
end

AD.@primitive function jacobian(b::AD.EnzymeForwardBackend, f, x)
    val = f(x)
    if val isa Real
        return adjoint.(AD.gradient(b, f, x))
    else
        if length(x) == 1 && length(val) == 1
            # Enzyme.jacobian returns a vector of length 1 in this case
            return (Matrix(adjoint(Enzyme.jacobian(Enzyme.Forward, f, x))),)
        else
            return (Enzyme.jacobian(Enzyme.Forward, f, x),)
        end
    end
end
function AD.jacobian(b::AD.EnzymeForwardBackend, f, x::Real)
    return AD.derivative(b, f, x)
end
function AD.gradient(::AD.EnzymeForwardBackend, f, x::AbstractArray)
    # Enzyme.gradient with Forward returns a tuple of the same length as the input
    return ([Enzyme.gradient(Enzyme.Forward, f, x)...],)
end
function AD.gradient(b::AD.EnzymeForwardBackend, f, x::Real)
    return AD.derivative(b, f, x)
end
function AD.derivative(::AD.EnzymeForwardBackend, f, x::Number)
    (Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated(x, one(x)))[1],)
end

AD.@primitive function jacobian(::AD.EnzymeReverseBackend, f, x)
    val = f(x)
    if val isa Real
        return (adjoint(Enzyme.gradient(Enzyme.Reverse, f, x)),)
    else
        if length(x) == 1 && length(val) == 1
            # Enzyme.jacobian returns an adjoint vector of length 1 in this case
            return (Matrix(Enzyme.jacobian(Enzyme.Reverse, f, x, Val(1))),)
        else
            return (Enzyme.jacobian(Enzyme.Reverse, f, x, Val(length(val))),)
        end
    end
end
function AD.gradient(::AD.EnzymeReverseBackend, f, x::AbstractArray)
    dx = similar(x)
    Enzyme.gradient!(Enzyme.Reverse, dx, f, x)
    return (dx,)
end
function AD.derivative(::AD.EnzymeReverseBackend, f, x::Number)
    (Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active(x))[1][1],)
end

end # module
