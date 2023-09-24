module AbstractDifferentiationEnzymeExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using Enzyme: Enzyme
else
    import ..AbstractDifferentiation as AD
    using ..Enzyme: Enzyme
end

struct Mutating{F}
    f::F
end
function (f::Mutating)(y, xs...)
    y .= f.f(xs...)
    return y
end

AD.@primitive function value_and_pullback_function(b::AD.EnzymeReverseBackend, f, xs...)
    y = f(xs...)
    return y,
    Δ -> begin
        Δ_xs = zero.(xs)
        dup = if y isa Real
            if Δ isa Real
                Enzyme.Duplicated([y], [Δ])
            elseif Δ isa Tuple{Real}
                Enzyme.Duplicated([y], [Δ[1]])
            else
                throw(ArgumentError("Unsupported cotangent type."))
            end
        else
            if Δ isa AbstractArray{<:Real}
                Enzyme.Duplicated(y, Δ)
            elseif Δ isa Tuple{AbstractArray{<:Real}}
                Enzyme.Duplicated(y, Δ[1])
            else
                throw(ArgumentError("Unsupported cotangent type."))
            end
        end
        Enzyme.autodiff(
            Enzyme.Reverse,
            Mutating(f),
            Enzyme.Const,
            dup,
            Enzyme.Duplicated.(xs, Δ_xs)...,
        )
        return Δ_xs
    end
end
function AD.pushforward_function(::AD.EnzymeReverseBackend, f, xs...)
    return AD.pushforward_function(AD.EnzymeForwardBackend(), f, xs...)
end

AD.@primitive function pushforward_function(b::AD.EnzymeForwardBackend, f, xs...)
    return ds ->
        Tuple(Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated.(xs, copy.(ds))...))
end
function AD.value_and_pullback_function(::AD.EnzymeForwardBackend, f, xs...)
    return AD.value_and_pullback_function(AD.EnzymeReverseBackend(), f, xs...)
end

end # module
