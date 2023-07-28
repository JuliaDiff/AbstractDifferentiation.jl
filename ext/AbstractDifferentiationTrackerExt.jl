module AbstractDifferentiationTrackerExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using Tracker: Tracker
else
    import ..AbstractDifferentiation as AD
    using ..Tracker: Tracker
end

function AD.second_lowest(::AD.TrackerBackend)
    return throw(ArgumentError("Tracker backend does not support nested differentiation."))
end

AD.primal_value(x::Tracker.TrackedReal) = Tracker.data(x)
AD.primal_value(x::Tracker.TrackedArray) = Tracker.data(x)
AD.primal_value(x::AbstractArray{<:Tracker.TrackedReal}) = Tracker.data.(x)

AD.@primitive function value_and_pullback_function(ba::AD.TrackerBackend, f, xs...)
    value, back = Tracker.forward(f, xs...)
    function value_and_pullback(ws)
        _ws = if ws isa Tuple && !(value isa Tuple)
            only(ws)
        else
            ws
        end
        return (value, map(Tracker.data, back(_ws)))
    end
    return value_and_pullback
end

function AD.derivative(::AD.TrackerBackend, f, xs::Number...)
    return Tracker.data.(Tracker.gradient(f, xs...))
end

function AD.gradient(::AD.TrackerBackend, f, xs::AbstractVector...)
    return Tracker.data.(Tracker.gradient(f, xs...))
end

end # module
