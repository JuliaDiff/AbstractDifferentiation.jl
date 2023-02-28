module AbstractDifferentiationTrackerExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using Tracker: Tracker
else
    using ..Tracker: Tracker
end

function AD.second_lowest(::TrackerBackend)
    return throw(ArgumentError("Tracker backend does not support nested differentiation."))
end

AD.primal_value(x::Tracker.TrackedReal) = Tracker.data(x)
AD.primal_value(x::Tracker.TrackedArray) = Tracker.data(x)
AD.primal_value(x::AbstractArray{<:Tracker.TrackedReal}) = Tracker.data.(x)

AD.@primitive function pullback_function(ba::AD.TrackerBackend, f, xs...)
    value, back = Tracker.forward(f, xs...)
    function pullback(ws)
        if ws isa Tuple && !(value isa Tuple) 
            map(Tracker.data, back(only(ws)))
        else
            map(Tracker.data, back(ws))
        end
    end
    return pullback
end

function AD.derivative(::AD.TrackerBackend, f, xs::Number...)
    return Tracker.data.(Tracker.gradient(f, xs...))
end

function AD.gradient(::AD.TrackerBackend, f, xs::AbstractVector...)
    return Tracker.data.(Tracker.gradient(f, xs...))
end

end # module
