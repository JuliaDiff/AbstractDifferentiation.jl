using .Tracker: Tracker

"""
    TrackerBackend

AD backend that uses reverse mode with Tracker.jl.
"""
struct TrackerBackend <: AbstractReverseMode end

function second_lowest(::TrackerBackend)
    return throw(ArgumentError("Tracker backend does not support nested differentiation."))
end

primal_value(x::Tracker.TrackedReal) = Tracker.value(x)
primal_value(x::Tracker.TrackedArray) = Tracker.value(x)
primal_value(x::AbstractArray{<:Tracker.TrackedReal}) = Tracker.value.(x)

@primitive function pullback_function(ba::TrackerBackend, f, xs...)
    value, back = Tracker.forward(f, xs...)
    function pullback(ws)
        if ws isa Tuple && !(value isa Tuple) 
            @assert length(ws) == 1
            map(Tracker.data, back(ws[1]))
        else
            map(Tracker.data, back(ws))
        end
    end
    return pullback
end

function derivative(ba::TrackerBackend, f, xs::Number...)
    return Tracker.data.(Tracker.gradient(f, xs...))
end

function gradient(ba::TrackerBackend, f, xs::AbstractVector...)
    return Tracker.data.(Tracker.gradient(f, xs...))
end
