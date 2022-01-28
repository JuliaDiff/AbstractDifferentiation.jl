using .Tracker: Tracker

primal_value(x::Tracker.TrackedArray) = Tracker.data(x)
primal_value(x::Tracker.TrackedReal) = Tracker.data(x)
primal_value(x::AbstractArray{<:Tracker.TrackedReal}) = Tracker.data.(x)

"""
    TrackerBackend

AD backend that uses reverse mode with Tracker.jl.
"""
struct TrackerBackend <: AbstractReverseMode end

function second_lowest(::TrackerBackend)
    return throw(ArgumentError("Tracker backend does not support nested differentiation."))
end

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
