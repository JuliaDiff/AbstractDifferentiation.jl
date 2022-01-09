using .ReverseDiff: ReverseDiff

struct ReverseDiffBackend <: AbstractBackend end

@primitive function pullback_function(ba::ReverseDiffBackend, f, xs...)
    return (ws) -> begin
        tape = ReverseDiff.InstructionTape()
        x_tracked = ReverseDiff.track.(xs, Ref(tape))
        y_tracked = f(x_tracked...)
        ReverseDiff.deriv!.(y_tracked, ws...)
        ReverseDiff.reverse_pass!(tape)
        return ReverseDiff.deriv.(x_tracked)
    end
end
