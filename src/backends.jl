"""
    FiniteDifferencesBackend{M}

AD backend that uses forward mode with FiniteDifferences.jl.

The type parameter `M` is the type of the method used to perform finite differences.

!!! note
    To be able to use this backend, you have to load FiniteDifferences.
"""
struct FiniteDifferencesBackend{M} <: AbstractFiniteDifference
    method::M
end

"""
    ForwardDiffBackend{CS}

AD backend that uses forward mode with ForwardDiff.jl.

The type parameter `CS` denotes the chunk size of the differentiation algorithm. If it is
`Nothing`, then ForwardiffDiff uses a heuristic to set the chunk size based on the input.

See also: [ForwardDiff.jl: Configuring Chunk Size](https://juliadiff.org/ForwardDiff.jl/dev/user/advanced/#Configuring-Chunk-Size)

!!! note
    To be able to use this backend, you have to load ForwardDiff.
"""
struct ForwardDiffBackend{CS} <: AbstractForwardMode end

"""
    ReverseDiffBackend

AD backend that uses reverse mode with ReverseDiff.jl.

!!! note
    To be able to use this backend, you have to load ReverseDiff.
"""
struct ReverseDiffBackend <: AbstractReverseMode end

"""
    TrackerBackend

AD backend that uses reverse mode with Tracker.jl.

!!! note
    To be able to use this backend, you have to load Tracker.
"""
struct TrackerBackend <: AbstractReverseMode end


"""
    ReverseRuleConfigBackend

AD backend that uses reverse mode with any ChainRules-compatible reverse-mode AD package.

!!! note
    On Julia >= 1.9, you have to load ChainRulesCore (possibly implicitly by loading
    a ChainRules-compatible AD package) to be able to use this backend.
"""
struct ReverseRuleConfigBackend{RC} <: AbstractReverseMode
    ruleconfig::RC
end

"""
    ZygoteBackend()

Create an AD backend that uses reverse mode with Zygote.jl.

It is a special case of [`ReverseRuleConfigBackend`](@ref).

!!! note
    To be able to use this backend, you have to load Zygote.
"""
function ZygoteBackend end
