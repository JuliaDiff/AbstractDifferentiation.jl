"""
    FiniteDifferencesBackend{M}

AD backend that uses forward mode with [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).

The type parameter `M` is the type of the method used to perform finite differences.

!!! note
    To be able to use this backend, you have to load FiniteDifferences.
"""
struct FiniteDifferencesBackend{M} <: AbstractFiniteDifference
    method::M
end

"""
    ForwardDiffBackend{CS}

AD backend that uses forward mode with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

The type parameter `CS` denotes the chunk size of the differentiation algorithm. 
If it is `Nothing`, then ForwardiffDiff uses a heuristic to set the chunk size based on the input.

See also: [ForwardDiff.jl: Configuring Chunk Size](https://juliadiff.org/ForwardDiff.jl/dev/user/advanced/#Configuring-Chunk-Size)

!!! note
    To be able to use this backend, you have to load ForwardDiff.
"""
struct ForwardDiffBackend{CS} <: AbstractForwardMode end

"""
    ReverseDiffBackend

AD backend that uses reverse mode with [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).

!!! note
    To be able to use this backend, you have to load ReverseDiff.
"""
struct ReverseDiffBackend <: AbstractReverseMode end

"""
    TrackerBackend

AD backend that uses reverse mode with [Tracker.jl](https://github.com/FluxML/Tracker.jl).

!!! note
    To be able to use this backend, you have to load Tracker.
"""
struct TrackerBackend <: AbstractReverseMode end

"""
    ReverseRuleConfigBackend

AD backend that uses reverse mode with any [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)-compatible reverse-mode AD package.

Constructed with a [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) object:

```julia
backend = AD.ReverseRuleConfigBackend(rc)
```

!!! note
    On Julia >= 1.9, you have to load ChainRulesCore (possibly implicitly by loading a ChainRules-compatible AD package) to be able to use this backend.
"""
struct ReverseRuleConfigBackend{RC} <: AbstractReverseMode
    ruleconfig::RC
end

# internal function for extracting the rule config
# falls back to returning the wrapped `ruleconfig` but can be specialized
# e.g., for Zygote to fix #69
ruleconfig(ba::ReverseRuleConfigBackend) = ba.ruleconfig

"""
    ZygoteBackend

Create an AD backend that uses reverse mode with [Zygote.jl](https://github.com/FluxML/Zygote.jl).

Alternatively, you can perform AD with Zygote using a special [`ReverseRuleConfigBackend`](@ref), namely `ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())`.
Note, however, that the behaviour of this backend is not equivalent to `ZygoteBackend()` since the former uses a generic implementation of jacobian etc. for ChainRules-compatible AD backends whereas `ZygoteBackend` uses implementations in Zygote.jl.

!!! note
    To be able to use this backend, you have to load Zygote.
"""
struct ZygoteBackend <: AbstractReverseMode end
