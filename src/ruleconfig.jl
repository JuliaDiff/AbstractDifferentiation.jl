"""
    ReverseRuleConfigBackend

AD backend that uses reverse mode with any ChainRules-compatible reverse-mode AD package.
"""
struct ReverseRuleConfigBackend{RC<:ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode}} <: AbstractReverseMode
    ruleconfig::RC
end

AD.@primitive function pullback_function(ab::ReverseRuleConfigBackend, f, xs...)
    _, back = ChainRulesCore.rrule_via_ad(ab.ruleconfig, f, xs...)
    pullback(vs) = Base.tail(back(vs))
    pullback(vs::Tuple{Any}) = Base.tail(back(first(vs))
    return pullback
end
