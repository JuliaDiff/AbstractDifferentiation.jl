"""
    ReverseRuleConfigBackend

AD backend that uses reverse mode with any ChainRules-compatible reverse-mode AD package.
"""
struct ReverseRuleConfigBackend{RC<:ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode}} <: AbstractReverseMode
    ruleconfig::RC
end

AD.@primitive function pullback_function(ab::ReverseRuleConfigBackend, f, xs...)
    _, back = ChainRulesCore.rrule_via_ad(ab.ruleconfig, f, xs...)
    function pullback(vs)
        grad = Base.tail(back(vs))
        empty_cache!(ab.ruleconfig)
        grad
    end
    pullback(vs::Tuple{Any}) = pullback(first(vs))
    return pullback
end

empty_cache!(ruleconfig::ChainRulesCore.RuleConfig) = ruleconfig