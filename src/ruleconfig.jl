using ChainRulesCore: RuleConfig, rrule_via_ad

"""
    ReverseRuleConfigBackend

AD backend that uses reverse mode with any ChainRules-compatible reverse-mode AD package.
"""
struct ReverseRuleConfigBackend{RC <: RuleConfig} <: AbstractReverseMode
    ruleconfig::RC
end

AD.@primitive function pullback_function(ab::ReverseRuleConfigBackend, f, xs...)
    return (vs) -> begin
        _, back = rrule_via_ad(ab.ruleconfig, f, xs...)
        if vs isa Tuple && length(vs) === 1
            return Base.tail(back(vs[1]))
        else
            return Base.tail(back(vs))
        end
    end
end
