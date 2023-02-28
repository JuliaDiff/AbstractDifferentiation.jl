module AbstractDifferentiationChainRulesCoreExt

using AbstractDifferentiation: AbstractDifferentiation, EXTENSIONS_SUPPORTED, ReverseRuleConfigBackend
if EXTENSIONS_SUPPORTED
    using ChainRulesCore: ChainRulesCore
else
    using .ChainRulesCore: ChainRulesCore
end

const AD = AbstractDifferentiation

AD.@primitive function pullback_function(ab::ReverseRuleConfigBackend, f, xs...)
    _, back = ChainRulesCore.rrule_via_ad(ab.ruleconfig, f, xs...)
    pullback(vs) = Base.tail(back(vs))
    pullback(vs::Tuple{Any}) = Base.tail(back(first(vs)))
    return pullback
end

end # module
