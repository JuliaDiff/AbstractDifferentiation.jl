module AbstractDifferentiationChainRulesCoreExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using ChainRulesCore: ChainRulesCore
else
    using .ChainRulesCore: ChainRulesCore
end

AD.@primitive function pullback_function(ab::AD.ReverseRuleConfigBackend, f, xs...)
    _, back = ChainRulesCore.rrule_via_ad(ab.ruleconfig, f, xs...)
    pullback(vs) = Base.tail(back(vs))
    pullback(vs::Tuple{Any}) = Base.tail(back(first(vs)))
    return pullback
end

end # module
