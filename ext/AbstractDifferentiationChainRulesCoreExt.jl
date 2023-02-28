module AbstractDifferentiationChainRulesCoreExt

import AbstractDifferentiation as AD
using ChainRulesCore: ChainRulesCore

AD.@primitive function pullback_function(ba::AD.ReverseRuleConfigBackend, f, xs...)
    _, back = ChainRulesCore.rrule_via_ad(ba.ruleconfig, f, xs...)
    pullback(vs) = Base.tail(back(vs))
    pullback(vs::Tuple{Any}) = Base.tail(back(first(vs)))
    return pullback
end

end # module
