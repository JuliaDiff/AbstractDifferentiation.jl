module AbstractDifferentiationChainRulesCoreExt

import AbstractDifferentiation as AD
using ChainRulesCore: ChainRulesCore

AD.@primitive function value_and_pullback_function(
    ba::AD.ReverseRuleConfigBackend, f, xs...
)
    value, back = ChainRulesCore.rrule_via_ad(AD.ruleconfig(ba), f, xs...)
    function rrule_pullback(vs...)
        _vs = if !(value isa Tuple)
            only(vs)
        else
            vs
        end
        @show vs _vs
        return Base.tail(back(_vs))
    end
    return value, rrule_pullback
end

end # module
