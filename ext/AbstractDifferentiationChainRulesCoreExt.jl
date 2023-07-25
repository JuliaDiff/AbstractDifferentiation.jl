module AbstractDifferentiationChainRulesCoreExt

import AbstractDifferentiation as AD
using ChainRulesCore: ChainRulesCore

AD.@primitive function value_and_pullback_function(ba::AD.ReverseRuleConfigBackend, f, xs...)
    value, back = ChainRulesCore.rrule_via_ad(AD.ruleconfig(ba), f, xs...)
    function value_and_pullback(vs)
        _vs = if vs isa Tuple && !(value isa Tuple)
            only(vs)
        else
            vs
        end
        return (value, Base.tail(back(_vs)))
    end
    return value_and_pullback
end

end # module
