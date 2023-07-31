module AbstractDifferentiationChainRulesCoreExt

import AbstractDifferentiation as AD
using ChainRulesCore: ChainRulesCore

AD.@primitive function value_and_pullback_function(ba::AD.ReverseRuleConfigBackend, f, xs...)
    value, back = ChainRulesCore.rrule_via_ad(AD.ruleconfig(ba), f, xs...)
    function value_and_pullback(vs)
        pb_value = if vs === nothing
            nothing
        else
            _vs = if vs isa Tuple && !(value isa Tuple)
                only(vs)
            else
                vs
            end
            Base.tail(back(_vs))
        end
        return value, pb_value
    end
    return value_and_pullback
end

end # module
