module AbstractDifferentiationZygoteExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using Zygote: Zygote
else
    using ..Zygote: Zygote
end
using ChainRulesCore: ChainRulesCore

AD.ZygoteBackend() = AD.ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())

# Context should not persist between different AD calls: fixes #69
function AD.pullback_function(ba::AD.ReverseRuleConfigBackend{<:Zygote.ZygoteRuleConfig}, f, xs...)
    _, back = ChainRulesCore.rrule_via_ad(Zygote.ZygoteRuleConfig(), f, xs...)
    pullback(vs) = Base.tail(back(vs))
    pullback(vs::Tuple{Any}) = Base.tail(back(first(vs)))
    return pullback
end

end # module
