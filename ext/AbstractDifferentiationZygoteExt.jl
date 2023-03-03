module AbstractDifferentiationZygoteExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using Zygote: Zygote
else
    using ..Zygote: Zygote
end

AD.@primitive function pullback_function(ba::AD.ZygoteBackend, f, xs...)
    @inline # @primitive doesn't support this before function keyword
    rc_backend = AD.ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())
    return AD.pullback_function(rc_backend, f, xs...)
end

end # module
