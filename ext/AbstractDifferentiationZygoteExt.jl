module AbstractDifferentiationZygoteExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using Zygote: Zygote
else
    using ..Zygote: Zygote
end

AD.ZygoteBackend() = AD.ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())

# Context should not persist between different AD calls: fixes #69
function AD.ruleconfig(::AD.ReverseRuleConfigBackend{<:Zygote.ZygoteRuleConfig})
    return Zygote.ZygoteRuleConfig()
end

end # module
