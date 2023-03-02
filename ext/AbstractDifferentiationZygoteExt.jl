module AbstractDifferentiationZygoteExt

import AbstractDifferentiation as AD
if AD.EXTENSIONS_SUPPORTED
    using Zygote: Zygote
else
    using ..Zygote: Zygote
end

AD.ZygoteBackend() = AD.ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())

end # module
