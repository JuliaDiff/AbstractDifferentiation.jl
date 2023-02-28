module AbstractDifferentiationZygoteExt

using AbstractDifferentiation: AbstractDifferentiation, EXTENSIONS_SUPPORTED, ReverseRuleConfigBackend
if EXTENSIONS_SUPPORTED
    using Zygote: Zygote
else
    using ..Zygote: Zygote
end

AbstractDifferentiation.ZygoteBackend() = ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())

end # module
