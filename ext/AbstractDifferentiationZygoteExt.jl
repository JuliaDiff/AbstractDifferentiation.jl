module AbstractDifferentiationZygoteExt

using AbstractDifferentiation: AbstractDifferentiation, EXTENSIONS_SUPPORTED, ReverseRuleConfigBackend

if EXTENSIONS_SUPPORTED
    using Zygote: Zygote
else
    using ..Zygote: Zygote
end

@static if isdefined(AbstractDifferentiation, :ZygoteBackend) && isdefined(Zygote, :ZygoteRuleConfig)
    AbstractDifferentiation.ZygoteBackend() = ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())
end

end # module
