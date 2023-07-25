module AbstractDifferentiationZygoteExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using Zygote: Zygote
else
    import ..AbstractDifferentiation as AD
    using ..Zygote: Zygote
end

AD.ZygoteBackend() = AD.ReverseRuleConfigBackend(Zygote.ZygoteRuleConfig())

# Context should not persist between different AD calls: fixes #69
function AD.ruleconfig(::AD.ReverseRuleConfigBackend{<:Zygote.ZygoteRuleConfig})
    return Zygote.ZygoteRuleConfig()
end

end # module
