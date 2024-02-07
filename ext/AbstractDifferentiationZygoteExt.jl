module AbstractDifferentiationZygoteExt

if isdefined(Base, :get_extension)
    import AbstractDifferentiation as AD
    using Zygote: Zygote
else
    import ..AbstractDifferentiation as AD
    using ..Zygote: Zygote
end

# Context should not persist between different AD calls: fixes #69
function AD.ruleconfig(::AD.ReverseRuleConfigBackend{<:Zygote.ZygoteRuleConfig})
    return Zygote.ZygoteRuleConfig()
end

function AD.value_and_pullback_function(::AD.ZygoteBackend, f, args...)
    return Zygote.pullback(f, args...)
end

AD.gradient(::AD.ZygoteBackend, f, args...) = Zygote.gradient(f, args...)
function AD.value_and_gradient(::AD.ZygoteBackend, f, args...)
    res = Zygote.withgradient(f, args...)
    return res.val, res.grad
end

AD.jacobian(::AD.ZygoteBackend, f, args...) = Zygote.jacobian(f, args...)
function AD.value_and_jacobian(::AD.ZygoteBackend, f, args...)
    res = Zygote.withjacobian(f, args...)
    return res.val, res.grad
end

AD.hessian(::AD.ZygoteBackend, f, arg) = Zygote.hessian(f, arg)

end # module
