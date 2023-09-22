# User guide

The list of functions on this page is the officially supported differentiation interface in `AbstractDifferentiation`.

## Loading `AbstractDifferentiation`

To load `AbstractDifferentiation`, it is recommended to use

```julia
import AbstractDifferentiation as AD
```

With the `AD` alias you can access names inside of `AbstractDifferentiation` using `AD.<>` instead of typing the long name `AbstractDifferentiation`.

## `AbstractDifferentiation` backends

To use `AbstractDifferentiation`, first construct a backend instance `ab::AD.AbstractBackend` using your favorite differentiation package in Julia that supports `AbstractDifferentiation`.

Here's an example:

```jldoctest
julia> import AbstractDifferentiation as AD, Zygote

julia> backend = AD.ZygoteBackend();

julia> f(x) = log(sum(exp, x));

julia> AD.gradient(backend, f, collect(1:3))
([0.09003057317038046, 0.2447284710547977, 0.665240955774822],)
```

The following backends are temporarily made available by `AbstractDifferentiation` as soon as their corresponding package is loaded (thanks to [weak dependencies](https://pkgdocs.julialang.org/dev/creating-packages/#Weak-dependencies) on Julia ≥ 1.9 and [Requires.jl](https://github.com/JuliaPackaging/Requires.jl) on older Julia versions):

```@docs
AD.ReverseDiffBackend
AD.ReverseRuleConfigBackend
AD.FiniteDifferencesBackend
AD.ZygoteBackend
AD.ForwardDiffBackend
AD.TrackerBackend
```

In the long term, these backend objects (and many more) will be defined within their respective packages to enforce the `AbstractDifferentiation` interface.
This is already the case for:

- `Diffractor.DiffractorForwardBackend()` for [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl) in forward mode

For higher order derivatives, you can build higher order backends using `AD.HigherOrderBackend`.

```@docs
AD.HigherOrderBackend
```

## Derivatives

The following list of functions can be used to request the derivative, gradient, Jacobian or Hessian without the function value.

```@docs
AD.derivative
AD.gradient
AD.jacobian
AD.hessian
```

## Value and derivatives

The following list of functions can be used to request the function value along with its derivative, gradient, Jacobian or Hessian. You can also request the function value, its gradient and Hessian for single-input functions.

```@docs
AD.value_and_derivative
AD.value_and_gradient
AD.value_and_jacobian
AD.value_and_hessian
AD.value_gradient_and_hessian
```

## Jacobian-vector products

This operation goes by a few names, like "pushforward". Refer to the [ChainRules documentation](https://juliadiff.org/ChainRulesCore.jl/stable/#The-propagators:-pushforward-and-pullback) for more on terminology. For a single input, single output function `f` with a Jacobian `J`, the pushforward operator `pf_f` is equivalent to applying the function `v -> J * v` on a (tangent) vector `v`.

The following functions can be used to request a function that returns the pushforward operator/function. In order to request the pushforward function `pf_f` of a function `f` at the inputs `xs`, you can use either of:

```@docs
AD.pushforward_function
AD.value_and_pushforward_function
```

## Vector-Jacobian products

This operation goes by a few names, like "pullback". Refer to the [ChainRules documentation](https://juliadiff.org/ChainRulesCore.jl/stable/#The-propagators:-pushforward-and-pullback) for more on terminology. For a single input, single output function `f` with a Jacobian `J`, the pullback operator `pb_f` is equivalent to applying the function `v -> v' * J` on a (co-tangent) vector `v`.

The following functions can be used to request the pullback operator/function with or without the function value. In order to request the pullback function `pb_f` of a function `f` at the inputs `xs`, you can use either of:

```@docs
AD.pullback_function
AD.value_and_pullback_function
```

## Lazy operators

You can also get a struct for the lazy derivative/gradient/Jacobian/Hessian of a function. You can then use the `*` operator to apply the lazy operator on a value or tuple of the correct shape. To get a lazy derivative/gradient/Jacobian/Hessian use any one of:

```@docs
AD.lazy_derivative
AD.lazy_gradient
AD.lazy_jacobian
AD.lazy_hessian
```

## Index

```@index
```
