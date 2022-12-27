# AbstractDifferentiation

[![CI](https://github.com/JuliaDiff/AbstractDifferentiation.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaDiff/AbstractDifferentiation.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/JuliaDiff/AbstractDifferentiation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaDiff/AbstractDifferentiation.jl)

## Motivation

This is a package that implements an abstract interface for differentiation in Julia. This is particularly useful for implementing abstract algorithms requiring derivatives, gradients, jacobians, Hessians or multiple of those without depending on specific automatic differentiation packages' user interfaces.

Julia has more (automatic) differentiation packages than you can count on 2 hands. Different packages have different user interfaces. Therefore, having a backend-agnostic interface to request the function value and its gradient for example is necessary to avoid a combinatorial explosion of code when trying to support every differentiation package in Julia in every algorithm package requiring gradients. For higher order derivatives, the situation is even more dire since you can combine any 2 differentiation backends together to create a new higher-order backend.

## Loading `AbstractDifferentiation`

To load `AbstractDifferentiation`, it is recommended to use
```julia
import AbstractDifferentiation as AD
```
on Julia ≥ 1.6 and
```julia
import AbstractDifferentiation
const AD = AbstractDifferentiation
```
on older Julia versions.
With the `AD` alias you can access names inside of `AbstractDifferentiation` using `AD.<>` instead of typing the long name `AbstractDifferentiation`.

## `AbstractDifferentiation` backends

To use `AbstractDifferentiation`, first construct a backend instance `ab::AD.AbstractBackend` using your favorite differentiation package in Julia that supports `AbstractDifferentiation`.
In particular, you may want to use `AD.ReverseRuleConfigBackend(ruleconfig)` for any [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible reverse mode differentiation package.

The following backends are temporarily made available by `AbstractDifferentiation` as soon as their corresponding package is loaded (thanks to [weak dependencies](https://pkgdocs.julialang.org/dev/creating-packages/#Weak-dependencies) on Julia ≥ 1.9 and [Requires.jl](https://github.com/JuliaPackaging/Requires.jl) on older Julia versions):

- `AD.ForwardDiffBackend()` for [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- `AD.FiniteDifferencesBackend()` for [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
- `AD.ReverseDiffBackend()` for [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
- `AD.TrackerBackend()` for [Tracker.jl](https://github.com/FluxML/Tracker.jl)
- `AD.ZygoteBackend()` for [Zygote.jl](https://github.com/FluxML/Zygote.jl), which is a special case of `AD.ReverseRuleConfigBackend`

In the long term, these backend objects (and many more) will be defined within their respective packages to enforce the `AbstractDifferentiation` interface.

Here's an example:

```julia
julia> import AbstractDifferentiation as AD, Zygote

julia> ab = AD.ZygoteBackend()
AbstractDifferentiation.ReverseRuleConfigBackend{Zygote.ZygoteRuleConfig{Zygote.Context}}(Zygote.ZygoteRuleConfig{Zygote.Context}(Zygote.Context(nothing)))

julia> f(x) = log(sum(exp, x))
f (generic function with 1 method)

julia> AD.gradient(ab, f, rand(10))
([0.07163448353282538, 0.08520350535348796, 0.09675622487503996, 0.1522744408520505, 0.12174662595572318, 0.07996969757526722, 0.07832665607158593, 0.11001685581681672, 0.06691909637037166, 0.1371524135968315],)
```

For higher order derivatives, you can build higher order backends using `AD.HigherOrderBackend`. For instance, let `ab_f` be a forward-mode automatic differentiation backend and let `ab_r` be a reverse-mode automatic differentiation backend. To construct a higher order backend for doing forward-over-reverse-mode automatic differentiation, use `AD.HigherOrderBackend((ab_f, ab_r))`. To construct a higher order backend for doing reverse-over-forward-mode automatic differentiation, use `AD.HigherOrderBackend((ab_r, ab_f))`.

## Backend-agnostic interface

The following list of functions is the officially supported differentiation interface in `AbstractDifferentiation`.

### Derivative/Gradient/Jacobian/Hessian

The following list of functions can be used to request the derivative, gradient, Jacobian or Hessian without the function value.

- `ds = AD.derivative(ab::AD.AbstractBackend, f, xs::Number...)`: computes the derivatives `ds` of `f` wrt the numbers `xs` using the backend `ab`. `ds` is a tuple of derivatives, one for each element in `xs`.
- `gs = AD.gradient(ab::AD.AbstractBackend, f, xs...)`: computes the gradients `gs` of `f` wrt the inputs `xs` using the backend `ab`. `gs` is a tuple of gradients, one for each element in `xs`.
- `js = AD.jacobian(ab::AD.AbstractBackend, f, xs...)`: computes the Jacobians `js` of `f` wrt the inputs `xs` using the backend `ab`. `js` is a tuple of Jacobians, one for each element in `xs`.
- `h = AD.hessian(ab::AD.AbstractBackend, f, x)`: computes the Hessian `h` of `f` wrt the input `x` using the backend `ab`. `hessian` currently only supports a single input.

### Value and Derivative/Gradient/Jacobian/Hessian

The following list of functions can be used to request the function value along with its derivative, gradient, Jacobian or Hessian. You can also request the function value, its gradient and Hessian for single-input functions.

- `(v, ds) = AD.value_and_derivative(ab::AD.AbstractBackend, f, xs::Number...)`: computes the function value `v = f(xs...)` and the derivatives `ds` of `f` wrt the numbers `xs` using the backend `ab`. `ds` is a tuple of derivatives, one for each element in `xs`.
- `(v, gs) = AD.value_and_gradient(ab::AD.AbstractBackend, f, xs...)`: computes the function value `v = f(xs...)` and the gradients `gs` of `f` wrt the inputs `xs` using the backend `ab`. `gs` is a tuple of gradients, one for each element in `xs`.
- `(v, js) = AD.value_and_jacobian(ab::AD.AbstractBackend, f, xs...)`: computes the function value `v = f(xs...)` and the Jacobians `js` of `f` wrt the inputs `xs` using the backend `ab`. `js` is a tuple of Jacobians, one for each element in `xs`.
- `(v, h) = AD.value_and_hessian(ab::AD.AbstractBackend, f, x)`: computes the function value `v = f(x)` and the Hessian `h` of `f` wrt the input `x` using the backend `ab`. `hessian` currently only supports a single input.
- `(v, g, h) = AD.value_gradient_and_hessian(ab::AD.AbstractBackend, f, x)`: computes the function value `v = f(x)` and the gradient `g` and Hessian `h` of `f` wrt the input `x` using the backend `ab`. `hessian` currently only supports a single input.

### Jacobian vector products (aka pushforward)

This operation goes by a few names. Refer to the [ChainRules documentation](https://juliadiff.org/ChainRulesCore.jl/stable/#The-propagators:-pushforward-and-pullback) for more on terminology. For a single input, single output function `f` with a Jacobian `J`, the pushforward operator `pf_f` is equivalent to applying the function `v -> J * v` on a (tangent) vector `v`.

The following functions can be used to request a function that returns the pushforward operator/function. In order to request the pushforward function `pf_f` of a function `f` at the inputs `xs`, you can use either of:
- `pf_f = AD.pushforward_function(ab::AD.AbstractBackend, f, xs...)`: returns the pushforward function `pf_f` of the function `f` at the inputs `xs`. `pf_f` is a function that accepts the tangents `vs` as input which is a tuple of length equal to the length of the tuple `xs`. If `f` has a single input, `pf_f` can also accept a single input instead of a 1-tuple.
- `value_and_pf_f = AD.value_and_pushforward_function(ab::AD.AbstractBackend, f, xs...)`: returns a function `value_and_pf_f` which accepts the tangent `vs` as input which is a tuple of length equal to the length of the tuple `xs`. If `f` has a single input, `value_and_pf_f` can accept a single input instead of a 1-tuple. `value_and_pf_f` returns a 2-tuple, namely the value `f(xs...)` and output of the pushforward operator.

### Vector Jacobian products (aka pullback)

This operation goes by a few names. Refer to the [ChainRules documentation](https://juliadiff.org/ChainRulesCore.jl/stable/#The-propagators:-pushforward-and-pullback) for more on terminology. For a single input, single output function `f` with a Jacobian `J`, the pullback operator `pb_f` is equivalent to applying the function `v -> v' * J` on a (co-tangent) vector `v`.

The following functions can be used to request the pullback operator/function with or without the function value. In order to request the pullback function `pb_f` of a function `f` at the inputs `xs`, you can use either of:
- `pb_f = AD.pullback_function(ab::AD.AbstractBackend, f, xs...)`: returns the pullback function `pb_f` of the function `f` at the inputs `xs`. `pb_f` is a function that accepts the co-tangents `vs` as input which is a tuple of length equal to the number of outputs of `f`. If `f` has a single output, `pb_f` can also accept a single input instead of a 1-tuple.
- `value_and_pb_f = AD.value_and_pullback_function(ab::AD.AbstractBackend, f, xs...)`: returns a function `value_and_pb_f` which accepts the co-tangent `vs` as input which is a tuple of length equal to the number of outputs of `f`. If `f` has a single output, `value_and_pb_f` can accept a single input instead of a 1-tuple. `value_and_pb_f` returns a 2-tuple, namely the value `f(xs...)` and output of the pullback operator.

### Lazy operators

You can also get a struct for the lazy derivative/gradient/Jacobian/Hessian of a function. You can then use the `*` operator to apply the lazy operator on a value or tuple of the correct shape. To get a lazy derivative/gradient/Jacobian/Hessian use any one of:
- `ld = lazy_derivative(ab::AbstractBackend, f, xs::Number...)`: returns an operator `ld` for multiplying by the derivative of `f` at `xs`. You can apply the operator by multiplication e.g. `ld * y` where `y` is a number if `f` has a single input, a tuple of the same length as `xs` if `f` has multiple inputs, or an array of numbers/tuples.
- `lg = lazy_gradient(ab::AbstractBackend, f, xs...)`: returns an operator `lg` for multiplying by the gradient of `f` at `xs`. You can apply the operator by multiplication e.g. `lg * y` where `y` is a number if `f` has a single input or a tuple of the same length as `xs` if `f` has multiple inputs.
- `lh = lazy_hessian(ab::AbstractBackend, f, x)`: returns an operator `lh` for multiplying by the Hessian of the scalar-valued function `f` at `x`. You can apply the operator by multiplication e.g. `lh * y` or `y' * lh` where `y` is a number or a vector of the appropriate length.
- `lj = lazy_jacobian(ab::AbstractBackend, f, xs...)`: returns an operator `lj` for multiplying by the Jacobian of `f` at `xs`. You can apply the operator by multiplication e.g. `lj * y` or `y' * lj` where `y` is a number, vector or tuple of numbers and/or vectors. If `f` has multiple inputs, `y` in `lj * y` should be a tuple. If `f` has multiply outputs, `y` in `y' * lj` should be a tuple. Otherwise, it should be a scalar or a vector of the appropriate length.

## Citing this package

If you use this package in your work, please cite the package:

```bib
@article{schafer2021abstractdifferentiation,
  title={AbstractDifferentiation. jl: Backend-Agnostic Differentiable Programming in Julia},
  author={Sch{\"a}fer, Frank and Tarek, Mohamed and White, Lyndon and Rackauckas, Chris},
  journal={NeurIPS 2021 Differentiable Programming Workshop},
  year={2021}
}
```
