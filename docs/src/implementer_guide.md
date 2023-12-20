# Implementer guide

!!! warning "Work in progress"
    
    Come back later!

## The macro `@primitive`

To implement the `AbstractDifferentiation` interface for your backend, you only _need_ to provide a "primitive" from which the rest of the functions can be deduced.
However, for performance reasons, you _can_ implement more of the interface to make certain calls faster.

At the moment, the only primitives supported are `AD.pushforward_function` and `AD.value_and_pullback_function`.
The `AD.@primitive` macro uses the provided function to implement `AD.jacobian`, and all the other functions follow.

```julia
AD.@primitive function AD.myprimitive(ab::MyBackend, f, xs...)
    # write your code here
end
```

See the backend-specific extensions in the `ext/` folder of the repository for example implementations.

## Function dependency graph

These details are not part of the public API and are expected to change.
They are just listed here to help readers figure out the code structure:

  - `jacobian` has no default implementation
  - `derivative` calls `jacobian`
  - `gradient` calls `jacobian`
  - `hessian` calls `jacobian` and `gradient`
  - `second_derivative` calls `derivative`
  - `value_and_jacobian` calls `jacobian`
  - `value_and_derivative` calls `value_and_jacobian`
  - `value_and_gradient` calls `value_and_jacobian`
  - `value_and_hessian` calls `jacobian` and `gradient`
  - `value_and_second_derivative` calls `second_derivative`
  - `value_gradient_and_hessian` calls `value_and_jacobian` and `gradient`
  - `value_and_derivatives` calls `value_and_derivative` and `second_derivative`
  - `pushforward_function` calls `jacobian`
  - `value_and_pushforward_function` calls `pushforward_function`
  - `pullback_function` calls `value_and_pullback_function`
  - `value_and_pullback_function` calls `gradient`
