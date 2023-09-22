# AbstractDifferentiation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiff.github.io/AbstractDifferentiation.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDiff.github.io/AbstractDifferentiation.jl/dev)
[![CI](https://github.com/JuliaDiff/AbstractDifferentiation.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaDiff/AbstractDifferentiation.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaDiff/AbstractDifferentiation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaDiff/AbstractDifferentiation.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## Motivation

This is a package that implements an abstract interface for differentiation in Julia. This is particularly useful for implementing abstract algorithms requiring derivatives, gradients, jacobians, Hessians or multiple of those without depending on specific automatic differentiation packages' user interfaces.

Julia has more (automatic) differentiation packages than you can count on 2 hands. Different packages have different user interfaces. Therefore, having a backend-agnostic interface to request the function value and its gradient for example is necessary to avoid a combinatorial explosion of code when trying to support every differentiation package in Julia in every algorithm package requiring gradients. For higher order derivatives, the situation is even more dire since you can combine any 2 differentiation backends together to create a new higher-order backend.

## Getting started

- If you are an autodiff user and want to write code in a backend-agnostic way, read the _user guide_ in the docs.
- If you are an autodiff developer and want your backend to implement the interface, read the _implementer guide_ in the docs.

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
