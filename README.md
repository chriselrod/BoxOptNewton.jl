# BoxOptNewton

[![Build Status](https://github.com/chriselrod/BoxOptNewton.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chriselrod/BoxOptNewton.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/chriselrod/BoxOptNewton.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/chriselrod/BoxOptNewton.jl)

See the tests for an example problem.
This package implements a box-optimizer for non-linear programs that does not avoid the boundaries. This package only supports `SVector`s.
The goal is to be more useful for small-sized branch and bound problems.
