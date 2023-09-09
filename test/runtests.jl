using BoxOptNewton, StaticArrays, ForwardDiff
using Test

# minimize fcore
# (2*u1 + u2 + u1*u2) / (u1 * u2)
# 2/u2 + 1/u1 + 1
# or maximize
# (u1 * u2) / (2*u1 + u2 + u1*u2)
@inline function fcore(u1, u2)
  # cost is
  # u1 cost considered roughly twice as high if unrolled
  # all else equal, we prefer larger slack variables
  (2 * u1 + u2 + u1 * u2) / (u1 * u2)
end
@inline function gcore(u1, u2)
  # must be <= 0
  # constraint is
  # u1 + u1*u2 <= 32
  # but we must specify equality constraint, g(x) = 0
  SVector((u1 + u1 * u2 + 1 - 32,))
end
_softplus(x) = log1p(exp(x))
function _softplus(x::ForwardDiff.Dual{T}) where {T}
  sx = _softplus(ForwardDiff.value(x))
  px = BoxOptNewton.sigmoid(ForwardDiff.value(x))
  ForwardDiff.Dual{T}(sx, px * ForwardDiff.partials(x))
end
softplus(x) = 0.125_softplus(8.0x)
function fsoft(x::NTuple{2})
  f = fcore(x...)
  g = gcore(x...)
  f + sum(softplus, g)
end
fsoft(x::SVector{2}) = fsoft((x[1], x[2]))

@testset "BoxOptNewton.jl" begin
  opt1 = BoxOptNewton.minimize(fsoft, (2, 2), (1, 1), (32, 32))
  @test SVector(opt1) ≈ SVector(3.4567718680186568, 7.799906157078232) rtol =
    1e-6
  opt2 = BoxOptNewton.minimize(fsoft, (2, 2), (1, 1), (3, 32))
  @test SVector(opt2) ≈ SVector(3.0, 9.132451832031007) rtol = 1e-6
end
