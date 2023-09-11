module BoxOptNewton

using StaticArrays, ForwardDiff
using LinearAlgebra: Diagonal, UnitLowerTriangular

sigmoid(x) = 1 / (1 + exp(-x))
function sigmoid(x::ForwardDiff.Dual{T}) where {T}
  s = sigmoid(x.value)
  ForwardDiff.Dual{T}(s, muladd(-s, s, s) * x.partials)
end

struct Transform{F,A,B}
  f::F
  a::A
  b::B
end
transform(x::Number, a, b) = muladd(sigmoid(x), a, b)
transform(t::Transform, x) = map(transform, x, t.a, t.b)

@inline (t::Transform)(x) = t.f(transform(t, x))

const EXTREME = 8.0 # the most extreme number

# get scale and offsets for bounds
function getmulfactor(lbs, ubs)
  slb = sigmoid(-EXTREME)
  sub = sigmoid(EXTREME)
  # sigmoid range is from slb to sub
  scale = map(Base.Fix2(/, (sub - slb)), map(-, ubs, lbs))
  offset = map(-, lbs, map(Base.Fix2(*, slb), scale))
  scale, offset
end
bound(f::F, lbs, ubs) where {F} = Transform(f, getmulfactor(lbs, ubs)...)

@inline function safeldl(S::SMatrix{N,N,T}) where {N,T}
  A = MMatrix{N,N,T}(S)
  @fastmath @inbounds for k = 1:N
    # force positive definite
    Akk = max(A[k, k], T(1e-2))
    # we calc `inv` here, which is slow, so may as well store
    A[k, k] = invAkk = 1.0 / Akk
    for i = k+1:N
      A[i, k] = A[i, k] * invAkk
    end
    for i = k+1:N
      Aik = A[i, k] * Akk
      for j = k+1:i
        A[i, j] -= Aik * A[j, k]
      end
    end
  end
  return SMatrix(A)
end
function delta(f::F, x::SVector{N,T}) where {F,N,T}
  g = MVector{N,T}(undef)
  # `ForwardDiff.hessian` with a `DiffResults.ImmutableHessianResult`
  # allocates. Rather that debug, just implement it ourselves.
  H = ForwardDiff.jacobian(x) do x
    @inline
    _g = ForwardDiff.gradient(f, x)
    g .= ForwardDiff.value.(_g)
    _g
  end
  LD = safeldl(H) # H ≈ L*inv(D)*L'
  # we solve for L*inv(D)*L' * delta = -g
  d = UnitLowerTriangular(LD)' \ (Diagonal(LD) * (UnitLowerTriangular(LD) \ -g))
  SVector(d), SVector(g)
end

logit(x) = log(x / (1 - x))
untransform(x::Number, a, b) = logit((x - b) / a)
untransform(x, as, bs) = map(untransform, x, as, bs)
function untrf(x, lbs, ubs)
  scale, offset = getmulfactor(lbs, ubs)
  clamp.(untransform(x, scale, offset), -EXTREME, EXTREME)
end

function minimize(
  f::F,
  x::SVector;
  c = 0.5,
  τ = 0.5,
  N = 1_000,
  gnorm = 1e-10,
  fnorm = 1e-10,
  xnorm = 1e-10
) where {F}
  n = 0
  Fx = f(x)
  α = 1.0
  while n < N
    n += 1
    d, grad = delta(f, x)
    sum(abs2, d) <= gnorm^2 && return x
    d = map(d, x) do s, y
      y == EXTREME && return min(s, 0.0)
      y == -EXTREME && return max(s, 0.0)
      s
    end
    all(isfinite, d) || return x
    sum(abs2, grad) <= gnorm^2 && return x
    m = grad' * d
    t = -c * m
    t < 0 && return x
    xnew = clamp.(muladd.(α, d, x), -EXTREME, EXTREME)
    Fxnew = f(xnew)
    cond = Fx - Fxnew >= α * t
    if cond
      (Fx - Fxnew) <= fnorm && return xnew
      sum(abs2, xnew - x) <= xnorm^2 && return xnew
      xold = x
      Fxold = Fx
      x = xnew
      Fx = Fxnew
      while true # try and grow
        αnew = α / τ
        xnew = clamp.(muladd.(αnew, d, xold), -EXTREME, EXTREME)
        Fxnew = f(xnew)
        (Fxold - Fxnew < αnew * t) && break
        (Fxold - Fxnew) <= fnorm && return xnew
        x = xnew
        Fx = Fxnew
        α = αnew
      end
    else
      while true
        α *= τ
        xnew = clamp.(muladd.(α, d, x), -EXTREME, EXTREME)
        Fxnew = f(xnew)
        ((Fx - Fxnew) < (α * t)) && continue
        (Fx - Fxnew) <= fnorm && return xnew
        sum(abs2, xnew - x) <= xnorm^2 && return xnew
        x = xnew
        Fx = Fxnew
        break
      end
    end
  end
  x
end
function minimize(
  ff::F,
  _x::SVector{L},
  lbs::SVector{L},
  ubs::SVector{L};
  kwargs...
) where {F,L}
  f = bound(ff, lbs, ubs)
  x = untransform(_x, f.a, f.b)
  ret = minimize(f, x; kwargs...)
  transform(f, ret)
end
function minimize(ff::F, x, lbs, ubs; kwargs...) where {F}
  @assert length(x) == length(lbs) == length(ubs)
  Tuple(minimize(ff, SVector(x), SVector(lbs), SVector(ubs); kwargs...))
end
end
