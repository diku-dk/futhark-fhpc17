-- The operations used in the benchmarks.

import "/futlib/math"

module sum = {
  let redop (x: i32) (y: i32): i32 = x + y
  let ne = 0
}

let sum [n] (xs: [n]i32): i32 =
  reduce sum.redop sum.ne xs

module index_of_max = {
  let redop ((xv, xi): (i32,i32)) ((yv, yi): (i32,i32)): (i32,i32) =
    if xv < yv then (yv,yi)
    else if yv < xv then (xv,xi)
                    else -- Prefer lowest index if the values are equal.
                     if xi < yi then (xv,xi) else (yv,yi)

  let ne = (0, -1000) -- not actually neutral, but good enough.
}

let index_of_max [n] (xs: [n]i32): i32 =
  let (_, i) = reduce_comm index_of_max.redop index_of_max.ne (zip xs (iota n))
  in i

module mss = {
  type quad = (i32,i32,i32,i32)

  let redop((bx, lx, rx, tx): quad)
           ((by, ly, ry, ty): quad): quad =
    ( i32.max bx (i32.max by (rx + ly))
    , i32.max lx (tx+ly)
    , i32.max ry (rx+ty)
    , tx + ty)

  let ne = (0,0,0,0)

  let mapop (x: i32): quad =
    ( i32.max x 0, i32.max x 0, i32.max x 0, x)
}

let mss [n] (xs: [n]i32): i32 =
  let (x, _, _, _) = reduce mss.redop mss.ne (map mss.mapop xs)
  in x

module blackscholes = {

  let horner (x: f64): f64 =
    let (c1,c2,c3,c4,c5) = (0.31938153,-0.356563782,1.781477937,-1.821255978,1.330274429)
    in x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))))

  let cnd0 (d: f64): f64 =
    let k        = 1.0 / (1.0 + 0.2316419 * f64.abs d)
    let p        = horner(k)
    let rsqrt2pi = 0.39894228040143267793994605993438
    in rsqrt2pi * f64.exp(-0.5*d*d) * p

  let cnd (d: f64): f64 =
    let c = cnd0(d)
    in if 0.0 < d then 1.0 - c else c

  let mapop (r: f64) (v: f64) (days: i32) (day: i32): f64 =
    let (call, price, strike, years) =
      (day % 2 == 0,
       58.0 + 4.0 * f64 (1+day) / f64 days,
       65.0,
       f64 (1+day) / 365.0)
    let v_sqrtT = v * f64.sqrt years
    let d1      = (f64.log(price / strike) + (r + 0.5 * v * v) * years) / v_sqrtT
    let d2      = d1 - v_sqrtT
    let cndD1   = cnd d1
    let cndD2   = cnd d2
    let x_expRT = strike * f64.exp (-r * years)
    in if call then price * cndD1 - x_expRT * cndD2
       else x_expRT * (1.0 - cndD2) - price * (1.0 - cndD1)

  let redop (x: f64) (y: f64) = x + y

  let ne = 0.0
}

let blackscholes (r: f64) (v: f64) (days: i32): f64 =
  reduce_comm blackscholes.redop blackscholes.ne (map (blackscholes.mapop r v days) (iota days))
