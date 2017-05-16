-- The operations used in the benchmarks.

import "/futlib/math"

module sum = {
  let redop (x: i32) (y: i32): i32 = x + y
  let ne = 0
}

let sum [n] (xs: [n]i32): i32 =
  reduce sum.redop sum.ne xs

module intense = {
  let redop (x : i32) (y : i32): i32 =
    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    let xx = f32 x * f32 x
    let x = i32 (f32.sqrt(xx))

    in x + y

  let ne = 0
}

let intense [n] (xs: [n]i32): i32 =
  reduce intense.redop intense.ne xs

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
