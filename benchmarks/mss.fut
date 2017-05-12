-- ==
-- input @ data/i32_2pow26
-- input @ data/i32_2pow18

import "/futlib/math"

type quad = (i32,i32,i32,i32)

let redOp((bx, lx, rx, tx): quad)
         ((by, ly, ry, ty): quad): quad =
  ( i32.max bx (i32.max by (rx + ly))
  , i32.max lx (tx+ly)
  , i32.max ry (rx+ty)
  , tx + ty)

let mapOp (x: i32): quad =
  ( i32.max x 0, i32.max x 0, i32.max x 0, x)

let main(xs: []i32): i32 =
  let (x, _, _, _) = reduce redOp (0,0,0,0) (map mapOp xs)
  in x
