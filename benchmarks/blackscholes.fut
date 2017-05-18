-- ==
-- input { 0.08f32 0.3f32 67108864 }
-- input { 0.08f32 0.3f32 262144 }

import "operations"

let main (r: f32) (v: f32) (days: i32): f32 =
  blackscholes r v days
