-- ==
-- input @ data/blackscholes_2pow18
-- input @ data/blackscholes_2pow26

import "operations"

let main (r: f32) (v: f32) (days: i32): f32 =
  blackscholes r v days
