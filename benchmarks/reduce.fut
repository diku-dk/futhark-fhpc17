-- ==
-- input @ data/i32_2pow26
-- input @ data/i32_2pow18

let main [n] (xs: [n]i32): i32 =
  reduce (+) 0 xs
