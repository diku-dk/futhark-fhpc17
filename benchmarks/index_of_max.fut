-- ==
-- input @ data/i32_2pow26
-- input @ data/i32_2pow18

import "operations"

let main [n] (xs: [n]i32): i32 =
  index_of_max xs
