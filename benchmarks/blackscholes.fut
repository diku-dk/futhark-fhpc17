-- ==
-- compiled input @ inputs/blackscholes_2pow18
-- output @ blackscholes_expected/blackscholes_2pow18
-- compiled input @ inputs/blackscholes_2pow26
-- output @ blackscholes_expected/blackscholes_2pow26

import "operations"

let main (r: f64) (v: f64) (days: i32): f64 =
  blackscholes r v days
