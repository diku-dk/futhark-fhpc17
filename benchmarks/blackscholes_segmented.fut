-- ==
-- input @ data/blackscholes_2pow0_2pow26
-- input @ data/blackscholes_2pow2_2pow24
-- input @ data/blackscholes_2pow4_2pow22
-- input @ data/blackscholes_2pow6_2pow20
-- input @ data/blackscholes_2pow8_2pow18
-- input @ data/blackscholes_2pow10_2pow16
-- input @ data/blackscholes_2pow12_2pow14
-- input @ data/blackscholes_2pow14_2pow12
-- input @ data/blackscholes_2pow16_2pow10
-- input @ data/blackscholes_2pow18_2pow8
-- input @ data/blackscholes_2pow20_2pow6
-- input @ data/blackscholes_2pow22_2pow4
-- input @ data/blackscholes_2pow24_2pow2
-- input @ data/blackscholes_2pow26_2pow0
--
-- input @ data/blackscholes_2pow0_2pow18
-- input @ data/blackscholes_2pow2_2pow16
-- input @ data/blackscholes_2pow4_2pow14
-- input @ data/blackscholes_2pow6_2pow12
-- input @ data/blackscholes_2pow8_2pow10
-- input @ data/blackscholes_2pow10_2pow8
-- input @ data/blackscholes_2pow12_2pow6
-- input @ data/blackscholes_2pow14_2pow4
-- input @ data/blackscholes_2pow16_2pow2
-- input @ data/blackscholes_2pow18_2pow0

import "operations"

let main [n] (rs: [n]f32) (vs: [n]f32) (days: i32): [n]f32 =
  map (\r v -> blackscholes r v days) rs vs
