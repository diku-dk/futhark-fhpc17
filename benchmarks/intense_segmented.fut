-- ==
-- input @ inputs/i32_2pow0_2pow26
-- input @ inputs/i32_2pow2_2pow24
-- input @ inputs/i32_2pow4_2pow22
-- input @ inputs/i32_2pow6_2pow20
-- input @ inputs/i32_2pow8_2pow18
-- input @ inputs/i32_2pow10_2pow16
-- input @ inputs/i32_2pow12_2pow14
-- input @ inputs/i32_2pow14_2pow12
-- input @ inputs/i32_2pow16_2pow10
-- input @ inputs/i32_2pow18_2pow8
-- input @ inputs/i32_2pow20_2pow6
-- input @ inputs/i32_2pow22_2pow4
-- input @ inputs/i32_2pow24_2pow2
-- input @ inputs/i32_2pow26_2pow0
--
-- input @ inputs/i32_2pow0_2pow18
-- input @ inputs/i32_2pow2_2pow16
-- input @ inputs/i32_2pow4_2pow14
-- input @ inputs/i32_2pow6_2pow12
-- input @ inputs/i32_2pow8_2pow10
-- input @ inputs/i32_2pow10_2pow8
-- input @ inputs/i32_2pow12_2pow6
-- input @ inputs/i32_2pow14_2pow4
-- input @ inputs/i32_2pow16_2pow2
-- input @ inputs/i32_2pow18_2pow0

import "operations"

let main [m] [n] (xss : [m][n]i32) : [m]i32 =
  map intense xss
