-- ==
-- compiled input @ inputs/blackscholes_2pow0_2pow26
-- output @ blackscholes_expected/blackscholes_2pow0_2pow26
-- compiled input @ inputs/blackscholes_2pow2_2pow24
-- output @ blackscholes_expected/blackscholes_2pow2_2pow24
-- compiled input @ inputs/blackscholes_2pow4_2pow22
-- output @ blackscholes_expected/blackscholes_2pow4_2pow22
-- compiled input @ inputs/blackscholes_2pow6_2pow20
-- output @ blackscholes_expected/blackscholes_2pow6_2pow20
-- compiled input @ inputs/blackscholes_2pow8_2pow18
-- output @ blackscholes_expected/blackscholes_2pow8_2pow18
-- compiled input @ inputs/blackscholes_2pow10_2pow16
-- output @ blackscholes_expected/blackscholes_2pow10_2pow16
-- compiled input @ inputs/blackscholes_2pow12_2pow14
-- output @ blackscholes_expected/blackscholes_2pow12_2pow14
-- compiled input @ inputs/blackscholes_2pow14_2pow12
-- output @ blackscholes_expected/blackscholes_2pow14_2pow12
-- compiled input @ inputs/blackscholes_2pow16_2pow10
-- output @ blackscholes_expected/blackscholes_2pow16_2pow10
-- compiled input @ inputs/blackscholes_2pow18_2pow8
-- output @ blackscholes_expected/blackscholes_2pow18_2pow8
-- compiled input @ inputs/blackscholes_2pow20_2pow6
-- output @ blackscholes_expected/blackscholes_2pow20_2pow6
-- compiled input @ inputs/blackscholes_2pow22_2pow4
-- output @ blackscholes_expected/blackscholes_2pow22_2pow4
-- compiled input @ inputs/blackscholes_2pow24_2pow2
-- output @ blackscholes_expected/blackscholes_2pow24_2pow2
-- compiled input @ inputs/blackscholes_2pow26_2pow0
-- output @ blackscholes_expected/blackscholes_2pow26_2pow0
--
-- compiled input @ inputs/blackscholes_2pow0_2pow18
-- output @ blackscholes_expected/blackscholes_2pow0_2pow18
-- compiled input @ inputs/blackscholes_2pow2_2pow16
-- output @ blackscholes_expected/blackscholes_2pow2_2pow16
-- compiled input @ inputs/blackscholes_2pow4_2pow14
-- output @ blackscholes_expected/blackscholes_2pow4_2pow14
-- compiled input @ inputs/blackscholes_2pow6_2pow12
-- output @ blackscholes_expected/blackscholes_2pow6_2pow12
-- compiled input @ inputs/blackscholes_2pow8_2pow10
-- output @ blackscholes_expected/blackscholes_2pow8_2pow10
-- compiled input @ inputs/blackscholes_2pow10_2pow8
-- output @ blackscholes_expected/blackscholes_2pow10_2pow8
-- compiled input @ inputs/blackscholes_2pow12_2pow6
-- output @ blackscholes_expected/blackscholes_2pow12_2pow6
-- compiled input @ inputs/blackscholes_2pow14_2pow4
-- output @ blackscholes_expected/blackscholes_2pow14_2pow4
-- compiled input @ inputs/blackscholes_2pow16_2pow2
-- output @ blackscholes_expected/blackscholes_2pow16_2pow2
-- compiled input @ inputs/blackscholes_2pow18_2pow0
-- output @ blackscholes_expected/blackscholes_2pow18_2pow0

import "operations"

let main [n] (rs: [n]f64) (vs: [n]f64) (days: i32): [n]f64 =
  let xs = map (\r v -> scan blackscholes.redop blackscholes.ne (map (blackscholes.mapop r v days) (iota days))) rs vs
  in xs[:,days-1]
