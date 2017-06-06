-- ==
-- compiled input @ inputs/i32_2pow0_2pow26
-- output @ mss_expected/i32_2pow0_2pow26
-- compiled input @ inputs/i32_2pow2_2pow24
-- output @ mss_expected/i32_2pow2_2pow24
-- compiled input @ inputs/i32_2pow4_2pow22
-- output @ mss_expected/i32_2pow4_2pow22
-- compiled input @ inputs/i32_2pow6_2pow20
-- output @ mss_expected/i32_2pow6_2pow20
-- compiled input @ inputs/i32_2pow8_2pow18
-- output @ mss_expected/i32_2pow8_2pow18
-- compiled input @ inputs/i32_2pow10_2pow16
-- output @ mss_expected/i32_2pow10_2pow16
-- compiled input @ inputs/i32_2pow12_2pow14
-- output @ mss_expected/i32_2pow12_2pow14
-- compiled input @ inputs/i32_2pow14_2pow12
-- output @ mss_expected/i32_2pow14_2pow12
-- compiled input @ inputs/i32_2pow16_2pow10
-- output @ mss_expected/i32_2pow16_2pow10
-- compiled input @ inputs/i32_2pow18_2pow8
-- output @ mss_expected/i32_2pow18_2pow8
-- compiled input @ inputs/i32_2pow20_2pow6
-- output @ mss_expected/i32_2pow20_2pow6
-- compiled input @ inputs/i32_2pow22_2pow4
-- output @ mss_expected/i32_2pow22_2pow4
-- compiled input @ inputs/i32_2pow24_2pow2
-- output @ mss_expected/i32_2pow24_2pow2
-- compiled input @ inputs/i32_2pow26_2pow0
-- output @ mss_expected/i32_2pow26_2pow0
--
-- compiled input @ inputs/i32_2pow0_2pow18
-- output @ mss_expected/i32_2pow0_2pow18
-- compiled input @ inputs/i32_2pow2_2pow16
-- output @ mss_expected/i32_2pow2_2pow16
-- compiled input @ inputs/i32_2pow4_2pow14
-- output @ mss_expected/i32_2pow4_2pow14
-- compiled input @ inputs/i32_2pow6_2pow12
-- output @ mss_expected/i32_2pow6_2pow12
-- compiled input @ inputs/i32_2pow8_2pow10
-- output @ mss_expected/i32_2pow8_2pow10
-- compiled input @ inputs/i32_2pow10_2pow8
-- output @ mss_expected/i32_2pow10_2pow8
-- compiled input @ inputs/i32_2pow12_2pow6
-- output @ mss_expected/i32_2pow12_2pow6
-- compiled input @ inputs/i32_2pow14_2pow4
-- output @ mss_expected/i32_2pow14_2pow4
-- compiled input @ inputs/i32_2pow16_2pow2
-- output @ mss_expected/i32_2pow16_2pow2
-- compiled input @ inputs/i32_2pow18_2pow0
-- output @ mss_expected/i32_2pow18_2pow0

import "operations"

let mss [n] (xs: [n]i32): [n]i32 =
  let (x, _, _, _) = unzip (scan mss.redop mss.ne (map mss.mapop xs))
  in x

let main [m][n] (xss: [m][n]i32) =
  (map mss xss)[:,n-1]
