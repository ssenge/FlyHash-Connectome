# FlyHash++ exploratory study

AP cutoff, splits and all hyperparameters are recorded in the JSON. Intervals are paired-seed t intervals, exploratory and unadjusted.
Timing uses full code scans (including OR); this is not an optimized ANN throughput benchmark.
Frontiers are selected on validation quality and use projection arithmetic plus fixed-width code storage.

| Dataset | Method/config | AP (mean) | Recall | Projection ops | Code bits |
|---|---|---:|---:|---:|---:|
| sift | fly:m128:s3:k4 | 0.1032 | 0.2138 | 384 | 28 |
| sift | balanced:m128:s3:k4 | 0.1015 | 0.2086 | 384 | 28 |
| sift | diverse:m128:s3:k4 | 0.1058 | 0.2193 | 384 | 28 |
| sift | calibrated:m128:s3:k4 | 0.0945 | 0.1948 | 384 | 28 |
| sift | zscore:m128:s3:k4 | 0.0948 | 0.1938 | 384 | 28 |
| sift | balanced_calibrated:m128:s3:k4 | 0.0961 | 0.1901 | 384 | 28 |
| sift | diverse_calibrated:m128:s3:k4 | 0.0969 | 0.1943 | 384 | 28 |
| sift | rank:m128:s3:k4 | 0.1211 | 0.2383 | 384 | 28 |
| sift | adaptive_relative:m128:s3:k4 | 0.0711 | 0.1552 | 384 | 128 |
| sift | adaptive_external:m128:s3:k4 | 0.0614 | 0.1354 | 384 | 128 |
| sift | scale_tag:m128:s3:k4 | 0.1051 | 0.2198 | 384 | 36 |
| sift | two_global:m128:s3:k4 | 0.1136 | 0.2250 | 384 | 28 |
| sift | two_concat:m128:s3:k4 | 0.0990 | 0.2029 | 384 | 28 |
| sift | two_or:m128:s3:k4 | 0.0988 | 0.2023 | 384 | 28 |
| sift | densefly:m128:s3:k4 | 0.3771 | 0.4958 | 384 | 128 |
| sift | densefly_probe:m128:s3:k4 | 0.3337 | 0.4370 | 384 | 144 |
| sift | biohash:m128:s3:k4 | 0.1153 | 0.2320 | 8192 | 28 |
| sift | biohash_untrained:m128:s3:k4 | 0.1148 | 0.2310 | 8192 | 28 |
| sift | sparse_bio:m128:s3:k4 | 0.0800 | 0.1839 | 768 | 28 |
| sift | sparse_untrained:m128:s3:k4 | 0.0785 | 0.1846 | 768 | 28 |
| sift | gaussian_real:m128:s3:k4 | 0.0918 | 0.2005 | 256 | 128 |
| sift | gaussian_sign:m128:s3:k4 | 0.0229 | 0.0784 | 256 | 4 |
| sift | gaussian_ops:m128:s3:k4 | 0.1369 | 0.2620 | 384 | 192 |
| sift | gaussian_storage:m128:s3:k4 | 0.1506 | 0.2786 | 1792 | 28 |
| sift | fly:m128:s3:k8 | 0.1386 | 0.2534 | 384 | 56 |
| sift | balanced:m128:s3:k8 | 0.1446 | 0.2622 | 384 | 56 |
| sift | diverse:m128:s3:k8 | 0.1391 | 0.2591 | 384 | 56 |
| sift | calibrated:m128:s3:k8 | 0.1404 | 0.2526 | 384 | 56 |
| sift | zscore:m128:s3:k8 | 0.1448 | 0.2599 | 384 | 56 |
| sift | balanced_calibrated:m128:s3:k8 | 0.1558 | 0.2693 | 384 | 56 |
| sift | diverse_calibrated:m128:s3:k8 | 0.1514 | 0.2604 | 384 | 56 |
| sift | rank:m128:s3:k8 | 0.1779 | 0.3031 | 384 | 56 |
| sift | adaptive_relative:m128:s3:k8 | 0.1185 | 0.2240 | 384 | 128 |
| sift | adaptive_external:m128:s3:k8 | 0.1055 | 0.1969 | 384 | 128 |
| sift | scale_tag:m128:s3:k8 | 0.1464 | 0.2685 | 384 | 64 |
| sift | two_global:m128:s3:k8 | 0.1431 | 0.2560 | 384 | 56 |
| sift | two_concat:m128:s3:k8 | 0.1427 | 0.2565 | 384 | 56 |
| sift | two_or:m128:s3:k8 | 0.1424 | 0.2563 | 384 | 56 |
| sift | densefly:m128:s3:k8 | 0.3771 | 0.4958 | 384 | 128 |
| sift | densefly_probe:m128:s3:k8 | 0.3337 | 0.4370 | 384 | 144 |
| sift | biohash:m128:s3:k8 | 0.1928 | 0.3263 | 8192 | 56 |
| sift | biohash_untrained:m128:s3:k8 | 0.1917 | 0.3268 | 8192 | 56 |
| sift | sparse_bio:m128:s3:k8 | 0.1763 | 0.3036 | 768 | 56 |
| sift | sparse_untrained:m128:s3:k8 | 0.1809 | 0.3102 | 768 | 56 |
| sift | gaussian_real:m128:s3:k8 | 0.1752 | 0.3086 | 512 | 256 |
| sift | gaussian_sign:m128:s3:k8 | 0.0539 | 0.1526 | 512 | 8 |
| sift | gaussian_ops:m128:s3:k8 | 0.1369 | 0.2620 | 384 | 192 |
| sift | gaussian_storage:m128:s3:k8 | 0.2749 | 0.4120 | 3584 | 56 |
| sift | fly:m128:s6:k4 | 0.1183 | 0.2333 | 768 | 28 |
| sift | balanced:m128:s6:k4 | 0.1176 | 0.2307 | 768 | 28 |
| sift | diverse:m128:s6:k4 | 0.1186 | 0.2385 | 768 | 28 |
| sift | calibrated:m128:s6:k4 | 0.1051 | 0.2102 | 768 | 28 |
| sift | zscore:m128:s6:k4 | 0.1093 | 0.2107 | 768 | 28 |
| sift | balanced_calibrated:m128:s6:k4 | 0.1112 | 0.2146 | 768 | 28 |
| sift | diverse_calibrated:m128:s6:k4 | 0.1152 | 0.2190 | 768 | 28 |
| sift | rank:m128:s6:k4 | 0.1200 | 0.2435 | 768 | 28 |
| sift | adaptive_relative:m128:s6:k4 | 0.0982 | 0.1836 | 768 | 128 |
| sift | adaptive_external:m128:s6:k4 | 0.0868 | 0.1604 | 768 | 128 |
| sift | scale_tag:m128:s6:k4 | 0.1227 | 0.2393 | 768 | 36 |
| sift | two_global:m128:s6:k4 | 0.1229 | 0.2375 | 768 | 28 |
| sift | two_concat:m128:s6:k4 | 0.1188 | 0.2266 | 768 | 28 |
| sift | two_or:m128:s6:k4 | 0.1188 | 0.2263 | 768 | 28 |
| sift | densefly:m128:s6:k4 | 0.4057 | 0.5214 | 768 | 128 |
| sift | densefly_probe:m128:s6:k4 | 0.3609 | 0.4612 | 768 | 144 |
| sift | biohash:m128:s6:k4 | 0.1153 | 0.2320 | 8192 | 28 |
| sift | biohash_untrained:m128:s6:k4 | 0.1148 | 0.2310 | 8192 | 28 |
| sift | sparse_bio:m128:s6:k4 | 0.0919 | 0.2023 | 1536 | 28 |
| sift | sparse_untrained:m128:s6:k4 | 0.0917 | 0.2003 | 1536 | 28 |
| sift | gaussian_real:m128:s6:k4 | 0.0918 | 0.2005 | 256 | 128 |
| sift | gaussian_sign:m128:s6:k4 | 0.0229 | 0.0784 | 256 | 4 |
| sift | gaussian_ops:m128:s6:k4 | 0.2313 | 0.3669 | 768 | 384 |
| sift | gaussian_storage:m128:s6:k4 | 0.1506 | 0.2786 | 1792 | 28 |
| sift | fly:m128:s6:k8 | 0.1860 | 0.3133 | 768 | 56 |
| sift | balanced:m128:s6:k8 | 0.1865 | 0.3125 | 768 | 56 |
| sift | diverse:m128:s6:k8 | 0.1803 | 0.3102 | 768 | 56 |
| sift | calibrated:m128:s6:k8 | 0.1846 | 0.3068 | 768 | 56 |
| sift | zscore:m128:s6:k8 | 0.1839 | 0.3120 | 768 | 56 |
| sift | balanced_calibrated:m128:s6:k8 | 0.1862 | 0.3081 | 768 | 56 |
| sift | diverse_calibrated:m128:s6:k8 | 0.1935 | 0.3182 | 768 | 56 |
| sift | rank:m128:s6:k8 | 0.2008 | 0.3318 | 768 | 56 |
| sift | adaptive_relative:m128:s6:k8 | 0.1755 | 0.2969 | 768 | 128 |
| sift | adaptive_external:m128:s6:k8 | 0.1519 | 0.2529 | 768 | 128 |
| sift | scale_tag:m128:s6:k8 | 0.1918 | 0.3177 | 768 | 64 |
| sift | two_global:m128:s6:k8 | 0.1974 | 0.3331 | 768 | 56 |
| sift | two_concat:m128:s6:k8 | 0.1842 | 0.3169 | 768 | 56 |
| sift | two_or:m128:s6:k8 | 0.1843 | 0.3174 | 768 | 56 |
| sift | densefly:m128:s6:k8 | 0.4057 | 0.5214 | 768 | 128 |
| sift | densefly_probe:m128:s6:k8 | 0.3609 | 0.4612 | 768 | 144 |
| sift | biohash:m128:s6:k8 | 0.1928 | 0.3263 | 8192 | 56 |
| sift | biohash_untrained:m128:s6:k8 | 0.1917 | 0.3268 | 8192 | 56 |
| sift | sparse_bio:m128:s6:k8 | 0.1573 | 0.2846 | 1536 | 56 |
| sift | sparse_untrained:m128:s6:k8 | 0.1527 | 0.2823 | 1536 | 56 |
| sift | gaussian_real:m128:s6:k8 | 0.1752 | 0.3086 | 512 | 256 |
| sift | gaussian_sign:m128:s6:k8 | 0.0539 | 0.1526 | 512 | 8 |
| sift | gaussian_ops:m128:s6:k8 | 0.2313 | 0.3669 | 768 | 384 |
| sift | gaussian_storage:m128:s6:k8 | 0.2749 | 0.4120 | 3584 | 56 |
| sift | fly:m256:s3:k4 | 0.1152 | 0.2258 | 768 | 32 |
| sift | balanced:m256:s3:k4 | 0.1194 | 0.2328 | 768 | 32 |
| sift | diverse:m256:s3:k4 | 0.1205 | 0.2336 | 768 | 32 |
| sift | calibrated:m256:s3:k4 | 0.1010 | 0.1951 | 768 | 32 |
| sift | zscore:m256:s3:k4 | 0.0995 | 0.1924 | 768 | 32 |
| sift | balanced_calibrated:m256:s3:k4 | 0.0879 | 0.1846 | 768 | 32 |
| sift | diverse_calibrated:m256:s3:k4 | 0.0894 | 0.1826 | 768 | 32 |
| sift | rank:m256:s3:k4 | 0.1404 | 0.2620 | 768 | 32 |
| sift | adaptive_relative:m256:s3:k4 | 0.0792 | 0.1419 | 768 | 256 |
| sift | adaptive_external:m256:s3:k4 | 0.0689 | 0.1268 | 768 | 256 |
| sift | scale_tag:m256:s3:k4 | 0.1229 | 0.2398 | 768 | 40 |
| sift | two_global:m256:s3:k4 | 0.1194 | 0.2391 | 768 | 32 |
| sift | two_concat:m256:s3:k4 | 0.1186 | 0.2349 | 768 | 32 |
| sift | two_or:m256:s3:k4 | 0.1184 | 0.2344 | 768 | 32 |
| sift | densefly:m256:s3:k4 | 0.4334 | 0.5380 | 768 | 256 |
| sift | densefly_probe:m256:s3:k4 | 0.3739 | 0.4737 | 768 | 272 |
| sift | biohash:m256:s3:k4 | 0.1245 | 0.2393 | 16384 | 32 |
| sift | biohash_untrained:m256:s3:k4 | 0.1247 | 0.2419 | 16384 | 32 |
| sift | sparse_bio:m256:s3:k4 | 0.0853 | 0.1901 | 1536 | 32 |
| sift | sparse_untrained:m256:s3:k4 | 0.0843 | 0.1896 | 1536 | 32 |
| sift | gaussian_real:m256:s3:k4 | 0.0918 | 0.2005 | 256 | 128 |
| sift | gaussian_sign:m256:s3:k4 | 0.0229 | 0.0784 | 256 | 4 |
| sift | gaussian_ops:m256:s3:k4 | 0.2313 | 0.3669 | 768 | 384 |
| sift | gaussian_storage:m256:s3:k4 | 0.1981 | 0.3354 | 2048 | 32 |
| sift | fly:m256:s3:k8 | 0.1571 | 0.2789 | 768 | 64 |
| sift | balanced:m256:s3:k8 | 0.1594 | 0.2826 | 768 | 64 |
| sift | diverse:m256:s3:k8 | 0.1777 | 0.3062 | 768 | 64 |
| sift | calibrated:m256:s3:k8 | 0.1464 | 0.2578 | 768 | 64 |
| sift | zscore:m256:s3:k8 | 0.1438 | 0.2547 | 768 | 64 |
| sift | balanced_calibrated:m256:s3:k8 | 0.1439 | 0.2518 | 768 | 64 |
| sift | diverse_calibrated:m256:s3:k8 | 0.1447 | 0.2565 | 768 | 64 |
| sift | rank:m256:s3:k8 | 0.1896 | 0.3135 | 768 | 64 |
| sift | adaptive_relative:m256:s3:k8 | 0.1112 | 0.2031 | 768 | 256 |
| sift | adaptive_external:m256:s3:k8 | 0.0963 | 0.1737 | 768 | 256 |
| sift | scale_tag:m256:s3:k8 | 0.1623 | 0.2880 | 768 | 72 |
| sift | two_global:m256:s3:k8 | 0.1812 | 0.2995 | 768 | 64 |
| sift | two_concat:m256:s3:k8 | 0.1661 | 0.2938 | 768 | 64 |
| sift | two_or:m256:s3:k8 | 0.1655 | 0.2922 | 768 | 64 |
| sift | densefly:m256:s3:k8 | 0.4334 | 0.5380 | 768 | 256 |
| sift | densefly_probe:m256:s3:k8 | 0.3739 | 0.4737 | 768 | 272 |
| sift | biohash:m256:s3:k8 | 0.2072 | 0.3422 | 16384 | 64 |
| sift | biohash_untrained:m256:s3:k8 | 0.2048 | 0.3393 | 16384 | 64 |
| sift | sparse_bio:m256:s3:k8 | 0.1766 | 0.3091 | 1536 | 64 |
| sift | sparse_untrained:m256:s3:k8 | 0.1774 | 0.3083 | 1536 | 64 |
| sift | gaussian_real:m256:s3:k8 | 0.1752 | 0.3086 | 512 | 256 |
| sift | gaussian_sign:m256:s3:k8 | 0.0539 | 0.1526 | 512 | 8 |
| sift | gaussian_ops:m256:s3:k8 | 0.2313 | 0.3669 | 768 | 384 |
| sift | gaussian_storage:m256:s3:k8 | 0.2926 | 0.4203 | 4096 | 64 |
| sift | fly:m256:s6:k4 | 0.1281 | 0.2456 | 1536 | 32 |
| sift | balanced:m256:s6:k4 | 0.1161 | 0.2365 | 1536 | 32 |
| sift | diverse:m256:s6:k4 | 0.1364 | 0.2602 | 1536 | 32 |
| sift | calibrated:m256:s6:k4 | 0.1086 | 0.2057 | 1536 | 32 |
| sift | zscore:m256:s6:k4 | 0.1259 | 0.2250 | 1536 | 32 |
| sift | balanced_calibrated:m256:s6:k4 | 0.1015 | 0.1917 | 1536 | 32 |
| sift | diverse_calibrated:m256:s6:k4 | 0.1125 | 0.2089 | 1536 | 32 |
| sift | rank:m256:s6:k4 | 0.1236 | 0.2414 | 1536 | 32 |
| sift | adaptive_relative:m256:s6:k4 | 0.0797 | 0.1406 | 1536 | 256 |
| sift | adaptive_external:m256:s6:k4 | 0.0794 | 0.1313 | 1536 | 256 |
| sift | scale_tag:m256:s6:k4 | 0.1218 | 0.2474 | 1536 | 40 |
| sift | two_global:m256:s6:k4 | 0.1325 | 0.2505 | 1536 | 32 |
| sift | two_concat:m256:s6:k4 | 0.1281 | 0.2448 | 1536 | 32 |
| sift | two_or:m256:s6:k4 | 0.1274 | 0.2427 | 1536 | 32 |
| sift | densefly:m256:s6:k4 | 0.5013 | 0.5961 | 1536 | 256 |
| sift | densefly_probe:m256:s6:k4 | 0.4206 | 0.5078 | 1536 | 272 |
| sift | biohash:m256:s6:k4 | 0.1245 | 0.2393 | 16384 | 32 |
| sift | biohash_untrained:m256:s6:k4 | 0.1247 | 0.2419 | 16384 | 32 |
| sift | sparse_bio:m256:s6:k4 | 0.1163 | 0.2370 | 3072 | 32 |
| sift | sparse_untrained:m256:s6:k4 | 0.1082 | 0.2299 | 3072 | 32 |
| sift | gaussian_real:m256:s6:k4 | 0.0918 | 0.2005 | 256 | 128 |
| sift | gaussian_sign:m256:s6:k4 | 0.0229 | 0.0784 | 256 | 4 |
| sift | gaussian_ops:m256:s6:k4 | 0.3745 | 0.4969 | 1536 | 768 |
| sift | gaussian_storage:m256:s6:k4 | 0.1981 | 0.3354 | 2048 | 32 |
| sift | fly:m256:s6:k8 | 0.1901 | 0.3247 | 1536 | 64 |
| sift | balanced:m256:s6:k8 | 0.1917 | 0.3185 | 1536 | 64 |
| sift | diverse:m256:s6:k8 | 0.2028 | 0.3344 | 1536 | 64 |
| sift | calibrated:m256:s6:k8 | 0.1870 | 0.3073 | 1536 | 64 |
| sift | zscore:m256:s6:k8 | 0.1907 | 0.3125 | 1536 | 64 |
| sift | balanced_calibrated:m256:s6:k8 | 0.1643 | 0.2844 | 1536 | 64 |
| sift | diverse_calibrated:m256:s6:k8 | 0.1814 | 0.2940 | 1536 | 64 |
| sift | rank:m256:s6:k8 | 0.1906 | 0.3146 | 1536 | 64 |
| sift | adaptive_relative:m256:s6:k8 | 0.1374 | 0.2221 | 1536 | 256 |
| sift | adaptive_external:m256:s6:k8 | 0.1281 | 0.1966 | 1536 | 256 |
| sift | scale_tag:m256:s6:k8 | 0.1922 | 0.3174 | 1536 | 72 |
| sift | two_global:m256:s6:k8 | 0.1946 | 0.3232 | 1536 | 64 |
| sift | two_concat:m256:s6:k8 | 0.1943 | 0.3268 | 1536 | 64 |
| sift | two_or:m256:s6:k8 | 0.1939 | 0.3258 | 1536 | 64 |
| sift | densefly:m256:s6:k8 | 0.5013 | 0.5961 | 1536 | 256 |
| sift | densefly_probe:m256:s6:k8 | 0.4206 | 0.5078 | 1536 | 272 |
| sift | biohash:m256:s6:k8 | 0.2072 | 0.3422 | 16384 | 64 |
| sift | biohash_untrained:m256:s6:k8 | 0.2048 | 0.3393 | 16384 | 64 |
| sift | sparse_bio:m256:s6:k8 | 0.1842 | 0.3115 | 3072 | 64 |
| sift | sparse_untrained:m256:s6:k8 | 0.1740 | 0.2971 | 3072 | 64 |
| sift | gaussian_real:m256:s6:k8 | 0.1752 | 0.3086 | 512 | 256 |
| sift | gaussian_sign:m256:s6:k8 | 0.0539 | 0.1526 | 512 | 8 |
| sift | gaussian_ops:m256:s6:k8 | 0.3745 | 0.4969 | 1536 | 768 |
| sift | gaussian_storage:m256:s6:k8 | 0.2926 | 0.4203 | 4096 | 64 |
| mnist | fly:m128:s3:k4 | 0.1567 | 0.2703 | 384 | 28 |
| mnist | balanced:m128:s3:k4 | 0.1419 | 0.2573 | 384 | 28 |
| mnist | diverse:m128:s3:k4 | 0.1455 | 0.2544 | 384 | 28 |
| mnist | calibrated:m128:s3:k4 | 0.1299 | 0.2341 | 384 | 28 |
| mnist | zscore:m128:s3:k4 | 0.1337 | 0.2333 | 384 | 28 |
| mnist | balanced_calibrated:m128:s3:k4 | 0.1239 | 0.2190 | 384 | 28 |
| mnist | diverse_calibrated:m128:s3:k4 | 0.1302 | 0.2365 | 384 | 28 |
| mnist | rank:m128:s3:k4 | 0.1572 | 0.2703 | 384 | 28 |
| mnist | adaptive_relative:m128:s3:k4 | 0.1205 | 0.1945 | 384 | 128 |
| mnist | adaptive_external:m128:s3:k4 | 0.0910 | 0.1557 | 384 | 128 |
| mnist | scale_tag:m128:s3:k4 | 0.1637 | 0.2781 | 384 | 36 |
| mnist | two_global:m128:s3:k4 | 0.1599 | 0.2714 | 384 | 28 |
| mnist | two_concat:m128:s3:k4 | 0.1494 | 0.2578 | 384 | 28 |
| mnist | two_or:m128:s3:k4 | 0.1459 | 0.2497 | 384 | 28 |
| mnist | densefly:m128:s3:k4 | 0.4807 | 0.5680 | 384 | 128 |
| mnist | densefly_probe:m128:s3:k4 | 0.4212 | 0.4940 | 384 | 144 |
| mnist | biohash:m128:s3:k4 | 0.1891 | 0.3052 | 8192 | 28 |
| mnist | biohash_untrained:m128:s3:k4 | 0.1829 | 0.2961 | 8192 | 28 |
| mnist | sparse_bio:m128:s3:k4 | 0.1515 | 0.2602 | 768 | 28 |
| mnist | sparse_untrained:m128:s3:k4 | 0.1510 | 0.2630 | 768 | 28 |
| mnist | gaussian_real:m128:s3:k4 | 0.1139 | 0.2102 | 256 | 128 |
| mnist | gaussian_sign:m128:s3:k4 | 0.0221 | 0.0849 | 256 | 4 |
| mnist | gaussian_ops:m128:s3:k4 | 0.1880 | 0.3047 | 384 | 192 |
| mnist | gaussian_storage:m128:s3:k4 | 0.2359 | 0.3549 | 1792 | 28 |
| mnist | fly:m128:s3:k8 | 0.2141 | 0.3294 | 384 | 56 |
| mnist | balanced:m128:s3:k8 | 0.2078 | 0.3182 | 384 | 56 |
| mnist | diverse:m128:s3:k8 | 0.2210 | 0.3326 | 384 | 56 |
| mnist | calibrated:m128:s3:k8 | 0.2018 | 0.3128 | 384 | 56 |
| mnist | zscore:m128:s3:k8 | 0.2016 | 0.3141 | 384 | 56 |
| mnist | balanced_calibrated:m128:s3:k8 | 0.2022 | 0.3122 | 384 | 56 |
| mnist | diverse_calibrated:m128:s3:k8 | 0.2228 | 0.3276 | 384 | 56 |
| mnist | rank:m128:s3:k8 | 0.2321 | 0.3435 | 384 | 56 |
| mnist | adaptive_relative:m128:s3:k8 | 0.1845 | 0.2859 | 384 | 128 |
| mnist | adaptive_external:m128:s3:k8 | 0.1437 | 0.2302 | 384 | 128 |
| mnist | scale_tag:m128:s3:k8 | 0.2277 | 0.3424 | 384 | 64 |
| mnist | two_global:m128:s3:k8 | 0.2231 | 0.3273 | 384 | 56 |
| mnist | two_concat:m128:s3:k8 | 0.2259 | 0.3315 | 384 | 56 |
| mnist | two_or:m128:s3:k8 | 0.2259 | 0.3318 | 384 | 56 |
| mnist | densefly:m128:s3:k8 | 0.4807 | 0.5680 | 384 | 128 |
| mnist | densefly_probe:m128:s3:k8 | 0.4212 | 0.4940 | 384 | 144 |
| mnist | biohash:m128:s3:k8 | 0.2833 | 0.3979 | 8192 | 56 |
| mnist | biohash_untrained:m128:s3:k8 | 0.2824 | 0.3971 | 8192 | 56 |
| mnist | sparse_bio:m128:s3:k8 | 0.2468 | 0.3612 | 768 | 56 |
| mnist | sparse_untrained:m128:s3:k8 | 0.2436 | 0.3578 | 768 | 56 |
| mnist | gaussian_real:m128:s3:k8 | 0.2390 | 0.3557 | 512 | 256 |
| mnist | gaussian_sign:m128:s3:k8 | 0.0632 | 0.1581 | 512 | 8 |
| mnist | gaussian_ops:m128:s3:k8 | 0.1880 | 0.3047 | 384 | 192 |
| mnist | gaussian_storage:m128:s3:k8 | 0.3894 | 0.5042 | 3584 | 56 |
| mnist | fly:m128:s6:k4 | 0.1616 | 0.2771 | 768 | 28 |
| mnist | balanced:m128:s6:k4 | 0.1632 | 0.2716 | 768 | 28 |
| mnist | diverse:m128:s6:k4 | 0.1711 | 0.2797 | 768 | 28 |
| mnist | calibrated:m128:s6:k4 | 0.1565 | 0.2497 | 768 | 28 |
| mnist | zscore:m128:s6:k4 | 0.1545 | 0.2555 | 768 | 28 |
| mnist | balanced_calibrated:m128:s6:k4 | 0.1518 | 0.2615 | 768 | 28 |
| mnist | diverse_calibrated:m128:s6:k4 | 0.1578 | 0.2638 | 768 | 28 |
| mnist | rank:m128:s6:k4 | 0.1788 | 0.2826 | 768 | 28 |
| mnist | adaptive_relative:m128:s6:k4 | 0.1193 | 0.2034 | 768 | 128 |
| mnist | adaptive_external:m128:s6:k4 | 0.0974 | 0.1581 | 768 | 128 |
| mnist | scale_tag:m128:s6:k4 | 0.1865 | 0.2969 | 768 | 36 |
| mnist | two_global:m128:s6:k4 | 0.1756 | 0.2836 | 768 | 28 |
| mnist | two_concat:m128:s6:k4 | 0.1677 | 0.2807 | 768 | 28 |
| mnist | two_or:m128:s6:k4 | 0.1645 | 0.2737 | 768 | 28 |
| mnist | densefly:m128:s6:k4 | 0.5046 | 0.5935 | 768 | 128 |
| mnist | densefly_probe:m128:s6:k4 | 0.4436 | 0.5201 | 768 | 144 |
| mnist | biohash:m128:s6:k4 | 0.1891 | 0.3052 | 8192 | 28 |
| mnist | biohash_untrained:m128:s6:k4 | 0.1829 | 0.2961 | 8192 | 28 |
| mnist | sparse_bio:m128:s6:k4 | 0.1481 | 0.2596 | 1536 | 28 |
| mnist | sparse_untrained:m128:s6:k4 | 0.1459 | 0.2523 | 1536 | 28 |
| mnist | gaussian_real:m128:s6:k4 | 0.1139 | 0.2102 | 256 | 128 |
| mnist | gaussian_sign:m128:s6:k4 | 0.0221 | 0.0849 | 256 | 4 |
| mnist | gaussian_ops:m128:s6:k4 | 0.2845 | 0.3982 | 768 | 384 |
| mnist | gaussian_storage:m128:s6:k4 | 0.2359 | 0.3549 | 1792 | 28 |
| mnist | fly:m128:s6:k8 | 0.2527 | 0.3701 | 768 | 56 |
| mnist | balanced:m128:s6:k8 | 0.2564 | 0.3724 | 768 | 56 |
| mnist | diverse:m128:s6:k8 | 0.2672 | 0.3828 | 768 | 56 |
| mnist | calibrated:m128:s6:k8 | 0.2383 | 0.3497 | 768 | 56 |
| mnist | zscore:m128:s6:k8 | 0.2429 | 0.3479 | 768 | 56 |
| mnist | balanced_calibrated:m128:s6:k8 | 0.2369 | 0.3516 | 768 | 56 |
| mnist | diverse_calibrated:m128:s6:k8 | 0.2499 | 0.3628 | 768 | 56 |
| mnist | rank:m128:s6:k8 | 0.2742 | 0.3846 | 768 | 56 |
| mnist | adaptive_relative:m128:s6:k8 | 0.2260 | 0.3336 | 768 | 128 |
| mnist | adaptive_external:m128:s6:k8 | 0.1680 | 0.2500 | 768 | 128 |
| mnist | scale_tag:m128:s6:k8 | 0.2720 | 0.3846 | 768 | 64 |
| mnist | two_global:m128:s6:k8 | 0.2602 | 0.3807 | 768 | 56 |
| mnist | two_concat:m128:s6:k8 | 0.2507 | 0.3687 | 768 | 56 |
| mnist | two_or:m128:s6:k8 | 0.2505 | 0.3682 | 768 | 56 |
| mnist | densefly:m128:s6:k8 | 0.5046 | 0.5935 | 768 | 128 |
| mnist | densefly_probe:m128:s6:k8 | 0.4436 | 0.5201 | 768 | 144 |
| mnist | biohash:m128:s6:k8 | 0.2833 | 0.3979 | 8192 | 56 |
| mnist | biohash_untrained:m128:s6:k8 | 0.2824 | 0.3971 | 8192 | 56 |
| mnist | sparse_bio:m128:s6:k8 | 0.2422 | 0.3638 | 1536 | 56 |
| mnist | sparse_untrained:m128:s6:k8 | 0.2435 | 0.3664 | 1536 | 56 |
| mnist | gaussian_real:m128:s6:k8 | 0.2390 | 0.3557 | 512 | 256 |
| mnist | gaussian_sign:m128:s6:k8 | 0.0632 | 0.1581 | 512 | 8 |
| mnist | gaussian_ops:m128:s6:k8 | 0.2845 | 0.3982 | 768 | 384 |
| mnist | gaussian_storage:m128:s6:k8 | 0.3894 | 0.5042 | 3584 | 56 |
| mnist | fly:m256:s3:k4 | 0.1531 | 0.2651 | 768 | 32 |
| mnist | balanced:m256:s3:k4 | 0.1761 | 0.2865 | 768 | 32 |
| mnist | diverse:m256:s3:k4 | 0.1762 | 0.2917 | 768 | 32 |
| mnist | calibrated:m256:s3:k4 | 0.1315 | 0.2227 | 768 | 32 |
| mnist | zscore:m256:s3:k4 | 0.1390 | 0.2378 | 768 | 32 |
| mnist | balanced_calibrated:m256:s3:k4 | 0.1376 | 0.2260 | 768 | 32 |
| mnist | diverse_calibrated:m256:s3:k4 | 0.1316 | 0.2234 | 768 | 32 |
| mnist | rank:m256:s3:k4 | 0.1969 | 0.3026 | 768 | 32 |
| mnist | adaptive_relative:m256:s3:k4 | 0.1142 | 0.1711 | 768 | 256 |
| mnist | adaptive_external:m256:s3:k4 | 0.0851 | 0.1365 | 768 | 256 |
| mnist | scale_tag:m256:s3:k4 | 0.1981 | 0.3094 | 768 | 40 |
| mnist | two_global:m256:s3:k4 | 0.1626 | 0.2698 | 768 | 32 |
| mnist | two_concat:m256:s3:k4 | 0.1553 | 0.2604 | 768 | 32 |
| mnist | two_or:m256:s3:k4 | 0.1546 | 0.2591 | 768 | 32 |
| mnist | densefly:m256:s3:k4 | 0.5504 | 0.6302 | 768 | 256 |
| mnist | densefly_probe:m256:s3:k4 | 0.4579 | 0.5247 | 768 | 272 |
| mnist | biohash:m256:s3:k4 | 0.1803 | 0.2828 | 16384 | 32 |
| mnist | biohash_untrained:m256:s3:k4 | 0.1789 | 0.2812 | 16384 | 32 |
| mnist | sparse_bio:m256:s3:k4 | 0.1603 | 0.2742 | 1536 | 32 |
| mnist | sparse_untrained:m256:s3:k4 | 0.1618 | 0.2724 | 1536 | 32 |
| mnist | gaussian_real:m256:s3:k4 | 0.1139 | 0.2102 | 256 | 128 |
| mnist | gaussian_sign:m256:s3:k4 | 0.0221 | 0.0849 | 256 | 4 |
| mnist | gaussian_ops:m256:s3:k4 | 0.2845 | 0.3982 | 768 | 384 |
| mnist | gaussian_storage:m256:s3:k4 | 0.2708 | 0.3911 | 2048 | 32 |
| mnist | fly:m256:s3:k8 | 0.2204 | 0.3263 | 768 | 64 |
| mnist | balanced:m256:s3:k8 | 0.2448 | 0.3555 | 768 | 64 |
| mnist | diverse:m256:s3:k8 | 0.2453 | 0.3607 | 768 | 64 |
| mnist | calibrated:m256:s3:k8 | 0.1998 | 0.3044 | 768 | 64 |
| mnist | zscore:m256:s3:k8 | 0.2026 | 0.3029 | 768 | 64 |
| mnist | balanced_calibrated:m256:s3:k8 | 0.2188 | 0.3211 | 768 | 64 |
| mnist | diverse_calibrated:m256:s3:k8 | 0.2179 | 0.3284 | 768 | 64 |
| mnist | rank:m256:s3:k8 | 0.2681 | 0.3714 | 768 | 64 |
| mnist | adaptive_relative:m256:s3:k8 | 0.1654 | 0.2440 | 768 | 256 |
| mnist | adaptive_external:m256:s3:k8 | 0.1271 | 0.1938 | 768 | 256 |
| mnist | scale_tag:m256:s3:k8 | 0.2581 | 0.3661 | 768 | 72 |
| mnist | two_global:m256:s3:k8 | 0.2310 | 0.3477 | 768 | 64 |
| mnist | two_concat:m256:s3:k8 | 0.2298 | 0.3404 | 768 | 64 |
| mnist | two_or:m256:s3:k8 | 0.2301 | 0.3409 | 768 | 64 |
| mnist | densefly:m256:s3:k8 | 0.5504 | 0.6302 | 768 | 256 |
| mnist | densefly_probe:m256:s3:k8 | 0.4579 | 0.5247 | 768 | 272 |
| mnist | biohash:m256:s3:k8 | 0.2947 | 0.3995 | 16384 | 64 |
| mnist | biohash_untrained:m256:s3:k8 | 0.2956 | 0.4021 | 16384 | 64 |
| mnist | sparse_bio:m256:s3:k8 | 0.2593 | 0.3792 | 1536 | 64 |
| mnist | sparse_untrained:m256:s3:k8 | 0.2630 | 0.3818 | 1536 | 64 |
| mnist | gaussian_real:m256:s3:k8 | 0.2390 | 0.3557 | 512 | 256 |
| mnist | gaussian_sign:m256:s3:k8 | 0.0632 | 0.1581 | 512 | 8 |
| mnist | gaussian_ops:m256:s3:k8 | 0.2845 | 0.3982 | 768 | 384 |
| mnist | gaussian_storage:m256:s3:k8 | 0.4003 | 0.5036 | 4096 | 64 |
| mnist | fly:m256:s6:k4 | 0.1844 | 0.2849 | 1536 | 32 |
| mnist | balanced:m256:s6:k4 | 0.1818 | 0.2891 | 1536 | 32 |
| mnist | diverse:m256:s6:k4 | 0.1881 | 0.2990 | 1536 | 32 |
| mnist | calibrated:m256:s6:k4 | 0.1560 | 0.2513 | 1536 | 32 |
| mnist | zscore:m256:s6:k4 | 0.1702 | 0.2714 | 1536 | 32 |
| mnist | balanced_calibrated:m256:s6:k4 | 0.1418 | 0.2333 | 1536 | 32 |
| mnist | diverse_calibrated:m256:s6:k4 | 0.1461 | 0.2448 | 1536 | 32 |
| mnist | rank:m256:s6:k4 | 0.1999 | 0.3164 | 1536 | 32 |
| mnist | adaptive_relative:m256:s6:k4 | 0.1024 | 0.1594 | 1536 | 256 |
| mnist | adaptive_external:m256:s6:k4 | 0.0657 | 0.1081 | 1536 | 256 |
| mnist | scale_tag:m256:s6:k4 | 0.2046 | 0.3104 | 1536 | 40 |
| mnist | two_global:m256:s6:k4 | 0.1748 | 0.2852 | 1536 | 32 |
| mnist | two_concat:m256:s6:k4 | 0.1744 | 0.2841 | 1536 | 32 |
| mnist | two_or:m256:s6:k4 | 0.1743 | 0.2839 | 1536 | 32 |
| mnist | densefly:m256:s6:k4 | 0.5829 | 0.6549 | 1536 | 256 |
| mnist | densefly_probe:m256:s6:k4 | 0.4735 | 0.5326 | 1536 | 272 |
| mnist | biohash:m256:s6:k4 | 0.1803 | 0.2828 | 16384 | 32 |
| mnist | biohash_untrained:m256:s6:k4 | 0.1789 | 0.2812 | 16384 | 32 |
| mnist | sparse_bio:m256:s6:k4 | 0.1734 | 0.2854 | 3072 | 32 |
| mnist | sparse_untrained:m256:s6:k4 | 0.1646 | 0.2768 | 3072 | 32 |
| mnist | gaussian_real:m256:s6:k4 | 0.1139 | 0.2102 | 256 | 128 |
| mnist | gaussian_sign:m256:s6:k4 | 0.0221 | 0.0849 | 256 | 4 |
| mnist | gaussian_ops:m256:s6:k4 | 0.4420 | 0.5440 | 1536 | 768 |
| mnist | gaussian_storage:m256:s6:k4 | 0.2708 | 0.3911 | 2048 | 32 |
| mnist | fly:m256:s6:k8 | 0.2658 | 0.3776 | 1536 | 64 |
| mnist | balanced:m256:s6:k8 | 0.2810 | 0.3904 | 1536 | 64 |
| mnist | diverse:m256:s6:k8 | 0.2858 | 0.3966 | 1536 | 64 |
| mnist | calibrated:m256:s6:k8 | 0.2577 | 0.3625 | 1536 | 64 |
| mnist | zscore:m256:s6:k8 | 0.2663 | 0.3742 | 1536 | 64 |
| mnist | balanced_calibrated:m256:s6:k8 | 0.2493 | 0.3555 | 1536 | 64 |
| mnist | diverse_calibrated:m256:s6:k8 | 0.2663 | 0.3680 | 1536 | 64 |
| mnist | rank:m256:s6:k8 | 0.2880 | 0.3974 | 1536 | 64 |
| mnist | adaptive_relative:m256:s6:k8 | 0.1789 | 0.2630 | 1536 | 256 |
| mnist | adaptive_external:m256:s6:k8 | 0.1309 | 0.1849 | 1536 | 256 |
| mnist | scale_tag:m256:s6:k8 | 0.2910 | 0.4008 | 1536 | 72 |
| mnist | two_global:m256:s6:k8 | 0.2673 | 0.3747 | 1536 | 64 |
| mnist | two_concat:m256:s6:k8 | 0.2639 | 0.3779 | 1536 | 64 |
| mnist | two_or:m256:s6:k8 | 0.2635 | 0.3763 | 1536 | 64 |
| mnist | densefly:m256:s6:k8 | 0.5829 | 0.6549 | 1536 | 256 |
| mnist | densefly_probe:m256:s6:k8 | 0.4735 | 0.5326 | 1536 | 272 |
| mnist | biohash:m256:s6:k8 | 0.2947 | 0.3995 | 16384 | 64 |
| mnist | biohash_untrained:m256:s6:k8 | 0.2956 | 0.4021 | 16384 | 64 |
| mnist | sparse_bio:m256:s6:k8 | 0.2557 | 0.3766 | 3072 | 64 |
| mnist | sparse_untrained:m256:s6:k8 | 0.2529 | 0.3695 | 3072 | 64 |
| mnist | gaussian_real:m256:s6:k8 | 0.2390 | 0.3557 | 512 | 256 |
| mnist | gaussian_sign:m256:s6:k8 | 0.0632 | 0.1581 | 512 | 8 |
| mnist | gaussian_ops:m256:s6:k8 | 0.4420 | 0.5440 | 1536 | 768 |
| mnist | gaussian_storage:m256:s6:k8 | 0.4003 | 0.5036 | 4096 | 64 |

## Paired ablations

| Dataset/config | Treatment − control | Δ AP | 95% interval |
|---|---|---:|---|
| sift/m128/s3/k4 | balanced − fly | -0.0018 | [-0.09193368673979689, 0.08841382970171473] |
| sift/m128/s3/k8 | balanced − fly | +0.0060 | [-0.08337788282618899, 0.09532004352693618] |
| sift/m128/s6/k4 | balanced − fly | -0.0006 | [-0.026165893092080512, 0.024869505982637055] |
| sift/m128/s6/k8 | balanced − fly | +0.0004 | [-0.02592985106921357, 0.026824889640517123] |
| sift/m256/s3/k4 | balanced − fly | +0.0042 | [-0.04067522893564259, 0.049097010987636976] |
| sift/m256/s3/k8 | balanced − fly | +0.0024 | [-0.011792617877546149, 0.01650005059593572] |
| sift/m256/s6/k4 | balanced − fly | -0.0120 | [-0.03560859842569143, 0.011660099180716268] |
| sift/m256/s6/k8 | balanced − fly | +0.0016 | [-0.07185717918834814, 0.07507130945307827] |
| mnist/m128/s3/k4 | balanced − fly | -0.0148 | [-0.05284817310487761, 0.023161273465022274] |
| mnist/m128/s3/k8 | balanced − fly | -0.0064 | [-0.04140565775513556, 0.028642235526797] |
| mnist/m128/s6/k4 | balanced − fly | +0.0015 | [-0.04631689342605092, 0.04940286519296796] |
| mnist/m128/s6/k8 | balanced − fly | +0.0037 | [-0.017450893720652654, 0.02488244356494677] |
| mnist/m256/s3/k4 | balanced − fly | +0.0230 | [-0.022760840822244447, 0.06878644405941402] |
| mnist/m256/s3/k8 | balanced − fly | +0.0244 | [-0.005084965866766197, 0.05387238345854477] |
| mnist/m256/s6/k4 | balanced − fly | -0.0027 | [-0.06502019041127971, 0.05964340815169703] |
| mnist/m256/s6/k8 | balanced − fly | +0.0152 | [-0.010009229603463706, 0.04040341675643079] |
| sift/m128/s3/k4 | diverse − balanced | +0.0044 | [-0.013402493452269584, 0.022133753231988622] |
| sift/m128/s3/k8 | diverse − balanced | -0.0055 | [-0.025962979994525084, 0.015027322765330711] |
| sift/m128/s6/k4 | diverse − balanced | +0.0010 | [-0.029207814562069508, 0.031170819752810303] |
| sift/m128/s6/k8 | diverse − balanced | -0.0062 | [-0.04704128096435543, 0.03473764460583808] |
| sift/m256/s3/k4 | diverse − balanced | +0.0010 | [-0.018324439513565556, 0.02039302870265592] |
| sift/m256/s3/k8 | diverse − balanced | +0.0183 | [-0.0016982234990361757, 0.03829353575454096] |
| sift/m256/s6/k4 | diverse − balanced | +0.0203 | [0.007335186249955267, 0.033206689679824604] |
| sift/m256/s6/k8 | diverse − balanced | +0.0111 | [-0.01338835780397774, 0.03560623618435853] |
| mnist/m128/s3/k4 | diverse − balanced | +0.0036 | [-0.06285218141186628, 0.07012976451212018] |
| mnist/m128/s3/k8 | diverse − balanced | +0.0133 | [-0.02535913085521011, 0.05192120451220685] |
| mnist/m128/s6/k4 | diverse − balanced | +0.0080 | [-0.004178891780257356, 0.020129731271755748] |
| mnist/m128/s6/k8 | diverse − balanced | +0.0108 | [-0.009274997156789207, 0.030824085258641894] |
| mnist/m256/s3/k4 | diverse − balanced | +0.0001 | [-0.04111376438687644, 0.04127504133867923] |
| mnist/m256/s3/k8 | diverse − balanced | +0.0005 | [-0.08157027673515338, 0.08255900112243622] |
| mnist/m256/s6/k4 | diverse − balanced | +0.0063 | [-0.045724946615654176, 0.05834110725141862] |
| mnist/m256/s6/k8 | diverse − balanced | +0.0048 | [-0.0283927585126463, 0.038027864647123606] |
| sift/m128/s3/k4 | calibrated − fly | -0.0087 | [-0.05945998352334542, 0.042043908809839395] |
| sift/m128/s3/k8 | calibrated − fly | +0.0018 | [-0.09230091339294437, 0.09580479710544361] |
| sift/m128/s6/k4 | calibrated − fly | -0.0132 | [-0.05191518106637712, 0.025553267127668478] |
| sift/m128/s6/k8 | calibrated − fly | -0.0014 | [-0.06351834878271224, 0.06067703214459317] |
| sift/m256/s3/k4 | calibrated − fly | -0.0143 | [-0.0445063459614774, 0.015990561786770528] |
| sift/m256/s3/k8 | calibrated − fly | -0.0107 | [-0.023701544091495056, 0.0022585986219677487] |
| sift/m256/s6/k4 | calibrated − fly | -0.0195 | [-0.048571187833935006, 0.009567275514063896] |
| sift/m256/s6/k8 | calibrated − fly | -0.0031 | [-0.02195014796343922, 0.01573353084723573] |
| mnist/m128/s3/k4 | calibrated − fly | -0.0268 | [-0.06748580492197284, 0.013822035685569146] |
| mnist/m128/s3/k8 | calibrated − fly | -0.0124 | [-0.041563133060235825, 0.016858101229804905] |
| mnist/m128/s6/k4 | calibrated − fly | -0.0051 | [-0.008685595125844095, -0.0014930707739058417] |
| mnist/m128/s6/k8 | calibrated − fly | -0.0144 | [-0.029119534913850637, 0.00041252892507462335] |
| mnist/m256/s3/k4 | calibrated − fly | -0.0216 | [-0.030772946203264064, -0.012516826327571267] |
| mnist/m256/s3/k8 | calibrated − fly | -0.0206 | [-0.03852055427335349, -0.002679453749168042] |
| mnist/m256/s6/k4 | calibrated − fly | -0.0285 | [-0.03957401025514087, -0.01734552715225449] |
| mnist/m256/s6/k8 | calibrated − fly | -0.0081 | [-0.04567287112882136, 0.029546478201018272] |
| sift/m128/s3/k4 | zscore − fly | -0.0084 | [-0.07495524873394334, 0.058206577119753154] |
| sift/m128/s3/k8 | zscore − fly | +0.0062 | [-0.09496841107611137, 0.1073818471273788] |
| sift/m128/s6/k4 | zscore − fly | -0.0090 | [-0.04087331865176165, 0.022797441250528683] |
| sift/m128/s6/k8 | zscore − fly | -0.0021 | [-0.062092859360528135, 0.05788124867222191] |
| sift/m256/s3/k4 | zscore − fly | -0.0157 | [-0.0483991617979714, 0.017039742869376782] |
| sift/m256/s3/k8 | zscore − fly | -0.0132 | [-0.042163924508295626, 0.015665316029714216] |
| sift/m256/s6/k4 | zscore − fly | -0.0022 | [-0.03626071692206638, 0.03193172828344393] |
| sift/m256/s6/k8 | zscore − fly | +0.0006 | [-0.010970720437560635, 0.012168202353345139] |
| mnist/m128/s3/k4 | zscore − fly | -0.0230 | [-0.08951877130452256, 0.043424003438820566] |
| mnist/m128/s3/k8 | zscore − fly | -0.0125 | [-0.04483452693892103, 0.019809501504706832] |
| mnist/m128/s6/k4 | zscore − fly | -0.0071 | [-0.04729566850668903, 0.03313469111029659] |
| mnist/m128/s6/k8 | zscore − fly | -0.0098 | [-0.03128440735615337, 0.011660793812261181] |
| mnist/m256/s3/k4 | zscore − fly | -0.0141 | [-0.059433655912911464, 0.03120347967431803] |
| mnist/m256/s3/k8 | zscore − fly | -0.0178 | [-0.034414145221969644, -0.0012355641135828599] |
| mnist/m256/s6/k4 | zscore − fly | -0.0142 | [-0.03687628241309941, 0.008393863688928572] |
| mnist/m256/s6/k8 | zscore − fly | +0.0006 | [-0.009753553342433543, 0.010931953397937873] |
| sift/m128/s3/k4 | balanced_calibrated − balanced | -0.0054 | [-0.033210364243058094, 0.022448134044331654] |
| sift/m128/s3/k8 | balanced_calibrated − balanced | +0.0112 | [-0.05325514962704903, 0.07571503532555461] |
| sift/m128/s6/k4 | balanced_calibrated − balanced | -0.0064 | [-0.025305830285615935, 0.012497585122071352] |
| sift/m128/s6/k8 | balanced_calibrated − balanced | -0.0002 | [-0.034967271447255956, 0.03449079690646736] |
| sift/m256/s3/k4 | balanced_calibrated − balanced | -0.0316 | [-0.08181509182070806, 0.018694800159092397] |
| sift/m256/s3/k8 | balanced_calibrated − balanced | -0.0155 | [-0.04895039799584145, 0.017947773867733573] |
| sift/m256/s6/k4 | balanced_calibrated − balanced | -0.0147 | [-0.0322974473651322, 0.002992852075266979] |
| sift/m256/s6/k8 | balanced_calibrated − balanced | -0.0274 | [-0.04813255491931008, -0.006607342845875504] |
| mnist/m128/s3/k4 | balanced_calibrated − balanced | -0.0180 | [-0.04407035172864303, 0.00815334946410547] |
| mnist/m128/s3/k8 | balanced_calibrated − balanced | -0.0055 | [-0.027265190456483058, 0.01618982550092998] |
| mnist/m128/s6/k4 | balanced_calibrated − balanced | -0.0114 | [-0.05040182970819968, 0.02758675178494404] |
| mnist/m128/s6/k8 | balanced_calibrated − balanced | -0.0195 | [-0.04845730117804889, 0.009547491474955972] |
| mnist/m256/s3/k4 | balanced_calibrated − balanced | -0.0386 | [-0.09602690346560716, 0.018884943289954104] |
| mnist/m256/s3/k8 | balanced_calibrated − balanced | -0.0260 | [-0.04147762726925418, -0.01045009319694828] |
| mnist/m256/s6/k4 | balanced_calibrated − balanced | -0.0399 | [-0.08895919279627013, 0.009078413045315738] |
| mnist/m256/s6/k8 | balanced_calibrated − balanced | -0.0316 | [-0.049725734553807646, -0.013547114252371258] |
| sift/m128/s3/k4 | diverse_calibrated − diverse | -0.0089 | [-0.024174093407344215, 0.006318850325889363] |
| sift/m128/s3/k8 | diverse_calibrated − diverse | +0.0123 | [-0.02433984155556572, 0.04897704097894934] |
| sift/m128/s6/k4 | diverse_calibrated − diverse | -0.0034 | [-0.03772258337053138, 0.03088730188644959] |
| sift/m128/s6/k8 | diverse_calibrated − diverse | +0.0132 | [-0.006546986536298047, 0.032957954580211606] |
| sift/m256/s3/k4 | diverse_calibrated − diverse | -0.0311 | [-0.07952787650485252, 0.01740853346318702] |
| sift/m256/s3/k8 | diverse_calibrated − diverse | -0.0330 | [-0.07079988516369436, 0.004807819112127534] |
| sift/m256/s6/k4 | diverse_calibrated − diverse | -0.0239 | [-0.0559170699131097, 0.008139922668083204] |
| sift/m256/s6/k8 | diverse_calibrated − diverse | -0.0214 | [-0.041410182769615965, -0.0014137674325511301] |
| mnist/m128/s3/k4 | diverse_calibrated − diverse | -0.0153 | [-0.04123161791774341, 0.010623702962704309] |
| mnist/m128/s3/k8 | diverse_calibrated − diverse | +0.0018 | [-0.015862021696399484, 0.019401950346199036] |
| mnist/m128/s6/k4 | diverse_calibrated − diverse | -0.0134 | [-0.03556730866421788, 0.008823038975143556] |
| mnist/m128/s6/k8 | diverse_calibrated − diverse | -0.0172 | [-0.033234051098372304, -0.0011921241437616997] |
| mnist/m256/s3/k4 | diverse_calibrated − diverse | -0.0446 | [-0.09304724952289796, 0.0038249727241893156] |
| mnist/m256/s3/k8 | diverse_calibrated − diverse | -0.0273 | [-0.11001219380811719, 0.05531609912121366] |
| mnist/m256/s6/k4 | diverse_calibrated − diverse | -0.0420 | [-0.05412542540046918, -0.029887286435929285] |
| mnist/m256/s6/k8 | diverse_calibrated − diverse | -0.0195 | [-0.06644632017465339, 0.02745552702763447] |
| sift/m128/s3/k4 | rank − balanced | +0.0197 | [-0.024628235909752372, 0.06396261344392914] |
| sift/m128/s3/k8 | rank − balanced | +0.0333 | [-0.020705063953241884, 0.0873526837648637] |
| sift/m128/s6/k4 | rank − balanced | +0.0023 | [-0.013180407804133728, 0.017831144194542153] |
| sift/m128/s6/k8 | rank − balanced | +0.0144 | [-0.03437753038896358, 0.06315072739588255] |
| sift/m256/s3/k4 | rank − balanced | +0.0209 | [0.012659328020179807, 0.029231284124580502] |
| sift/m256/s3/k8 | rank − balanced | +0.0302 | [-0.0010907723647976715, 0.06146621410912004] |
| sift/m256/s6/k4 | rank − balanced | +0.0074 | [-0.0034621465875177915, 0.01831926240565579] |
| sift/m256/s6/k8 | rank − balanced | -0.0011 | [-0.016655702889592745, 0.01445959812800083] |
| mnist/m128/s3/k4 | rank − balanced | +0.0153 | [0.0020224605434984375, 0.028665223988637557] |
| mnist/m128/s3/k8 | rank − balanced | +0.0244 | [-0.007601439993182111, 0.05638048953044328] |
| mnist/m128/s6/k4 | rank − balanced | +0.0157 | [-0.005758715867711966, 0.037060378876454594] |
| mnist/m128/s6/k8 | rank − balanced | +0.0178 | [0.010624708713242144, 0.024924843867318573] |
| mnist/m256/s3/k4 | rank − balanced | +0.0208 | [-0.003488238070051927, 0.04505665878520497] |
| mnist/m256/s3/k8 | rank − balanced | +0.0233 | [0.013216274055929945, 0.033403514810496845] |
| mnist/m256/s6/k4 | rank − balanced | +0.0182 | [-0.011445210729238191, 0.04783094621605245] |
| mnist/m256/s6/k8 | rank − balanced | +0.0071 | [-0.007689389779266524, 0.021876148701476854] |
| sift/m128/s3/k4 | adaptive_relative − balanced | -0.0304 | [-0.06429874316697079, 0.0035675647418706193] |
| sift/m128/s3/k8 | adaptive_relative − balanced | -0.0261 | [-0.05931350897426699, 0.007087553208559406] |
| sift/m128/s6/k4 | adaptive_relative − balanced | -0.0195 | [-0.06070712964643207, 0.021798947988818856] |
| sift/m128/s6/k8 | adaptive_relative − balanced | -0.0109 | [-0.03066578600232668, 0.00882483563226691] |
| sift/m256/s3/k4 | adaptive_relative − balanced | -0.0402 | [-0.056167560290074386, -0.024231973383453065] |
| sift/m256/s3/k8 | adaptive_relative − balanced | -0.0482 | [-0.0814030619494133, -0.01502586380257024] |
| sift/m256/s6/k4 | adaptive_relative − balanced | -0.0364 | [-0.06728993965490784, -0.005581744006914628] |
| sift/m256/s6/k8 | adaptive_relative − balanced | -0.0543 | [-0.08384793092643061, -0.02479144437951059] |
| mnist/m128/s3/k4 | adaptive_relative − balanced | -0.0214 | [-0.07286164554129776, 0.030142474102965163] |
| mnist/m128/s3/k8 | adaptive_relative − balanced | -0.0232 | [-0.03192024838368013, -0.014496635932052825] |
| mnist/m128/s6/k4 | adaptive_relative − balanced | -0.0438 | [-0.06625186508700953, -0.021410429945278746] |
| mnist/m128/s6/k8 | adaptive_relative − balanced | -0.0304 | [-0.04768185554168373, -0.013027302805039928] |
| mnist/m256/s3/k4 | adaptive_relative − balanced | -0.0620 | [-0.07957244120394827, -0.04436937102534214] |
| mnist/m256/s3/k8 | adaptive_relative − balanced | -0.0794 | [-0.1148116839346098, -0.043963790236024446] |
| mnist/m256/s6/k4 | adaptive_relative − balanced | -0.0794 | [-0.13953010881643518, -0.01922259901727595] |
| mnist/m256/s6/k8 | adaptive_relative − balanced | -0.1020 | [-0.14576915964502013, -0.05824182348850796] |
| sift/m128/s3/k4 | adaptive_external − balanced | -0.0401 | [-0.07239825191222292, -0.007794722067576226] |
| sift/m128/s3/k8 | adaptive_external − balanced | -0.0391 | [-0.07407889266137, -0.0040382650114668545] |
| sift/m128/s6/k4 | adaptive_external − balanced | -0.0309 | [-0.04556996135660925, -0.016176631309947582] |
| sift/m128/s6/k8 | adaptive_external − balanced | -0.0346 | [-0.07773264608391481, 0.008550444525189459] |
| sift/m256/s3/k4 | adaptive_external − balanced | -0.0506 | [-0.06616755209901207, -0.03493391548816079] |
| sift/m256/s3/k8 | adaptive_external − balanced | -0.0632 | [-0.07111897040004204, -0.05525425052946277] |
| sift/m256/s6/k4 | adaptive_external − balanced | -0.0367 | [-0.06362216033899654, -0.0098362168450724] |
| sift/m256/s6/k8 | adaptive_external − balanced | -0.0636 | [-0.07863956241334959, -0.048603862633807574] |
| mnist/m128/s3/k4 | adaptive_external − balanced | -0.0509 | [-0.10825350667318104, 0.006543729372326466] |
| mnist/m128/s3/k8 | adaptive_external − balanced | -0.0640 | [-0.0978477361896454, -0.03022798124039184] |
| mnist/m128/s6/k4 | adaptive_external − balanced | -0.0657 | [-0.10196309381733795, -0.029507919367104028] |
| mnist/m128/s6/k8 | adaptive_external − balanced | -0.0884 | [-0.1320944781232502, -0.04470960284515931] |
| mnist/m256/s3/k4 | adaptive_external − balanced | -0.0910 | [-0.13575890228059567, -0.0462528440042011] |
| mnist/m256/s3/k8 | adaptive_external − balanced | -0.1177 | [-0.15756663257608405, -0.07775145867460928] |
| mnist/m256/s6/k4 | adaptive_external − balanced | -0.1161 | [-0.20629545140389405, -0.025862034736878634] |
| mnist/m256/s6/k8 | adaptive_external − balanced | -0.1500 | [-0.19212412503997095, -0.10796521603476093] |
| sift/m128/s3/k4 | scale_tag − balanced | +0.0036 | [0.0006248792861064775, 0.006614930028550466] |
| sift/m128/s3/k8 | scale_tag − balanced | +0.0018 | [-0.009153872953845224, 0.012660764199102366] |
| sift/m128/s6/k4 | scale_tag − balanced | +0.0050 | [-0.010235516550625117, 0.020250589884136458] |
| sift/m128/s6/k8 | scale_tag − balanced | +0.0053 | [-0.012820643110223667, 0.023488639781630755] |
| sift/m256/s3/k4 | scale_tag − balanced | +0.0035 | [-0.007153731337460948, 0.014176254377483558] |
| sift/m256/s3/k8 | scale_tag − balanced | +0.0029 | [-0.010596754192890086, 0.01630131589088492] |
| sift/m256/s6/k4 | scale_tag − balanced | +0.0057 | [-0.010326748590857901, 0.021663854921615086] |
| sift/m256/s6/k8 | scale_tag − balanced | +0.0005 | [-0.015099004765724608, 0.01610177892905411] |
| mnist/m128/s3/k4 | scale_tag − balanced | +0.0218 | [0.008902503587721228, 0.034732631670503236] |
| mnist/m128/s3/k8 | scale_tag − balanced | +0.0199 | [0.019588797946606454, 0.020274857433871248] |
| mnist/m128/s6/k4 | scale_tag − balanced | +0.0233 | [0.00479394549024776, 0.04177659578460368] |
| mnist/m128/s6/k8 | scale_tag − balanced | +0.0156 | [0.0063014363874132635, 0.02490919867505037] |
| mnist/m256/s3/k4 | scale_tag − balanced | +0.0220 | [-0.00592983667677275, 0.0499059893439084] |
| mnist/m256/s3/k8 | scale_tag − balanced | +0.0133 | [0.0011920266887834041, 0.0253621565921922] |
| mnist/m256/s6/k4 | scale_tag − balanced | +0.0228 | [0.008623083501374489, 0.03699218924492088] |
| mnist/m256/s6/k8 | scale_tag − balanced | +0.0100 | [-0.003507567658353814, 0.02356368656998991] |
| sift/m128/s3/k4 | two_concat − two_global | -0.0146 | [-0.039807624613021654, 0.010632825605538342] |
| sift/m128/s3/k8 | two_concat − two_global | -0.0004 | [-0.03218566991457692, 0.03143007159482177] |
| sift/m128/s6/k4 | two_concat − two_global | -0.0041 | [-0.013104076357326768, 0.004984908418354239] |
| sift/m128/s6/k8 | two_concat − two_global | -0.0133 | [-0.033925546889574464, 0.007367496290792459] |
| sift/m256/s3/k4 | two_concat − two_global | -0.0009 | [-0.009926531222403227, 0.008191861167602097] |
| sift/m256/s3/k8 | two_concat − two_global | -0.0151 | [-0.05810194201720278, 0.02791581923717281] |
| sift/m256/s6/k4 | two_concat − two_global | -0.0044 | [-0.038786679354171814, 0.029984141236920607] |
| sift/m256/s6/k8 | two_concat − two_global | -0.0003 | [-0.015518552628092906, 0.014906436185686704] |
| mnist/m128/s3/k4 | two_concat − two_global | -0.0105 | [-0.021672203138107943, 0.0006985610157501532] |
| mnist/m128/s3/k8 | two_concat − two_global | +0.0027 | [-0.01839352674443328, 0.023807824237266954] |
| mnist/m128/s6/k4 | two_concat − two_global | -0.0079 | [-0.023863924689452982, 0.008036528833334689] |
| mnist/m128/s6/k8 | two_concat − two_global | -0.0095 | [-0.03344379871741014, 0.014455232935657796] |
| mnist/m256/s3/k4 | two_concat − two_global | -0.0073 | [-0.01742627917747373, 0.002874447902417792] |
| mnist/m256/s3/k8 | two_concat − two_global | -0.0012 | [-0.011933225985189692, 0.009573030160524975] |
| mnist/m256/s6/k4 | two_concat − two_global | -0.0004 | [-0.036381732305286935, 0.03564194931324241] |
| mnist/m256/s6/k8 | two_concat − two_global | -0.0034 | [-0.025565677558304786, 0.018690581380044945] |
| sift/m128/s3/k4 | two_or − two_concat | -0.0002 | [-0.0005600804239799486, 0.00022599534482906582] |
| sift/m128/s3/k8 | two_or − two_concat | -0.0003 | [-0.001245433738733962, 0.0007432281507238991] |
| sift/m128/s6/k4 | two_or − two_concat | -0.0000 | [-0.0008383665234145984, 0.0007862735069252762] |
| sift/m128/s6/k8 | two_or − two_concat | +0.0001 | [-9.940665717665751e-05, 0.0002885178005375003] |
| sift/m256/s3/k4 | two_or − two_concat | -0.0002 | [-0.0005006666772784372, 0.0001833532113218911] |
| sift/m256/s3/k8 | two_or − two_concat | -0.0006 | [-0.002114787762856451, 0.0009167623172981286] |
| sift/m256/s6/k4 | two_or − two_concat | -0.0007 | [-0.0027209233377458587, 0.0012556981725279216] |
| sift/m256/s6/k8 | two_or − two_concat | -0.0003 | [-0.00041684870813594955, -0.00020749062006608234] |
| mnist/m128/s3/k4 | two_or − two_concat | -0.0035 | [-0.007673498929378712, 0.0006075198967297906] |
| mnist/m128/s3/k8 | two_or − two_concat | +0.0000 | [-0.00015810033364686968, 0.0002538417552155026] |
| mnist/m128/s6/k4 | two_or − two_concat | -0.0032 | [-0.007404869277433399, 0.0009662229322709701] |
| mnist/m128/s6/k8 | two_or − two_concat | -0.0002 | [-0.0006590709541703231, 0.0002410688174182221] |
| mnist/m256/s3/k4 | two_or − two_concat | -0.0008 | [-0.0041280404848615965, 0.0026111529574888736] |
| mnist/m256/s3/k8 | two_or − two_concat | +0.0003 | [-0.0013651445529930147, 0.0019501400103399667] |
| mnist/m256/s6/k4 | two_or − two_concat | -0.0001 | [-0.00046668768229457953, 0.00024154210365151947] |
| mnist/m256/s6/k8 | two_or − two_concat | -0.0004 | [-0.0012146100832274855, 0.0004794996099446264] |
| sift/m128/s3/k4 | densefly − fly | +0.2739 | [0.2330078995714231, 0.3147717495120427] |
| sift/m128/s3/k8 | densefly − fly | +0.2385 | [0.16001621037491454, 0.3169406104718299] |
| sift/m128/s6/k4 | densefly − fly | +0.2874 | [0.2531183733067335, 0.3217012676485015] |
| sift/m128/s6/k8 | densefly − fly | +0.2197 | [0.1700878079725085, 0.26929591744818715] |
| sift/m256/s3/k4 | densefly − fly | +0.3182 | [0.26507022782695383, 0.3712440991702412] |
| sift/m256/s3/k8 | densefly − fly | +0.2763 | [0.21051671503309838, 0.34206361370822097] |
| sift/m256/s6/k4 | densefly − fly | +0.3732 | [0.2794105967436883, 0.46698734492503025] |
| sift/m256/s6/k8 | densefly − fly | +0.3112 | [0.2023890822617127, 0.41999842292656175] |
| mnist/m128/s3/k4 | densefly − fly | +0.3239 | [0.287212733062328, 0.3606201027129542] |
| mnist/m128/s3/k8 | densefly − fly | +0.2665 | [0.24052735712022133, 0.2925104483714469] |
| mnist/m128/s6/k4 | densefly − fly | +0.3430 | [0.3124998199251308, 0.3734245001272369] |
| mnist/m128/s6/k8 | densefly − fly | +0.2519 | [0.22316975119400845, 0.2806685441625241] |
| mnist/m256/s3/k4 | densefly − fly | +0.3973 | [0.34187211188148475, 0.4527740394965381] |
| mnist/m256/s3/k8 | densefly − fly | +0.3300 | [0.2876019332888472, 0.3724880063807775] |
| mnist/m256/s6/k4 | densefly − fly | +0.3984 | [0.38542427395258966, 0.41146461846993515] |
| mnist/m256/s6/k8 | densefly − fly | +0.3171 | [0.295161854671129, 0.3390911111922882] |
| sift/m128/s3/k4 | densefly_probe − densefly | -0.0434 | [-0.05505480243656542, -0.03172427993885504] |
| sift/m128/s3/k8 | densefly_probe − densefly | -0.0434 | [-0.05505480243656542, -0.03172427993885504] |
| sift/m128/s6/k4 | densefly_probe − densefly | -0.0448 | [-0.05003907786266424, -0.03949352325529385] |
| sift/m128/s6/k8 | densefly_probe − densefly | -0.0448 | [-0.05003907786266424, -0.03949352325529385] |
| sift/m256/s3/k4 | densefly_probe − densefly | -0.0595 | [-0.11189247927896101, -0.007018478694738768] |
| sift/m256/s3/k8 | densefly_probe − densefly | -0.0595 | [-0.11189247927896101, -0.007018478694738768] |
| sift/m256/s6/k4 | densefly_probe − densefly | -0.0807 | [-0.08870113664066702, -0.07268706342859922] |
| sift/m256/s6/k8 | densefly_probe − densefly | -0.0807 | [-0.08870113664066702, -0.07268706342859922] |
| mnist/m128/s3/k4 | densefly_probe − densefly | -0.0594 | [-0.10017583383660753, -0.018647621101217163] |
| mnist/m128/s3/k8 | densefly_probe − densefly | -0.0594 | [-0.10017583383660753, -0.018647621101217163] |
| mnist/m128/s6/k4 | densefly_probe − densefly | -0.0610 | [-0.09824940986740859, -0.02366560946398316] |
| mnist/m128/s6/k8 | densefly_probe − densefly | -0.0610 | [-0.09824940986740859, -0.02366560946398316] |
| mnist/m256/s3/k4 | densefly_probe − densefly | -0.0926 | [-0.11973211009481914, -0.0654127736296649] |
| mnist/m256/s3/k8 | densefly_probe − densefly | -0.0926 | [-0.11973211009481914, -0.0654127736296649] |
| mnist/m256/s6/k4 | densefly_probe − densefly | -0.1094 | [-0.12823683519771323, -0.09053127373359947] |
| mnist/m256/s6/k8 | densefly_probe − densefly | -0.1094 | [-0.12823683519771323, -0.09053127373359947] |
| sift/m128/s3/k4 | biohash − biohash_untrained | +0.0005 | [-0.026078172237514956, 0.027080148771007555] |
| sift/m128/s3/k8 | biohash − biohash_untrained | +0.0011 | [-0.013505383576368024, 0.015683540865665976] |
| sift/m128/s6/k4 | biohash − biohash_untrained | +0.0005 | [-0.026078172237514956, 0.027080148771007555] |
| sift/m128/s6/k8 | biohash − biohash_untrained | +0.0011 | [-0.013505383576368024, 0.015683540865665976] |
| sift/m256/s3/k4 | biohash − biohash_untrained | -0.0002 | [-0.021917750301592955, 0.021551671799932507] |
| sift/m256/s3/k8 | biohash − biohash_untrained | +0.0024 | [-0.01382194932993862, 0.0186687740474217] |
| sift/m256/s6/k4 | biohash − biohash_untrained | -0.0002 | [-0.021917750301592955, 0.021551671799932507] |
| sift/m256/s6/k8 | biohash − biohash_untrained | +0.0024 | [-0.01382194932993862, 0.0186687740474217] |
| mnist/m128/s3/k4 | biohash − biohash_untrained | +0.0062 | [-0.013415241159415892, 0.025891580345434838] |
| mnist/m128/s3/k8 | biohash − biohash_untrained | +0.0009 | [-0.013237441458527347, 0.01508940857081028] |
| mnist/m128/s6/k4 | biohash − biohash_untrained | +0.0062 | [-0.013415241159415892, 0.025891580345434838] |
| mnist/m128/s6/k8 | biohash − biohash_untrained | +0.0009 | [-0.013237441458527347, 0.01508940857081028] |
| mnist/m256/s3/k4 | biohash − biohash_untrained | +0.0014 | [-0.010529314673021647, 0.01325009313266184] |
| mnist/m256/s3/k8 | biohash − biohash_untrained | -0.0009 | [-0.01623433599899842, 0.014462750082191935] |
| mnist/m256/s6/k4 | biohash − biohash_untrained | +0.0014 | [-0.010529314673021647, 0.01325009313266184] |
| mnist/m256/s6/k8 | biohash − biohash_untrained | -0.0009 | [-0.01623433599899842, 0.014462750082191935] |
| sift/m128/s3/k4 | sparse_bio − sparse_untrained | +0.0015 | [-0.0026667188526726332, 0.005612345861041528] |
| sift/m128/s3/k8 | sparse_bio − sparse_untrained | -0.0046 | [-0.014772492931845804, 0.00552939404103504] |
| sift/m128/s6/k4 | sparse_bio − sparse_untrained | +0.0001 | [-0.016105799014620288, 0.01633706914690386] |
| sift/m128/s6/k8 | sparse_bio − sparse_untrained | +0.0046 | [-0.00912779134211202, 0.018303846501925423] |
| sift/m256/s3/k4 | sparse_bio − sparse_untrained | +0.0010 | [-0.0014838026254663686, 0.0035279240455552865] |
| sift/m256/s3/k8 | sparse_bio − sparse_untrained | -0.0008 | [-0.005239390560559046, 0.003631571761853961] |
| sift/m256/s6/k4 | sparse_bio − sparse_untrained | +0.0081 | [-0.006436589670781192, 0.022685767721187973] |
| sift/m256/s6/k8 | sparse_bio − sparse_untrained | +0.0102 | [-0.006794882370384831, 0.027185374658794594] |
| mnist/m128/s3/k4 | sparse_bio − sparse_untrained | +0.0005 | [-0.012312101394100184, 0.013252827898632786] |
| mnist/m128/s3/k8 | sparse_bio − sparse_untrained | +0.0032 | [-0.005510212320564971, 0.01196237497550191] |
| mnist/m128/s6/k4 | sparse_bio − sparse_untrained | +0.0022 | [-0.010550366300132188, 0.015046459470216725] |
| mnist/m128/s6/k8 | sparse_bio − sparse_untrained | -0.0013 | [-0.008954485171304151, 0.006379118096553913] |
| mnist/m256/s3/k4 | sparse_bio − sparse_untrained | -0.0015 | [-0.010044765237321207, 0.00713457638497631] |
| mnist/m256/s3/k8 | sparse_bio − sparse_untrained | -0.0037 | [-0.019644672970711697, 0.012311101482305195] |
| mnist/m256/s6/k4 | sparse_bio − sparse_untrained | +0.0088 | [0.004841803060555439, 0.012754114635618641] |
| mnist/m256/s6/k8 | sparse_bio − sparse_untrained | +0.0028 | [-0.017292783260933517, 0.0228458737843341] |
