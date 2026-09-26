# FlyHash++ exploratory study

AP cutoff, splits and all hyperparameters are recorded in the JSON. Intervals are paired-seed t intervals, exploratory and unadjusted.
Timing uses full code scans (including OR); this is not an optimized ANN throughput benchmark.
Frontiers are selected on validation quality and use projection arithmetic plus fixed-width code storage.

| Dataset | Method/config | AP (mean) | Recall | Projection ops | Code bits |
|---|---|---:|---:|---:|---:|
| sift | measured_fly:m3751:s0:k8 | 0.1931 | 0.3151 | 19858 | 96 |
| sift | measured_two_concat:m3751:s0:k8 | 0.1875 | 0.3083 | 19858 | 96 |
| sift | measured_two_or:m3751:s0:k8 | 0.1873 | 0.3078 | 19858 | 96 |
| sift | measured_sparse_bio:m3751:s0:k8 | 0.1475 | 0.2701 | 39716 | 96 |
| sift | measured_sparse_untrained:m3751:s0:k8 | 0.1477 | 0.2690 | 39716 | 96 |
| sift | null_fly:m3751:s0:k8 | 0.1681 | 0.2891 | 19858 | 96 |
| sift | null_two_concat:m3751:s0:k8 | 0.1714 | 0.2852 | 19858 | 96 |
| sift | null_two_or:m3751:s0:k8 | 0.1714 | 0.2852 | 19858 | 96 |
| sift | null_sparse_bio:m3751:s0:k8 | 0.1183 | 0.2437 | 39716 | 96 |
| sift | null_sparse_untrained:m3751:s0:k8 | 0.1173 | 0.2435 | 39716 | 96 |

## Paired ablations

| Dataset/config | Treatment − control | Δ AP | 95% interval |
|---|---|---:|---|
| sift/m3751/s0/k8 | measured_two_concat − measured_fly | -0.0056 | [-0.01729673884391504, 0.006047223187411666] |
| sift/m3751/s0/k8 | measured_two_or − measured_two_concat | -0.0002 | [-0.0006731234206779368, 0.000262548371945006] |
| sift/m3751/s0/k8 | measured_two_concat − null_two_concat | +0.0161 | [-0.005231706255474332, 0.03742702473495735] |
| sift/m3751/s0/k8 | measured_two_or − null_two_or | +0.0159 | [-0.005906015977824558, 0.0376820788530191] |
| sift/m3751/s0/k8 | measured_sparse_bio − measured_sparse_untrained | -0.0002 | [-0.010858990574171978, 0.01046367743886028] |
| sift/m3751/s0/k8 | null_sparse_bio − null_sparse_untrained | +0.0010 | [-0.005371357290905964, 0.007279527261612244] |
