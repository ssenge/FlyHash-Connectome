# FlyHash++ exploratory study

AP cutoff, splits and all hyperparameters are recorded in the JSON. Intervals are paired-seed t intervals, exploratory and unadjusted.
Timing uses full code scans (including OR); this is not an optimized ANN throughput benchmark.
Frontiers are selected on validation quality and use projection arithmetic plus fixed-width code storage.

| Dataset | Method/config | AP (mean) | Recall | Projection ops | Code bits |
|---|---|---:|---:|---:|---:|
| sift | fly:m256:s6:k8 | 0.1920 | 0.3221 | 1536 | 64 |
| sift | balanced:m256:s6:k8 | 0.1907 | 0.3197 | 1536 | 64 |
| sift | diverse:m256:s6:k8 | 0.1939 | 0.3223 | 1536 | 64 |
| sift | rank:m256:s6:k8 | 0.1986 | 0.3256 | 1536 | 64 |
| sift | scale_tag:m256:s6:k8 | 0.1929 | 0.3205 | 1536 | 72 |
| sift | densefly:m256:s6:k8 | 0.4830 | 0.5862 | 1536 | 256 |
| sift | gaussian_ops:m256:s6:k8 | 0.3612 | 0.4859 | 1536 | 768 |
| sift | gaussian_storage:m256:s6:k8 | 0.2968 | 0.4266 | 4096 | 64 |
| mnist | fly:m256:s6:k8 | 0.2792 | 0.3937 | 1536 | 64 |
| mnist | balanced:m256:s6:k8 | 0.2877 | 0.3984 | 1536 | 64 |
| mnist | diverse:m256:s6:k8 | 0.2862 | 0.4013 | 1536 | 64 |
| mnist | rank:m256:s6:k8 | 0.2936 | 0.4048 | 1536 | 64 |
| mnist | scale_tag:m256:s6:k8 | 0.3042 | 0.4141 | 1536 | 72 |
| mnist | densefly:m256:s6:k8 | 0.5895 | 0.6615 | 1536 | 256 |
| mnist | gaussian_ops:m256:s6:k8 | 0.4477 | 0.5442 | 1536 | 768 |
| mnist | gaussian_storage:m256:s6:k8 | 0.4095 | 0.5133 | 4096 | 64 |
| glove | fly:m256:s6:k8 | 0.0234 | 0.0686 | 1536 | 64 |
| glove | balanced:m256:s6:k8 | 0.0226 | 0.0733 | 1536 | 64 |
| glove | diverse:m256:s6:k8 | 0.0219 | 0.0684 | 1536 | 64 |
| glove | rank:m256:s6:k8 | 0.0227 | 0.0692 | 1536 | 64 |
| glove | scale_tag:m256:s6:k8 | 0.0215 | 0.0638 | 1536 | 72 |
| glove | densefly:m256:s6:k8 | 0.0512 | 0.1202 | 1536 | 256 |
| glove | gaussian_ops:m256:s6:k8 | 0.0493 | 0.1249 | 1536 | 768 |
| glove | gaussian_storage:m256:s6:k8 | 0.0349 | 0.0926 | 4096 | 64 |

## Paired ablations

| Dataset/config | Treatment − control | Δ AP | 95% interval |
|---|---|---:|---|
| sift/m256/s6/k8 | balanced − fly | -0.0014 | [-0.01745798545870752, 0.01469759470701211] |
| mnist/m256/s6/k8 | balanced − fly | +0.0085 | [-0.003431222555866765, 0.02050863335308572] |
| glove/m256/s6/k8 | balanced − fly | -0.0008 | [-0.0061435891232712045, 0.004517252915538666] |
| sift/m256/s6/k8 | diverse − balanced | +0.0032 | [-0.006808386803036074, 0.013212433673402121] |
| mnist/m256/s6/k8 | diverse − balanced | -0.0015 | [-0.015517384708951549, 0.01251394112877981] |
| glove/m256/s6/k8 | diverse − balanced | -0.0006 | [-0.002435983865402168, 0.0011991416661712318] |
| sift/m256/s6/k8 | rank − balanced | +0.0079 | [-0.0024094194267653586, 0.018206251551281062] |
| mnist/m256/s6/k8 | rank − balanced | +0.0059 | [0.001771744219027505, 0.010031995432005636] |
| glove/m256/s6/k8 | rank − balanced | +0.0001 | [-0.0019015723970781562, 0.002169655098180155] |
| sift/m256/s6/k8 | scale_tag − balanced | +0.0023 | [-0.0030807611938222573, 0.007593545749200664] |
| mnist/m256/s6/k8 | scale_tag − balanced | +0.0165 | [0.010700219884545586, 0.022222107897187433] |
| glove/m256/s6/k8 | scale_tag − balanced | -0.0011 | [-0.002936978430779766, 0.0007466922129502492] |
| sift/m256/s6/k8 | densefly − fly | +0.2909 | [0.2646279687863245, 0.31725895940040544] |
| mnist/m256/s6/k8 | densefly − fly | +0.3103 | [0.2956961573482832, 0.32490900008109297] |
| glove/m256/s6/k8 | densefly − fly | +0.0279 | [0.018892020946358713, 0.03683943497409682] |
