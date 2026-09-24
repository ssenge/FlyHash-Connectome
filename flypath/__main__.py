"""flypath: the fly hashing algorithm on measured mushroom body wiring.

Every figure, table and number in the paper is regenerated from results/*.json
by `flypath report`; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flypath", description=__doc__.splitlines()[0])
    p.add_argument("--config", help="a YAML config overriding config.example.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="download the MaleCNS data and build the graph")

    a = sub.add_parser("analyse", help="primary analysis, architecture comparison, "
                                       "curveball diagnostics (~45 min)")
    a.add_argument("--nulls", type=int, default=200)
    a.add_argument("--bootstrap", type=int, default=200)
    a.add_argument("--skip-primary", action="store_true")
    a.add_argument("--refresh", action="store_true",
                   help="only recompute power and metadata of results/primary.json")

    r = sub.add_parser("robustness", help="one-factor-at-a-time sensitivity grid (~35 min)")
    r.add_argument("--nulls", type=int, default=40)
    r.add_argument("--only", help="comma-separated condition ids")

    c = sub.add_parser("coverage", help="coverage of the bootstrap interval (~30 min)")
    c.add_argument("--datasets", type=int, default=100)

    b = sub.add_parser("replicate", help="the 2017 benchmarks (SIFT, GloVe, MNIST): random "
                                         "matrices, then the measured wiring (~3 h)")
    b.add_argument("--trials", type=int, default=50)
    b.add_argument("--connectome-trials", type=int, default=20)
    b.add_argument("--nulls", type=int, default=50)
    b.add_argument("--skip-random", action="store_true")

    n = sub.add_parser("connectomes", help="the same analyses on MaleCNS, hemibrain, FlyWire "
                                            "and BANC, every hemisphere (~3 h)")
    n.add_argument("--only", help="comma-separated hemispheres, e.g. flywire_L,banc_R")

    sub.add_parser("report", help="regenerate figure, LaTeX numbers, README block")
    sub.add_parser("checksums", help="verify input data against DATA_CHECKSUMS.txt")

    args = p.parse_args(argv)

    from . import config
    cfg = config.load(args.config)

    if args.cmd == "build":
        from . import data
        data.build(cfg)
        return 0

    if args.cmd == "analyse":
        from . import experiments as ex
        if args.refresh:
            ex.refresh_power(cfg)
            return 0
        if not args.skip_primary:
            ex.primary(cfg, B=args.nulls, R=args.bootstrap)
        ex.architecture(cfg)
        ex.convergence(cfg)
        return 0

    if args.cmd == "robustness":
        from . import experiments as ex
        ex.robustness(cfg, B=args.nulls,
                      only=args.only.split(",") if args.only else None)
        return 0

    if args.cmd == "coverage":
        from . import experiments as ex
        ex.coverage(cfg, datasets=args.datasets)
        return 0

    if args.cmd == "replicate":
        from . import replication as rp
        if not args.skip_random:
            rp.replicate(cfg, trials=args.trials)
        rp.connectome(cfg, trials=args.connectome_trials, B=args.nulls)
        return 0

    if args.cmd == "connectomes":
        from . import connectomes
        connectomes.compare(cfg, only=args.only.split(",") if args.only else None)
        return 0

    if args.cmd == "report":
        from . import report
        report.build_all()
        return 0

    if args.cmd == "checksums":
        from . import report
        ok = report.verify_checksums(cfg)
        print("all input files match" if ok else "MISMATCH, see above")
        return 0 if ok else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
