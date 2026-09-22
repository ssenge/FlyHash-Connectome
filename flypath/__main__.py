"""Command line: build the connectome graph, then run the hashing experiment.

Subcommand modules import lazily so that --help works before a build.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flypath", description=__doc__.splitlines()[0])
    p.add_argument("--config",
                   help="a YAML config overriding config.example.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="download the MaleCNS data and build the graph")

    fh = sub.add_parser(
        "flyhash", help="is the mushroom body a better hash than chance?")
    fh.add_argument("--side", choices=["R", "L"], default="R",
                    help="which hemisphere's mushroom body to use")
    fh.add_argument("--dataset", choices=["mixtures", "odours", "synthetic"],
                    default="mixtures",
                    help="odours: the ~172 measured odorants, which is real but "
                         "underpowered. mixtures: blends of them, enough items to "
                         "resolve a few-percent effect.")
    fh.add_argument("--items", type=int, default=4000)
    fh.add_argument("--seeds", type=int, default=20,
                    help="random matrices drawn per control")
    fh.add_argument("--neighbours", type=int, default=10)
    fh.add_argument("--weighted", action="store_true",
                    help="use synapse counts instead of binary connections")
    fh.add_argument("--json", action="store_true")
    fh.add_argument("--save", action="store_true",
                    help="write results/flyhash.json and results/flyhash.png")

    args = p.parse_args(argv)

    from . import config
    cfg = config.load(args.config)

    if args.cmd == "build":
        from . import data
        data.build(cfg)
        return 0

    if args.cmd == "flyhash":
        from . import data, flyhash
        res = flyhash.experiment(data.load_graph(cfg), cfg, side=args.side,
                                 dataset=args.dataset, seeds=args.seeds,
                                 neighbours=args.neighbours,
                                 weighted=args.weighted, n_items=args.items)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            flyhash.print_experiment(res)
        if args.save:
            from .config import ROOT
            out = ROOT / "results"
            out.mkdir(parents=True, exist_ok=True)
            (out / "flyhash.json").write_text(json.dumps(res, indent=2))
            flyhash.figure(res, out / "flyhash.png")
            print(f"\n  wrote {out}/flyhash.json and flyhash.png")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
