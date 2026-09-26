"""Paper figures, generated from results/*.json.

Style after the conventions of figures4papers by Chen Liu
(https://github.com/ChenLiu-1996/figures4papers, "scientific-figure-making"):
Helvetica/Arial, no top/right spines, frameless legends, the blue-green-red-
neutral semantic palette (blue: the fly hash / connectome, greens: variants
that improve on it, reds: baselines and contrasts, neutral: references),
black-edged annotated bars, 3-pt lines with markers, tight layout, 300 dpi.
The conventions are reimplemented here, not copied (that repository is
CC BY-NC 4.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

from .config import ROOT  # noqa: E402

RES = ROOT / "results"

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "grey": "#767676", "dark": "#272727",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}
NAMES = {"sift": "SIFT", "glove": "GloVe", "mnist": "MNIST", "odours": "Odours"}
DATASETS = ("sift", "glove", "mnist", "odours")


def apply_style(font_size: int = 15, axes_linewidth: float = 2.0) -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": font_size, "axes.linewidth": axes_linewidth,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "xtick.major.width": axes_linewidth * 0.8, "ytick.major.width": axes_linewidth * 0.8,
        "xtick.major.size": 5, "ytick.major.size": 5,
    })


def finalize(fig, name: str, pad: float = 1.0) -> None:
    fig.tight_layout(pad=pad)
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def _load(name: str):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


def _log2_axis(ax, ks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()


def _line(ax, x, y, color, label, ls="-", marker="o", sd=None, alpha=0.9):
    y = np.asarray(y)
    ax.plot(x, y, ls, color=color, lw=3, marker=marker, ms=7, alpha=alpha, label=label,
            markeredgecolor="white", markeredgewidth=1)
    if sd is not None:
        sd = np.asarray(sd)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15, lw=0)


def _bar_text(ax, bars, values, fmt, dy, fontsize=11, color="black"):
    for b, v in zip(bars, values):
        y = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, y + (dy if y >= 0 else -dy), fmt.format(v),
                ha="center", va="bottom" if y >= 0 else "top", fontsize=fontsize, color=color)


# ------------------------------------------------------------------ Fig. 1

def fig_concept(an: dict) -> None:
    """(a) the hash; (b) glomerulus fan-out in the MaleCNS connectome, sorted,
    against the even fan-out at the same number of connections; (c) inputs
    per Kenyon cell against the six of the 2017 construction."""
    apply_style(15, 2.0)
    fig = plt.figure(figsize=(12, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.25, 1.0])

    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 11.6)
    rng = np.random.default_rng(3)
    gy = np.linspace(1.2, 8.8, 7)
    ky = np.linspace(0.2, 9.8, 16)
    x_in = rng.uniform(0.2, 1.0, len(gy))
    wiring = [rng.choice(len(gy), rng.integers(2, 4), replace=False) for _ in ky]
    drive = np.array([x_in[w].sum() for w in wiring])
    win = set(np.argsort(-drive)[:3])
    for i, w in enumerate(wiring):
        for j in w:
            ax.plot([2.3, 5.2], [gy[j], ky[i]], color=PALETTE["blue_main"] if i in win else PALETTE["neutral"],
                    lw=1.6 if i in win else 0.8, alpha=0.9 if i in win else 0.7, zorder=1)
    for j, y in enumerate(gy):
        ax.barh(y, x_in[j] * 1.4, height=0.7, left=0.3, color=PALETTE["blue_secondary"], alpha=0.35)
        ax.add_patch(Circle((2.3, y), 0.33, color=PALETTE["blue_secondary"], zorder=2))
    for i, y in enumerate(ky):
        ax.add_patch(Circle((5.2, y), 0.24, zorder=2,
                            color=PALETTE["blue_main"] if i in win else "white",
                            ec=PALETTE["dark"], lw=1))
        ax.add_patch(FancyBboxPatch((7.2, y - 0.22), 0.44, 0.44, boxstyle="round,pad=0.02",
                                    fc=PALETTE["blue_main"] if i in win else "white",
                                    ec=PALETTE["dark"], lw=1))
    ax.annotate("", xy=(7.05, 5), xytext=(5.7, 5),
                arrowprops=dict(arrowstyle="->", lw=2, color=PALETTE["dark"]))
    ax.text(1.3, 10.2, "glomeruli ($d$)", ha="center", va="bottom", fontsize=13)
    ax.text(4.9, 10.2, "Kenyon cells ($m$)", ha="center", va="bottom", fontsize=13)
    ax.text(7.9, 10.2, "tag", ha="center", va="bottom", fontsize=13)
    ax.text(6.35, 5.35, "top $k$", ha="center", va="bottom", fontsize=12)
    ax.text(3.75, -0.55, "$y = M^\\top x$", ha="center", va="bottom", fontsize=13)
    ax.set_title("(a) fly hash", loc="left", fontsize=15)

    h = an["hemispheres"]["malecns_R"]
    ax = fig.add_subplot(gs[1])
    fz = _load("fanout.json")
    M = np.array(fz["fanout_rel"])
    order = np.argsort(-M.mean(0))
    shades = {"malecns": PALETTE["blue_main"], "hemibrain": PALETTE["red_strong"],
              "flywire": PALETTE["green_3"], "banc": PALETTE["violet"]}
    seen = set()
    for row, key in zip(M, fz["hemispheres"]):
        ds = key.split("_")[0]
        ax.plot(np.arange(len(order)), row[order], "-o", ms=3, lw=1.3, alpha=0.8, color=shades[ds],
                label={"malecns": "MaleCNS", "hemibrain": "hemibrain", "flywire": "FlyWire",
                       "banc": "BANC"}[ds] if ds not in seen else None)
        seen.add(ds)
    ax.axhline(1, color=PALETTE["grey"], ls="--", lw=2.5, label="even")
    names = [fz["common"][i] for i in order]
    ax.set_xticks([0, len(names) - 1])
    ax.set_xticklabels([names[0], names[-1]], fontsize=12)
    ax.set_xlabel(f"{len(names)} glomeruli, by mean fan-out")
    ax.set_ylabel("fan-out / mean")
    ax.legend(loc="upper right", fontsize=11, ncol=2)
    ax.set_title("(b) fan-out, 7 hemispheres", loc="left", fontsize=15)

    ax = fig.add_subplot(gs[2])
    ins = np.array(h["inputs"], float)
    xs = np.arange(len(ins))
    ax.bar(xs, ins / ins.sum(), width=0.85, color=PALETTE["blue_secondary"], edgecolor="black", lw=0.6)
    ax.axvline(6, color=PALETTE["red_strong"], ls="--", lw=3, label="2017: six")
    mean = (xs * ins).sum() / ins.sum()
    ax.axvline(mean, color=PALETTE["green_3"], ls=":", lw=3, label=f"mean {mean:.1f}")
    ax.set_xlim(0, 16)
    ax.set_xlabel("glomerular inputs per cell")
    ax.set_ylabel("fraction of cells")
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title("(c) inputs per cell", loc="left", fontsize=15)
    finalize(fig, "fig_concept")


# ------------------------------------------------------------------ Fig. 2

def fig_replication(rp: dict) -> None:
    """The 2017 protocol with random matrices: AP@200 against k, per dataset."""
    apply_style(15, 2.0)
    ks = rp["protocol"]["hash_lengths"]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.62]})
    series = [("lsh", "LSH, $k$ projections", PALETTE["red_strong"], "s"),
              ("random_20k", "random selection, $m=20k$", PALETTE["grey"], "v"),
              ("fly_20k", "fly hash, $m=20k$", PALETTE["green_3"], "^"),
              ("fly_10d", "fly hash, $m=10d$", PALETTE["blue_main"], "o")]
    for ax, name in zip(axes, DATASETS):
        r = rp["datasets"][name]
        for key, label, color, mk in series:
            _line(ax, ks, r[key]["ap"]["mean"], color, label, marker=mk, sd=r[key]["ap"]["sd"])
        _log2_axis(ax, ks)
        ax.set_title(f"{NAMES[name]} ($d={r['d']}$)", loc="left", fontsize=15)
        ax.set_xlabel("hash length $k$")
    axes[0].set_ylabel("AP@200")
    h, lab = axes[0].get_legend_handles_labels()
    axes[-1].set_axis_off()
    axes[-1].legend(h, lab, loc="center left", fontsize=13)
    finalize(fig, "fig_replication")


# ------------------------------------------------------------------ Fig. 3

def fig_budget(rp: dict, cb: dict) -> None:
    """(a) one dataset at several input dimensions (MNIST by PCA), k = 4;
    (b, c) the fly hash's best AP over the tested k grid against real-valued
    LSH with the same projection arithmetic: (b) random matrices on the
    original inputs, (c) the MaleCNS connectome on its own inputs (PCA to one
    component per glomerulus; odours on measured glomeruli). The inputs and
    ground truth differ between (b) and (c), so only pairs within a panel are
    comparable."""
    apply_style(16, 2.0)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ds = rp["dimension_sweep"]
    d = [r["d"] for r in ds["rows"]]
    _line(a1, d, [r["lsh"]["ap"]["mean"][0] for r in ds["rows"]], PALETTE["red_strong"],
          "LSH, 4 projections", marker="s")
    _line(a1, d, [r["fly_10d"]["ap"]["mean"][0] for r in ds["rows"]], PALETTE["blue_main"],
          "fly hash, $m=10d$, $k=4$")
    _line(a1, d, [r["lsh_ops_10d"]["ap"]["mean"] for r in ds["rows"]], PALETTE["red_strong"],
          "LSH, same arithmetic", ls="--", marker="D")
    _log2_axis(a1, d)
    a1.set_xlabel("input dimension $d$ (MNIST, PCA)")
    a1.set_ylabel("AP@200")
    a1.set_ylim(0, 1.05)
    a1.legend(loc="upper left", fontsize=13)
    a1.set_title("(a) varying $d$, one dataset", loc="left", fontsize=16)

    x = np.arange(len(DATASETS))
    w = 0.38
    for ax, fly, lsh, flab, title, hatch in (
            (a2, [max(rp["datasets"][n]["fly_10d"]["ap"]["mean"]) for n in DATASETS],
             [rp["datasets"][n]["lsh_ops_10d"]["ap"]["mean"] for n in DATASETS],
             "random fly, $m=10d$", "(b) random matrices", None),
            (a3, [max(r["real"] for r in cb["datasets"][n]["ap"]["rows"]) for n in DATASETS],
             [cb["datasets"][n]["ap"]["lsh_ops"] for n in DATASETS],
             "connectome fly", "(c) MaleCNS connectome", "//")):
        b1 = ax.bar(x - w / 2, fly, w, color=PALETTE["blue_main"] if hatch is None else PALETTE["blue_secondary"],
                    edgecolor="black", lw=1.5, hatch=hatch, label=f"{flab} (best $k$)")
        b2 = ax.bar(x + w / 2, lsh, w, color=PALETTE["red_2"], edgecolor="black", lw=1.5,
                    label="LSH, same arithmetic")
        _bar_text(ax, b1, fly, "{:.2f}", 0.015, fontsize=11)
        _bar_text(ax, b2, lsh, "{:.2f}", 0.015, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[n] for n in DATASETS], fontsize=14)
        ax.set_ylim(0, 1.3)
        ax.legend(loc="upper left", fontsize=13)
        ax.set_title(title, loc="left", fontsize=16)
    a2.set_ylabel("AP@200")
    finalize(fig, "fig_budget")


# ------------------------------------------------------------------ Fig. 4

def fig_connectomes(pr: dict, an: dict) -> None:
    """(a) the MaleCNS odour analysis against 200 nulls; (b) every hemisphere
    against its own null under the 2017 protocol; (c) the odour analysis in
    every hemisphere."""
    apply_style(15, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw={"width_ratios": [1, 1.45, 1.2]})
    ax = axes[0]
    null, real = np.array(pr["null_scores"]), np.array(pr["real_scores"])
    rng = np.random.default_rng(0)
    for i, k in enumerate(pr["sizes"]):
        rel = 100 * (null[:, i] / null[:, i].mean() - 1)
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(rel)), rel, s=10, color=PALETTE["red_2"],
                   alpha=0.45, lw=0)
        r = 100 * (real[i] / null[:, i].mean() - 1)
        ax.scatter([i], [r], s=110, marker="D", color=PALETTE["blue_main"], edgecolor="white",
                   lw=1.2, zorder=3)
    ax.axhline(0, color=PALETTE["grey"], lw=1.5)
    ax.set_xticks(range(len(pr["sizes"])))
    ax.set_xticklabels([f"{k}{'*' if k == pr['primary_k'] else ''}" for k in pr["sizes"]])
    ax.set_xlabel("hash length $k$ (* primary)")
    ax.set_ylabel("AP@10 relative to null (%)")
    ax.set_title(f"(a) odours, MaleCNS R", loc="left", fontsize=15)

    hs = list(an["hemispheres"].values())
    labels = [h["label"].replace("Hemibrain", "Hemibr.") for h in hs]
    ax = axes[1]
    marks = {"sift": "o", "glove": "s", "mnist": "^", "odours": "D"}
    for i, h in enumerate(hs):
        for name, per in h["protocol"].items():
            v = [100 * r["relative_difference"] for r in per["ap"]["rows"]]
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(v)), v, s=22, marker=marks[name],
                       color=PALETTE["blue_secondary"], alpha=0.45, lw=0,
                       label=NAMES[name] if i == 0 else None)
        med = np.median([100 * r["relative_difference"] for per in h["protocol"].values()
                         for r in per["ap"]["rows"]])
        ax.plot([i - 0.35, i + 0.35], [med, med], color=PALETTE["blue_main"], lw=3.5,
                label="median" if i == 0 else None)
    ax.axhline(0, color=PALETTE["grey"], lw=1.5)
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("AP@200 relative to null (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), fontsize=12, ncol=5,
              handletextpad=0.2, columnspacing=0.8)
    ax.set_title("(b) 2017 protocol, 7 hemispheres", loc="left", fontsize=15)

    ax = axes[2]
    x = np.arange(len(hs))
    own = [next(r for r in h["odours"] if r["primary"]) for h in hs]
    mat = [next(r for r in h["odours_matched"] if r["primary"]) for h in hs]
    for off, rows, color, hatch, label in ((-0.2, own, PALETTE["blue_main"], None, "own glomeruli"),
                                           (0.2, mat, PALETTE["blue_secondary"], "//", "shared glomeruli")):
        vals = [100 * r["relative_difference"] for r in rows]
        bars = ax.bar(x + off, vals, 0.38, color=color, edgecolor="black", lw=1.2, hatch=hatch,
                      label=label)
        for b, r in zip(bars, rows):
            if r["p_two_sided"] < 0.05:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.12, "*", ha="center",
                        va="top", fontsize=16)
    ax.axhline(0, color=PALETTE["grey"], lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("AP@10 relative to null (%)")
    ax.set_ylim(min(100 * r["relative_difference"] for r in own + mat) - 0.9, 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), fontsize=12, ncol=2)
    ax.text(0.98, 0.04, "* unadjusted $p<0.05$", transform=ax.transAxes, ha="right", fontsize=12)
    ax.set_title("(c) odours, primary $k$", loc="left", fontsize=15)
    finalize(fig, "fig_connectomes")


# ------------------------------------------------------------------ Fig. 5

def fig_controls(ct: dict) -> None:
    """Equal-connection degree controls with 95% bootstrap intervals over
    trials (paired with each trial's null mean). (a) MaleCNS R at the largest
    hash size per dataset; (b) every hemisphere, averaged over datasets and
    hash sizes."""
    from .report import pooled_control
    apply_style(16, 2.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.1, 1.3]})
    series = [("real", "connectome", PALETTE["blue_main"], None),
              ("in_equal", "inputs per cell even", PALETTE["red_2"], None),
              ("out_equal", "fan-out even", PALETTE["green_3"], None),
              ("both_equal", "both even", PALETTE["green_2"], "//"),
              ("random_2017", "2017: six inputs", PALETTE["neutral"], "..")]
    h = ct["hemispheres"]["malecns_R"]
    names = [n for n in DATASETS if n in h]
    x = np.arange(len(names))
    w = 0.16
    for i, (key, label, color, hatch) in enumerate(series):
        est, lo, hi = [], [], []
        for n in names:
            c = h[n]["contrasts"][key][-1]                     # largest hash size
            est.append(c["estimate"])
            lo.append(c["estimate"] - c["ci95"][0])
            hi.append(c["ci95"][1] - c["estimate"])
        a1.bar(x + (i - 2) * w, est, w, yerr=[lo, hi], color=color, edgecolor="black", lw=1.2,
               hatch=hatch, label=label, error_kw=dict(lw=1.5, capsize=2.5))
    a1.axhline(0, color=PALETTE["grey"], lw=1.5)
    a1.set_xticks(x)
    a1.set_xticklabels([f"{NAMES[n]}\n$k={h[n]['sizes'][-1]}$" for n in names], fontsize=14)
    a1.set_ylabel("AP@200 relative to null (%)")
    top = a1.get_ylim()[1]
    a1.set_ylim(a1.get_ylim()[0], top * 1.55)
    a1.legend(loc="upper left", fontsize=12, ncol=3, columnspacing=0.8, handletextpad=0.4)
    a1.set_title("(a) MaleCNS R, largest $k$", loc="left", fontsize=16)

    hs = [(k, ct["hemispheres"][k]) for k in LABELS_SHORT
          if k in ct["hemispheres"] and all(n in ct["hemispheres"][k] for n in DATASETS)]
    labels = [LABELS_SHORT.get(k, k) for k, _ in hs]
    xs = np.arange(len(hs))
    w = 0.2
    for i, (key, label, color, hatch) in enumerate(series[1:]):
        vals = [pooled_control(v, key) for _, v in hs]
        est = [p["estimate"] for p in vals]
        err = [[p["estimate"] - p["ci95"][0] for p in vals], [p["ci95"][1] - p["estimate"] for p in vals]]
        a2.bar(xs + (i - 1.5) * w, est, w, yerr=err, color=color, edgecolor="black", lw=1.2, hatch=hatch,
               label=label, error_kw=dict(lw=1.5, capsize=2.5))
    a2.axhline(0, color=PALETTE["grey"], lw=1.5)
    a2.set_xticks(xs)
    a2.set_xticklabels(labels, rotation=30, ha="right", fontsize=13)
    a2.set_ylabel("mean AP@200 relative to null (%)")
    a2.set_title("(b) every hemisphere, all datasets and $k$", loc="left", fontsize=16)
    finalize(fig, "fig_controls")


LABELS_SHORT = {"malecns_R": "MaleCNS R", "malecns_L": "MaleCNS L", "hemibrain_R": "Hemibr. R",
                "flywire_R": "FlyWire R", "flywire_L": "FlyWire L", "banc_R": "BANC R", "banc_L": "BANC L"}


def build() -> list[str]:
    pr, rp = _load("primary.json"), _load("replication.json")
    cb, an = _load("connectome_benchmarks.json"), _load("connectomes.json")
    done = []
    if an and an.get("hemispheres", {}).get("malecns_R", {}).get("fan_out") and _load("fanout.json"):
        fig_concept(an)
        done.append("fig_concept")
    if rp:
        fig_replication(rp)
        done.append("fig_replication")
    if rp and cb and rp.get("dimension_sweep"):
        fig_budget(rp, cb)
        done.append("fig_budget")
    if pr and an and an.get("hemispheres"):
        fig_connectomes(pr, an)
        done.append("fig_connectomes")
    ct = _load("controls.json")
    if ct and "malecns_R" in ct.get("hemispheres", {}):
        fig_controls(ct)
        done.append("fig_controls")
    return done
