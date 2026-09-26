"""Standalone scientific plots for completed FlyHash++ studies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result')
    args = p.parse_args()
    path = Path(args.result)
    result = json.loads(path.read_text())
    if result['status'] != 'complete':
        raise ValueError('wait for a complete experiment')
    datasets = result['config']['datasets']
    families = {
        'fly': ('Original FlyHash', '#555555', 'o'),
        'balanced': ('Balanced', '#228833', 's'),
        'diverse': ('Balanced + diversity', '#66aa55', '^'),
        'diverse_calibrated': ('Diversity + calibration', '#0077bb', 'D'),
        'rank': ('Rank-coded balanced', '#44aaaa', '<'),
        'scale_tag': ('Balanced + 8-bit scale', '#998833', '>'),
        'biohash': ('BioHash (pilot training)', '#aa3377', 'P'),
        'densefly': ('DenseFly', '#ee7733', 'v'),
        'gaussian_ops': ('Real Gaussian, matched ops', '#cc3311', 'X'),
        'gaussian_storage': ('Gaussian signs, matched bits', '#aa4499', '*'),
    }
    fig, axes = plt.subplots(len(datasets), 3, figsize=(16, 4 * len(datasets)), squeeze=False)
    for i, dataset in enumerate(datasets):
        rows = [r for r in result['summary'] if r['dataset'] == dataset]
        for method, (label, color, marker) in families.items():
            rs = [r for r in rows if r['key'].split(':')[0] == method]
            if not rs:
                continue
            for j, cost in enumerate(('projection_operations', 'code_bits_fixed_width', 'test_search_seconds')):
                values = [r[cost] if j < 2 else 1000 * r[cost]['mean'] / result['config']['queries'] for r in rs]
                axes[i, j].scatter(values, [r['validation_ap']['mean'] for r in rs],
                                   label=label, color=color, marker=marker, alpha=.8, s=45)
        for j, xlabel in enumerate(('Analytical projection operations / item', 'Estimated fixed-width code bits / item',
                                   'Measured search ms/query (reference code)')):
            ax = axes[i, j]
            ax.set(xscale='log', xlabel=xlabel, ylabel=f"Validation AP@{result['config']['top']}",
                   title=f'{dataset.upper()}: selected methods, all configurations')
            if j == 1:
                limits = ax.get_xlim()
                ax.xaxis.set_major_locator(FixedLocator([32, 64, 128, 256, 512, 1024]))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xlim(limits)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle('Exploratory FlyHash++ trade-offs — no optimized ANN latency claim')
    fig.tight_layout(rect=(0, .15, 1, .95))
    fig.savefig(path.with_name(path.stem + '_tradeoffs.png'), dpi=160)
    fig.savefig(path.with_name(path.stem + '_tradeoffs.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
