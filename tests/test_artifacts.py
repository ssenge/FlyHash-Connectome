"""The paper, README and supplement must agree with results/*.json.

These fail if someone edits a number by hand or forgets `flypath report`.
Skipped until the analysis has been run.
"""

from __future__ import annotations

import json

import pytest

from flypath.config import ROOT

PRIMARY = ROOT / "results" / "primary.json"


@pytest.fixture(scope="module")
def primary():
    if not PRIMARY.exists():
        pytest.skip("run: python -m flypath analyse && python -m flypath report")
    return json.loads(PRIMARY.read_text())


def _row(primary):
    return next(r for r in primary["rows"] if r["primary"])


def test_generated_tex_matches_results(primary):
    tex = (ROOT / "paper" / "generated.tex").read_text()
    row = _row(primary)
    assert f"\\newcommand{{\\PrimaryPtwo}}{{{row['randomization']['p_two_sided']:.3f}}}" in tex
    assert f"\\newcommand{{\\PrimaryReal}}{{{row['real']:.4f}}}" in tex
    lo, hi = primary["bootstrap"]["relative_ci"]
    assert f"\\newcommand{{\\BootLo}}{{{100 * lo:+.1f}\\%}}" in tex
    assert f"\\newcommand{{\\BootHi}}{{{100 * hi:+.1f}\\%}}" in tex


def test_readme_block_matches_results(primary):
    readme = (ROOT / "README.md").read_text()
    row = _row(primary)
    assert f"| {row['real']:.4f} |" in readme
    assert f"{row['randomization']['p_two_sided']:.3f}" in readme


def test_paper_uses_macros_not_literals():
    """The headline numbers must come from generated.tex."""
    tex = (ROOT / "paper" / "flyhash.tex").read_text()
    assert "\\input{generated.tex}" in tex
    for macro in ("\\PrimaryPtwo", "\\BootLo", "\\BootHi", "\\AnimalRows", "\\TaskNovConn"):
        assert macro in tex
