"""The shared projection builder used for every connectome."""

import numpy as np
import pandas as pd

from flypath import connectomes as C


def test_build_sums_per_glomerulus_and_drops_uninnervated_cells():
    pn = pd.DataFrame({"id": [1, 2, 3, 4, 5], "glom": ["DA1", "DA1", "VM2", "VP1d", "Z"]})
    edges = pd.DataFrame({"pre":  [1, 2, 3, 3, 4, 5, 1],
                          "post": [10, 10, 11, 11, 12, 12, 13],
                          "weight": [2, 3, 1, 4, 7, 7, 1]})
    p = C.build(pn, [10, 11, 12, 13, 99], edges, "toy", "R")
    assert p.glomeruli == ["DA1", "VM2"]                 # VP (thermo) and Z excluded
    assert p.matrix.shape == (2, 3)                      # cells 12 and 99 have no olfactory input
    np.testing.assert_array_equal(p.matrix, [[5, 0, 1], [0, 5, 0]])
    np.testing.assert_array_equal(p.meta["pn_partners"], [2, 1, 1])
    q = C.fh.align(p, ["VM2"])                          # partners follow the restriction
    np.testing.assert_array_equal(q.meta["pn_partners"], [1])


def test_response_breadth_excludes_unmeasured_odorants():
    x = np.array([[0.5, 0.0], [np.nan, 0.5], [np.nan, 0.0], [0.1, 0.5]])
    observed = ~np.isnan(x)
    # glomerulus 0: measured 0.5, 0.1 -> 1 of 2; glomerulus 1: 0, .5, 0, .5 -> 2 of 4
    np.testing.assert_allclose(C.response_breadth(np.nan_to_num(x), observed), [0.5, 0.5])
    # the unmasked comparison would count the two unmeasured entries as non-responses (1/4)
    assert np.nanmean(np.where(observed, x, np.nan) > 0.2, 0)[0] == 0.25
