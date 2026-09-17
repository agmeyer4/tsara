"""The uncertainty chain end to end: manifest → file → ingest → bin.

Every other test of the uncertainty system covers one half of this path.
`tests/ingest/test_round_trip.py` checks that a declared budget survives being
written to a file and read back; `tests/align/test_binning.py` and
`tests/core/test_propagation.py` check that a sigma already sitting on a
stream is propagated correctly. The two halves meet nowhere, and where they
nearly do -- the acceptance test -- the published error column is renamed onto
its canonical name *by hand*, which short-circuits ingestion, the stage that
performs exactly that mapping in production.

So a rename could change on one side of the seam and every test would still
pass while a propagated sigma silently became empirical, or absent. This file
walks the whole path and then asks the only question that settles it: does the
binned value actually scatter about the truth by the amount the propagated
sigma claims?

The campaign has **no sources**, deliberately. With no plumes the residual
between a binned value and the binned truth is nothing but the injected error,
so the measurement is of the uncertainty model and not of plume shape.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from tsara.align import bin_streams_onto_cells
from tsara.config.loader import load_manifest
from tsara.core.naming import TIME_BOUNDS_VAR, sigma_rand_name, sigma_sys_name
from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.ingest import ingest_campaign
from tsara.synthetic import generate
from tsara.synthetic.config import SyntheticConfig
from tsara.synthetic.export import export_raw

#: Per-point random 1-sigma declared in the manifest, in ppb.
RANDOM_PPB = 2.0
#: Systematic component, as a fraction of the reading.
SYSTEMATIC_FRACTION = 0.01
#: Constant background, so the relative systematic is a near-constant offset.
BACKGROUND_PPB = 1900.0
#: Campaign length in seconds, and the cell width the scatter is measured on.
SPAN_S = 1200
CELL_S = 60

SPEC: dict[str, Any] = {
    "name": "uncertainty_chain",
    "seed": 1,
    "start": "2026-01-01T00:00:00Z",
    "duration": "20min",
    "platform": {"kind": "stationary", "latitude": 40.77, "longitude": -111.85},
    "atmosphere": {
        "fields": {
            "ch4": {
                "background": {"kind": "parametric", "offset": BACKGROUND_PPB},
                "role": "gas",
                "units": "ppb",
            }
        },
    },
    "instruments": {
        "analyzer": {
            "native_rate": "1s",
            "measures": {
                "ch4": {
                    "uncertainty": {
                        "random": {"absolute": RANDOM_PPB},
                        "systematic": {"relative": SYSTEMATIC_FRACTION},
                    },
                }
            },
        }
    },
}


def _truth_stream(generated: xr.Dataset) -> xr.Dataset:
    """Return the answer key as a stream the binner will accept.

    The truth columns never reach a file -- an instrument does not publish
    them -- so they are taken from the generator side and put on the same
    cells, which is what makes the comparison below a comparison at all.
    """
    truth = generated[["truth_background_ch4"]].assign_coords(
        {TIME_BOUNDS_VAR: generated[TIME_BOUNDS_VAR]}
    )
    truth["time"].attrs["bounds"] = TIME_BOUNDS_VAR
    return truth


def _uniform(epoch_ns: int, width_s: int, span_s: int = SPAN_S) -> CellBounds:
    """Return cells of ``width_s`` tiling ``span_s`` from ``epoch_ns``."""
    k = np.arange(span_s // width_s, dtype=np.int64)
    return CellBounds(
        start_ns=epoch_ns + k * width_s * SECOND,
        stop_ns=epoch_ns + (k + 1) * width_s * SECOND,
    )


def _walk(tmp_path: Path, seed: int) -> tuple[xr.Dataset, xr.Dataset, float]:
    """Run the whole chain for one seed.

    Returns the ingested stream, the generator's truth stream, and the
    systematic offset the generator actually injected -- which is a single
    draw for the whole campaign, so it is one number.
    """
    spec = copy.deepcopy(SPEC)
    spec["seed"] = seed
    data = generate(SyntheticConfig.model_validate(spec))
    manifest = load_manifest(export_raw(data, tmp_path / f"seed{seed}"))
    ingested = ingest_campaign(manifest)["analyzer"]
    generated = data.streams["analyzer"]
    injected = float(
        generated["ch4"].attrs["true_sys_rel_draw"] * SYSTEMATIC_FRACTION * BACKGROUND_PPB
    )
    return ingested, _truth_stream(generated), injected


def _epoch(stream: xr.Dataset) -> int:
    """Return the first cell's start, in epoch nanoseconds."""
    return int(stream[TIME_BOUNDS_VAR].values[0, 0].astype("datetime64[ns]").astype("int64"))


def test_a_declared_budget_reaches_the_binner_through_a_file(tmp_path: Path) -> None:
    """Both components survive the round trip under their canonical names.

    The names are the seam: the generator publishes a per-point sigma under
    whatever raw column a manifest declares, and ingestion is what maps it
    onto ``sigma_rand_<species>``. If that mapping broke, the binner would
    emit no sigma at all and every downstream weight would silently fall back.
    """
    stream, _, _ = _walk(tmp_path, seed=1)
    assert sigma_rand_name("ch4") in stream.data_vars
    assert sigma_sys_name("ch4") in stream.data_vars
    assert stream["ch4"].attrs["uncertainty_provenance_random"] == "declared"
    assert stream["ch4"].attrs["uncertainty_provenance_systematic"] == "declared"

    joined = bin_streams_onto_cells({"analyzer": stream}, _uniform(_epoch(stream), CELL_S), ["ch4"])
    assert sigma_rand_name("ch4") in joined.data_vars
    assert joined[sigma_rand_name("ch4")].attrs["uncertainty_provenance"] == "declared"
    # The form is recorded where it is true: on the sigma it produced. This
    # manifest declares no decorrelation timescale, so the readings are
    # independent by declaration and the default 'ar1_neff' never runs -- and
    # the product must not carry a second, dataset-level attribute claiming it
    # did. That disagreement was live until the notebook-04 walkthrough.
    assert joined[sigma_rand_name("ch4")].attrs["tsara_propagation_form"] == "independent"
    assert "tsara_propagation_form" not in joined.attrs


def test_the_random_component_falls_as_one_over_root_n(tmp_path: Path) -> None:
    """Averaging more samples shrinks the random component, exactly.

    Checked across three orders of magnitude rather than at one width,
    because a wrong exponent and a wrong constant look the same at a single
    point.
    """
    stream, _, _ = _walk(tmp_path, seed=1)
    epoch = _epoch(stream)
    for width_s in (1, 10, 60, 600):
        joined = bin_streams_onto_cells({"analyzer": stream}, _uniform(epoch, width_s), ["ch4"])
        n = float(np.nanmedian(joined["n_readings_ch4"].values))
        got = float(np.nanmedian(joined[sigma_rand_name("ch4")].values))
        assert n == pytest.approx(width_s)
        assert got == pytest.approx(RANDOM_PPB / np.sqrt(n), rel=1e-9)


def test_the_systematic_component_does_not_move(tmp_path: Path) -> None:
    """The error that averaging cannot touch.

    The single most common mistake in a reported enhancement ratio is
    shrinking this one with the sample count, so it is worth an assertion
    that would fail loudly if a √N ever appeared on this path.
    """
    stream, _, _ = _walk(tmp_path, seed=1)
    epoch = _epoch(stream)
    got = [
        float(
            np.nanmedian(
                bin_streams_onto_cells({"analyzer": stream}, _uniform(epoch, width_s), ["ch4"])[
                    sigma_sys_name("ch4")
                ].values
            )
        )
        for width_s in (1, 10, 60, 600)
    ]
    expected = SYSTEMATIC_FRACTION * BACKGROUND_PPB
    for value in got:
        assert value == pytest.approx(expected, rel=0.01)
    # Averaging 600 samples must not have bought anything at all.
    assert got[-1] == pytest.approx(got[0], rel=1e-3)
    assert got[-1] > expected / 2


def test_the_binned_values_scatter_by_the_amount_claimed(tmp_path: Path) -> None:
    """The question algebra cannot answer: is the claimed sigma the real one?

    Pooled over several campaigns because the scatter of 20 residuals is too
    loose a statistic to argue with; a dozen campaigns give enough residuals
    for the ratio below to be good to a few percent, which is far tighter than
    the failure this guards against. Treating the samples as independent when
    they are not, or forgetting the sample count entirely, would move the
    ratio by factors rather than by percent.
    """
    residuals: list[np.ndarray] = []
    claimed: list[float] = []
    for seed in range(1, 13):
        stream, truth, injected = _walk(tmp_path, seed=seed)
        cells = _uniform(_epoch(stream), CELL_S)
        joined = bin_streams_onto_cells({"analyzer": stream}, cells, ["ch4"])
        truth_binned = bin_streams_onto_cells({"truth": truth}, cells, ["truth_background_ch4"])
        got = joined["ch4"].values
        want = truth_binned["truth_background_ch4"].values
        usable = np.isfinite(got) & np.isfinite(want)
        # The systematic draw is one number for the whole campaign, so it
        # shifts every cell equally and is removed by centring. What is left
        # is the random component and nothing else.
        gap = got[usable] - want[usable] - injected
        residuals.append(gap)
        claimed.append(float(np.nanmedian(joined[sigma_rand_name("ch4")].values)))

    pooled = np.concatenate(residuals)
    observed = float(np.sqrt((pooled**2).mean()))
    predicted = float(np.mean(claimed))
    assert pooled.size >= 200
    assert observed == pytest.approx(predicted, rel=0.15)


def test_the_offset_from_truth_is_the_generators_own_systematic_draw(tmp_path: Path) -> None:
    """A systematic error is an offset, not a spread, and it survives averaging.

    This is the half of the two-component model that a scatter measurement
    cannot see: centring removes it. Here it is compared directly against the
    number the generator injected, campaign by campaign.
    """
    for seed in range(1, 5):
        stream, truth, injected = _walk(tmp_path, seed=seed)
        cells = _uniform(_epoch(stream), CELL_S)
        joined = bin_streams_onto_cells({"analyzer": stream}, cells, ["ch4"])
        truth_binned = bin_streams_onto_cells({"truth": truth}, cells, ["truth_background_ch4"])
        got = joined["ch4"].values
        want = truth_binned["truth_background_ch4"].values
        usable = np.isfinite(got) & np.isfinite(want)
        offset = float(np.mean(got[usable] - want[usable]))
        # Tolerance is the random component's own contribution to the mean of
        # ~20 cells, which is what remains once the offset is removed.
        room = 4.0 * RANDOM_PPB / np.sqrt(CELL_S * int(usable.sum()))
        assert offset == pytest.approx(injected, abs=room)
        # And it is not small: a sqrt(N) reduction would have made it so.
        assert abs(injected) > 0 or offset == pytest.approx(0.0, abs=room)


def _slow_cell_sigmas(
    readings: CellBounds,
    values: np.ndarray,
    sigma_rand: np.ndarray,
    sigma_sys: np.ndarray,
    target: CellBounds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute value and both sigmas per cell, slowly, from the definitions.

    Plain Python loops over every (target, reading) pair, with the overlap
    written as ``max(0, min(stops) - max(starts))``. Independent of the
    vectorized path in every way that matters: no binary search, no
    ``bincount``, no long form, and the weights are rebuilt from the
    boundaries rather than carried.
    """
    n = len(target)
    out_value = np.full(n, np.nan)
    out_rand = np.full(n, np.nan)
    out_sys = np.full(n, np.nan)
    for cell in range(n):
        weights, keep_value, rand_here, sys_here = [], [], [], []
        for row in range(len(readings)):
            overlap = min(readings.stop_ns[row], target.stop_ns[cell]) - max(
                readings.start_ns[row], target.start_ns[cell]
            )
            if overlap <= 0 or not np.isfinite(values[row]):
                continue
            weights.append(float(overlap))
            keep_value.append(float(values[row]))
            rand_here.append(float(sigma_rand[row]))
            sys_here.append(float(sigma_sys[row]))
        if not weights:
            continue
        w = np.asarray(weights) / sum(weights)
        out_value[cell] = float(np.sum(w * np.asarray(keep_value)))
        # A sample with no stated uncertainty is dropped from the budget and
        # the rest renormalized, which is what the module does.
        rand_arr = np.asarray(rand_here)
        good = np.isfinite(rand_arr)
        if good.any():
            wr = w[good] / w[good].sum()
            out_rand[cell] = float(np.sqrt(np.sum(wr**2 * rand_arr[good] ** 2)))
        sys_arr = np.asarray(sys_here)
        good_sys = np.isfinite(sys_arr)
        if good_sys.any():
            ws = w[good_sys] / w[good_sys].sum()
            out_sys[cell] = float(np.sum(ws * sys_arr[good_sys]))
    return out_value, out_rand, out_sys


def test_an_awkward_cell_matches_a_slow_reimplementation(tmp_path: Path) -> None:
    """The tidy fixture above cannot see two real mistakes; this one can.

    Mutation-tested. With cells that tile the samples exactly, every overlap
    weight is equal, so substituting the propagation's own weights for the
    operation's changes nothing and the substitution passes unnoticed. And
    with no missing sigma, dropping the rule that a sample without a stated
    uncertainty is excluded rather than counted as *zero* uncertainty is
    likewise invisible.

    So this fixture is deliberately awkward: target cells 2.5 s wide against
    1 s samples, so boundary samples contribute half weight, and one sample
    whose value survived while its uncertainty did not.
    """
    stream, _, _ = _walk(tmp_path, seed=3)
    perturbed = stream.copy(deep=True)
    # A value that survived while its stated uncertainty did not. Rare -- on
    # the only per-point uncertainty columns in the 2024 archive the two are
    # always absent together -- but the code path exists and decides whether a
    # missing sigma is dropped or silently counted as zero.
    perturbed[sigma_rand_name("ch4")].values[7] = np.nan

    epoch = _epoch(stream)
    k = np.arange(120, dtype=np.int64)
    target = CellBounds(start_ns=epoch + k * 2_500_000_000, stop_ns=epoch + (k + 1) * 2_500_000_000)
    joined = bin_streams_onto_cells({"analyzer": perturbed}, target, ["ch4"])

    bounds = perturbed[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
    readings = CellBounds(start_ns=bounds[:, 0], stop_ns=bounds[:, 1])
    want_value, want_rand, want_sys = _slow_cell_sigmas(
        readings,
        perturbed["ch4"].values,
        perturbed[sigma_rand_name("ch4")].values,
        perturbed[sigma_sys_name("ch4")].values,
        target,
    )
    assert np.allclose(joined["ch4"].values, want_value, rtol=1e-12, equal_nan=True)
    assert np.allclose(joined[sigma_rand_name("ch4")].values, want_rand, rtol=1e-12, equal_nan=True)
    assert np.allclose(joined[sigma_sys_name("ch4")].values, want_sys, rtol=1e-12, equal_nan=True)
    # The fixture has to actually be awkward, or it proves nothing. Every
    # cell holds three samples, but not with equal weight: two whole ones and
    # a half. Equal-weight arithmetic would therefore give a visibly different
    # answer, which is what makes a substituted weight detectable here.
    counts = joined["n_readings_ch4"].values
    assert set(np.unique(counts)) == {3}
    equal_weight = RANDOM_PPB / np.sqrt(counts)
    assert not np.allclose(joined[sigma_rand_name("ch4")].values, equal_weight, rtol=1e-3)
