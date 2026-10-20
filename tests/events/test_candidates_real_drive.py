"""METHODS §6.8's candidates on a real drive, re-measured from the archive (opt-in).

The Wyoming van's Aeris on the afternoon of 2026-02-03, methane and ethane
from one spectrum, through the whole chain with the example sweep. Needs the
campaign archive mounted, so it runs only when ``TSARA_ARCHIVE`` names the
directory holding its ``2024/`` and ``2026/`` trees (about half a minute):

    TSARA_ARCHIVE=/path/to/Data pytest tests/events/test_candidates_real_drive.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.config.loader import load_manifest
from tsara.events import event_candidates, event_catalog, event_state
from tsara.ingest import ingest_campaign

ARCHIVE = os.environ.get("TSARA_ARCHIVE")

pytestmark = pytest.mark.skipif(
    not ARCHIVE,
    reason="Set TSARA_ARCHIVE to the directory holding the archive's 2024/ and 2026/ trees "
    "to re-measure METHODS §6.8's real drive.",
)

#: The afternoon file alone: the morning file of the same day ends in a
#: calibration sequence nothing in it marks (METHODS §6.8).
MANIFEST = """
name: wyo_van_20260203_afternoon
base_path: {aligned}
platform: {{kind: mobile, gps_instrument: gps, lat_variable: latitude, lon_variable: longitude}}
instruments:
  aeris:
    loader:
      format: parquet
      path_template: "WYO_aerisultra460/Eng/Ultra100460_260203_195917Eng.parquet"
      exclude: ["**/bad/**", "**/bad_timestamp/**"]
    variables:
      ch4:
        column: CH4_ppm
        role: gas
        units: ppm
        convert: {{from_unit: ppm, to_unit: ppb, scale: 1000.0}}
      c2h6: {{column: C2H6_ppb, role: gas, units: ppb}}
  gps:
    loader:
      format: parquet
      path_template: "WYO_sprinter/UWTR_WX_Sprinter_20260203_*.parquet"
      exclude: ["**/bad/**", "**/bad_timestamp/**"]
    variables:
      latitude: {{column: lat_deg, role: gps_lat, units: degrees_north}}
      longitude: {{column: lon_deg, role: gps_lon, units: degrees_east}}
"""


def test_the_wyoming_van_on_the_afternoon_of_2026_02_03(tmp_path: Path) -> None:
    """The rule METHODS states: the example sweep (the baseline at 2, 10 and 60 min
    x q 0.01, 0.05 and 0.10; events at entry 3, 4 and 5 x exit 1, 1.5 and 2; the
    events configuration's defaults otherwise), no QA/QC, one record. Per gas and
    window: candidates, those whose highest peak z reaches 5, and the median
    highest peak z; per gas, the clean spread at q 0.05 (median over readings)
    and the chance events the closed form expects at q 0.05, entry 5, exit 1."""
    assert ARCHIVE is not None
    aligned = Path(ARCHIVE) / "2026" / "03_instrument_aligned"
    path = tmp_path / "manifest.yaml"
    path.write_text(MANIFEST.format(aligned=aligned))
    aeris = ingest_campaign(load_manifest(path))["aeris"]
    baseline = baseline_state(
        aeris,
        instrument="aeris",
        baseline=BaselineConfig(windows=("2min", "10min", "60min"), quantiles=(0.01, 0.05, 0.10)),
        variables=["ch4", "c2h6"],
    )
    events = event_state(
        baseline,
        instrument="aeris",
        events=EventsConfig(enter_multiple=(3.0, 4.0, 5.0), exit_multiple=(1.0, 1.5, 2.0)),
    )
    catalog = event_catalog({"aeris": events}, {"aeris": baseline})
    candidates = event_candidates(catalog, {"aeris": events})

    finite = np.isfinite(aeris["ch4"].values)
    times = aeris["time"].values[finite]
    assert int(finite.sum()) == 14656
    assert round(float((times[-1] - times[0]) / np.timedelta64(1, "h")), 2) == 4.06
    record = events["record_ch4"].values
    assert np.unique(record[record >= 0]).size == 1

    expected = {
        "ch4": ([138, 90, 49], [92, 53, 31], [8.9, 7.5, 7.2], [2.51, 3.44, 4.77], [69, 37, 20]),
        "c2h6": ([160, 105, 73], [58, 43, 28], [4.0, 4.0, 4.3], [0.25, 0.28, 0.39], [37, 29, 18]),
    }
    point = {"baseline_quantile": 0.05, "enter_multiple": 5.0, "exit_multiple": 1.0}
    for variable, (counts, tall, medians, spreads, found) in expected.items():
        rows = [
            candidates[(candidates["species"] == variable) & (candidates["baseline_window"] == w)]
            for w in (120.0, 600.0, 3600.0)
        ]
        assert [len(r) for r in rows] == counts, variable
        assert [int((r["max_peak_z"] >= 5).sum()) for r in rows] == tall, variable
        assert [round(float(r["max_peak_z"].median()), 1) for r in rows] == medians, variable
        spread = events[f"clean_spread_{variable}"].sel(baseline_quantile=0.05).values
        assert [round(float(np.nanmedian(spread[:, k])), 2) for k in range(3)] == spreads
        assert events[f"n_events_{variable}"].sel(point).values.tolist() == found
        assert (events[f"chance_events_{variable}"].sel(point).values < 0.01).all()
