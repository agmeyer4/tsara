"""Tests for the ICARTT FFI-1001 reader and its filename conventions.

The fixtures are written by hand rather than copied from the archive: no real
data ever enters the repository. But every awkward case below was *found* in
the 1122-file campaign archive and is reproduced faithfully — the datetime
strings in a file whose header promises seconds, the non-UTF-8 bytes, the
per-variable missing sentinels, the truncated data rows, and the filename
shapes that a naive revision policy silently discards.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import CSVLoader, ICARTTLoader
from tsara.ingest.base import TIME_INDEX_NAME, TsaraIngestError
from tsara.ingest.icartt import (
    parse_icartt_filename,
    parse_icartt_header,
    read_icartt,
    select_latest_revisions,
)

LOADER = ICARTTLoader(path_template="*.ict")


def _build(
    *,
    variables: list[tuple[str, str, str]] | None = None,
    scales: list[float] | None = None,
    missings: list[float] | None = None,
    independent: str = "Time_Start, seconds_past_midnight, Seconds from midnight UTC",
    data_date: str = "2024, 08, 15",
    special: list[str] | None = None,
    normal: list[str] | None = None,
    rows: list[str] | None = None,
) -> str:
    """Assemble a syntactically valid FFI-1001 file.

    NLHEAD is computed rather than hard-coded, so a test that changes the
    number of variables or comments cannot accidentally desynchronize the
    header from its own declared length.
    """
    variables = variables or [("CH4_ppb", "ppb", "Methane"), ("CO2_ppm", "ppm", "Carbon dioxide")]
    scales = scales if scales is not None else [1.0] * len(variables)
    missings = missings if missings is not None else [-9999.0] * len(variables)
    special = special if special is not None else ["SPECIAL COMMENTS:"]
    column_header = ", ".join([independent.split(",")[0].strip(), *(v[0] for v in variables)])
    normal = (normal if normal is not None else ["PLATFORM: Test Van"]) + [column_header]
    rows = rows if rows is not None else ["42480.0, 1900.0, 415.0", "42481.0, 1901.0, 415.5"]

    body = [
        f"{', '.join(str(v) for v in scales)}",
        f"{', '.join(str(v) for v in missings)}",
        *[f"{n}, {u}, {d}" for n, u, d in variables],
        str(len(special)),
        *special,
        str(len(normal)),
        *normal,
    ]
    head = [
        "PLACEHOLDER, 1001",
        "Doe, Jane",
        "Test Organization",
        "Test Instrument",
        "TESTMISSION",
        "1, 1",
        f"{data_date}, 2024, 12, 21",
        "1",
        independent,
        str(len(variables)),
    ]
    n_header = len(head) + len(body)
    head[0] = f"{n_header}, 1001"
    return "\n".join([*head, *body, *rows]) + "\n"


def _write(tmp_path: Path, text: str, name: str = "f.ict", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_header_fields(tmp_path: Path) -> None:
    path = _write(tmp_path, _build())
    header = parse_icartt_header(path.read_text().splitlines(), path)

    assert header.file_format_index == 1001
    assert header.pi_name == "Doe, Jane"
    assert header.organization == "Test Organization"
    assert header.mission == "TESTMISSION"
    assert header.volume == (1, 1)
    assert header.data_date == date(2024, 8, 15)
    assert header.revision_date == date(2024, 12, 21)
    assert header.interval == 1.0
    assert header.independent_variable.name == "Time_Start"
    assert [v.name for v in header.variables] == ["CH4_ppb", "CO2_ppm"]


def test_data_begins_at_declared_header_length(tmp_path: Path) -> None:
    """NLHEAD is the format's one self-describing guarantee."""
    text = _build()
    path = _write(tmp_path, text)
    header = parse_icartt_header(text.splitlines(), path)
    assert text.splitlines()[header.n_header_lines].startswith("42480.0")


def test_column_names_come_from_the_last_normal_comment(tmp_path: Path) -> None:
    """A genuine quirk: the data header hides at the end of a comment block."""
    path = _write(tmp_path, _build())
    header = parse_icartt_header(path.read_text().splitlines(), path)
    assert header.column_names == ("Time_Start", "CH4_ppb", "CO2_ppm")


def test_variable_description_may_contain_commas(tmp_path: Path) -> None:
    """Descriptions are free text in a comma-delimited header line."""
    path = _write(tmp_path, _build(variables=[("CH4_ppb", "ppb", "Methane, dry air, 1 Hz")]))
    header = parse_icartt_header(path.read_text().splitlines(), path)
    assert header.variables[0].description == "Methane, dry air, 1 Hz"


def test_metadata_harvested_from_comment_blocks(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _build(special=["ULOD_FLAG: -7777", "LLOD_FLAG: -8888"], normal=["REVISION: R0"]),
    )
    header = parse_icartt_header(path.read_text().splitlines(), path)
    assert header.metadata["ULOD_FLAG"] == "-7777"
    assert header.metadata["LLOD_FLAG"] == "-8888"
    assert header.metadata["REVISION"] == "R0"


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="is empty"):
        parse_icartt_header([], tmp_path / "f.ict")


def test_missing_nlhead_line_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="NLHEAD"):
        parse_icartt_header(["not a header"], tmp_path / "f.ict")


def test_non_1001_format_is_rejected(tmp_path: Path) -> None:
    """TSARA implements one FFI; saying so beats mis-parsing another."""
    with pytest.raises(TsaraIngestError, match="FFI-1001 only"):
        parse_icartt_header(["30, 2110", "x"], tmp_path / "f.ict")


def test_truncated_header_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="declares a 44-line header"):
        parse_icartt_header(["44, 1001", "Doe, Jane"], tmp_path / "f.ict")


def test_malformed_header_body_is_rejected(tmp_path: Path) -> None:
    lines = _build().splitlines()
    lines[6] = "not, a, date"
    with pytest.raises(TsaraIngestError, match="Malformed ICARTT header"):
        parse_icartt_header(lines, Path("f.ict"))


def test_non_numeric_comment_count_is_tolerated(tmp_path: Path) -> None:
    """A deviant comment block must not cost the whole file."""
    lines = _build().splitlines()
    header_len = int(lines[0].split(",")[0])
    lines[14] = "NOT A NUMBER"
    header = parse_icartt_header(lines, Path("f.ict"))
    assert header.n_header_lines == header_len


def test_comment_block_mismatch_logs_debug_and_trusts_nlhead(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A comment-count mismatch is reported, but only at DEBUG.

    It was a warning until the Phase-3 walkthrough measured it: on the target
    archive it fires on 44 of 1055 files and diagnoses none of them, because
    43 PTR-MS files carry one extra blank-named definition line that offsets
    the walk harmlessly. The provable case (NLHEAD smaller than 12 + NV) is a
    warning instead — see the test below.
    """
    lines = _build().splitlines()
    lines[14] = "0"  # understate the special-comment count
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        parse_icartt_header(lines, Path("f.ict"))
    assert caplog.text == ""

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="tsara.ingest.icartt"):
        parse_icartt_header(lines, Path("f.ict"))
    assert "trusting NLHEAD" in caplog.text


# ---------------------------------------------------------------------------
# Reading data
# ---------------------------------------------------------------------------


def test_reads_values_and_time(tmp_path: Path) -> None:
    path = _write(tmp_path, _build())
    table = read_icartt(path, LOADER)

    assert list(table.frame.columns) == ["Time_Start", "CH4_ppb", "CO2_ppm"]
    assert table.frame["CH4_ppb"].tolist() == [1900.0, 1901.0]
    # 42480 s past midnight on 2024-08-15 = 11:48:00 UTC
    assert table.frame.index[0] == pd.Timestamp("2024-08-15 11:48:00")


def test_index_is_naive_utc_nanoseconds(tmp_path: Path) -> None:
    path = _write(tmp_path, _build())
    index = read_icartt(path, LOADER).frame.index
    assert index.dtype == "datetime64[ns]"
    assert pd.DatetimeIndex(index).tz is None
    assert index.name == TIME_INDEX_NAME


def test_seconds_past_86400_cross_midnight(tmp_path: Path) -> None:
    """Records spanning midnight keep counting; no special case needed."""
    path = _write(tmp_path, _build(rows=["86401.0, 1900.0, 415.0"]))
    assert read_icartt(path, LOADER).frame.index[0] == pd.Timestamp("2024-08-16 00:00:01")


def test_declared_missing_values_become_nan(tmp_path: Path) -> None:
    path = _write(tmp_path, _build(rows=["42480.0, -9999.0, 415.0"]))
    frame = read_icartt(path, LOADER).frame
    assert np.isnan(frame["CH4_ppb"].iloc[0])
    assert frame["CO2_ppm"].iloc[0] == 415.0


def test_missing_sentinels_are_per_variable(tmp_path: Path) -> None:
    """Different columns legitimately declare different sentinels."""
    path = _write(
        tmp_path,
        _build(missings=[-9999.0, -99999.0], rows=["42480.0, -9999.0, -99999.0"]),
    )
    frame = read_icartt(path, LOADER).frame
    assert frame["CH4_ppb"].isna().all()
    assert frame["CO2_ppm"].isna().all()


def test_sentinel_spelling_variants_are_matched(tmp_path: Path) -> None:
    """'-9999' in the header and '-9999.0' in the data are the same sentinel."""
    path = _write(tmp_path, _build(missings=[-9999, -9999], rows=["42480.0, -9999.000, 415.0"]))
    assert read_icartt(path, LOADER).frame["CH4_ppb"].isna().all()


def test_scale_factors_are_applied(tmp_path: Path) -> None:
    path = _write(tmp_path, _build(scales=[1000.0, 1.0], rows=["42480.0, 1.9, 415.0"]))
    assert read_icartt(path, LOADER).frame["CH4_ppb"].iloc[0] == pytest.approx(1900.0)


def test_sentinels_are_masked_before_scaling(tmp_path: Path) -> None:
    """Order matters: scaling first turns -9999 into an ordinary-looking number."""
    path = _write(tmp_path, _build(scales=[1000.0, 1.0], rows=["42480.0, -9999.0, 415.0"]))
    frame = read_icartt(path, LOADER).frame
    assert frame["CH4_ppb"].isna().all()


def test_provenance_is_carried_in_attrs(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _build(special=["ULOD_FLAG: -7777", "LLOD_FLAG: -8888"], normal=["REVISION: R1"]),
    )
    attrs = read_icartt(path, LOADER).attrs
    assert attrs["icartt_pi"] == "Doe, Jane"
    assert attrs["icartt_mission"] == "TESTMISSION"
    assert attrs["icartt_data_date"] == "2024-08-15"
    # LOD flags are carried, not acted on: below-detection is an upper bound,
    # not a missing value, and a later stage may want to model it.
    assert attrs["icartt_llod_flag"] == "-8888"
    assert attrs["icartt_revision"] == "R1"


def test_non_utf8_bytes_are_tolerated(tmp_path: Path) -> None:
    """30 archive files carry stray bytes in PI names; losing them is absurd."""
    path = _write(tmp_path, _build().replace("Doe, Jane", "Doe, Jos\xe9"), encoding="latin-1")
    assert read_icartt(path, LOADER).frame.shape[0] == 2


def test_datetime_strings_where_the_header_promises_seconds(tmp_path: Path) -> None:
    """43 PTR-MS VOC files do exactly this; refusing them loses 35 species."""
    path = _write(
        tmp_path,
        _build(
            independent="Time_UTC, number_of_seconds_from_0000_UTC",
            rows=["2024-08-15 19:26:00, 1900.0, 415.0", "2024-08-15 19:26:01, 1901.0, 415.5"],
        ),
    )
    table = read_icartt(path, LOADER)
    assert table.frame.index[0] == pd.Timestamp("2024-08-15 19:26:00")


def test_malformed_rows_are_skipped_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One corrupt line must not cost a whole measurement day."""
    path = _write(
        tmp_path,
        _build(rows=["42480.0, 1900.0, 415.0", "42481.0, 1, 2, 3, 4, 5", "42482.0, 1902.0, 416.0"]),
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        table = read_icartt(path, LOADER)

    assert len(table.frame) == 2
    assert "Dropped 1 of 3 rows" in caplog.text
    assert "malformed data row (wrong field count)" in caplog.text


def test_unparseable_timestamps_are_dropped_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(
        tmp_path,
        _build(rows=["42480.0, 1900.0, 415.0", "not-a-time, 1901.0, 415.5"]),
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        table = read_icartt(path, LOADER)
    assert len(table.frame) == 1
    assert "timestamp did not parse" in caplog.text


def test_column_header_disagreeing_with_nv_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The variable definitions are authoritative; NV guarantees their count."""
    text = _build()
    lines = text.splitlines()
    lines[-3] = "Time_Start, CH4_ppb"  # column header missing a name
    path = _write(tmp_path, "\n".join(lines) + "\n")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        table = read_icartt(path, LOADER)
    assert list(table.frame.columns) == ["Time_Start", "CH4_ppb", "CO2_ppm"]
    assert "using the variable definitions" in caplog.text


def test_header_without_data_rows_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, _build(rows=[""]))
    with pytest.raises(TsaraIngestError, match="no data rows"):
        read_icartt(path, LOADER)


def test_independent_variable_absent_from_data_is_an_error(tmp_path: Path) -> None:
    text = _build()
    lines = text.splitlines()
    lines[8] = "Absent_Column, seconds_past_midnight, nope"
    path = _write(tmp_path, "\n".join(lines) + "\n")
    with pytest.raises(TsaraIngestError, match="declares independent variable"):
        read_icartt(path, LOADER)


def test_wholly_unparseable_time_column_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, _build(rows=["zzz, 1900.0, 415.0", "yyy, 1901.0, 415.5"]))
    with pytest.raises(TsaraIngestError, match="neither numeric seconds nor parseable"):
        read_icartt(path, LOADER)


def test_wrong_loader_type_is_refused(tmp_path: Path) -> None:
    bad: Any = CSVLoader.model_validate(
        {"path_template": "*.csv", "time": {"column": "t", "format": "unix"}}
    )
    with pytest.raises(TsaraIngestError, match="registered under the wrong format"):
        read_icartt(tmp_path / "f.ict", bad)


# ---------------------------------------------------------------------------
# Filenames and revision selection
# ---------------------------------------------------------------------------


def test_standard_filename() -> None:
    parsed = parse_icartt_filename(Path("USOS-HCHO_MobileLab_20240815_R0.ict"))
    assert parsed is not None
    assert parsed.identifier == "USOS-HCHO_MobileLab"
    assert parsed.file_date == date(2024, 8, 15)
    assert parsed.revision == "R0"
    assert parsed.comment == ""


def test_identifier_may_span_more_than_two_parts() -> None:
    """147 archive files look like this; counting underscores fails on them."""
    parsed = parse_icartt_filename(Path("SLCSOS-ROZE-O3_UWyo_Sprinter_20240802_RA_L1.ict"))
    assert parsed is not None
    assert parsed.identifier == "SLCSOS-ROZE-O3_UWyo_Sprinter"
    assert parsed.revision == "RA"
    assert parsed.comment == "L1"


def test_multi_part_comment_is_preserved() -> None:
    parsed = parse_icartt_filename(
        Path("SLC-SOS-UMT_PTR-MS-VOC_20240801_RA_Drive01_LakeBreeze.ict")
    )
    assert parsed is not None
    assert parsed.comment == "Drive01_LakeBreeze"


def test_filename_without_revision() -> None:
    parsed = parse_icartt_filename(Path("Data_Thing_20240815.ict"))
    assert parsed is not None
    assert parsed.revision is None
    assert parsed.comment == ""


def test_filename_accepts_a_plain_string() -> None:
    parsed = parse_icartt_filename("USOS-HCHO_MobileLab_20240815_R0.ict")
    assert parsed is not None
    assert parsed.file_date == date(2024, 8, 15)


def test_filename_without_a_date_is_unparseable() -> None:
    assert parse_icartt_filename(Path("Miro_Data_0820.ict")) is None


def test_eight_digit_non_date_is_skipped() -> None:
    """A serial number is not a date; keep scanning rather than give up."""
    parsed = parse_icartt_filename(Path("Sensor_99999999_20240815_R0.ict"))
    assert parsed is not None
    assert parsed.file_date == date(2024, 8, 15)


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("X_Y_20240815_RA.ict", "X_Y_20240815_RB.ict"),
        ("X_Y_20240815_R0.ict", "X_Y_20240815_R1.ict"),
        ("X_Y_20240815_R1.ict", "X_Y_20240815_R2.ict"),
        # Numeric revisions are final data; alphabetic are preliminary field
        # data. 41 archive groups hold both, so this ordering is load-bearing.
        ("X_Y_20240815_RA.ict", "X_Y_20240815_R0.ict"),
        ("X_Y_20240815.ict", "X_Y_20240815_RA.ict"),
        # Plain string ordering would put R10 before R2.
        ("X_Y_20240815_R2.ict", "X_Y_20240815_R10.ict"),
    ],
)
def test_revision_ranking(lower: str, higher: str) -> None:
    low = parse_icartt_filename(Path(lower))
    high = parse_icartt_filename(Path(higher))
    assert low is not None and high is not None
    assert low.revision_rank < high.revision_rank


def test_latest_revision_is_selected() -> None:
    paths = [Path("X_Y_20240815_R0.ict"), Path("X_Y_20240815_R1.ict")]
    assert select_latest_revisions(paths) == [Path("X_Y_20240815_R1.ict")]


def test_files_differing_only_by_comment_are_all_kept() -> None:
    """The bug this exists to prevent: three drives collapsing into one."""
    paths = [
        Path("SLC_PTR_20240802_RA_Drive02_Canyon.ict"),
        Path("SLC_PTR_20240802_RA_Drive03_LakeBreeze.ict"),
        Path("SLC_PTR_20240802_RA_Stationary01.ict"),
    ]
    assert len(select_latest_revisions(paths)) == 3


def test_processing_levels_are_kept_but_revisions_within_them_are_not() -> None:
    """L1 and L2 are different products; R0 supersedes RA within each."""
    paths = [
        Path("X_Y_20240802_RA_L1.ict"),
        Path("X_Y_20240802_R0_L1.ict"),
        Path("X_Y_20240802_RA_L2.ict"),
        Path("X_Y_20240802_R0_L2.ict"),
    ]
    assert select_latest_revisions(paths) == [
        Path("X_Y_20240802_R0_L1.ict"),
        Path("X_Y_20240802_R0_L2.ict"),
    ]


def test_different_dates_are_independent() -> None:
    paths = [Path("X_Y_20240815_R0.ict"), Path("X_Y_20240816_R0.ict")]
    assert len(select_latest_revisions(paths)) == 2


def test_unparseable_names_are_always_kept() -> None:
    """An unparseable name is not evidence of duplication."""
    paths = [Path("Miro_Data_0820.ict"), Path("Other_Thing.ict")]
    assert select_latest_revisions(paths) == sorted(paths)


def test_selection_is_deterministic_regardless_of_input_order() -> None:
    a = [Path("X_Y_20240815_R0.ict"), Path("X_Y_20240815_R1.ict")]
    assert select_latest_revisions(a) == select_latest_revisions(list(reversed(a)))


def test_identical_ranks_break_ties_stably() -> None:
    """Same key and rank in two directories: pick one, but always the same one."""
    paths = [Path("b/X_Y_20240815_R0.ict"), Path("a/X_Y_20240815_R0.ict")]
    assert select_latest_revisions(paths) == select_latest_revisions(list(reversed(paths)))
    assert len(select_latest_revisions(paths)) == 1


def test_superseded_count_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    paths = [Path("X_Y_20240815_R0.ict"), Path("X_Y_20240815_R1.ict")]
    with caplog.at_level(logging.INFO, logger="tsara.ingest.icartt"):
        select_latest_revisions(paths)
    assert "superseded 1 ICARTT file(s)" in caplog.text


def test_empty_input() -> None:
    assert select_latest_revisions([]) == []


def test_uniformly_too_wide_rows_do_not_shift_the_columns(tmp_path: Path) -> None:
    """The silent-corruption case that `index_col=False` exists to prevent.

    When *every* data row carries more fields than the header declares,
    pandas' default is to treat the surplus leading fields as an index. The
    frame then looks perfectly healthy while every value sits one or more
    columns to the left of where it belongs — the first declared column
    receiving the third field. Nothing downstream could detect that, so it
    has to be prevented here.
    """
    path = _write(
        tmp_path,
        _build(rows=["42480.0, 1900.0, 415.0, 99, 98", "42481.0, 1901.0, 415.5, 97, 96"]),
    )
    # pandas warns that the surplus fields are discarded, which is exactly the
    # intended outcome here: dropped is correct, silently shifted is not.
    with pytest.warns(pd.errors.ParserWarning):
        frame = read_icartt(path, LOADER).frame

    assert frame.index[0] == pd.Timestamp("2024-08-15 11:48:00")
    assert frame["CH4_ppb"].tolist() == [1900.0, 1901.0]
    assert frame["CO2_ppm"].tolist() == [415.0, 415.5]


def test_declared_variable_absent_from_the_data_is_skipped(tmp_path: Path) -> None:
    """A column header naming different columns must not crash the read.

    The count matches NV, so the header line is trusted; a declared variable
    that simply is not there is left alone rather than fabricated.
    """
    lines = _build().splitlines()
    lines[-3] = "Time_Start, CH4_ppb, SOMETHING_ELSE"
    path = _write(tmp_path, "\n".join(lines) + "\n")
    frame = read_icartt(path, LOADER).frame
    assert list(frame.columns) == ["Time_Start", "CH4_ppb", "SOMETHING_ELSE"]
    assert frame["CH4_ppb"].tolist() == [1900.0, 1901.0]


def test_nan_missing_sentinel_masks_nothing(tmp_path: Path) -> None:
    """A file declaring no usable sentinel must not mask real measurements."""
    path = _write(
        tmp_path,
        _build(missings=[float("nan"), float("nan")], rows=["42480.0, -9999.0, 415.0"]),
    )
    frame = read_icartt(path, LOADER).frame
    assert frame["CH4_ppb"].iloc[0] == -9999.0


# ---------------------------------------------------------------------------
# Walkthrough stage 4: the four parse fixes measured against the real archive
# ---------------------------------------------------------------------------


def test_mixed_time_column_follows_the_majority_not_a_stray_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A couple of stray numerics must not redefine a datetime time axis.

    This is the archive's most expensive misparse reproduced in miniature.
    Two PTR-MS VOC files hold thousands of datetime strings plus exactly two
    numeric tokens that leaked in from a mis-declared header block. The old
    test — "is *any* value numeric?" — sent them down the seconds-past-
    midnight branch, where every real timestamp then failed to convert and
    was dropped, leaving 2 rows out of 10,235 behind a mere warning.
    """
    rows = [f"2024-08-15 19:{m:02d}:00, 1900.0, 415.0" for m in range(10)]
    rows += ["1, 1901.0, 415.5", "19, 1902.0, 416.0"]  # the two stray numerics
    path = _write(tmp_path, _build(rows=rows))

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        frame = read_icartt(path, LOADER).frame

    # The ten datetime rows survive; only the two numeric intruders drop out.
    assert len(frame) == 10
    assert frame.index[0] == pd.Timestamp("2024-08-15 19:00:00")
    assert "is mixed" in caplog.text
    assert "Reading it as timestamps (majority)" in caplog.text


def test_mixed_time_column_with_numeric_majority_reads_as_seconds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same vote, decided the other way, keeps the spec-compliant reading."""
    rows = [f"{42480 + i}.0, 1900.0, 415.0" for i in range(10)]
    rows += ["not-a-time, 1901.0, 415.5"]
    path = _write(tmp_path, _build(rows=rows))

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        frame = read_icartt(path, LOADER).frame

    assert len(frame) == 10
    assert frame.index[0] == pd.Timestamp("2024-08-15") + pd.Timedelta(seconds=42480)
    assert "Reading it as seconds past midnight (majority)" in caplog.text


def test_unmixed_time_columns_do_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The two unambiguous cases short-circuit before the vote."""
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        read_icartt(_write(tmp_path, _build(), name="a.ict"), LOADER)
        read_icartt(
            _write(
                tmp_path,
                _build(rows=["2024-08-15 19:00:00, 1.0, 2.0", "2024-08-15 19:00:01, 1.0, 2.0"]),
                name="b.ict",
            ),
            LOADER,
        )
    assert "is mixed" not in caplog.text


def test_column_names_follow_the_data_width_not_nv(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The one hard failure in the archive: NV is wrong, the header line is not.

    A real file declares NV=1 with its independent *and* its single dependent
    variable both named ``Time_UTC``, while its column-header line carries the
    7 names its 7-field rows need. Trusting NV handed pandas a duplicated
    ``names=`` list, which it refuses with an untyped ``ValueError``.
    """
    lines = _build().splitlines()
    # NV=1, one definition, and a column-header line that matches the rows.
    lines[9] = "1"
    lines[10] = "1"
    lines[11] = "-9999"
    lines[12] = "Time_Start, seconds, Duplicated on purpose"
    del lines[13]
    lines[0] = f"{len(lines) - 2}, 1001"
    path = _write(tmp_path, "\n".join(lines) + "\n")

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        frame = read_icartt(path, LOADER).frame

    assert list(frame.columns) == ["Time_Start", "CH4_ppb", "CO2_ppm"]
    assert len(frame) == 2
    assert "using the variable definitions" not in caplog.text


def test_duplicate_column_names_are_mangled_pandas_style(tmp_path: Path) -> None:
    """A real ground-site file repeats two of its own column names.

    Mangling rather than refusing: TSARA cannot know which ``NO_ppb`` is
    which, but the manifest maps raw names to canonical ones, so a
    distinguishable name is all the reader owes the next stage. Left
    unmangled, pandas refuses the read outright.
    """
    lines = _build().splitlines()
    lines[-3] = "Time_Start, CH4_ppb, CH4_ppb"  # column header repeats a name
    path = _write(tmp_path, "\n".join(lines) + "\n")

    frame = read_icartt(path, LOADER).frame
    assert list(frame.columns) == ["Time_Start", "CH4_ppb", "CH4_ppb.1"]


def test_nlhead_smaller_than_its_variable_definitions_is_corrected(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """NLHEAD can be arithmetically impossible, and then it must not be trusted.

    Two archive files declare NLHEAD=36 with NV=35, but 12 fixed lines plus 35
    definition lines cannot fit in 36. Trusting the number admits header text
    into the data block, which is where the stray numerics of the mixed-time
    test above come from.
    """
    lines = _build().splitlines()
    lines[0] = "13, 1001"  # 12 + NV = 14, so 13 is provably impossible
    path = _write(tmp_path, "\n".join(lines) + "\n")

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        header = parse_icartt_header(path.read_text().splitlines(), path)

    assert header.n_header_lines == 14
    assert "provably wrong" in caplog.text


def test_impossible_nlhead_longer_than_the_file_is_an_error(tmp_path: Path) -> None:
    """Clamping NLHEAD up must not walk off the end of a truncated file."""
    lines = _build().splitlines()[:13]
    lines[0] = "1, 1001"
    path = _write(tmp_path, "\n".join(lines) + "\n")

    with pytest.raises(TsaraIngestError, match="needs at least a 14-line header"):
        parse_icartt_header(path.read_text().splitlines(), path)


def test_near_total_row_loss_is_an_error_not_a_warning(tmp_path: Path) -> None:
    """A read returning a handful of rows is a misparse, not a thin dataset.

    Downstream, "2 rows" and "this instrument barely ran" are indistinguishable,
    and a warning in a thousand-file run scrolls past unread.
    """
    rows = ["42480.0, 1900.0, 415.0"] + ["oops, 1, 2, 3, 4"] * 20
    path = _write(tmp_path, _build(rows=rows))

    with pytest.raises(TsaraIngestError, match="exceeds max_dropped_fraction"):
        read_icartt(path, LOADER)

    # ...and the threshold is the manifest's call, not a hard-coded rule.
    tolerant = ICARTTLoader(path_template="*.ict", max_dropped_fraction=1.0)
    assert len(read_icartt(path, tolerant).frame) == 1


def test_repeated_basenames_after_revision_selection_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Date-less names bypass de-duplication, so at least say so.

    39 basenames in the target archive exist in two or three directories at
    once (a dated directory, ``Calibrated Data/``, and ``Calibrated Data
    (Updated)/``). They carry no ``YYYYMMDD`` token, so revision selection
    cannot compare them and keeps them all — correctly, but silently, and a
    recursive template then ingests every copy.
    """
    paths = [
        Path("MIRO/20240809/Miro_Data_0809.ict"),
        Path("MIRO/Calibrated Data/Miro_Data_0809.ict"),
        Path("MIRO/Calibrated Data (Updated)/Miro_Data_0809.ict"),
    ]
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        selected = select_latest_revisions(paths)

    assert len(selected) == 3  # kept, as they must be
    assert "appear in more than one directory" in caplog.text
    assert "Miro_Data_0809.ict" in caplog.text


def test_distinct_basenames_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    paths = [Path("a/Miro_Data_0809.ict"), Path("b/Miro_Data_0810.ict")]
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        select_latest_revisions(paths)
    assert "appear in more than one directory" not in caplog.text


def test_names_matching_neither_the_rows_nor_nv_fall_back_to_definitions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The residual case: the row width settles nothing, so NV decides again.

    No file in the target archive reaches here — every one of the 1055 matches
    on at least one list — but the branch is what keeps a width disagreement
    from being fatal. Uniformly-too-wide rows land here and are handled by
    ``index_col=False`` truncating the surplus, which is the behaviour the
    reader has always promised.
    """
    lines = _build(rows=["42480.0, 1900.0, 415.0, 99, 98"]).splitlines()
    lines[-2] = "Time_Start, CH4_ppb"  # 2 declared names vs 3 variables vs 5 fields
    path = _write(tmp_path, "\n".join(lines) + "\n")

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
        with pytest.warns(pd.errors.ParserWarning):
            frame = read_icartt(path, LOADER).frame

    assert list(frame.columns) == ["Time_Start", "CH4_ppb", "CO2_ppm"]
    assert "matching neither" in caplog.text


# ---------------------------------------------------------------------------
# Limit-of-detection sentinels
# ---------------------------------------------------------------------------
#
# These matter more than their line count suggests. Measured over the 2024
# archive, 10.03% of every numeric value is an LOD sentinel, and in the
# PTR-MS VOC files it reaches 67-70% of samples -- far enough past half that
# the *median* of a real benzene record was the sentinel rather than a
# concentration, so a rolling low-quantile baseline would have called
# -88888 ppbv the background (docs/METHODS.md 9.2.1).


def test_llod_sentinel_is_masked(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _build(
            special=["SPECIAL COMMENTS:", "LLOD_FLAG: -8888"],
            rows=["42480.0, 1900.0, 415.0", "42481.0, -8888, 415.5"],
        ),
    )
    table = read_icartt(path, LOADER)
    assert np.isnan(table.frame["CH4_ppb"].to_numpy()[1])
    assert table.attrs["icartt_lod_masked"] == {"CH4_ppb": 1}


def test_ulod_sentinel_is_masked_too(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _build(
            special=["SPECIAL COMMENTS:", "ULOD_FLAG: -7777"],
            rows=["42480.0, -7777, 415.0", "42481.0, 1901.0, 415.5"],
        ),
    )
    assert np.isnan(read_icartt(path, LOADER).frame["CH4_ppb"].to_numpy()[0])


def test_lod_sentinel_is_masked_before_scaling(tmp_path: Path) -> None:
    """Scaling first would turn -8888 into a plausible-looking number, which
    is exactly how a sentinel becomes silent data."""
    path = _write(
        tmp_path,
        _build(
            scales=[1000.0, 1.0],
            special=["SPECIAL COMMENTS:", "LLOD_FLAG: -8888"],
            rows=["42480.0, -8888, 415.0"],
        ),
    )
    assert np.isnan(read_icartt(path, LOADER).frame["CH4_ppb"].to_numpy()[0])


def test_non_numeric_lod_flags_are_ignored(tmp_path: Path) -> None:
    """'N/A' and 'NaN' both mean the flag is unused."""
    path = _write(
        tmp_path,
        _build(special=["SPECIAL COMMENTS:", "LLOD_FLAG: N/A", "ULOD_FLAG: NaN"]),
    )
    table = read_icartt(path, LOADER)
    assert "icartt_lod_masked" not in table.attrs
    assert not table.frame["CH4_ppb"].isna().any()


def test_per_variable_lod_flags_are_matched_by_position(tmp_path: Path) -> None:
    """A list as long as NV names one sentinel per variable."""
    path = _write(
        tmp_path,
        _build(
            special=["SPECIAL COMMENTS:", "LLOD_FLAG: -8888, -7777"],
            rows=["42480.0, -8888, -7777", "42481.0, -7777, -8888"],
        ),
    )
    frame = read_icartt(path, LOADER).frame
    # Row 0 carries each variable's *own* sentinel; row 1 has them swapped,
    # so a positional match masks exactly the diagonal.
    assert np.isnan(frame["CH4_ppb"].to_numpy()[0])
    assert np.isnan(frame["CO2_ppm"].to_numpy()[0])
    assert frame["CH4_ppb"].to_numpy()[1] == -7777.0
    assert frame["CO2_ppm"].to_numpy()[1] == -8888.0


def test_a_single_lod_flag_applies_to_every_variable(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _build(
            special=["SPECIAL COMMENTS:", "LLOD_FLAG: -8888"],
            rows=["42480.0, -8888, -8888"],
        ),
    )
    frame = read_icartt(path, LOADER).frame
    assert np.isnan(frame["CH4_ppb"].to_numpy()[0])
    assert np.isnan(frame["CO2_ppm"].to_numpy()[0])


def test_lod_masking_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A species mostly below detection is a fact worth knowing before a
    ratio is fitted to it."""
    path = _write(
        tmp_path,
        _build(
            special=["SPECIAL COMMENTS:", "LLOD_FLAG: -8888"],
            rows=["42480.0, -8888, 415.0"],
        ),
    )
    with caplog.at_level(logging.INFO, logger="tsara.ingest.icartt"):
        read_icartt(path, LOADER)
    assert "out-of-detection-range" in caplog.text
    assert "CH4_ppb:1" in caplog.text


def test_metadata_is_scraped_from_the_whole_header(tmp_path: Path) -> None:
    """The comment blocks are located by 12 + NV arithmetic that a malformed
    variable count invalidates. Measured: the 43 PTR-MS files in the archive
    carry an extra blank-named definition line, which offset the walk and
    left the metadata EMPTY -- on exactly the files declaring the LOD flags
    with the highest below-detection fractions in the archive."""
    text = _build(special=["SPECIAL COMMENTS:", "LLOD_FLAG: -8888", "REVISION: R2"])
    header = parse_icartt_header(text.splitlines(), Path("f.ict"))
    assert header.metadata["LLOD_FLAG"] == "-8888"
    assert header.metadata["REVISION"] == "R2"
