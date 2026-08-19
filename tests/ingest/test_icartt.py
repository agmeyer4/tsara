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


def test_comment_block_mismatch_warns_but_trusts_nlhead(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lines = _build().splitlines()
    lines[14] = "0"  # understate the special-comment count
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.icartt"):
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
    assert "Skipped 1 of 3 malformed data row(s)" in caplog.text


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
