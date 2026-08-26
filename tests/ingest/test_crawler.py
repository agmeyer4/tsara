"""Tests for the path-template crawler.

The layouts exercised here are the ones the target archive actually uses: a
flat directory of ICARTT files, one directory per measurement day, an
institution/instrument hierarchy, and a per-instrument tree where one
instrument nests its files a level deeper than its siblings. Each is built
as empty files under ``tmp_path`` — the crawler never opens anything, so
content is irrelevant and the tests stay fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from tsara.config.manifest import CSVLoader, ICARTTLoader, ParquetLoader
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.crawler import FileMatch, compile_template, crawl


def _touch(root: Path, *relative: str) -> None:
    """Create empty files at the given relative paths."""
    for item in relative:
        path = root / item
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _icartt(**kwargs: Any) -> ICARTTLoader:
    return ICARTTLoader(**kwargs)


def _parquet(**kwargs: Any) -> ParquetLoader:
    return ParquetLoader(**kwargs)


def _csv(**kwargs: Any) -> CSVLoader:
    """Build a CSVLoader with a time block, which the crawler never consults."""
    kwargs.setdefault("time", {"column": "t", "format": "iso8601"})
    return CSVLoader(**kwargs)


def _names(matches: list[FileMatch]) -> list[str]:
    return [match.path.name for match in matches]


# ---------------------------------------------------------------------------
# compile_template
# ---------------------------------------------------------------------------


def test_literal_template() -> None:
    pattern, regex = compile_template("data/file.ict")
    assert pattern == "data/file.ict"
    assert regex.fullmatch("data/file.ict")


def test_literal_dots_are_escaped_in_the_regex() -> None:
    """'.' is a wildcard to re and a literal to glob; they must not diverge."""
    _, regex = compile_template("a.b/f.ict")
    assert regex.fullmatch("a.b/f.ict")
    assert regex.fullmatch("axb/fxict") is None


def test_field_becomes_a_wildcard_and_a_capture() -> None:
    pattern, regex = compile_template("{institution}/*.ict")
    assert pattern == "*/*.ict"
    matched = regex.fullmatch("Univ_Montana/x.ict")
    assert matched is not None
    assert matched.group("institution") == "Univ_Montana"


def test_field_matches_only_one_segment() -> None:
    _, regex = compile_template("{institution}/*.ict")
    assert regex.fullmatch("a/b/x.ict") is None


def test_fixed_metadata_becomes_a_literal() -> None:
    """A fixed value narrows the filesystem scan instead of being harvested."""
    pattern, regex = compile_template("{institution}/*.ict", {"institution": "uutah"})
    assert pattern == "uutah/*.ict"
    assert regex.fullmatch("uutah/x.ict")
    assert regex.fullmatch("other/x.ict") is None


def test_repeated_field_becomes_a_back_reference() -> None:
    """The same placeholder twice means the same value twice, not an error."""
    _, regex = compile_template("{site}/{site}/*.ict")
    assert regex.fullmatch("a/a/x.ict")
    assert regex.fullmatch("a/b/x.ict") is None


@pytest.mark.parametrize(
    ("token", "digits"),
    [("%Y", 4), ("%y", 2), ("%m", 2), ("%d", 2), ("%H", 2), ("%M", 2), ("%S", 2), ("%j", 3)],
)
def test_strftime_tokens_match_fixed_width_digits(token: str, digits: int) -> None:
    pattern, regex = compile_template(f"{token}/x.ict")
    assert pattern == "[0-9]" * digits + "/x.ict"
    assert regex.fullmatch("1" * digits + "/x.ict")
    assert regex.fullmatch("1" * (digits + 1) + "/x.ict") is None


def test_strftime_tokens_are_not_harvested() -> None:
    """A date in a directory is structure, not campaign metadata."""
    _, regex = compile_template("%Y%m%d/x.ict")
    matched = regex.fullmatch("20240815/x.ict")
    assert matched is not None
    assert matched.groupdict() == {}


def test_double_percent_is_a_literal() -> None:
    pattern, regex = compile_template("100%%/x.ict")
    assert pattern == "100%/x.ict"
    assert regex.fullmatch("100%/x.ict")


def test_single_star_stays_within_a_segment() -> None:
    _, regex = compile_template("a/*.ict")
    assert regex.fullmatch("a/x.ict")
    assert regex.fullmatch("a/b/x.ict") is None


def test_double_star_spans_segments() -> None:
    pattern, regex = compile_template("a/**/x.ict")
    assert pattern == "a/**/x.ict"
    assert regex.fullmatch("a/b/c/x.ict")


def test_question_mark_matches_one_character() -> None:
    _, regex = compile_template("f?.ict")
    assert regex.fullmatch("fa.ict")
    assert regex.fullmatch("fab.ict") is None


def test_character_class_passes_through() -> None:
    pattern, regex = compile_template("f[ab].ict")
    assert pattern == "f[ab].ict"
    assert regex.fullmatch("fa.ict")
    assert regex.fullmatch("fc.ict") is None


def test_unterminated_brace_is_an_error() -> None:
    with pytest.raises(TsaraIngestError, match="Unterminated '{'"):
        compile_template("{institution/*.ict")


def test_unterminated_bracket_is_an_error() -> None:
    with pytest.raises(TsaraIngestError, match=r"Unterminated '\['"):
        compile_template("f[ab.ict")


def test_unsupported_strftime_token_is_an_error() -> None:
    """Treating '%q' as a literal would match nothing, with no clue why."""
    with pytest.raises(TsaraIngestError, match="Unsupported strftime token '%q'"):
        compile_template("%q/x.ict")


# ---------------------------------------------------------------------------
# crawl: the real layouts
# ---------------------------------------------------------------------------


def test_flat_directory(tmp_path: Path) -> None:
    _touch(tmp_path, "NOAA_ARC/a.ict", "NOAA_ARC/b.ict", "elsewhere/c.ict")
    found = crawl(tmp_path, _icartt(path_template="NOAA_ARC/*.ict"))
    assert _names(found) == ["a.ict", "b.ict"]


def test_one_directory_per_day(tmp_path: Path) -> None:
    _touch(tmp_path, "drives/20240815/a.ict", "drives/20240816/b.ict", "drives/notadate/c.ict")
    found = crawl(tmp_path, _icartt(path_template="drives/%Y%m%d/*.ict"))
    assert _names(found) == ["a.ict", "b.ict"]


def test_institution_and_instrument_are_harvested(tmp_path: Path) -> None:
    _touch(tmp_path, "Univ_Montana/PTR_ToF_MS/a.ict", "CSU/Picarro/b.ict")
    found = crawl(tmp_path, _icartt(path_template="{institution}/{instrument}/*.ict"))

    assert len(found) == 2
    by_name = {match.path.name: match.fields for match in found}
    assert by_name["a.ict"] == {"institution": "Univ_Montana", "instrument": "PTR_ToF_MS"}
    assert by_name["b.ict"] == {"institution": "CSU", "instrument": "Picarro"}


def test_fixed_metadata_filters_and_is_recorded(tmp_path: Path) -> None:
    _touch(tmp_path, "Univ_Montana/PTR/a.ict", "CSU/Picarro/b.ict")
    found = crawl(
        tmp_path,
        _icartt(path_template="{institution}/{instrument}/*.ict"),
        {"institution": "Univ_Montana"},
    )
    assert _names(found) == ["a.ict"]
    assert found[0].fields == {"institution": "Univ_Montana", "instrument": "PTR"}


def test_metadata_beyond_template_fields_is_still_attached(tmp_path: Path) -> None:
    """Instrument metadata describes the file even when unused for matching."""
    _touch(tmp_path, "x/a.parquet")
    found = crawl(tmp_path, _parquet(path_template="x/*.parquet"), {"campaign": "slv2026"})
    assert found[0].fields == {"campaign": "slv2026"}


def test_nested_subdirectory_layout(tmp_path: Path) -> None:
    """One instrument nests a level deeper than its siblings; both are describable."""
    _touch(tmp_path, "LANL_pico/Eng/a.parquet", "WYO_picarro/b.parquet")
    found = crawl(tmp_path, _parquet(path_template="{instrument}/Eng/*.parquet"))
    assert _names(found) == ["a.parquet"]
    assert found[0].fields == {"instrument": "LANL_pico"}


def test_double_star_reaches_varying_depths(tmp_path: Path) -> None:
    """'**' spans zero or more directories, so the shallowest file counts too."""
    _touch(tmp_path, "a/x.ict", "a/b/y.ict", "a/b/c/z.ict")
    found = crawl(tmp_path, _icartt(path_template="a/**/*.ict"))
    assert set(_names(found)) == {"x.ict", "y.ict", "z.ict"}


def test_double_star_regex_admits_the_zero_directory_case() -> None:
    """Compiling '**' and '/' separately yields '.*/', which needs a separator."""
    _, regex = compile_template("a/**/*.ict")
    assert regex.fullmatch("a/x.ict")
    assert regex.fullmatch("a/b/c/z.ict")


def test_directories_are_not_returned(tmp_path: Path) -> None:
    """A directory named like a data file would otherwise be handed to a reader."""
    (tmp_path / "d" / "looks_like.ict").mkdir(parents=True)
    _touch(tmp_path, "d/real.ict")
    assert _names(crawl(tmp_path, _icartt(path_template="d/*.ict"))) == ["real.ict"]


def test_results_are_sorted(tmp_path: Path) -> None:
    """Filesystem order is arbitrary; a run must not depend on it."""
    _touch(tmp_path, "d/c.ict", "d/a.ict", "d/b.ict")
    assert _names(crawl(tmp_path, _icartt(path_template="d/*.ict"))) == ["a.ict", "b.ict", "c.ict"]


def test_regex_rejects_what_the_glob_admits(tmp_path: Path) -> None:
    """A glob '*' matches the empty string; a metadata field must not."""
    _touch(tmp_path, "x.ict", "xA.ict")
    found = crawl(tmp_path, _icartt(path_template="x{tag}.ict"))
    assert _names(found) == ["xA.ict"]
    assert found[0].fields == {"tag": "A"}


# ---------------------------------------------------------------------------
# crawl: several templates
# ---------------------------------------------------------------------------


def test_templates_are_merged(tmp_path: Path) -> None:
    _touch(tmp_path, "one/a.ict", "two/b.ict")
    found = crawl(tmp_path, _icartt(path_templates=["one/*.ict", "two/*.ict"]))
    assert _names(found) == ["a.ict", "b.ict"]


def test_a_file_matched_twice_is_ingested_once(tmp_path: Path) -> None:
    """Overlapping layouts are real; double-ingesting the same air is not ok."""
    _touch(tmp_path, "one/a.ict")
    found = crawl(tmp_path, _icartt(path_templates=["one/*.ict", "**/*.ict"]))
    assert len(found) == 1
    assert found[0].template == "one/*.ict"


def test_disagreeing_templates_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """First declared wins silently otherwise, and may be the wrong one."""
    _touch(tmp_path, "site_a/inst/x.ict")
    loader = _icartt(path_templates=["{site}/inst/*.ict", "site_a/{site}/*.ict"])
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.crawler"):
        found = crawl(tmp_path, loader)
    assert found[0].fields == {"site": "site_a"}
    assert "disagree on" in caplog.text


def test_agreeing_templates_do_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _touch(tmp_path, "one/a.ict")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.crawler"):
        crawl(tmp_path, _icartt(path_templates=["one/*.ict", "one/a*.ict"]))
    assert "disagree on" not in caplog.text


def test_a_template_matching_nothing_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Silence here is how a typo'd layout goes unnoticed for a whole run."""
    _touch(tmp_path, "one/a.ict")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.crawler"):
        crawl(tmp_path, _icartt(path_templates=["one/*.ict", "missing/*.ict"]))
    assert "matched no files" in caplog.text


# ---------------------------------------------------------------------------
# crawl: format-specific selection and failures
# ---------------------------------------------------------------------------


def test_icartt_revision_policy_latest_is_applied(tmp_path: Path) -> None:
    _touch(tmp_path, "d/X_Y_20240815_R0.ict", "d/X_Y_20240815_R1.ict")
    found = crawl(tmp_path, _icartt(path_template="d/*.ict"))
    assert _names(found) == ["X_Y_20240815_R1.ict"]


def test_icartt_revision_policy_all_keeps_everything(tmp_path: Path) -> None:
    _touch(tmp_path, "d/X_Y_20240815_R0.ict", "d/X_Y_20240815_R1.ict")
    found = crawl(tmp_path, _icartt(path_template="d/*.ict", revision_policy="all"))
    assert len(found) == 2


def test_revision_policy_keeps_distinct_comments(tmp_path: Path) -> None:
    """The drives-collapsing bug, guarded at the crawler boundary too."""
    _touch(
        tmp_path,
        "d/X_Y_20240802_RA_Drive02.ict",
        "d/X_Y_20240802_RA_Drive03.ict",
        "d/X_Y_20240802_RA_Stationary01.ict",
    )
    assert len(crawl(tmp_path, _icartt(path_template="d/*.ict"))) == 3


def test_non_icartt_loaders_are_untouched_by_revision_logic(tmp_path: Path) -> None:
    """A parquet file named like an ICARTT revision must not be dropped."""
    _touch(tmp_path, "d/X_Y_20240815_R0.parquet", "d/X_Y_20240815_R1.parquet")
    assert len(crawl(tmp_path, _parquet(path_template="d/*.parquet"))) == 2


def test_missing_base_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="not an existing directory"):
        crawl(tmp_path / "nope", _icartt(path_template="*.ict"))


def test_base_path_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    _touch(tmp_path, "afile")
    with pytest.raises(TsaraIngestError, match="not an existing directory"):
        crawl(tmp_path / "afile", _icartt(path_template="*.ict"))


def test_no_matches_at_all_is_an_error(tmp_path: Path) -> None:
    """The most common manifest mistake deserves the most helpful message."""
    _touch(tmp_path, "d/a.txt")
    with pytest.raises(TsaraIngestError, match=r"No files found.*\['d/\*\.ict'\]"):
        crawl(tmp_path, _icartt(path_template="d/*.ict"))


def test_crawl_accepts_a_string_base_path(tmp_path: Path) -> None:
    _touch(tmp_path, "d/a.ict")
    assert len(crawl(str(tmp_path), _icartt(path_template="d/*.ict"))) == 1


def test_paths_returned_are_absolute(tmp_path: Path) -> None:
    """Readers are handed these directly, from whatever the cwd happens to be."""
    _touch(tmp_path, "d/a.ict")
    assert crawl(tmp_path, _icartt(path_template="d/*.ict"))[0].path.is_absolute()


def test_csv_loader_crawls_too(tmp_path: Path) -> None:
    """Nothing about the crawler is format-specific except revision policy."""
    _touch(tmp_path, "wyo/Picarro/20240801/a.dat")
    loader = CSVLoader.model_validate(
        {
            "path_template": "wyo/{instrument}/%Y%m%d/*.dat",
            "delimiter": r"\s+",
            "time": {"column": "EPOCH_TIME", "format": "unix"},
        }
    )
    found = crawl(tmp_path, loader)
    assert found[0].fields == {"instrument": "Picarro"}


def test_trailing_double_star_matches_everything_below(tmp_path: Path) -> None:
    """'**' with no separator after it is still a valid, if blunt, template."""
    pattern, regex = compile_template("a/**")
    # '**/*' not '**': through Python 3.12 a glob ending in '**' yields only
    # directories, so the bare form would silently match no files at all.
    assert pattern == "a/**/*"
    assert regex.fullmatch("a/x.ict")
    assert regex.fullmatch("a/b/c/z.ict")

    _touch(tmp_path, "a/x.ict", "a/b/c/z.ict", "other/y.ict")
    found = crawl(tmp_path, _icartt(path_template="a/**"))
    assert set(_names(found)) == {"x.ict", "z.ict"}


@pytest.mark.parametrize("template", ["a**/b.ict", "a**"])
def test_double_star_must_be_a_whole_component(template: str) -> None:
    """glob only treats '**' as recursive when it stands alone."""
    with pytest.raises(TsaraIngestError, match="whole path component"):
        compile_template(template)


# ---------------------------------------------------------------------------
# Walkthrough stage 5: exclusion, resource forks, and negated classes
# ---------------------------------------------------------------------------


def test_exclude_removes_quarantined_subdirectories(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The layout that motivated `exclude`, in miniature.

    The target archive's instrument-aligned stage keeps rejected files in
    ``bad/`` and ``bad_timestamp/`` directories sitting directly beneath the
    good ones — 187 of its 608 files. Templates are otherwise include-only,
    so the '**' pattern recommended for varying depth swept every one of
    them in silently.
    """
    _touch(
        tmp_path,
        "aeris/Eng/good1.parquet",
        "aeris/Eng/good2.parquet",
        "aeris/Eng/bad/reject1.parquet",
        "aeris/Raw/bad_timestamp/reject2.parquet",
    )
    swept = crawl(tmp_path, _parquet(path_template="**/*.parquet"))
    assert len(swept) == 4  # the problem

    with caplog.at_level(logging.INFO, logger="tsara.ingest.crawler"):
        kept = crawl(
            tmp_path,
            _parquet(path_template="**/*.parquet", exclude=["**/bad/**", "**/bad_timestamp/**"]),
        )
    assert set(_names(kept)) == {"good1.parquet", "good2.parquet"}
    assert "Excluded 2 matched file(s)" in caplog.text


def test_exclude_accepts_a_bare_string(tmp_path: Path) -> None:
    """Same shorthand `path_template` already allows."""
    _touch(tmp_path, "a/keep.ict", "a/skip/drop.ict")
    found = crawl(tmp_path, _icartt(path_template="**/*.ict", exclude="**/skip/**"))
    assert _names(found) == ["keep.ict"]


def test_exclude_rejects_an_empty_pattern() -> None:
    with pytest.raises(ValueError, match="non-empty path patterns"):
        ICARTTLoader(path_template="*.ict", exclude=["  "])


def test_excluding_everything_is_an_error_that_says_so(tmp_path: Path) -> None:
    """Distinct from 'no files found': the templates worked, the filter did not.

    Reported separately because the two have opposite fixes — widen the
    template, or narrow the exclusion.
    """
    _touch(tmp_path, "a/bad/x.ict", "a/bad/y.ict")
    with pytest.raises(TsaraIngestError, match="exclusions are too broad"):
        crawl(tmp_path, _icartt(path_template="**/*.ict", exclude=["**/bad/**"]))


def test_a_template_matching_only_excluded_files_still_reports_matches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """'Matched no files' must keep meaning 'this layout is absent'.

    A template whose every hit is excluded found the layout perfectly well;
    saying it matched nothing would send the reader after the wrong bug.
    """
    _touch(tmp_path, "a/keep.ict", "b/bad/drop.ict")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.crawler"):
        crawl(
            tmp_path,
            _icartt(path_templates=["a/*.ict", "b/**/*.ict"], exclude=["**/bad/**"]),
        )
    assert "matched no files" not in caplog.text


def test_appledouble_resource_forks_are_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """macOS metadata is not data, and pathlib.glob hands it over anyway.

    `pathlib.Path.glob` matches dotfiles where `glob.glob` does not, so a
    '*.csv' template over a Mac-touched archive picks up a binary '._*.csv'
    shadow for every real file. 18 sit in the target archive's aerosol
    directories, each one an unreadable "data file" in the run log.
    """
    _touch(tmp_path, "grimm/2024-02-22.csv", "grimm/._2024-02-22.csv")
    loader = _csv(path_template="grimm/*.csv")

    with caplog.at_level(logging.DEBUG, logger="tsara.ingest.crawler"):
        found = crawl(tmp_path, loader)

    assert _names(found) == ["2024-02-22.csv"]
    assert "Skipped 1 AppleDouble resource fork" in caplog.text


def test_ordinary_dotfiles_are_not_skipped(tmp_path: Path) -> None:
    """Only '._' is unambiguously a resource fork.

    A leading dot on its own just means hidden, and a template pointed at one
    deliberately should still find it.
    """
    _touch(tmp_path, "d/.hidden.ict")
    found = crawl(tmp_path, _icartt(path_template="d/*.ict"))
    assert _names(found) == [".hidden.ict"]


def test_glob_negated_character_class_reaches_the_regex(tmp_path: Path) -> None:
    """`[!x]` is glob negation; the harvesting regex must agree.

    Passed through untouched it read as a literal '!' or 'x', so the glob
    matched the file and the regex then threw it away — a correct template
    reporting "no files found".
    """
    pattern, regex = compile_template("[!x]*.ict")
    assert pattern == "[!x]*.ict"
    assert regex.pattern == r"[^x][^/]*\.ict"

    _touch(tmp_path, "a1.ict", "x1.ict")
    found = crawl(tmp_path, _icartt(path_template="[!x]*.ict"))
    assert _names(found) == ["a1.ict"]


def test_regex_spelling_of_negation_is_refused() -> None:
    """'[^x]' negates in re but is a literal class in glob, so it cannot work."""
    with pytest.raises(TsaraIngestError, match=r"write '\[!x\]'"):
        compile_template("[^x]*.ict")


def test_ordinary_character_classes_still_pass_through() -> None:
    pattern, regex = compile_template("[0-9]*.ict")
    assert pattern == "[0-9]*.ict"
    assert regex.pattern == r"[0-9][^/]*\.ict"
