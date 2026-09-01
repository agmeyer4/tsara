"""Finding an instrument's files, and reading metadata out of where they sit.

The problem this solves
-----------------------
Campaign archives encode real information in their directory layout. A path
like ``SLC-SOS/2024_mobile/Univ_Montana/PTR_ToF_MS/...`` records which
institution operated the instrument and which instrument it was, and nothing
inside the files says so. Meanwhile the *same* campaign stores another
instrument as ``NOAA_MobileLab_Drives/20240815/*.ict``, with the measurement
date in a directory name, and a third as
``LANL_aerispico017/Eng/*.parquet``, nested one level deeper than its
siblings.

TSARA's position (CLAUDE.md §5) is that **directory and naming conventions
are data, not code**. A path template describes one layout; a loader carries
a list of them, because one instrument's files routinely span several
conventions at once — a mid-campaign reorganization, an archive mirror
alongside a local copy. Supporting a new layout is then a YAML edit.

How a template is interpreted
-----------------------------
A template is matched against paths *relative to* ``Manifest.base_path``,
and mixes three kinds of token:

``{field}``
    A metadata placeholder. Matches one path segment and **harvests** it, so
    ``{institution}/{instrument}/*.ict`` turns a matched path into
    ``{"institution": "Univ_Montana", "instrument": "PTR_ToF_MS"}``. If the
    instrument's ``metadata:`` block fixes a value for that field, the value
    is substituted into the search instead — which both narrows the scan and
    filters out everything else.
``%Y %m %d`` and friends
    strftime tokens, which constrain matching to digits of the right width.
    They are **not** harvested: ``_BaseLoader.template_fields`` defines a
    template's metadata as its ``{field}`` placeholders, and a date embedded
    in a directory name is structure rather than campaign metadata.
``*``, ``**``, ``?``, ``[abc]``
    Ordinary glob wildcards. ``*`` stays within one path segment; ``**``
    spans segments, which is the escape hatch for archives whose depth
    varies. Negated classes use the glob spelling ``[!abc]``, not the regex
    ``[^abc]``.

Why there is also an exclude list
---------------------------------
Templates are include-only, and that is not enough for archives that
**quarantine data in place**. The target archive's instrument-aligned stage
keeps rejected files in ``bad/`` and ``bad_timestamp/`` subdirectories
sitting directly beneath the good ones — 187 of its 608 files. A ``**``
template, which is exactly what the varying-depth advice above recommends,
sweeps all of them in without a word. Careful per-directory templates avoid
it, since ``Eng/*.parquet`` does not cross into ``Eng/bad/``, but only if
you already know the quarantine is there.

``_BaseLoader.exclude`` takes patterns in the same syntax and removes what
they match, reporting the count. Saying "not that" is a layout fact, so it
belongs in the manifest alongside the layout facts that find files in the
first place.

Why both a glob and a regex
---------------------------
Each template compiles to a *pair*. The glob drives the filesystem walk,
because only the filesystem can enumerate what exists; the regex then runs
against each matched path to harvest ``{field}`` values, because a glob
cannot report *which* text a wildcard consumed. The regex is also stricter
than the glob (a glob ``*`` happily crosses nothing, but ``[^/]+`` demands a
non-empty segment), so it doubles as a second filter — matches that survive
the glob but fail the regex are discarded rather than mis-harvested.
"""

from __future__ import annotations

import glob as globlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tsara.config.manifest import ICARTTLoader
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.icartt import select_latest_revisions

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from tsara.config.manifest import LoaderConfig

logger = logging.getLogger(__name__)

__all__ = ["FileMatch", "compile_template", "crawl"]

#: strftime tokens the crawler understands, and how many digits each matches.
#: Deliberately a closed set: silently treating an unknown ``%q`` as a literal
#: would make a template match nothing, with no indication why.
_STRFTIME_WIDTHS = {
    "%Y": 4,
    "%y": 2,
    "%m": 2,
    "%d": 2,
    "%H": 2,
    "%M": 2,
    "%S": 2,
    "%j": 3,
}


@dataclass(frozen=True)
class FileMatch:
    """One file the crawler found, with what its location says about it.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to the file.
    fields : dict
        Metadata for this file: values harvested from ``{field}``
        placeholders, plus any fixed values the instrument's ``metadata:``
        block supplied. This is the "what did its location say about it?"
        half of a file's provenance; the other half — what the file said
        about itself — arrives later as :attr:`~tsara.ingest.base.RawTable.attrs`.
    template : str
        The path template that matched, kept so that an archive spanning
        several layouts can be debugged when one of them misbehaves.
    """

    path: Path
    fields: dict[str, str] = field(default_factory=dict)
    template: str = ""


def compile_template(
    template: str, metadata: Mapping[str, str] | None = None
) -> tuple[str, re.Pattern[str]]:
    """Compile one path template into a glob pattern and a harvesting regex.

    Exposed publicly because it is independently useful and independently
    testable: given a template, you can check what it will search for and
    what it will extract without touching a filesystem.

    Parameters
    ----------
    template : str
        Path pattern relative to ``Manifest.base_path``.
    metadata : Mapping, optional
        Fixed values for ``{field}`` placeholders. A field named here is
        substituted as a literal rather than harvested.

    Returns
    -------
    tuple
        ``(glob_pattern, regex)``.

    Raises
    ------
    TsaraIngestError
        If the template contains an unterminated ``{``/``[`` or an
        unsupported ``%`` token.
    """
    fixed = dict(metadata or {})
    glob_parts: list[str] = []
    regex_parts: list[str] = []
    # Tracks fields already captured, so a template repeating a placeholder
    # becomes a back-reference ("the same value in both places") instead of a
    # duplicate group name, which is a hard regex error.
    captured: set[str] = set()

    index = 0
    length = len(template)
    while index < length:
        char = template[index]

        if char == "{":
            close = template.find("}", index)
            if close == -1:
                raise TsaraIngestError(f"Unterminated '{{' in path template '{template}'.")
            name = template[index + 1 : close]
            if name in fixed:
                value = fixed[name]
                glob_parts.append(globlib.escape(value))
                regex_parts.append(re.escape(value))
            elif name in captured:
                regex_parts.append(f"(?P={name})")
                glob_parts.append("*")
            else:
                captured.add(name)
                # One segment, non-empty: a metadata field that matched the
                # empty string would silently produce blank provenance.
                regex_parts.append(f"(?P<{name}>[^/]+)")
                glob_parts.append("*")
            index = close + 1

        elif char == "%":
            token = template[index : index + 2]
            if token == "%%":
                glob_parts.append("%")
                regex_parts.append("%")
            elif token in _STRFTIME_WIDTHS:
                width = _STRFTIME_WIDTHS[token]
                glob_parts.append("[0-9]" * width)
                regex_parts.append(rf"\d{{{width}}}")
            else:
                raise TsaraIngestError(
                    f"Unsupported strftime token '{token}' in path template "
                    f"'{template}'. Supported: {sorted(_STRFTIME_WIDTHS)} (and '%%')."
                )
            index += 2

        elif char == "*":
            if template[index : index + 3] == "**/":
                # Consumed together with its separator, because '**' spans
                # ZERO or more directories: 'a/**/*.ict' must find 'a/x.ict'
                # as well as 'a/b/c/z.ict'. Compiling '**' and '/' separately
                # would emit '.*/', which demands at least one separator and
                # silently skips every file sitting directly in 'a'.
                if index != 0 and template[index - 1] != "/":
                    raise TsaraIngestError(
                        f"'**' must be a whole path component in path template "
                        f"'{template}' (write 'a/**/b', not 'a**/b')."
                    )
                glob_parts.append("**/")
                regex_parts.append("(?:[^/]+/)*")
                index += 3
            elif template[index : index + 2] == "**":
                # A trailing '**'. Emitted as '**/*' rather than '**' because
                # pathlib disagrees with itself across the versions TSARA
                # supports: through Python 3.12 a pattern ending in '**'
                # yields directories ONLY, so it would match nothing once
                # non-files are filtered out, while 3.13+ also yields files.
                # '**/*' means "every file at any depth" on both.
                if index != 0 and template[index - 1] != "/":
                    raise TsaraIngestError(
                        f"'**' must be a whole path component in path template "
                        f"'{template}' (write 'a/**/b', not 'a**b')."
                    )
                glob_parts.append("**/*")
                regex_parts.append(".*")
                index += 2
            else:
                glob_parts.append("*")
                regex_parts.append("[^/]*")
                index += 1

        elif char == "?":
            glob_parts.append("?")
            regex_parts.append("[^/]")
            index += 1

        elif char == "[":
            close = template.find("]", index)
            if close == -1:
                raise TsaraIngestError(f"Unterminated '[' in path template '{template}'.")
            group = template[index : close + 1]
            # A character class means ALMOST the same thing to glob and to re.
            # The exception is negation, and it is silent: glob spells it
            # '[!abc]' while re spells it '[^abc]'. Passing the class through
            # untouched made '[!x]*.csv' glob-match 'a1.csv' and then be
            # rejected by its own harvesting regex, which reads '[!x]' as a
            # literal '!' or 'x' -- so the template found nothing and the
            # crawl reported "no files found" for a pattern that was correct.
            if group.startswith("[!"):
                regex_parts.append("[^" + group[2:])
            elif group.startswith("[^"):
                # The reverse divergence, refused rather than translated:
                # '[^abc]' is negation to re but a literal class containing
                # '^' to glob, so the two halves would disagree whichever way
                # it were interpreted. Naming the supported spelling is more
                # use than silently picking one.
                raise TsaraIngestError(
                    f"Character class '{group}' in path template '{template}' uses the "
                    "regex spelling of negation. Path templates are glob patterns: "
                    f"write '[!{group[2:-1]}]' instead."
                )
            else:
                regex_parts.append(group)
            glob_parts.append(group)
            index = close + 1

        elif char == "/":
            glob_parts.append("/")
            regex_parts.append("/")
            index += 1

        else:
            glob_parts.append(char)
            regex_parts.append(re.escape(char))
            index += 1

    return "".join(glob_parts), re.compile("".join(regex_parts))


def crawl(
    base_path: str | Path,
    loader: LoaderConfig,
    metadata: Mapping[str, str] | None = None,
) -> list[FileMatch]:
    """Find every file an instrument's loader describes.

    Searches all of the loader's templates, merges the results, removes
    duplicates and anything the loader's ``exclude`` patterns match, and
    applies format-specific selection (currently ICARTT's revision policy).

    AppleDouble resource forks (``._*``) are skipped unconditionally: they
    are macOS metadata, not data, and ``pathlib.Path.glob`` matches them
    where ``glob.glob`` would not.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Root of the archive; templates are relative to it. Already resolved
        to an absolute path by the config loader.
    loader : LoaderConfig
        The instrument's loader, supplying ``path_templates`` and any
        format-specific selection rules.
    metadata : Mapping, optional
        Fixed values for ``{field}`` placeholders, from
        ``InstrumentConfig.metadata``.

    Returns
    -------
    list of FileMatch
        Matched files sorted by path, so a run is reproducible regardless of
        the order the filesystem happens to report entries in.

    Raises
    ------
    TsaraIngestError
        If ``base_path`` is not a directory, a template is malformed, no
        template matched anything at all, or every matched file was removed
        by an ``exclude`` pattern.
    """
    base = Path(base_path)
    if not base.is_dir():
        raise TsaraIngestError(f"Manifest base_path '{base}' is not an existing directory.")

    fixed = dict(metadata or {})
    # Compiled once rather than per template: exclusion is a property of the
    # instrument, not of the layout that happened to find the file.
    exclusions = [compile_template(pattern, fixed)[1] for pattern in loader.exclude]
    # Keyed by path so a file described by two templates is ingested once.
    # First template wins, matching the declaration order in the manifest.
    matches: dict[Path, FileMatch] = {}
    n_excluded = 0
    n_resource_forks = 0

    for template in loader.path_templates:
        pattern, regex = compile_template(template, fixed)
        found = 0
        for path in base.glob(pattern):
            if not path.is_file():
                continue
            if path.name.startswith("._"):
                # AppleDouble resource forks. `pathlib.Path.glob` matches
                # dotfiles where `glob.glob` does not, so an archive touched
                # by macOS hands every '*.csv' template a shadow '._*.csv'
                # of binary metadata, which then fails the read and reports
                # itself as an unreadable data file. 18 of these sit in the
                # target archive's aerosol directories.
                #
                # Only this prefix, not every dotfile: '._' is unambiguously
                # a resource fork, whereas a leading dot in general is just
                # a hidden file and could in principle be real data someone
                # deliberately pointed a template at.
                n_resource_forks += 1
                continue
            relative = path.relative_to(base).as_posix()
            matched = regex.fullmatch(relative)
            if matched is None:
                # Survived the glob but not the stricter regex — e.g. a
                # `{field}` that would have to match an empty segment.
                continue
            # Counted before exclusion, so that "this template matched
            # nothing" keeps meaning "this layout is not present" rather
            # than "everything it found was excluded", which is a different
            # problem with a different fix.
            found += 1
            if any(exclusion.fullmatch(relative) for exclusion in exclusions):
                n_excluded += 1
                continue
            if path in matches:
                _warn_if_ambiguous(matches[path], template, matched.groupdict())
                continue
            matches[path] = FileMatch(
                path=path,
                fields={**fixed, **matched.groupdict()},
                template=template,
            )
        logger.debug("Template '%s' matched %d file(s) under %s", template, found, base)
        if found == 0:
            logger.warning("Path template '%s' matched no files under %s.", template, base)

    if n_resource_forks:
        logger.debug(
            "Skipped %d AppleDouble resource fork(s) ('._*') under %s.", n_resource_forks, base
        )
    if n_excluded:
        # Info, not debug: a run that quietly drops a third of an archive
        # should say so at the level people actually read.
        logger.info(
            "Excluded %d matched file(s) under %s via %s.", n_excluded, base, list(loader.exclude)
        )

    if not matches:
        if n_excluded:
            raise TsaraIngestError(
                f"Every one of the {n_excluded} file(s) matched under '{base}' by "
                f"{list(loader.path_templates)} was removed by exclude patterns "
                f"{list(loader.exclude)}. The templates are finding data; the "
                "exclusions are too broad."
            )
        raise TsaraIngestError(
            f"No files found under '{base}' for any of the templates "
            f"{list(loader.path_templates)}. Check base_path, the template "
            "spelling, and any fixed metadata values used to filter it."
        )

    selected = sorted(matches.values(), key=lambda match: match.path)
    return _apply_format_selection(selected, loader)


def _warn_if_ambiguous(existing: FileMatch, template: str, harvested: dict[str, str]) -> None:
    """Warn when two templates match one file but disagree about its metadata.

    Matching the same file twice is harmless on its own — archives really do
    contain overlapping layouts, and the duplicate is dropped. Two templates
    *disagreeing* about what the path means is not harmless: whichever was
    declared first silently wins, and the losing interpretation may be the
    intended one.
    """
    conflicting = {
        key: value
        for key, value in harvested.items()
        if key in existing.fields and existing.fields[key] != value
    }
    if conflicting:
        logger.warning(
            "%s matches both '%s' and '%s', which disagree on %s; keeping the "
            "first template's values.",
            existing.path,
            existing.template,
            template,
            conflicting,
        )


def _apply_format_selection(matches: list[FileMatch], loader: LoaderConfig) -> list[FileMatch]:
    """Apply any format-specific rule about which matched files to keep.

    ICARTT is currently the only format with such a rule, because it is the
    only one whose *filenames* encode revisions of the same data
    (``..._R0.ict`` superseded by ``..._R1.ict``). Ingesting every revision
    would double-count the same air, so ``revision_policy='latest'`` — the
    default — keeps only the newest of each.

    The coupling to one format is deliberate and explicit rather than hidden
    behind a general hook: there is exactly one rule, and a hook with a
    single implementation would be harder to follow than the ``isinstance``
    check it replaced.
    """
    if not isinstance(loader, ICARTTLoader) or loader.revision_policy != "latest":
        return matches

    keep = set(select_latest_revisions([match.path for match in matches]))
    return [match for match in matches if match.path in keep]
