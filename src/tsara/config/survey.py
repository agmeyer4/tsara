"""Settings for the survey: what to measure before the pipeline, and how.

The survey (:mod:`tsara.survey`) measures what a manifest can declare but files
do not state, and writes it for a person to review (``docs/METHODS.md`` §12).
Its first measurement is clocks: for each instrument, how many seconds later its
files put the same air than a partner's, segment by segment, chained to one
reference clock (§12.4).

These settings live in a YAML of their own, not inside the analysis
configuration, because the survey runs before the pipeline and its product is
an input to the next ingestion, not a stage of this one.

The clock settings, in one picture::

    reference_clock: airmar          # the instrument whose clock the others are put on
    segment: 30min                   # one lag is measured per segment of this length
    tracers:
      picarro:                       # an instrument to survey, then its tiers in order
        - partner: airmar            # tier 1: the partner, and the tracer pairs
          pairs: [[h2o, rh]]         #   (a variable on this instrument, one on the partner)
      aeris:
        - partner: picarro
          pairs: [[ch4, ch4]]        # one gas on two analyzers first ...
        - partner: picarro
          pairs: [[h2o, h2o]]        # ... then a backup

Every instrument's partners must lead, through the others, to the reference
clock, so that its lag composes to it; the reference clock's own instrument has
no tiers.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, ValidationInfo, field_validator, model_validator

from tsara.config.base import StrictModel as _StrictModel
from tsara.config.base import validate_positive_timedelta as _validate_duration
from tsara.config.base import validate_signed_timedelta as _validate_signed
from tsara.config.base import validate_stream_name as _validate_name
from tsara.core.timebase import SECOND_NS

__all__ = ["SurveyConfig", "TracerTier"]

#: One tracer pair: a variable on the surveyed instrument, then one on its partner.
_TracerPair = Annotated[tuple[str, str], Field(description="(variable here, variable on partner)")]


class TracerTier(_StrictModel):
    """One tier of an instrument's tracer pairs: a partner, and the variables compared.

    A tier with several pairs pools them: their correlation curves are averaged
    before the best lag is chosen (``docs/METHODS.md`` §12.4).
    """

    partner: str = Field(
        description=(
            "The instrument this one is measured against: another surveyed instrument "
            "or the reference clock's own."
        )
    )
    pairs: tuple[_TracerPair, ...] = Field(
        min_length=1,
        description=(
            "Tracer pairs, each [variable on this instrument, variable on the partner], "
            "expected to show the same plumes."
        ),
    )
    min_pairs: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Minimum paired seconds in a segment for this tier, overriding the survey's; "
            "for a partner reporting every few seconds, which pairs fewer seconds."
        ),
    )

    @field_validator("partner")
    @classmethod
    def _partner_is_a_name(cls, value: str) -> str:
        _validate_name(value, field="TracerTier.partner")
        return value

    @field_validator("pairs")
    @classmethod
    def _variables_are_names(
        cls, value: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        for own, theirs in value:
            _validate_name(own, field="TracerTier.pairs")
            _validate_name(theirs, field="TracerTier.pairs")
        return value


class SurveyConfig(_StrictModel):
    """What the clock survey measures, against what, and how (``docs/METHODS.md`` §12.4)."""

    reference_clock: str = Field(
        description="The instrument whose clock every other is put on; it has no tiers."
    )
    tracers: dict[str, tuple[TracerTier, ...]] = Field(
        min_length=1,
        description=(
            "For each instrument to survey, its tiers in order: the first tier that "
            "decides a segment sets its lag, the others check it."
        ),
    )
    segment: str = Field(
        default="30min",
        description="Length of the segments records are cut into; one lag is measured on each.",
    )
    search: str = Field(
        default="120s",
        description=(
            "How far either way a lag is searched for. A best lag at the edge of the "
            "search is not decided, since the true one may lie beyond."
        ),
    )
    resolution: str = Field(
        default="1s",
        description=(
            "The step the lag is measured to, at most one second: the whole-second search "
            "finds the peak, and the lag is then measured on the readings' own cells at "
            "multiples of this within two seconds of it. The table's shifts are written to it."
        ),
    )
    max_bridged: str = Field(
        default="0s",
        description=(
            "Up to how long a stretch of undecided segments a row may bridge, when the lags "
            "on either side agree within the tolerance and the instrument's readings run "
            "across it without a logging stop (a gap of a minute or more). 0s, the default, "
            "never bridges: an undecided stretch keeps its recorded times and is named."
        ),
    )
    min_pairs: int = Field(
        default=300,
        gt=0,
        description="Minimum paired seconds in a segment for a lag to be measured at all.",
    )
    min_r: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="The correlation at the best lag a segment needs to be decided.",
    )
    tolerance: str = Field(
        default="2s",
        description=(
            "Lags agreeing within this form one row of the table; a larger jump is a "
            "clock step, and a steady change larger than this across a row, a clock drift."
        ),
    )
    running_median: str = Field(
        default="121s",
        description=(
            "Window of the centred running median removed from every series first, so "
            "that plumes rather than the background drive the correlation."
        ),
    )

    @field_validator("segment", "search", "tolerance", "running_median", "resolution")
    @classmethod
    def _durations_are_positive(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"SurveyConfig.{info.field_name}")
        return value

    @field_validator("max_bridged")
    @classmethod
    def _not_negative(cls, value: str) -> str:
        import pandas as pd

        _validate_signed(value, field="SurveyConfig.max_bridged")
        if pd.Timedelta(value) < pd.Timedelta(0):
            raise ValueError(f"max_bridged ({value}) must not be negative.")
        return value

    @field_validator("search", "running_median")
    @classmethod
    def _whole_seconds(cls, value: str, info: ValidationInfo) -> str:
        # Both are read as whole seconds; a fraction would be cut off without a
        # word (measured: a search of "20.7s" ran as 20, and "500ms" as 0, so
        # every segment sat at the edge of a search with no width).
        import pandas as pd

        if pd.Timedelta(value).value % SECOND_NS:
            raise ValueError(f"{info.field_name} ({value}) must be a whole number of seconds.")
        return value

    @field_validator("resolution")
    @classmethod
    def _at_most_a_second(cls, value: str) -> str:
        import pandas as pd

        if pd.Timedelta(value) > pd.Timedelta("1s"):
            raise ValueError(
                f"resolution ({value}) must be at most 1s: the whole-second search already "
                "gives the lag to a second."
            )
        return value

    @field_validator("reference_clock")
    @classmethod
    def _reference_is_a_name(cls, value: str) -> str:
        _validate_name(value, field="SurveyConfig.reference_clock")
        return value

    @field_validator("tracers")
    @classmethod
    def _instruments_are_names_with_tiers(
        cls, value: dict[str, tuple[TracerTier, ...]]
    ) -> dict[str, tuple[TracerTier, ...]]:
        for name, tiers in value.items():
            _validate_name(name, field="SurveyConfig.tracers")
            if not tiers:
                raise ValueError(f"tracers: instrument '{name}' has no tiers.")
        return value

    @model_validator(mode="after")
    def _every_chain_reaches_the_reference_clock(self) -> SurveyConfig:
        """Refuse a partner that leads nowhere, a loop, and a reference clock with tiers.

        An instrument's lag is measured against its partner and composed to the
        reference clock through the partner's own; a chain that loops, or ends
        at an instrument nobody surveys, has nothing to compose to.
        """
        import pandas as pd

        if pd.Timedelta(self.search) >= pd.Timedelta(self.segment):
            raise ValueError(
                f"search ({self.search}) must be shorter than a segment ({self.segment})."
            )
        segment_s = pd.Timedelta(self.segment).total_seconds()
        for name, tiers in self.tracers.items():
            for number, tier in enumerate(tiers, start=1):
                needed = tier.min_pairs or self.min_pairs
                if needed > segment_s:
                    # Paired seconds cannot outnumber a segment's seconds, so
                    # this tier could never decide; measured before this check,
                    # every segment came out "too short" and nothing was said.
                    raise ValueError(
                        f"tracers: '{name}' tier {number} needs {needed} paired seconds, "
                        f"more than a segment of {self.segment} holds, so it could never "
                        "decide. Lengthen the segment or lower min_pairs."
                    )
        if self.reference_clock in self.tracers:
            raise ValueError(
                f"reference_clock '{self.reference_clock}' has tiers: the instrument "
                "keeping the reference clock is what the others are put on, not surveyed."
            )
        for name, tiers in self.tracers.items():
            for tier in tiers:
                if tier.partner == name:
                    raise ValueError(f"tracers: '{name}' is its own partner.")
                if tier.partner != self.reference_clock and tier.partner not in self.tracers:
                    raise ValueError(
                        f"tracers: '{name}' has partner '{tier.partner}', which is neither "
                        f"the reference clock '{self.reference_clock}' nor surveyed, so its "
                        "lag could not be composed to the reference clock."
                    )
        self.composition_order()  # raises on a loop
        return self

    def composition_order(self) -> tuple[str, ...]:
        """Return the surveyed instruments, each after every partner of it.

        Raises
        ------
        ValueError
            If the partners loop, so that no order composes.
        """
        order: list[str] = []
        placed = {self.reference_clock}
        remaining = dict(self.tracers)
        while remaining:
            ready = [
                name
                for name, tiers in remaining.items()
                if all(tier.partner in placed for tier in tiers)
            ]
            if not ready:
                raise ValueError(
                    f"tracers: the partners of {sorted(remaining)} form a loop, so their "
                    "lags cannot be composed to the reference clock."
                )
            for name in ready:
                order.append(name)
                placed.add(name)
                del remaining[name]
        return tuple(order)
