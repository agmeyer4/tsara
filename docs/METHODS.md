# TSARA Methods

**The scientific methods document for TSARA.** Every algorithmic choice in the
package — estimators, propagation rules, thresholds, approximations — is
specified *here*, with its mathematics, its rationale, and the alternatives it
was chosen over. Code implements this document; this document is not
reverse-engineered from code.

**Contract:** each development phase updates its sections of this file as a
deliverable of that phase. Swappable algorithms are registered by name in the
code (the same decorator-registry pattern used for file readers), and **every
registered algorithm name must have a section here**. Saved TSARA outputs
self-describe: xarray/parquet attributes record the package version and the
stage that wrote it (`tsara_version`, `tsara_stage`), the resolved
configuration, the algorithm names used, and the uncertainty provenance of
every interval (see §2.4).

Sections marked **[stub — Phase N]** are placeholders that will be written
when that phase is built. Nothing in a stub section is decided beyond what the
stub says.

---

## 1. Data model and alignment

### 1.1 Native-rate streams ("synchronize late")

Ingestion produces one `xarray.Dataset` **per instrument stream**, on that
instrument's own native timestamps. Since Phase 3.5 each of those timestamps
carries the *interval of air it describes* rather than standing for an
instant — see §10, which this section should be read alongside. Species measured by the same instrument
share a clock and live in the same Dataset (so per-instrument operations remain
vectorized over species). No resampling of any kind happens at ingestion.

The pipeline defers any change of clock to the last possible moment:

| Stage | Clock used | Support (§10) |
|---|---|---|
| QA/QC, unit conversion | native | unchanged; both are pointwise |
| Rolling baseline, enhancement Δ | native (time-based windows) | windows are durations, so cells of any width fit |
| Plume detection | native → events are time **intervals** | an event's bounds are the union of the cells above threshold; the smallest resolvable event is one cell |
| Ratio regression | **pairing clock** (§1.3), per event/window | the **wider-supported** stream's cells, with its partner averaged onto them by overlap |
| Continuous rolling state, PMF matrix | **output grid** (§1.4) | grid cells, overlap-weighted |

Rationale: rolling quantiles and MAD thresholds are well-defined on irregular
native timestamps; resampling before those stages either destroys information
(downsampling a fast stream) or fabricates it (upsampling a slow one). Only
*cross-species pairing* genuinely requires a common clock, so only pairing
pays for one.

The rejected middle path — a single Dataset on the union of all native
timestamps, NaN-padded — is called out explicitly because it looks like a
compromise: it corrupts rolling-window valid-sample counts, bloats memory, and
makes "minimum valid fraction" semantics incoherent. It is not used.

Note that early synchronization remains *expressible* in this architecture
(bin every stream to the output grid first, then run the identical per-stream
code); the reverse is not true. Late synchronization is therefore the more
general design, not merely the more cautious one.

### 1.2 The interpolation rule

> **Quantified species are never interpolated — only bin-averaged.
> Smooth auxiliary fields (GPS position, temperature, pressure, wind via
> circular statistics) may be interpolated, guarded by `max_interp_gap`.**

Phase 3.5 generalized this to the rule it was a special case of: **TSARA never
evaluates a value on a finer support than it was delivered on** (§10).
Interpolating a gas is one way to break that; treating a 1-minute mean as an
instant is another, and the archive is full of the second.

A concentration inside a plume is not a smooth field; linearly interpolating
it invents structure exactly where the science happens. Platform position and
ambient met vary smoothly on sampling timescales, so interpolation onto gas
timestamps is physically justified there — but never across gaps longer than
`max_interp_gap` (config: `AlignmentConfig.max_interp_gap`, `tsara.config.analysis`).

### 1.3 Pairing for regression (fast → slow, never the reverse)

To regress species *y* against species *x* within an event or rolling window,
when they come from instruments with different rates:

1. The **pairing clock** is the **wider-supported** of the two instruments'
   own **cells**, restricted to the event/window. Phase 3.5 sharpened this:
   before cells existed the rule read "the slower instrument", and rate and
   support can disagree. Measured on the 2024 drives, the iWAS canisters
   sample every 530 s (median over all 261 fills of the ten 2024 drive days)
   but each sample integrates for only 14.9 s, so against a 60 s stationary
   mean the canister is about nine times slower by rate (530 s against 60 s)
   and four times *narrower* by support (14.9 s against 60 s). Pairing on the
   canister's clock would
   split a 60 s mean onto 15 s, which the interval model forbids; pairing on
   the mean's clock is admissible, and the coverage of 0.25 is what says how
   much to trust it. When the two widths are equal, the member with fewer
   measured values where the records overlap is the clock (§11.4.1).
2. The faster stream is averaged onto those cells **weighted by overlap**
   (`tsara.align.binning.bin_streams_onto_cells`), with uncertainty propagated per §3,
   the count of contributing samples recorded, and the fraction of each cell
   actually covered recorded alongside it. Before Phase 3.5 the cell was
   *assumed* to be the slow instrument's sampling period centred on its
   timestamp; now it is read from the stream's own boundaries, so a canister
   that integrated for 14.1 s is paired over exactly 14.1 s (§10).
3. Cells with `n_source_<name> = 0` for either species are dropped — a pair is never
   fabricated. Cells with *some* data are kept, with their coverage recorded;
   `PairingConfig.min_coverage` (default 0.0, i.e. drop nothing) is the one
   knob that acts on it, because how much of a cell must be measured is a
   question about the science and not about the arithmetic.

Consequences: every regression point contains at least one real measurement
of *each* species. This eliminates interpolation pseudo-replication
(interpolated points posing as independent samples and silently inflating
degrees of freedom). It does not by itself make every pair independent: a
sparse member whose cells straddle the clock's boundaries can put one reading
into two pairs, so the product records how many distinct readings of each
species stand behind its pairs, and that — not the pair count — is the ceiling
on a regression's N (§11.4.1).

Different species pairs may therefore be paired on different clocks (each pair
uses its own wider-supported member). Each ratio is an independent slope estimate with
its own honest CI; no shared clock across pairs is required for the ratios to
be comparable as estimates.

### 1.4 Output grid

A single uniform master grid — the familiar `(time × species)` cube — is
constructed **only** for the continuous rolling state and the PMF export
matrix, which inherently require one. Construction is binning-only, by
overlap-weighted mean and no other statistic (§11.7), with per-cell
propagated uncertainties and `n_source_<name>` counts carried alongside values. Grid cells carry CF boundaries like any
other stream, and `cell_methods = "time: mean (interval: <native>)"` records
the resolution of the data that went into them (§10.2). Validation refuses a
period so fine that one selected reading would cover **two grid cells' worth of
time**, the post-Phase-3.5 form of "≥ the slowest stream's native period"
(§11.7; §1.3 makes the same correction for pairing); cells with no native samples are NaN with `n_source_<name> = 0`, never
interpolated. Config: `OutputGridConfig`
(`tsara.config.analysis`) — deliberately not named "the grid" or paired with
the aux-interpolation guard, since neither streams nor cross-species pairing
(§1.3) use it; it exists solely for this output boundary.

### 1.5 Circular statistics for angular variables

Wind direction and other angular quantities are averaged as unit vectors, never
arithmetically — on the real drive data the arithmetic mean is wrong by more
than 45° in 26.7 % of 60 s cells. A binned direction carries the mean
resultant length *R* alongside it, and the dispersion reported is the **exact**
circular standard deviation rather than the Yamartino (1984) single-pass
approximation, which is bounded where the exact form correctly is not.
Specified in full, with the measurements behind each choice, in **§11.5**.

---

## 2. Measurement uncertainty model

### 2.1 Two components

Every measured value is modeled as

$$x_i = x^{\mathrm{true}}_i + e^{\mathrm{rand}}_i + e^{\mathrm{sys}}_i$$

- **random** ($\sigma^{\mathrm{rand}}_i$): uncorrelated point-to-point
  (instrument noise). Averages down with the number of *effective* samples.
- **systematic** ($\sigma^{\mathrm{sys}}_i$): correlated across points on
  averaging timescales (calibration scale/offset, drift, airmass-dependent
  retrieval artifacts). **Does not average down.**

The two components are carried as *separate* variables through the entire
pipeline (`sigma_rand`, `sigma_sys` alongside each species) and are only
combined at the point of use, because downstream operations treat them
differently (§4.3, §5).

### 2.2 Three specification modes (manifest)

Each component of each variable may be specified as:

| Mode | Form | Example |
|---|---|---|
| `declared`, constant/relative | $\sigma_i = \sqrt{a^2 + (r\,x_i)^2}$ | Picarro CH₄: a = 0.5 ppb noise floor |
| `reported` | per-point column read from the data file | EM27 retrievals: an error column that changes every 8 s |
| *(fallback)* `empirical` | robust first-difference estimator `diff_mad` (§2.5) | any variable with no declared budget |

An optional **decorrelation timescale** τ on the random component covers the
in-between world (errors correlated over minutes but not hours); see §3.3.

**Implemented schema** (settled 2026-07-22, `tsara.config.manifest`):
`UncertaintySpec.random` and `.systematic` are each an optional discriminated
union tagged by `mode` — `DeclaredUncertainty` (`absolute`/`relative`,
quadrature) or `ReportedUncertainty` (`column`, the raw-file column holding
that component's per-point sigma) — matching the `kind`/`format`-discriminator
convention already used for QA/QC rules, loaders, and platforms.
`UncertaintySpec.decorrelation_timescale` is an optional duration string
carrying τ for the random component (§3.4).
`DeclaredUncertainty.at_width` (added in Phase 3.5) records the averaging
interval the `absolute`/`relative` figures were *quoted* at, for the ordinary
case of a spec-sheet precision stated at one second being applied to a
one-minute product; §10.8 sets out what ingestion records about it and why the
arithmetic that reconciles the two belongs to the stage that consumes a sigma
rather than to the one that reads it. Omitting a component means "not
modeled here": an omitted `systematic` is zero; an omitted `random` falls back
to the empirical estimator (§2.5) at runtime. A `ReportedUncertainty.column`
is scaled by the parent variable's `convert.scale` at ingestion (a spread has
no origin, so `convert.offset` never applies to it) — that scaling is Phase-3
ingestion logic, not part of this schema.

### 2.3 No silent assumptions

If no budget is declared, TSARA does not invent one: it falls back to the
empirical `diff_mad` estimate (§2.5) and **labels** the result. There is no
code path in which an uncertainty of unstated origin enters a confidence
interval. The direction of dependency is fixed: plume detection *consumes*
this uncertainty system (§6); it does not maintain a private, parallel
definition of noise.

### 2.4 Provenance

Every product carries an `uncertainty_source` label per species:
`declared | reported | empirical`. A reader of any TSARA output can always
determine what pedigree of uncertainty produced each interval.

Provenance is recorded **per component**, in
`uncertainty_source_random` and `uncertainty_source_systematic`, because real
manifests mix modes freely (the shipped example pairs a *reported* random
component with a *declared* systematic one). The species-level
`uncertainty_source` is then `mixed` when the two components disagree, and the
sigma companions carry `uncertainty_component` saying which one they are. Two further component values are needed to keep §2.3
honest, and they are not interchangeable:

| Component value | Meaning |
|---|---|
| `declared` | computed at ingestion from `absolute`/`relative` |
| `reported` | read at ingestion from the instrument's per-point sigma column |
| `empirical` | deferred to the stage holding the analysis config, which owns the estimator name and window (§2.5) |
| `zero` | a budget was given and *deliberately* omitted this component ("an omitted `systematic` is zero", §2.2) |
| `unknown` | no budget at all. The random component falls back to `empirical`; the systematic component cannot, since `diff_mad` is structurally blind to it (§2.5) |

Collapsing `zero` into `unknown` would let an undeclared calibration become a
silent claim of perfect calibration, which is exactly what §2.3 forbids.

### 2.5 The empirical noise estimator (`diff_mad`)

When noise must be estimated from the data itself, the default estimator is
the robust first-difference ("derivative noise") estimator over a rolling
window:

$$\hat\sigma^{\mathrm{rand}}
= \frac{1.4826 \cdot \mathrm{median}_i\big(|x_{i+1} - x_i|\big)}{\sqrt{2}}$$

(1.4826 is the Gaussian consistency constant for the MAD; the √2 because
$\mathrm{Var}(x_{i+1}-x_i) = 2\sigma^2$ for white noise.)

Why differences of the signal rather than the signal itself (`mad`):

- **Plume immunity.** A broad, smooth plume has point-to-point differences of
  noise size even at large amplitude, so it barely contaminates the estimate.
  Rolling MAD of the *signal* holds only up to its 50% breakdown point —
  once enhancements occupy more than half the noise window (routine in
  plume-dense records: near-source stationary sites, mobile transects through
  producing fields), it starts measuring plume variability, inflating
  detection thresholds exactly where there is the most to detect.
- **Honest scope.** Differencing cancels anything slowly varying, so
  `diff_mad` estimates *only the random component* and is structurally blind
  to systematic error. That matches what an empirical fallback is entitled to
  claim: it can measure noise; it cannot know the calibration. The systematic
  component of an undeclared budget is simply *unknown*, and outputs are
  labeled accordingly (§2.4).
- **Caveat.** For autocorrelated (red) instrument noise, `diff_mad` measures
  the high-frequency noise floor rather than total low-frequency variability
  — the right scale for detecting features above the noise at the sampling
  timescale; slower variability is the baseline's job to absorb.

**Quantization guard.** Any median-based estimator collapses to zero when
more than half the window shares a single value — which happens whenever
data are reported at a resolution comparable to the noise (e.g., a logger
writing 0.01-ppm steps). A zero noise scale makes every point a "plume".
All estimated sigmas are therefore floored at the quantization scale,
$\hat\sigma \ge \delta/\sqrt{12}$, where δ is the declared or detected
reporting resolution (δ/√12 = the standard deviation of uniform rounding
error).

Registered estimator names: `diff_mad` (default), `mad` (rolling MAD of the
signal, kept for comparison). MAD-family estimators are only ~37% efficient
at the Gaussian (their own sampling jitter is ~1.6× that of a standard
deviation on clean data); if threshold jitter ever proves limiting, the
Rousseeuw–Croux $Q_n$ estimator (~82% efficiency at the same 50% breakdown,
no symmetry assumption) is the designated upgrade path — a new registered
name, not a redesign.

---

## 3. Uncertainty propagation under averaging

### 3.1 Master equation

For a weighted mean $\bar{x} = \sum_i w_i x_i$ with $\sum_i w_i = 1$, the
variance is (GUM, JCGM 100:2008, eq. 13):

$$\mathrm{Var}(\bar{x}) \;=\; \sum_i \sum_j w_i\, w_j\, \rho_{ij}\, \sigma_i\, \sigma_j$$

where $\rho_{ij}$ is the error correlation between samples *i* and *j*.
Everything below is a special case of this equation; when in doubt, the engine
may always fall back to evaluating the double sum directly.

### 3.2 Independent errors (ρ = 0 off-diagonal) — the random component

$$\mathrm{Var}(\bar{x}) = \sum_i w_i^2 \sigma_i^2$$

With inverse-variance weights $w_i \propto 1/\sigma_i^2$ (the minimum-variance
unbiased choice):

$$\sigma_{\bar{x}}^2 = \Big(\sum_i 1/\sigma_i^2\Big)^{-1}
\quad\xrightarrow{\;\text{equal }\sigma\;}\quad \sigma^2/N .$$

This is the familiar "quadrature, scaled by N" rule — valid **only** here.

**The weights are the operation's, not the estimator's.** Inverse-variance
weights are the minimum-variance choice for *estimating a constant*, and it is
tempting to reach for them here. TSARA does not. The value being reported is an
overlap-weighted time average, because that is what "this stream's mean over
that interval" means (§1.3); an uncertainty computed with different weights
would not describe the number it is attached to. `tsara.core.propagation`
therefore takes the caller's weights and uses exactly those.

### 3.3 Fully correlated errors (ρ = 1) — the systematic component

$$\mathrm{Var}(\bar{x}) = \sum_i\sum_j w_i w_j \sigma_i \sigma_j
= \Big(\sum_i w_i \sigma_i\Big)^2
\quad\Longrightarrow\quad
\sigma_{\bar{x}} = \sum_i w_i\,\sigma_i .$$

**The uncertainty of the mean is the weighted mean of the uncertainties — no
√N reduction.** This is the exact algebra behind the "weighted mean of the
errors" treatment used in practice for EM27 averaging: it is the correct
propagation for reported errors that are declared (or known) to be correlated,
not an ad-hoc convention.

### 3.4 Partial correlation — decorrelation timescale τ

For a random component with a declared decorrelation timescale τ (§2.2), TSARA
models the error autocorrelation as AR(1)-like,
$\rho(\Delta t) = e^{-|\Delta t|/\tau}$, and reduces the sample count
accordingly:

$$\sigma_{\bar{x}}^2 = \sigma^2 / N_{\mathrm{eff}}, \qquad
N_{\mathrm{eff}} \in [1, N].$$

Where **no** τ is declared the samples are treated as independent (§3.2). That
is a *declaration*, not an assumption TSARA invents: a component declared
random is by definition uncorrelated point to point, and τ is the refinement
that says "less so than that". Every propagated σ carries the name of the form
that produced it, `independent` included, so the two cases are distinguishable
in the product rather than only by the absence of a label.

**Implementation.** `tsara.core.propagation` — one module, because §1.3
binning, §5 rolling and §4.3 fit weighting all need this answered identically,
and because moving a declared σ onto a different support (§10.8) is the same
arithmetic again. Ingestion deliberately performs none of it (§10.8, §9.6).

**Registered forms.** All three assume the same AR(1) autocorrelation and
differ only in how faithfully they evaluate the resulting double sum.

| name | cost | exact when |
|---|---|---|
| `ar1_neff` | O(N) | samples equally spaced with equal σ — **the default** |
| `ar1_asymptotic` | O(1) | N is many correlation times |
| `ar1_double_sum` | O(N²) | always, given AR(1) |

`ar1_neff` evaluates the equal-weight, equally-spaced case of §3.1 exactly. There
$\rho_{ij} = \rho_1^{|i-j|}$ collapses the $N^2$ terms onto $N$ distinct lags:

$$\frac{\mathrm{Var}(\bar{x})}{\sigma^2} = \frac{1}{N^2}\Big(N + 2\sum_{k=1}^{N-1}(N-k)\,\rho_1^{\,k}\Big),
\qquad N_{\mathrm{eff}} = \Big(\frac{\mathrm{Var}(\bar{x})}{\sigma^2}\Big)^{-1}.$$

The sum is evaluated term by term rather than through its algebraic closed
form, which is numerically hopeless as $\rho_1 \to 1$: at $\rho_1 = 0.999$ it
subtracts two quantities of order $10^4$ that agree to four figures, through a
denominator of $10^{-6}$. The terms decay geometrically, so the sum is
truncated once $\rho_1^k$ stops moving a float64 accumulator. Unequal weights
reach the same formula through their Kish sample size,
$(\sum w)^2 / \sum w^2$, which is exact for equal weights and an
approximation otherwise; `ar1_double_sum` is how to find out what that
approximation costs on a given cell.

`ar1_asymptotic` is the large-$N$ limit, $N_{\mathrm{eff}} = N(1-\rho_1)/(1+\rho_1)$.
It is the form this document specified before the finite-$N$ version existed,
and it is kept because reproducing its error is worth being able to do:

| case | `ar1_asymptotic` | `ar1_neff` | σ overstated by |
|---|---|---|---|
| 1 Hz, N = 200, τ = 100 s | 1.00 | 1.76 | 33 % |
| 1 Hz, N = 60, τ = 20 s | 1.50 | 2.19 | 21 % |
| 1 Hz, N = 600, τ = 2 s | 146.9 | 147.4 | 0.2 % |

So the choice matters exactly when a window holds only a few correlation
times, which is the normal case for a plume. **This resolves the "AR(1)
approximation vs. exact double sum" flag in CLAUDE.md §5 for the arithmetic**:
the finite-$N$ form is exact for the case it is used on and costs no more than
linear time, so there is no reason to prefer the asymptotic one. It does not
resolve whether the AR(1) *model* describes real instrument error; that is
measured against synthetic ground truth with known τ in §11.1 and §11.8.

**What "exact when equally spaced" costs when they are not.** The default form
summarises a cell by two numbers, how many samples it holds and how far apart
they are, so it cannot see *where* in the cell they sit. A dropout that splits
a cell into two clumps is where that matters, and the size of the difference
was measured rather than left as a caveat. Thirty samples drawn from an AR(1)
error with τ = 20 s, 20 000 realizations, arranged two ways:

| thirty samples arranged as | observed σ of the mean | `ar1_neff` | `ar1_double_sum` |
|---|---|---|---|
| one consecutive run | 1.608 | 1.604 (0.997×) | 1.604 (0.997×) |
| two blocks of 15, 30 s apart | 1.344 | 1.604 (**1.194×**) | 1.343 (0.999×) |

The same thirty samples carry *more* information when spread across the cell,
because the two blocks have had time to decorrelate from each other. The cheap
form does not know that and overstates σ by about a fifth. The error is in the
conservative direction and is bounded by how badly a cell's samples clump,
which `n_source` and `coverage` already report — and `ar1_double_sum` is
selectable wherever a form is selectable, which is what makes the reference
form useful rather than decorative.

Limits: τ → 0 recovers §3.2, and τ → ∞ is properly handled by declaring the
component systematic (§3.3) rather than by an enormous τ.

---

## 4. Regression estimators

Registered names: `ols`, `york`, `odr`. Defaults: `("ols", "york")` (config:
`RegressionConfig.methods`, `tsara.config.analysis`).

### 4.1 `ols` — ordinary least squares (statsmodels)

Kept for cheap, familiar diagnostics (R², residual structure). **Known bias:**
with noise on the x-axis, OLS attenuates slopes toward zero (regression
dilution); it is never the preferred ratio estimator here. See Cantrell (2008)
and Wu & Yu (2018) for the atmospheric-chemistry context of this choice.

### 4.2 `york` — York (2004) errors-in-both-variables fit (own implementation)

The maximum-likelihood straight line $Y = a + bX$ given per-point standard
errors $\sigma_{x,i}, \sigma_{y,i}$ **and per-point correlation $r_i$ between
the x- and y-errors** — the parameter scipy's ODR cannot accept. $r_i \neq 0$
is the physically expected case whenever numerator and denominator species
come from the same instrument (e.g., EM27 gases retrieved from one spectrum
share spectral noise, continuum fit, and airmass systematics).

With weights $\omega_{x,i} = 1/\sigma_{x,i}^2$, $\omega_{y,i} = 1/\sigma_{y,i}^2$,
$\alpha_i = \sqrt{\omega_{x,i}\,\omega_{y,i}}$, iterate on the slope *b*:

$$W_i = \frac{\omega_{x,i}\,\omega_{y,i}}
{\omega_{x,i} + b^2\,\omega_{y,i} - 2\,b\,r_i\,\alpha_i}$$

$$U_i = X_i - \bar{X},\quad V_i = Y_i - \bar{Y}
\quad(\bar{X},\bar{Y}\ \text{are } W\text{-weighted means})$$

$$\beta_i = W_i\!\left[\frac{U_i}{\omega_{y,i}} + \frac{b\,V_i}{\omega_{x,i}}
- (b\,U_i + V_i)\,\frac{r_i}{\alpha_i}\right],
\qquad
b_{\text{new}} = \frac{\sum_i W_i\,\beta_i\,V_i}{\sum_i W_i\,\beta_i\,U_i}$$

iterated to convergence (typically < 10 iterations), then
$a = \bar{Y} - b\bar{X}$, with analytic standard errors
$\sigma_b^2 = 1/\sum_i W_i u_i^2$ (where $u_i = x_i - \bar{x}$ for the
adjusted points $x_i = \bar{X} + \beta_i$) and goodness of fit
$S = \sum_i W_i (Y_i - bX_i - a)^2 \sim \chi^2_{N-2}$.

Full equations: York, Evensen, Martínez & Delgado (2004). The implementation
is ~60 lines, MIT-clean (precedent: TSARA's own ICARTT parser), and is
validated in tests against the canonical Pearson (1901) dataset with York's
weights — the standard cross-implementation benchmark for this estimator.

v1 default: $r_i = 0$ (reduces York to per-point-weighted ODR for the linear
case). How $r_i$ is declared per instrument species-pair, or estimated, is an
open design flag.

### 4.3 Which σ enters the point weights

York's derivation assumes point errors are **independent between points**.
Therefore:

- **Point weights use the random component** ($\sigma^{\mathrm{rand}}$, after
  any pairing-bin propagation), which satisfies that assumption.
- **The systematic component is common-mode within an event** and must *not*
  be stuffed into per-point weights (it violates the independence assumption
  and corrupts both the fit and its reported error). It is propagated to the
  *ratio* analytically after the fit:
  - *offset-type* systematics largely cancel in the enhancement
    Δ = x − baseline (the baseline subtracts the common offset on window
    timescales);
  - *scale-type* systematics propagate directly: a relative scale uncertainty
    $s$ on either axis contributes relative uncertainty $s$ to the slope,
    added in quadrature at the event level.

### 4.4 `odr` — scipy.odr

Retained as a numerical cross-check on `york` (they agree when all
$r_i = 0$) and for possible future nonlinear models. Not the default.

---

## 5. Combined uncertainty on reported ratios

Three terms, composed in quadrature, each with recorded provenance:

$$\sigma^2_{\text{ratio}} \;=\;
\underbrace{\sigma^2_{\text{fit}}}_{\substack{\text{York analytic error;}\\
\text{measurement-aware via §4.3 weights}}}
\;+\;
\underbrace{\sigma^2_{\text{sys}}}_{\substack{\text{event-level systematic}\\
\text{(scale terms, §4.3)}}}
\;+\;
\underbrace{\sigma^2_{\text{method}}}_{\substack{\text{variance across the}\\
\text{parameter-sweep hypercube}}}$$

The first term already contains propagated measurement noise (it is *not* an
independent "measurement" term — adding one separately would double-count).
The precise estimator for $\sigma^2_{\text{method}}$ (which cube axes, which
dispersion statistic) is specified in Phase 7. **[partial stub — Phase 7]**

### 5.1 Error-budget closure: within-fit SE vs. between-fit scatter

If per-fit standard errors are honest, they must statistically explain the
observed spread of slopes across repeated fits of the same (stable) ratio.
For K slope estimates $b_k$ with reported standard errors $\sigma_{b,k}$ and
their inverse-variance weighted mean $\bar{b}$:

$$R_B^2 \;=\; \frac{1}{K-1} \sum_k \frac{(b_k - \bar{b})^2}{\sigma_{b,k}^2}$$

— the **Birge ratio**, equivalently a reduced χ² with K−1 degrees of freedom;
the metrological standard (e.g., CODATA practice) for testing whether stated
uncertainties account for observed dispersion. The informal version — compare
the typical per-fit SE to the plain standard deviation of the $b_k$ — is the
quick-look equivalent.

Interpretation:

- **R_B ≈ 1** — the budget closes: reported fit errors explain the
  event-to-event scatter.
- **R_B ≫ 1** — excess scatter: understated per-fit errors (missing error
  components, unmodeled correlation) *or* genuine variability of the source
  ratio. TSARA cannot distinguish these two on its own; that judgment is
  scientific context, and outputs must not pretend otherwise.
- **R_B ≪ 1** — overstated errors, or correlated fits (see caveats).

Two uses in TSARA:

1. **Stability-cube layer (Phase 7):** R_B is computed per sweep point, so a
   user can see *where in parameter space the error budget closes*.
2. **Methodology selection:** choosing sweep parameters such that per-fit SE
   matches aggregate scatter (R_B ≈ 1) — the criterion used in the project
   owner's EM27 enhancement-ratio work — is a principled, self-consistency
   basis for picking the primary combination(s), and is the leading candidate
   criterion for the open "primary combinations" question (CLAUDE.md §5,
   open flags).

This is the same closure logic as York's per-event goodness of fit
$S/(N-2)$ (§4.2), applied one aggregation level up: point errors vs. residual
scatter *within* an event there; fit errors vs. slope scatter *across* events
here. A budget that closes at both levels, with small sweep spread
($\sigma_{\text{method}}$), is the strongest defensibility statement TSARA
can make about a reported ratio.

Caveats: valid as stated only for (quasi-)independent fits sampling an
approximately stable true ratio — discrete, non-overlapping plume events
qualify; overlapping rolling-window fits are serially correlated (effective
K < K), biasing R_B low. And forcing R_B → 1 as a hard optimization target
when the source ratio genuinely varies would drive selection toward
over-conservative parameters: use closure as a constraint and diagnostic,
not a sole objective. **[estimator details — Phase 7]**

---

## 6. Baselines, detection, smoothing, clustering **[stubs]**

- **Rolling low-quantile baseline** **[stub — Phase 5]**: quantile q of a
  centered time-based window; window/quantile lists are sweep dimensions.
  Baseline *uncertainty* (order-statistic variance or block bootstrap) to be
  specified in Phase 5 — it feeds Δ uncertainty.
- **Plume detection** **[partial stub — Phase 6]**: two-threshold hysteresis
  segmentation of the enhancement Δ (config: `DetectionConfig.enter_sigma`
  (sweep dim) / `exit_sigma`, both in noise-σ units, plus `min_duration` and
  `max_internal_gap` for internal-gap bridging). Decided 2026-07-09: the noise
  scale σ comes from the measurement-uncertainty system in provenance order —
  declared or reported $\sigma^{\mathrm{rand}}$ when available, else the
  empirical estimator named by `DetectionConfig.noise_estimator` (default
  `diff_mad`, §2.5); detection has no private definition of noise, and the
  §2.5 quantization floor applies to whichever source is used. Also decided:
  **quantile-offset correction** — because the baseline
  is a low quantile q, even pure noise has a positive median enhancement of
  $-z_q\,\sigma$ (≈ 1.64σ at q = 0.05, Gaussian), so thresholds are applied
  to the offset-corrected enhancement; otherwise the effective threshold
  silently depends on the swept quantile and false-positive rates differ
  across sweep points. Exact segmentation details specified in Phase 6.
  Nested events (an event at a short baseline window inside an event at a
  longer one) are recorded with parent–child links in the catalog —
  detection-level bookkeeping only; no area mathematics (§7).
- **Smoothing** **[stub — Phase 8]**: zero-phase Butterworth per stream at its
  native nominal rate, segment-wise around gaps (filtfilt requires uniform
  sampling; each instrument is nominally uniform between gaps).
- **Source-complex clustering** **[stub — Phase 8]**: DBSCAN on scaled
  space-time coordinates.

---

## 7. Deferred science (future avenues)

TSARA's pipeline is designed so new science stages can be inserted at any
point later without rearchitecting: stages consume and produce the documented
products of §1 (native-rate streams, event catalog, output grid), estimators
are registered by name, and the catalog schema reserves room for
stage-specific columns. Candidates already identified, deliberately **not** in
v1:

- **Peak-area integration & nested-area subtraction** (descoped 2026-07-09):
  trapezoidal integration of Δ over event intervals, with gross/net-area
  parent-event bookkeeping so nested micro-plume mass is not double-counted in
  macro-plumes — and the area-ratio $A_y/A_x$ as a pairing-free ratio
  cross-check. Deferred because integration is scientifically fraught
  (acutely sensitive to baseline placement, event-boundary choices, and data
  gaps) and plume detection must be trustworthy first. The catalog keeps
  parent–child event links (§6) precisely so this can be added later.
- **Alternative plume detectors** beyond threshold + hysteresis: changepoint
  segmentation (e.g., PELT), matched filtering against plume templates, HMM
  background/plume state models. Statistically interesting but heavier, more
  opaque, and less sweep-friendly than hysteresis; since the detector is a
  registered algorithm name, these can be added without touching the
  pipeline.
- **Drive-path-aware plume deconvolution** for mobile platforms.
- **Clock-offset estimation by cross-correlation.** `InstrumentConfig.time_shift`
  (§10.7) lets a manifest *declare* an offset; deriving one by correlating a
  species against a trusted reference is a registered estimator waiting to be
  written. The 2026 campaign's own alignment stage does exactly this.
- **Decorrelation-timescale estimation.** τ is what every $N_{\mathrm{eff}}$
  correction needs (§3.4) and what no file declares: `UncertaintySpec.
  decorrelation_timescale` lets a manifest *state* one, but establishing it for
  an instrument is a measurement, and a field most data owners cannot fill
  honestly should not be what decides whether a stored sigma is right. That is
  why ingestion records declarations instead of acting on them (§10.8), and it
  is the strongest argument for the package estimating τ itself: TSARA is a
  time-series package whose subject is uncertainty, and this is a time-series
  measurement. The classic instrument tool is the Allan deviation — average a
  record over increasing windows and find where the deviation stops falling as
  $1/\sqrt{N}$ — with a variogram or a fit to the first-difference
  autocorrelation as siblings. One constraint is known in advance and shapes
  the design: the campaign this package was built for has **no plume-free
  stretches**, so an estimator that assumes a quiet segment will measure the
  atmosphere rather than the instrument. It must work on plume-dense records —
  robust, difference-based, or fitted on the quietest quantile of windows —
  and it must report a provenance the way every other estimate here does.
  Registered by name, like the noise estimators of §2.5.
- **Instrument response deconvolution.** A laser analyzer's reading is a
  convolution of the truth with a response function, which is why `point` is
  the honest description of it rather than `mean` (§10.3). Recovering the
  underlying signal is a deconvolution problem and a stage of its own.
- **Non-uniform canister fill weighting.** A whole-air sampler's flow is not
  constant over its fill, so a canister is a *weighted* mean of the interval
  rather than a flat one. TSARA assumes uniform and records that it did
  (§10.10); the weighting is undeclared in every file seen.

---

## 8. Synthetic data generation (Phase 2)

TSARA has **no controlled-release measurements** to validate against
(CLAUDE.md §5): the campaign archive is ambient field data, plume-dense
throughout, with no independently documented emission rates. Injected
synthetic ground truth is therefore the *only* arbiter of detection and
enhancement-ratio correctness in v1. That makes the generator a scientific
instrument in its own right, and this section specifies it to the same
standard as the estimators above.

Implementation: `tsara.synthetic` (config, profiling, background, plumes,
noise, platform, generator, bundle, timebase).

### 8.1 What is manufactured, and what is recorded

One `SyntheticConfig` yields per-instrument `xarray.Dataset` streams on their
own native, irregular clocks (§1.1) plus a `GroundTruth` catalog. Each stream
carries the observable variable, the exact `truth_background_*` /
`truth_enhancement_*` decomposition, the true `truth_sigma_rand_*` /
`truth_sigma_sys_*` budget, and any instrument-*reported* sigma column under
its configured raw-file name. Everything prefixed `truth_` is the answer key
and is excluded from the pipeline-visible view (`SyntheticDataset.observable`).

The catalog is deliberately schema-compatible with a subset of the future
Phase-6 `PlumeCatalog`, so scoring detection is a column-wise diff rather than
a translation layer. It records both `true_amplitude` (the continuous peak the
source produced) and `sampled_peak_amplitude` (the largest value the
instrument's clock could have seen, NaN if the event fell entirely inside a
gap). These answer different questions, and a detector cannot be faulted for
the difference between them.

**Events are drawn before any instrument is rendered.** A plume is one
physical release: the same leak must appear on the 1 Hz and the 10 Hz analyzer
with consistent amplitudes and one consistent ratio. Drawing per-instrument
would destroy exactly the cross-species covariance TSARA exists to measure,
and every regression test built on such data would be measuring an artifact.

**Sources, ratios, and nesting.** A `SourceSpec` is a family of correlated
multi-species events, not a per-species plume model. Each event draws its
reference species' peak amplitude from `amplitude`, and every other
participating species gets `amplitude × ratio` with the ratio drawn from that
species' `RatioSpec`. Ratios are specified as a *distribution* rather than a
constant because real emission ratios vary between encounters of the same
source type; `relative_spread = 0` collapses to the fixed-ratio textbook case,
and non-zero spread is what gives the Phase-7 methodological-variance and
Birge-ratio diagnostics (§5.1) something to detect. Draws are lognormal,
parameterized so the configured `mean` is the arithmetic mean:

$$\sigma_{\log} = \sqrt{\ln(1 + s^2)}, \qquad \mu = \ln m - \tfrac{1}{2}\sigma_{\log}^2$$

which keeps "did the estimator recover the true ratio?" a well-posed question
with an unambiguous target.

A **nested** child is a short, sharp plume riding inside a broader parent — the
multi-scale case from CLAUDE.md §1. The child is modelled as a *distinct
physical source encountered inside the parent*, so when `nested.ratios` is set
it may name species the parent never emits: a broad landfill plume (methane,
no ethane) carrying a thermogenic blip (methane *and* ethane) is the canonical
example, and forbidding it would make the package's own motivating case
inexpressible. Species the child does not mention inherit the parent's realized
ratio; leaving `nested.ratios` unset inherits the parent's chemistry entirely,
describing finer temporal structure within one source rather than a second
source. The reference species may not appear in either ratio mapping — its
ratio to itself is 1 by definition, and a declared entry would double-count it.
Names are validated campaign-wide against the declared `role="gas"` species
rather than against the parent's list, which is what still catches typos.

Scientifically this is the case that matters most for v1: a regression that
lumps child and parent samples together measures neither source's ratio.
Phase 6 records the parent–child link in the catalog; the area mathematics that
would separate their masses is deferred (§7).

### 8.2 Plume shapes

Two registered shapes. **Gaussian** (`sigma`) is the symmetric textbook case.
**EMG** (`sigma`, `tau`) is the physically motivated one: a Gaussian
(turbulent dispersion) convolved with a decaying exponential (residence time
in the source volume, inlet and cavity flushing), producing the sharp rise and
long trailing tail real transects show. The asymmetric tail is what makes
baseline placement genuinely hard — it decays asymptotically, so "where the
plume ends" has no crisp answer.

The textbook EMG form,

$$f(t) \propto \exp\!\Big(\tfrac{\sigma^2}{2\tau^2} - \tfrac{t-\mu}{\tau}\Big)\,
\mathrm{erfc}\!\Big(\tfrac{\sigma/\tau - u}{\sqrt 2}\Big), \qquad u = \tfrac{t-\mu}{\sigma}$$

overflows catastrophically in float64 — the exponential diverges while `erfc`
underflows to 0, so the product evaluates to `inf * 0 = nan` precisely in the
tail that matters. TSARA evaluates it through the *scaled* complementary error
function $\mathrm{erfcx}(z) = e^{z^2}\mathrm{erfc}(z)$, which cancels the two
divergences analytically:

$$f(t) \propto e^{-u^2/2}\;\mathrm{erfcx}\!\Big(\tfrac{\sigma/\tau - u}{\sqrt 2}\Big)$$

stable until `erfcx` itself overflows near $z = -26$ (since $e^{709}$ is the
float64 ceiling). Beyond that, $\mathrm{erfcx}(z) \to 2e^{z^2}$ gives the exact
closed form $\log f \to \log 2 + \sigma^2/(2\tau^2) - u\sigma/\tau$ — a pure
exponential decay of timescale τ, which is the physical tail behaviour the EMG
was chosen for. The two branches are continuous to <1e-6 in the log-shape.

Kernels are normalized to unit peak on a dense reference grid with a
refinement pass, so `sampled_peak_amplitude ≤ true_amplitude` holds exactly
rather than to within a normalization artifact (verified at 0 violations
across the shipped example's catalog). Support is truncated at 4σ (and +6τ
for the EMG tail), keeping rendering O(events × window); the resulting step
at the support edge is 3.4e-4 of peak for a Gaussian and 8.4e-4 for the
example's EMG — 0.07σ of measurement noise for a median plume and 0.34σ for
the largest, so far below any hysteresis threshold that smoothing it away
would buy nothing.

**Record-edge asymmetry (harness limitation).** Event centers are drawn
strictly inside the record, so a plume landing near the end is truncated but
a record can never *open* part-way through a plume whose peak already passed.
Real records routinely do. This matters only to stages with distinct
boundary behaviour — Phase 5 baselines and Phase 6 detection both operate on
half-empty windows at the edges — and is recorded in `schedule_events` with
the change that would lift it, should either phase need the case.

### 8.3 Backgrounds

**Parametric**: `offset` + diurnal + linear drift + random walk, deliberately
separable so a test can switch on one term at a time. The diurnal term is
phased off the Unix epoch (midnight-aligned) rather than each stream's start,
so two instruments in one run breathe in phase as they physically must.
Random-walk increments scale as $\sqrt{\Delta t}$ so the configured
one-day wander magnitude is independent of sampling rate.

**Bootstrap** (real-data-driven): fluctuations are resampled in contiguous
*blocks* from a `RealDataProfile` (§8.4). Block resampling rather than
point-wise is the whole point — drawing points independently would destroy the
residual's autocorrelation and hand back white noise, defeating the purpose of
using real data.

Two documented limitations:

- **Blocks are mean-centred**, so only *within-block* (high-frequency)
  structure survives; between-block low-frequency structure is discarded to
  avoid step discontinuities at the stitching seams. Slow structure is
  supplied by the optional parametric `base` instead. On records with strong
  slow structure the discarded fraction can be large: a residual dominated by
  between-block drift (ρ₁ near 1) can lose several-fold of its robust spread
  to centring alone.
- Mean-centring equalizes block *levels*, so no step in the mean appears where
  two blocks meet, but the samples either side of a seam remain independent
  draws: a seam carries a sample-to-sample step of order the residual σ,
  empirically ~4× the typical interior step. This is accepted rather than
  blended away, because seams are only `1/block_length` of adjacent pairs
  (0.8 % at the default 128) and every downstream noise estimator is
  median-based, so `diff_mad` shifts by well under 1 %; overlap-blending would
  smooth exactly the high-frequency structure the bootstrap exists to
  preserve. When reading a generated record, an isolated sharp step every
  `block_length` samples is a stitching artifact, not injected signal.
- Because the source records are plume-dense, real plume energy leaks through
  the profiling baseline into the residual. This is treated as a **feature**:
  it is precisely the adversarial "is `diff_mad` really plume-immune on my
  instrument?" test case (§2.5). But it means the substrate is not a pure
  noise realization, and `RealDataProfile.lag1_autocorr` describes the
  *substrate*, not the instrument noise — it must not be fed to an N_eff
  calculation as though it were a noise decorrelation timescale.

### 8.4 Profiling real data

Deliberately **not** called "calibration" — in this domain that word means
referencing an instrument against gas standards, an operation the campaign
archive already uses the name for (`04_calibrated/`, `calibration_coefs.json`).
"Profiling" is the standard term for summarizing a dataset's statistical shape.

`profile_series` fits a plain rolling low-quantile background (no sweep, no
uncertainty propagation — explicitly *not* the Phase-5 baseline engine),
subtracts it, and characterizes the residual: robust spread, the plume-immune
`diff_mad` noise scale (§2.5), lag-1 autocorrelation and its implied AR(1) τ,
then cuts gap-free mean-centred blocks. Segments are split on gaps before
blocking, so no block straddles a dropout and no resampled substrate can
contain a jump the real instrument never made.

**Gap structure is handled where it changes the generated data, and nowhere
else.** Blocking is that place: a fabricated jump becomes part of the output.
The scalar statistics deliberately are not segmented — they are median- and
correlation-based, so 20 % data loss across 60 gaps moves `noise_sigma` by
only ~1.6 %, and on this project's records the question does not arise at all
(`03_instrument_aligned` Picarro data is already on a regular 2 s grid, with
0 gaps in 12 450 intervals across a measurement day). Segment-wise variants
were implemented, measured against the real archive, and removed as unearned
complexity. Revisit only if Phase 3's QA/QC masking begins feeding
hole-punched series into `profile_series`.

`residual_sigma / noise_sigma` is a useful diagnostic in its own right: values
far above 1 indicate a plume-dense record rather than a noisy one — an ambient
trace-gas archive with no plume-free stretches can easily sit an order of
magnitude or more above 1, since broad plumes leak through a simple quantile
baseline while `diff_mad` stays immune to them by construction (§2.5).

**τ is ill-conditioned near ρ₁ → 1**, which is a distinct problem from the
interpretive caveat above and applies even when ρ₁ is measured perfectly.
Differentiating $\tau = -\Delta t/\ln\rho$ gives

$$\frac{d\tau/\tau}{d\rho/\rho} = \frac{-1}{\rho\ln\rho}$$

an amplification of ~334× at ρ₁ = 0.997 — a value plume-dense ambient records
can plausibly reach. At a typical Δt = 2 s that is τ = 666 s for ρ₁ = 0.997
versus 499 s for ρ₁ = 0.996 — a one-part-in-a-thousand shift moving the answer
by a quarter. `decorrelation_timescale_s` is therefore
an order-of-magnitude indicator on strongly autocorrelated records, never a
calibrated timescale, and must not enter an N_eff calculation without an
uncertainty of its own. This bears directly on the open N_eff estimator
question (§3.4).

**No real data ships with TSARA, ever.** Profiles are computed from a live
mount, referenced *by name* in configs, and passed to the generator at call
time — never embedded, so every `SyntheticConfig` stays losslessly
YAML-round-trippable and no real-derived numbers can reach a committed file.
Tests touching real data skip unless `TSARA_REAL_DATA` is set.

### 8.5 Error injection with known decomposition

The generator's `TrueUncertainty` is the *inverse* of the manifest's
`UncertaintySpec` (§2.2): the manifest describes how to **read** an
uncertainty an instrument reports; the generator needs parameters from which
to **manufacture** it. `TrueComponent` mirrors `DeclaredUncertainty`'s
$\sigma_i = \sqrt{a^2 + (r x_i)^2}$ form and adds `report_as` (the raw-file
column an instrument would publish its per-point sigma under) and
`report_bias` (so an instrument that *understates* its own error by 20 % is
expressible, and downstream UQ can be tested against one).
`TrueUncertainty.to_manifest_uncertainty()` converts back, keeping the two
schemas provably aligned.

The components differ in **how they are drawn**, which is the entire point:

- **random** — one independent draw per sample, or, with a configured
  `decorrelation_timescale`, an AR(1) process with $\rho = e^{-\Delta t/\tau}$
  (§3.4). The *standardized* series carries the correlation and is then scaled
  by the per-point sigma, so marginal variance is unchanged and τ can be
  varied alone. This is the only way to obtain data with a **known** τ, which
  the open N_eff estimator question has no other way to be tested against.
  Uniform sampling uses a vectorized IIR filter seeded from a stationary draw
  (no burn-in transient); irregular sampling uses the exact per-point
  recursion rather than approximating ρ from a median interval — silently
  assuming regularity is exactly the class of hidden assumption this package
  refuses to make.
- **systematic** — two standard normals drawn **once per species per run**,
  applied as $e^{\mathrm{sys}}_i = a\,g_{\mathrm{abs}} + r\,x_i\,g_{\mathrm{rel}}$.
  This is rank-1: correlation exactly 1 between every pair of points, i.e.
  §3.3's fully-correlated case. Averaging a million samples does not reduce it
  at all, and a pipeline claiming otherwise is caught by data generated here.
  The realized coefficients are recorded in the variable's attrs, so a test
  can verify systematic error was correctly *propagated*, not merely present.

**Quantization** rounds to a configured reporting step, the required
adversarial case: once more than half a window shares one value every
median-based estimator collapses to exactly zero, making every point a
detection. `quantization_floor(δ) = δ/√12` exposes the §2.5 guard constant so
detection tests compare against the same number the generator used.

### 8.6 Clocks, gaps, and platforms

Instruments carry their own `native_rate`, optional `timestamp_jitter`
(schema-bounded under half the nominal interval so the clock can never run
backwards), and optional dropouts. Outages **delete** samples rather than
NaN-filling them — that is what a logger which stops writing produces, and the
resulting irregular timestamps are what §1.1's claim about rolling machinery
must actually survive. Outage onsets may predate the record start, since an
instrument can already be down when logging begins; restricting them to the
record would leave the first samples artificially immune.

Platforms are stationary (scalar lat/lon coordinates) or mobile. A mobile
track is emitted as its **own stream at its own rate**, which is the canonical
§1.2 case: position is a smooth auxiliary field that may be interpolated onto
gas timestamps, while the gases may not be. Putting GPS on the gas clock would
leave that asymmetry untestable. Two track patterns: `random_walk` (constant
speed, diffusing heading — a vehicle's actual behaviour, unlike a
position-space walk which would reverse instantaneously) and `circuit` (a
closed circle, which *revisits* coordinates and therefore produces genuinely
clusterable data for Phase 8).

In `random_walk`, heading increments are drawn as $\mathcal{N}(0,
\sigma\sqrt{\Delta t})$, so the heading's spread after elapsed time $T$ grows
as $\sigma\sqrt{T}$ — the `heading_volatility` parameter $\sigma$ is a Wiener
diffusion coefficient with units rad·s^(−1/2), **not** radians per second. The
$\sqrt{\Delta t}$ scaling is what makes the drive independent of the GPS
sampling rate: sampling the same 400 s span at 1 s, 500 ms and 250 ms yields
mean net displacements agreeing to better than 0.5 %.

Position primitives live in `tsara.core.geodesy` rather than with the
generator, because real ingested GPS (Phase 4) and plume clustering (Phase 8)
need the same metre↔degree mapping and the same track interpolation; only the
*manufacturing* of a fake track is synthetic-specific. Tracks integrate in a
local flat-Earth (equirectangular) approximation using a single constant of
111 320 m per degree on both axes — the WGS-84 equatorial degree of longitude,
$2\pi a/360$. It is deliberately also used for latitude, where the true mean
meridional degree is 111 133 m: the 0.17 % difference is two orders of
magnitude below GPS noise at survey scale, and one constant keeps the
metre↔degree mapping invertible and single-valued. Generated coordinates are
bounded before release — longitude wrapped into $[-180, 180)$, latitude
clamped to $[\pm 90]$ — since offsets are integrated without bound and a track
crossing the antimeridian would otherwise emit 180.08°, which is not a
coordinate and would propagate silently into the ground truth. Polar platforms
are outside the supported domain; the longitude-scale floor keeps the
arithmetic finite there but does not make it meaningful.

**One time representation, everywhere.** Timestamps enter from two directions —
as clocks (`DatetimeIndex`, built in `generator._build_times`) and as event
boundaries (scalar `Timestamp`, born in `plumes.schedule_events`) — and both
are normalized to **tz-naive UTC at nanosecond resolution** through
`tsara.core.timebase`. Both halves matter and both are load-bearing:

* *Timezone.* pandas raises `TypeError` on any comparison between an aware and
  a naive timestamp, so if the catalog kept the config's timezone while the
  clocks were normalized, the harness's central operation — slicing a stream
  with a ground-truth event window to score a detector against it — would fail
  outright on any config declaring a `Z` suffix, which the shipped example
  does. A tz-aware axis also cannot be encoded to netCDF.
* *Resolution.* Left to itself, a clock inherits its unit from the config's
  start (microseconds, for a `datetime.datetime`) while the jitter branch casts
  explicitly to nanoseconds — so one dataset could hold streams at two
  resolutions depending on which instruments declared jitter, and netCDF
  (which stores nanoseconds) would change the dtype on every save/load. The
  catalog is pinned the same way, on both the populated and empty paths, so a
  plume-free control run stays concatenable with a plume-dense one.

The consequence is that tz-aware and tz-naive configs produce byte-identical
streams *and* byte-identical catalogs, and every persisted file carries the
same time representation it had in memory.

**Names are filenames.** Species *and* instrument names must be valid Python
identifiers (`config.base.validate_stream_name`). For species the reason is
that names become `xarray` variables; for instruments it is stronger — they
become `streams/<name>.nc` inside a bundle, so a name carrying a path
separator would send the write into a directory that was never created and
fail only at save time, after a full generate, with a backend error naming
neither the instrument nor the rule it broke.

**Cells.** Since Phase 3.5 every generated stream also carries the interval
each value describes (§10.9). The default reproduces the original behaviour
exactly — one sample per cell, centred on its own timestamp — so existing
configurations emit byte-identical output; `method: mean` is what manufactures
the averaged products most of the real archive contains.

**Known limitation:** plume timing is *not* derived from track geometry —
there is no dispersion model placing sources in space and computing when the
vehicle drives through them. Ground-truth event coordinates are the platform
position at each event's peak, which is what a real mobile catalog records
anyway (a drive-by localizes the *encounter*, not the source).

### 8.7 Persistence

`SyntheticDataset.save/load` implement the CLAUDE.md §5 bundle convention for
the products this phase introduces, establishing the layout later phases
extend: `bundle.json` (contents + format version), `config.yaml` (the exact
config, so a bundle reproduces itself), `ground_truth.parquet` (catalog-shaped,
so ground truth and detections are directly comparable on disk), and
`streams/<instrument>.nc`. Streams self-describe as synthetic in their attrs —
a synthetic file mistaken for a measurement is a scientific hazard.

The module-level entry points are `save_bundle` / `load_bundle`, deliberately
*not* `save_synthetic` / `load_synthetic`: the latter name already belongs to
`tsara.config.loader.load_synthetic`, which reads the YAML *config* describing
a dataset to manufacture rather than the manufactured dataset itself. Both take
a path and return something plausible, so sharing a name would have made the
meaning of a notebook line depend on which import happened to be in scope.


---

## 9. Ingestion (Phase 3)

Ingestion turns a validated `Manifest` into one native-rate
`xarray.Dataset` per instrument. Nothing here resamples anything: streams
stay on each instrument's own timestamps until Phase 4 pairs them (§1.1).

Almost every design decision in this section was settled by measuring the
campaign archive it was built for rather than by reasoning about the format
specifications, and the counts below all refer to that archive: the 2024
tree of **1122 ICARTT files** and the 2026 processed tree of **623 parquet
files**. As of the Phase-3 review, **all 1122 ICARTT files parse, yielding
35,275,729 rows** — a number worth stating because it started at 1054 files
and 30,190,402 rows, and every file and row recovered since came from
letting the data rather than the header settle an ambiguity (§9.2).

### 9.1 The reader seam

Ingestion has two halves with different shapes. Reading a file is
format-specific and irreducibly fiddly; everything after it — masking, unit
conversion, uncertainty resolution, assembly — depends only on the manifest.
The seam between them is the `RawTable` contract: a reader's whole job is

```
(path, loader config) -> RawTable
```

where the returned frame is indexed by **tz-naive UTC nanosecond**
timestamps and keeps every column under **the name the raw file uses**.
Canonical renaming is a manifest concern handled downstream; a reader that
renamed columns would have to be taught the manifest, which is the coupling
the seam exists to prevent. The contract is enforced at runtime for every
reader, TSARA's own and anyone else's.

Phase 3.5 added one optional element to it: a reader **may** return two
reserved columns holding each row's cell boundaries, when the file itself
states them (§10.4). Both or neither — one alone describes no interval — and
they travel as columns precisely so that concatenation, sorting and
de-duplication carry them along with the rows they belong to, for free.

Readers are selected by name from a registry (`@register_reader("csv")`),
the same convention this document fixes for noise and regression estimators.
Registered names: `csv`, `icartt`, `parquet`.

### 9.1.1 Finding the files: path templates, and saying "not that"

Directory and naming conventions are **data, not code** (CLAUDE.md §5). A
path template describes one layout; a loader carries a list of them, because
one instrument's files routinely span several conventions at once. Each
template compiles to a *pair* — a glob to drive the filesystem walk, since
only the filesystem can enumerate what exists, and a regex to harvest
`{field}` values, since a glob cannot report which text a wildcard consumed.
The regex is the stricter of the two and so doubles as a second filter.

That pairing only works while the two halves agree, and there is one place
they silently did not: negation. Glob spells it `[!abc]`, regex spells it
`[^abc]`, and the class was passed through to both untouched — so `[!x]*.csv`
glob-matched `a1.csv` and was then discarded by its own harvesting regex,
which read `[!x]` as a literal `!` or `x`. A correct template reported "no
files found". The glob spelling is now translated for the regex, and the
regex spelling is refused with a message naming the supported one, because
`[^abc]` cannot be made to mean the same thing to both halves.

**Templates are include-only, and that is not sufficient.** Archives
quarantine data in place: the target archive's instrument-aligned stage
keeps rejected files in `bad/` and `bad_timestamp/` subdirectories sitting
directly beneath the good ones, **187 of its 623 files**. A `**` template —
exactly what varying archive depth calls for — sweeps every one of them in
without a word. Per-directory templates avoid it, since `Eng/*.parquet` does
not descend into `Eng/bad/`, but only if the quarantine is already known
about. `_BaseLoader.exclude` therefore takes patterns in the same syntax and
removes what they match, reporting the count at INFO: a run that drops a
third of an archive should say so at a level people read. Excluding
*everything* a template found is reported as its own error, since "the
templates found nothing" and "the exclusions removed everything" have
opposite fixes.

**AppleDouble resource forks (`._*`) are skipped unconditionally.**
`pathlib.Path.glob` matches dotfiles where `glob.glob` does not, so a
Mac-touched archive hands every `*.csv` template a binary `._*.csv` shadow
of each real file, which then fails the read and reports itself as an
unreadable data file — 18 of them in the target archive's aerosol
directories. Only this prefix is skipped, not every dotfile: `._` is
unambiguously macOS metadata, whereas a leading dot in general only means
hidden, and could be data someone deliberately pointed a template at.

**Known gap, deliberately not closed here.** The same archive publishes a
`quality_manifest.yaml` marking individual files `good`/`bad`. 31 of its 34
`bad` entries already sit in quarantine directories, so honouring the file
would add only 3 files beyond what `exclude` catches — too little to justify
a file-level accept/reject mechanism inside Phase 3. It is recorded as an
open design flag instead.

### 9.2 What each reader must get right

**`csv`** — comma, tab, whitespace-run and general regex delimiters;
headerless files whose column names come from the manifest as a *prefix*
(a wide instrument log should not require enumerating every spectral bin);
multi-line preambles; declared missing-value tokens.

Two delimited-text decisions are load-bearing enough to state explicitly.

**Data rows wider than the header.** Two different file shapes produce
records with more fields than the header names, and both are common enough
in real logger output to be handled rather than documented as unsupported.
In one campaign archive surveyed, 21% of delimited-text files were affected;
within each affected instrument family *every* file was, which is the usual
pattern — this is a property of the logger, not of the day.

*The shapes.* Either the logger terminates each record with the separator,
leaving an empty surplus field; or its header is genuinely one or more names
short, with real measurements recorded under no name at all. They look
identical to a parser and differ only in whether the surplus field is empty.

*Why it cannot be ignored.* pandas resolves a header/data width mismatch by
promoting column 0 to the index. Every remaining name then lands on its
neighbour's values and the final column is dropped — silently. A species
read that way reports the channel beside it, with nothing raised anywhere.

*The treatment,* in two parts:

1. `index_col=False`, unconditionally. TSARA always builds its time index
   afterwards from a *named* column, so an inferred index is never wanted
   whatever the file's shape. This stops the shift but, on its own, makes
   pandas discard the unnamed surplus.
2. Surplus columns are **named, not dropped** — `column_N`, continuing the
   convention headerless files already use, so a manifest addresses such a
   column the same way in both cases. An empty trailing field becomes an
   all-NaN column that costs nothing; an unnamed real column is preserved.
   A well-formed file is left completely alone, because supplying an
   explicit name list would also disable pandas' duplicate-name mangling.

The width is measured by tokenizing the header line and the first data line
*separately*. Reading the first few rows with `header=None` does not work on
a file with a preamble: pandas fixes the field count from the first row it
sees, so a two-column preamble above a 25-column table decides the width for
everything below it.

**Known limit:** a file whose *interior* rows are malformed — a logger
interrupted mid-write, or two records run together where a newline was never
emitted — is still rejected in full, because the width is established from
the first data row. Rows like that are rare (in the surveyed archive, 15
lines in 29,522, costing one file of 24) but they cost the whole file rather
than the affected rows. Recovering them needs bad-line handling with a
reported count, which the `icartt` reader already does for ragged rows.

**`header_row` counts lines after blank and comment lines are discarded**,
not physical line numbers. A file with two preamble lines, a blank line, and
then its header on physical line 4 needs `header_row: 2`. This follows from
`skip_blank_lines`/`comment` being applied first and is easy to get wrong by
one; it fails loudly (the error lists the column names actually found), but
the field description says so to save the round trip.

**Float parsing is left at pandas' default**, not `float_precision="round_trip"`.
The default parser is not guaranteed bitwise round-trip exact — it can differ
in the last unit in the last place — and the exact parser is available for
about a third more parse time. Measured on a real campaign archive across
three instrument families, over a million parsed values showed **zero**
differences between the two, so the guarantee buys nothing observable on
real instrument output, whose precision is far coarser than the ULP in
question. Revisit only if a specific dataset is shown to be affected.

**`icartt`** — TSARA's own FFI-1001 parser, because the PyPI `icartt`
package is GPL-3.0 and unmaintained. Owning it also buys tolerance for what
real archives contain: non-UTF-8 bytes, per-variable missing sentinels in
several spellings, ragged rows (skipped and counted), and files that
contradict their own header by declaring seconds-past-midnight and then
writing datetime strings. **The time axis is therefore built from the
values, not the labels** — numeric means seconds past midnight, anything
else is parsed as timestamps — because archives spell that unit at least a
dozen ways, including names that falsely suggest a different epoch.

Keying off the values raises the question of what to do when the values
disagree with each other, and the answer has to be a **majority vote**, not
an existence test. Asking "is *any* value numeric?" lets one token decide a
whole file: two PTR-MS VOC files in the surveyed archive hold ten thousand
datetime strings alongside exactly two numeric tokens that leaked in from a
mis-declared header block, and the existence test sent both down the
seconds-past-midnight branch, where every genuine timestamp then failed to
convert and was discarded. The result was 2 surviving rows out of 10,235,
across 35 VOC species, reported only as a warning. Counting both
interpretations and taking the larger costs a second parse **only for
genuinely mixed columns** — all-numeric and none-numeric short-circuit
first — and ties favour the spec-compliant seconds reading, since a tie
means the evidence does not actually distinguish them.

**`NLHEAD` is checked against arithmetic before it is trusted.** The first
twelve lines are fixed by the format and each of the `NV` dependent
variables needs its own definition line, so any valid header is at least
`12 + NV` lines and a file claiming fewer is provably wrong about itself.
Two archive files declare `NLHEAD = 36` with `NV = 35`; trusting that admits
header text into the data block, which is exactly where the stray numerics
above come from. The floor is raised to `12 + NV` with a warning naming the
arithmetic. This is a partial repair by construction — those files' true
header is longer still (70 lines), so a residue of comment text remains, and
it is the majority vote plus ragged-row skipping that contain it. It is a
no-op for every other file in the archive.

The complementary diagnostic — walking the two comment blocks and comparing
where they end against `NLHEAD` — is logged at **debug**, not warning. On
the surveyed archive it fires on 44 of 1122 files and correctly diagnoses
none of them: 43 are PTR-MS files carrying one extra, blank-named definition
line that offsets the walk harmlessly. A warning that is a false positive
every time it fires teaches its reader to ignore warnings.

**Column names are chosen by the width of the data, not by `NV`.** An
FFI-1001 file states its column names twice — the variable definitions, and
the last normal comment line, which the format designates as the data column
header — and the two disagree often enough to need a rule. The rule "trust
`NV`" is right 1121 times in 1122 and wrong once, and the once is a hard
failure rather than a degradation: a file declaring `NV = 1` with its
independent *and* its single dependent variable both named `Time_UTC` yields
a duplicated name list, which pandas refuses outright with an untyped
exception escaping a reader contracted to raise `TsaraIngestError`, while
that file's column-header line carries the 7 correct names its 7-field rows
need. So the arbiter is the modal field count of the data rows, preferring
the declared header line, then the definitions. Measurement is what makes
this safe: on the 1078 files where *both* lists match the row width their
contents are byte-identical, so the preference is provably content-neutral
across the archive. Where neither matches, the disagreement is about the
rows rather than the names — the uniformly-too-wide case that `index_col=False`
already handles — so the file is not refused. Duplicate names are mangled
`name`, `name.1` as pandas would, since a real ground-site file repeats two
of its own column names and an unmangled list cannot be read at all.

Scale factors are applied *after* missing-value sentinels are masked. The
reverse order turns a `-9999` sentinel into `-9999 * scale`, which no longer
matches the declared sentinel and enters the data as a plausible number.

**`parquet`** — the usual storage for a campaign's processed stages. Because
parquet stores the dataframe index, its `time:` block is optional: the
normal case has no timestamp to parse. Storing the index does not make the
reader trivial, though — stored indexes are timezone-aware and appear at
both microsecond and nanosecond resolution, and neither satisfies the
contract untouched.

### 9.2.1 Two kinds of sentinel: missing versus below detection

ICARTT files declare **two** unrelated families of sentinel, and conflating
them is the difference between a stream that can be analyzed and one that
cannot.

`VMISS` (header line 12, one per variable) marks *missing* data. TSARA has
always masked it, before applying `VSCAL`, for the reason given in §9.2.

`ULOD_FLAG` and `LLOD_FLAG` are declared in the special-comment block and
mark samples that fell **outside the instrument's detection range**. These
are scientifically not missing: a below-LOD benzene is an upper bound, while
a dropout is no information at all. That distinction is real, and for a long
while it was the argument for carrying the flags forward untouched and
leaving the values in place.

Measurement settled it the other way. Over the 2024 archive:

| quantity | value |
|---|---|
| files declaring a numeric LOD sentinel | 913 of 1122 |
| distinct sentinel magnitudes in use | `-8888`, `-88888`, `-8.888e50` |
| LOD sentinel values present in the data | 23,814,123 |
| share of every numeric value in the archive | **10.03%** |
| share within the PTR-MS VOC files | **33-56%** |

For a real PTR-MS record, benzene was 67% sentinel and propyne 70%, which is
far enough past half that the **median of the record was the sentinel** rather
than a concentration. A rolling low-quantile baseline (§6) would therefore
have reported a background of `-88888 ppbv`, and every enhancement ratio
built on it would have been meaningless. Leaving the values in place was not
a conservative choice; it was a silent one.

So the sentinels are masked to NaN alongside `VMISS`, and — because masking
must not destroy the information that motivated keeping them — the flag
values and a **per-variable count of masked samples** travel on into the
stream, where the count is attached to the species it censors as
`n_lod_masked`. A later phase can still substitute LOD/2 or fit a censored
model, which is what the original caveat was protecting, without any file
being re-read.

Three declaration shapes occur and all are handled: one value for every
variable, one value per variable (a list as long as `NV`, matched by
position), and a non-numeric placeholder (`N/A`, `NaN`) meaning the flag is
unused. When a list is neither length the union of the declared values
applies to all variables — the conservative reading, since these values are
chosen precisely to be impossible measurements.

**A parsing prerequisite that had to be fixed first.** The flags were
unreachable even in principle, because header metadata was scraped only from
the two comment blocks, which are located by `12 + NV` arithmetic. The 43
PTR-MS files carry one extra, blank-named variable-definition line, which
offsets that walk; both blocks were then read from the wrong place and the
metadata came back **empty** — on exactly the files with the highest
below-detection fractions in the archive. Metadata is now scraped from the
whole header, since a `KEY: value` line means the same thing wherever it
sits, and a data row (which begins with a numeric time field) can never
match a pattern anchored on an uppercase key. Where `NLHEAD` is *provably*
wrong (§9.2), the scrape extends a bounded distance past it, because
clamping to `12 + NV` recovers a lower bound on the header rather than the
header; two files in the archive keep their comment block past that bound,
and they held the last 19,398 unmasked sentinels.

### 9.2.2 Float precision: what "reading a number" costs

Converting decimal text to a binary double is not free of choices. pandas'
default CSV parser is fast and *usually* exact; it is not guaranteed to
return the nearest double to the digits written. `float_precision:
"round_trip"` is guaranteed, and slower. Both readers that parse text (CSV
and ICARTT) expose this as `float_precision: fast | exact`, defaulting to
`fast`.

The default is a measurement, not a guess. Over a 60-file sample of the 2024
ICARTT archive:

| quantity | value |
|---|---|
| values with 9 or fewer significant digits | 94.7% (parsed identically either way) |
| values with 14-17 significant digits | 5.4% |
| values where the two modes disagree | **0.358%** |
| size of the disagreement | ~1 unit in the last place, **1.2e-16** relative |

End to end on the 43-file PTR-MS instrument, `exact` cost about 20-34% more
ingestion time and changed 0.021% of finite benzene values by at most
9.0e-17 ppb.

So the trade is a fraction of a percent of values moving by roughly thirteen
orders of magnitude less than any instrument's precision, against a
double-digit percentage of the slowest step in the workflow. `fast` is
therefore the default, and `exact` exists because "my ingestion is bitwise
reproducible" is a legitimate thing to need — for a regression test, a
published dataset, or an argument with a collaborator's pipeline — and
because it should be one line of YAML rather than a patch.

Note the asymmetry with *writing*: `tsara.synthetic.export` writes at
`repr` precision, which is always round-trip exact, so a synthetic value is
never lost on the way out. About 41% of noisy synthetic values come back one
ULP away under `fast`, which is why the round-trip tests compare with a
relative tolerance rather than exact equality, and why one of them sets
`exact` and asserts bitwise recovery.

### 9.3 ICARTT revision selection

Archives hold several revisions of one day's data, and ingesting all of them
double-counts the same air. `revision_policy: latest` keeps the newest of
each. Three properties of the real filename convention make this less
obvious than it looks:

1. Revisions are **alphabetic as well as numeric**. Per the specification,
   alphabetic (`RA`, `RB`) is preliminary field data and numeric (`R0`,
   `R1`) is final, so any `R#` supersedes any `R<letter>`.
2. A **trailing comment field** follows the revision and distinguishes
   genuinely different products — processing levels, separate drives on one
   day. It is part of a file's identity, not decoration.
3. `dataID` and `locationID` are **not reliably one token each**, so the
   parse locates the `YYYYMMDD` field rather than counting underscores.

De-duplication therefore keys on `(everything-before-the-date, date,
comment)`. Keying without the comment collapses distinct products into one
another and silently discards real data.

**The blind spot, and why it reports rather than decides.** Selection can
only compare files whose names carry a `YYYYMMDD` token; a name without one
is kept unconditionally, since an unparseable name is not evidence of
duplication. But 147 of the 1122 names in the surveyed archive have no date
token, and 39 of those basenames exist in two or three directories at once —
a dated directory, a `Calibrated Data/` directory, and a `Calibrated Data
(Updated)/` directory holding the same filename. A recursive template then
ingests every copy. Whether that is triple-counted air or three genuinely
distinct products is a question only the data owner can answer, so the
selector warns and names the repeats instead of guessing.

### 9.3.1 Row loss as an error, not a warning

Every reader discards rows it cannot place on a time axis, and every reader
already refused a file where *no* row survived. That left the gap exactly
where it hurts most: a file yielding 2 rows out of 10,235 is a misparse, but
it produced only a warning, and three stages later it is indistinguishable
from "this instrument barely ran that day". Warnings scroll past in a run
over a thousand files.

`LoaderConfig.max_dropped_fraction` (default `0.5`) generalizes the
all-or-nothing rule to a threshold, applied by a single helper shared by all
three readers so the policy cannot drift between formats. The default has
wide headroom by measurement rather than by assumption: with the parsing
fixes above in place, the worst-affected file in the surveyed archive loses
0.29% of its rows and only four files lose anything at all, while the
pathology the threshold exists to catch loses 99.98%. Setting it to `1.0`
restores warn-only behaviour for archives where heavy loss is expected.

### 9.4 Order of operations per variable

Fixed, and each step depends on the previous one:

| Step | Why here |
|---|---|
| convert units | so everything downstream reads canonical numbers |
| wrap a `circular` variable into [0, 360) | a direction's canonical value is on one turn |
| apply QA/QC | bounds are written in the units the author thinks in |
| resolve uncertainty | `absolute` is declared in canonical units (§2.2) |

`range` bounds are physical statements in canonical units — the shipped
example converts ppm→ppb and then bounds in ppb, so masking before
conversion would compare ppb bounds against ppm numbers and reject the whole
record. `flag` reads a separate instrument status column, which is never a
converted quantity.

**The wrap is a step of its own because a conversion can carry a direction
across north.** The commonest conversion for a direction is an offset — a
magnetic bearing to a true one, +10.3° at Salt Lake City — and the most
natural QA/QC rule for a direction is `range: [0, 360]`. Without the wrap,
352° becomes 362.3° and the range rule masks it, along with every other
reading within 10.3° west of north: a whole sector of the wind rose deleted,
silently, rather than a random fraction of the record. The permitted 2026
archive carries magnetic and true directions side by side
(`Wind Direction (Deg Mag)` beside `wind_dir_true` on the Wyoming van,
`mag_dir_deg` and `mag_course_deg` on the LANL instruments), so the conversion
is one a manifest will plausibly declare. Found in the Phase-4 walkthrough;
before it, ingestion wrapped nothing, and a logger's `-5` or `365` also
reached the stream unwrapped. The wrap is exact and changes no direction, so
it is a consequence of the declaration rather than a model, and it applies
only to variables declaring `circular: true`.

Rules **mask rather than delete**. Rows are the instrument's clock, and a
rolling window that closes over a removed sample computes a different answer
than one that sees a gap; "no valid measurement" and "no measurement
attempted" must stay distinguishable. Counts are reported per rule, because
a range rule masking everything means the bounds are in the wrong units
while a flag rule masking everything means the polarity is inverted, and one
combined number cannot tell them apart.

A `flag` rule must list at least one value. An empty `good_values` validates
trivially and then masks the *entire* record, since nothing can be a member
of an empty list; an empty `bad_values` is a rule that looks active in the
manifest and does nothing. Both are refused at config load, on the same
principle that refuses the identity `UnitConversion`.

**One ordering constraint lives inside the uncertainty step, not between the
steps.** A `reported` sigma column is checked for negative values — almost
always an undeclared `-9999` missing-value sentinel — and that check must
run *before* the unit conversion is applied to it. `convert_spread` takes an
absolute value, correctly, because a negative `scale` is a legitimate
sign-convention flip whose magnitude must survive; so a negativity test
applied afterwards has nothing left to find. Ordered the wrong way, the
guard protected only variables with no conversion — failing precisely where
the manifest was doing more work — and a `-9999` under a ppm→ppb conversion
entered the budget as a silent 9,999,000 ppb "1σ". For a random component
that drives the point's inverse-variance weight to zero; for a systematic
component, combined as a weighted mean of sigmas (§3.3) rather than in
inverse variance, a single such value dominates the entire bin.

### 9.5 Why there is no spike rule

There was one, and it was removed on 2026-08-26 (owner decision, Phase-3
walkthrough). It is documented here rather than deleted silently, because
"we considered an outlier filter and rejected it" is a methodological
statement a reader of this package needs.

The rule was a centered rolling median/MAD (Hampel) test, thresholding at
`n_mad` times the raw MAD, intended for sub-second electronic glitches and
kept deliberately distinct from plume detection. The problem is that its
operating definition — *a short excursion, large relative to a local robust
scale* — is also the definition of a plume in mobile trace-gas data. The two
are not merely similar; on this data they are the same test.

Measurement settled it. On real 2-second analyzer records:

| window | windows with zero MAD | plume samples masked | quiet samples masked | enrichment |
|---|---|---|---|---|
| 5 s | 71.2 % | 1.26 % | 1.11 % | 1.14× |
| 11 s | 49.7 % | 3.88 % | 3.61 % | 1.08× |
| 31 s | 14.3 % | 6.40 % | 4.92 % | 1.30× |
| 61 s | 5.4 % | 6.85 % | 2.62 % | **2.61×** |
| 300 s | 0.0 % | 5.88 % | 2.82 % | 2.08× |

The window is bounded on both sides and the safe range between them is
narrow and undiscoverable. Too short, and the rolling MAD degenerates to
zero across most of the record, so the rule silently declines to test
anything while still reporting a plausible masked count. Too long, and it
masks plume samples at up to 2.6× the quiet-air rate — it has stopped being
a glitch filter and become a plume clipper.

The decisive number is the width of real features. In the same records,
**27–29 % of clear enhancement events are two samples wide or fewer**, and
even among events exceeding 100σ over baseline, 18 % are that narrow. A
filter tuned to reject 1–2 sample excursions cannot distinguish a glitch
from the signal this package exists to find. No choice of `window` and
`n_mad` escapes that, because the ambiguity is in the data, not the
parameters.

What replaces it: nothing, deliberately. Genuine instrument glitches are
better rejected where the information to identify them actually exists — an
instrument status `flag` column, a physical `range` bound, or a later stage
that already knows what a plume looks like and can judge an excursion in
that context. An outlier filter that runs *before* anything understands the
signal is guessing.

A consequence worth noting: with the rolling rule gone, every remaining
QA/QC rule is pointwise, so QA/QC no longer depends on record order at all.
Sorting is purely the orchestration stage's concern.

### 9.6 Uncertainty at ingestion, and what it refuses to invent

*(Phase 3.5 added one thing ingestion may now do with a declared budget:
move it onto the cells it describes, when — and only when — the manifest has
supplied the timescale that makes the correction knowable. See §10.8.)*

Ingestion knows the manifest; it does not know the analysis config. So it
computes exactly the budgets a manifest can state — `declared` and
`reported` — and **labels** everything else. The empirical estimator's name
and window belong to `DetectionConfig` (§2.5), so computing it here would
mean reading a config this stage has no business reading. The obligation is
recorded instead, which is the shape of §2.3's promise.

A `reported` column is scaled by `convert.scale` and never by
`convert.offset`: an uncertainty is a difference on the axis, so the origin
cancels. Applying the offset would add 273.15 to every sigma in a °C→K
conversion. A negative reported sigma is masked — in practice an undeclared
missing-value sentinel rather than a real spread.

### 9.7 Assembly, and the substitutability requirement

A stream built from an archive must be shaped exactly like one the generator
manufactures, because every later phase consumes both through one code path
and synthetic truth is the only correctness arbiter available (§9.9). The
variable-name convention (`sigma_rand_<name>`, `sigma_sys_<name>`) therefore
lives in one module both producers build from, rather than in two matching
string literals — a coupling that would break silently, since a rename would
not fail anything until a later stage found no sigma and fell back to an
empirical estimate, which is a *plausible* answer rather than an error.

**Cells.** Assembly is also where the boundaries resolved in orchestration
stop being table columns and become the CF representation: a `time_bnds`
coordinate, a `cell_methods` string on every variable, and attributes saying
what the support is and how each part of it was established (§10.2, §10.4).
The time axis becomes the cell midpoint here, and the declared clock
correction has already been applied upstream, before ordering, because
centring can reorder a stream when widths vary (§10.2).

**Platforms.** A stationary site has one position, so attaching it to any
clock is exact and free. A mobile platform's position lives on the GPS
instrument's clock, and putting it onto a gas instrument's clock is
*interpolation* — permitted for smooth auxiliary fields, but only under the
`max_interp_gap` guard, which belongs to Phase 4 (§1.2). Ingestion therefore
loads GPS as an ordinary stream, records the binding in attrs, and leaves
the join to the stage that owns the guard, so the interpolation rule stays
enforced in exactly one place.

### 9.8 Orchestration

`crawl → read → concatenate → sort → de-duplicate → assemble`.

**Concatenate before assembling.** QA/QC windows and uncertainty are
campaign-level quantities; evaluated per file they would give a different
answer at every file boundary, so an archive split into hourly files would
mask differently than the same data in daily files.

**Sort.** Files crawled across several directory layouts arrive in path
order, not time order, and an instrument's own timestamps cannot be assumed
sorted either — logger clock corrections, buffered writes and merge steps in
an upstream processing chain all produce records that step backwards
occasionally. Everything downstream assumes a monotonic axis.

**Duplicate timestamps keep the first, and say how many were dropped.** A
policy, not a truth: overlapping files may genuinely disagree, averaging
would silently invent a value, and erroring would reject archives that
legitimately overlap.

The warning **names the cause instead of guessing it**, because the two
causes call for opposite responses. Measured on the 43-file PTR-MS set, all
7,242 dropped rows were duplicated *within* a single file and none came from
overlap between files — while the message asked "Overlapping files?", which
points at the crawler and the revision policy, both innocent. Within-file
duplicates mean the instrument wrote two records under one timestamp (there,
a nominally 1 Hz logger with 1 s resolution, whose duplicate rows carry
genuinely different values), so the remedy is a resolution or averaging
decision; overlap between files means the archive really does hold the same
period twice, and the remedy is in the manifest's path templates.

The split is counted **per file, on the axis the rows are actually dropped
on**, and both halves of that sentence are load-bearing. Per file, because
after concatenation nothing else can attribute a duplicate. On the final
axis, because centring is not a translation once cells have per-row widths —
a wide cell's timestamp moves further than its neighbour's, so two rows that
shared a raw timestamp and declared different stops land on *different*
midpoints and stop being duplicates. Counting the within-file half before
centring and the total after it mixed two axes: the first could exceed the
second, and the reported overlap came out **negative**, telling the reader to
go and check the path templates of a single-file instrument. Counted on one
axis the split is exact and provably non-negative — for an instant held by
`c_i` rows in each of `k` files, the total is `sum(c_i) - 1` and the
within-file part is `sum(c_i - 1)`, so the remainder is `k - 1`: the number
of *extra files* holding that instant, which is what "overlap" means.

**Cells are centred before the record is sorted.** Ingestion moves each
timestamp onto its cell's midpoint (§10.2), and with per-row widths that can
genuinely reorder a stream: a 100 s cell starting at `:00` is centred after a
10 s cell starting at `:10`. Sorting the raw axis and centring afterwards
therefore leaves a non-monotonic stream, which assembly refuses — failing an
archive that is perfectly legitimate. So the sort must see the axis the
stream will actually carry.

**What a file said about itself reaches the stream.** A reader returns the
file's own declarations — an ICARTT header's PI, mission, revision, platform
and LOD flags — in `RawTable.attrs`, and orchestration reconciles them across
an instrument's files: keys that agree are carried through, keys that
disagree are *joined rather than picked*, since silently choosing one of two
PIs would put a false statement into a product whose purpose is to be
self-describing. Past a threshold the disagreement is summarized as
`first ... last (N distinct values)`, because some keys differ in every file
by design and a thousand-file instrument would otherwise write an attr that
is useless as provenance. Counts are summed instead of reconciled, a tally
over files being exactly the tally over the concatenated record.

**A bundle does not accumulate streams that are no longer its own.**
`ingest_campaign(..., instruments=[...])` exists so a campaign can be re-run
for a subset, and saving that subset over an existing bundle would otherwise
leave the previous run's stream files behind — nothing misreads them, since
the loader takes its list from `bundle.json`, but the directory would then
contradict its own descriptor.

**A file that will not read is logged and skipped; an instrument that loses
every file is an error.** Aborting a campaign on the first bad file is the
wrong trade for a few thousand files on a cluster.

### 9.9 The round-trip harness

The only check that bears on whether ingestion is *correct* rather than
self-consistent. `export_raw()` writes a generated dataset as raw CSV plus
the manifest describing it — using the same `TrueUncertainty ->
UncertaintySpec` seam the generator was built with — so that synthetic data
travels the road real data does: written to files, crawled, parsed,
converted, masked, reassembled. The generator's answer key then supplies
expectations that ingestion had no part in writing.

**How strong is it? Measured, by mutation.** Five realistic bugs were
injected into `tsara.ingest` and the round-trip file was run alone against
each. In its original form it caught **one of five**. The full suite caught
all five, so nothing was broken — but the harness was weaker than its own
docstring claimed, and three of the four misses had a single cause: a
default export declares no unit conversion, because it writes every species
in its own canonical units under its own name. There was nothing to convert.

`export_raw` therefore takes `raw_units`, which writes a species in
*non-canonical* units and declares the conversion back — the shape a real
archive has, instrument units on disk and canonical units after the
manifest. With it, the harness catches **three of five**, and additionally
catches a bug that compares QA/QC bounds before conversion instead of after.

Two misses remain, and both are deliberate:

* **Duplicate-timestamp policy.** Exercising it would require the exporter
  to fabricate overlapping files, which is a property of an archive rather
  than of an instrument. Campaign-level unit tests cover it (§9.8).
* **The nanosecond resolution pin.** Unobservable here, because CSV
  timestamp parsing already yields nanoseconds. The pin is defensive code
  guarding a path this harness cannot reach.

**One non-obvious requirement.** A test conversion must carry both a scale
and an offset. With a zero offset, `value * scale + offset` and
`(value + offset) * scale` are the same function, so an ordering bug in
`convert_values` survives; with a non-zero offset it does not. This was
found by mutation, not by reading the code.

**What the conversion path is really for.** Beyond conversion itself, it is
the only end-to-end check of the asymmetry in §2.2: a declared `absolute`
sigma is *already* in canonical units and must survive untouched, while a
reported sigma column is in the file's units and must be scaled (and never
offset — an offset shifts a measurement, not its spread). That asymmetry is
where the Stage-6 sentinel bug lived, and it was previously verified only
against hand-written expectations.

Only observable variables are exported; the `truth_`-prefixed answer key
stays behind, and a test asserts no exported header contains it.

**A second mutation round, on temporal support**, was run in Phase 3.5 and is
recorded in §10.9. It scored three of five and reached five of five after two
fixes — one of which was the same defect this section describes, in a new
place: a fixture that only ever wrote UTC could not test a reader that ignored
the declared timezone.

Two limits stated honestly:

- **CSV only.** A round trip constrains the reader only when the writer is
  trivially correct. An ICARTT writer would be more TSARA-authored code, and
  a trip through it would show writer and reader agreeing with each other
  rather than either matching FFI-1001.
- **Values compare to ~1 ULP, not bitwise.** pandas' default CSV parser is
  not round-trip exact. Far below any measurement resolution, and fully
  deterministic, but not zero.

The harness also exposed a genuine asymmetry worth recording: a *relative*
uncertainty term is a fraction of something, and the two sides necessarily
choose differently. The generator scales the **true** signal, because that
is what produced the error it injected; ingestion can only scale the
**reading**, because a manifest describes a file and the true value is
exactly what is unavailable. The two agree to the fractional size of the
error itself — second order, and the standard reading of "percent of
reading" in an instrument specification.

---

## 10. Temporal support (Phase 3.5)

TSARA's first data model treated every timestamp as an instant. Most of the
target archive labels **intervals**: the whole NOAA stationary suite publishes
1-minute means, the iWAS canisters integrate for roughly 15 s each, and the
1 s ICARTT records carry a start time by specification. Treating a 1-minute
mean as an instant invents 59 s of resolution the instrument never had, which
is the same error as interpolating a gas, stated more generally. One principle
covers both:

> **TSARA never evaluates a value on a finer support than it was delivered
> on.**

Everything in this section is the bookkeeping that makes that principle
checkable.

### 10.1 Vocabulary

A **cell** is one row of a stream: a value together with the time interval of
air it describes. That interval is its **support**. Three facts describe it:

| Fact | Question it answers |
|---|---|
| **width** | How long an interval? |
| **label** | Where in the interval does the file's timestamp sit — start, mid, end? |
| **method** | Is the value an *average over* the interval, or a *sample within* it? |

The three are established independently and are therefore recorded
independently (§10.3).

### 10.2 Representation: CF cell boundaries

Cells are stored in the Climate and Forecast convention rather than in a TSARA
invention, so a saved stream is readable by `ncview`, CDO and `cf_xarray`
without a translation layer. Per stream:

- a `time_bnds` coordinate of shape `(time, 2)`, holding each cell's start and
  stop;
- `time.attrs["bounds"] = "time_bnds"`, plus `standard_name` and `axis` so a
  generic CF tool can find the axis by role rather than by our choice of name;
- `cell_methods = "time: mean"` or `"time: point"` on every time-varying
  variable **except** the `sigma_rand_`/`sigma_sys_` companions.

The exception is a correctness matter rather than a fastidious one.
`cell_methods` says what operation produced a value *from* its cell, so
`time: mean` on `sigma_rand_ch4` asserts the stored number is the mean of the
random sigmas over that cell. It is not: a sigma describes the uncertainty
*of the cell's value*, and where that value is an average the two differ by
exactly √N_eff (§3.4) — the factor that makes averaging worth doing at all.
Stamping `time: mean` on a sigma would put a false claim in the file, wrong by
the one quantity the two-component design exists to track.

The systematic companion happens to satisfy `time: mean` exactly, since a
fully correlated error does not average down and every within-cell value is
the same number. It is excluded anyway: true by coincidence is not a reason to
assert it, and stamping one string on both invites a reader to treat two
components that behave oppositely under averaging as though they were alike —
the one confusion the two-component design exists to prevent. What the
companions *are* is carried by their names and by `uncertainty_component`; a
cell method is the wrong vocabulary for it.

**A bounds variable carries no units of its own.** CF says it inherits its
parent's, and xarray implements that — `time_bnds` is written with no `units`
attribute and encoded with whatever `time` was pinned to. So the two cannot
disagree about an epoch. `TIME_ENCODING` is pinned for a different reason,
measured: an unpinned datetime coordinate that has a bounds variable picks its
units from the data (`"minutes since 2024-07-01 00:01:00"` on a four-row
stream), making a file's resolution depend on when its record starts. And
because the bounds inherit that pinned unit, a bounds array stored coarser
than nanoseconds is written as the NaT sentinel for every value, with no error
anywhere — so `pin_time_encoding` widens before it pins.

**`time` is the cell midpoint on every stream.** Left at whatever each file
happened to use, `time` would mean a start on one instrument and an end on
another, and every operation that is not bounds-aware — a plot, a `sel`,
someone else's code — would carry up to a full cell of silent bias. Centred,
the worst case is half a cell and it is unbiased. Nothing is lost: the
original label is in `tsara_support_label` and the boundaries are exact.
Note that centring can *reorder* a stream when widths vary per file (a wide
cell starting just before a narrow one ends up with the later midpoint), so
sorting must follow it, not precede it.

**Bounds are always present, including for `point` data.** CF permits this
explicitly, while being clear about what it means: for point data "the cell is
irrelevant to the data and the bounds are arbitrary. Nonetheless, the bounds
may still be included." So a `point` stream's bounds are TSARA's tiling
convention, not an instrument claim — which is exactly what the per-field
provenance says out loud. Zero width is never used: measure zero means zero
weight in every overlap, so such a cell would vanish from the analysis without
a word while still sitting in the stream looking like data.

One real instrument does declare stop equal to start, on 644 of the 90,673
rows of the TwinOtter AMAX-DOAS record, 0.71 %. Those cells are widened and
the count is written to `tsara_cells_widened`, so the repair is visible rather
than silent. They are widened to the **median of the file's own other cells**,
not to the spacing between cells: a cell narrower than that spacing is not a
defect but the definition of a duty-cycled instrument, and repairing at the
spacing would inflate a 15 s canister fill to the ten minutes between
canisters and quietly claim the sampler had been collecting throughout. The
spacing is used only for the degenerate file whose cells are *all* zero-width,
where nothing else is available.

**Operational rule: never `resample`, `rolling` or `coarsen` a stream carrying
bounds.** Measured, xarray fails on this in two ways and raises on neither.
Stored as a coordinate, which is TSARA's layout, the bounds variable is
*dropped* while `time.attrs["bounds"]` goes on naming it — a dangling CF
reference. Stored as a data variable, the boundary timestamps are *averaged*
into cells no instrument measured. It also silently coarsens the time axis
from nanoseconds to microseconds. `tsara.core.support.check_bounds_intact`
turns that rule into a check, applied at every persistence boundary.

**Two absences that look identical on disk and mean opposite things.** Bundle
format version 2 is what added cells, and a version-1 bundle is *migrated*
rather than refused: the older layout has an exact honest reading — cells of
the record's own nominal cadence, centred on each timestamp, label and method
`assumed` and width `inferred`, since the cadence really was measured. But a
**version-2** stream without `time_bnds` is not missing them. It means
ingestion could not know: no declared width, and no file long enough to
measure a cadence, and it said so rather than inventing a number (§10.5).

Both loaders ran the migration on every stream regardless of version, so an
instrument whose files hold one row each — no measurable cadence per file, but
three timestamps a minute apart once concatenated — came back from its own
bundle with 60 s cells and `tsara_support_width_source` moved from `assumed`
to `inferred`. A stream was promoted up the provenance ladder by nothing but a
trip through disk, and the log line announced it had been "written before cell
boundaries existed" about a bundle written moments earlier at version 2. The
migration is now gated on the format version, which is the only thing that
distinguishes the two absences.

### 10.3 `point` versus `mean` is a claim about arithmetic

A cavity ring-down analyzer is not a point sampler. Gas in the cavity is a
mixture of what entered over the previous seconds, species are measured
sequentially in a cycle, and a `_Sync` log is the analyzer's own resampling of
asynchronous fits onto a grid. But it is not a box-car mean either. So the
line TSARA draws is operational rather than physical:

- **`mean`** — an explicit averaging operation over a *known* interval was
  performed. The value times the width is the integral over the cell. This
  licenses interval arithmetic.
- **`point`** — the value is a sample, whatever instrumental smoothing lies
  beneath it.

Declaring `point` is therefore a *guard*, not a shrug: it is what stops a
later stage rescaling a 1 s precision figure onto a 2 s cell as though
averaging had happened — a duty §10.8 leaves explicitly with that stage. Instrument response and cavity residence are
deliberately not modelled (§10.9); they are a deconvolution problem, not a
cell method.

The distinction is not academic and it is not TSARA's invention. The
stationary Picarro's own header states both cases in one sentence:

> Species are measured at 0.5 Hz and reported as 1-minute averages for
> stationary measurements.

The same analyzer is a point sample on a drive and a genuine 1-minute mean
when stationary.

### 10.4 The provenance ladder, recorded per field

Support is resolved the way the uncertainty budget is (§2.4): a manifest says
what it can, the file says what it can, TSARA measures the rest, and every
answer is labelled.

| Rung | Meaning |
|---|---|
| `reported` | Per-row boundary columns in the file itself |
| `declared` | The manifest states it |
| `inferred` | TSARA read it from the file (column names, measured cadence) |
| `assumed` | Nothing said; a default applied and labelled |

Recorded **per field** — `tsara_support_label_source`,
`tsara_support_width_source` and `tsara_support_method_source` — because the
three are established independently. A stationary analyzer whose file carries
a stop column and whose manifest declares `method: mean` is honestly reported
/ reported / declared, and one label per stream could not say that. Beside
them the stream records the answers themselves: `tsara_support_label`, and
`tsara_nominal_cell_width_s` where one nominal width applies to the whole
instrument. The method needs no attribute of its own — CF already has the
vocabulary, and it is on every variable as `cell_methods` (§10.2).

**Nothing is reconciled by vote.** A manifest declares support per *loader*,
so every file of an instrument shares whatever it says, and the two things
that genuinely vary file to file are handled more strictly than a majority
would handle them. A **width** measured per file stays per row, and the
stream reports no single nominal width rather than an average nobody could
use. A **label** inferred from an ICARTT independent variable is used only
when every file agrees — one disagreement, or one file offering no hint at
all, drops the whole instrument to `unknown` and a centred cell, since
picking a winner would put half the files half a cell out. An instrument is
only as documented as its least documented file.

(An earlier draft reconciled a *list* of provenance values by taking the
weakest of them. It was removed when the audit in §10.9 showed nothing ever
built such a list: support being per loader, there was never more than one
value to reconcile.)

Defaults admit undeclared files rather than refusing them: an unknown label is
treated as centred, which minimises the worst-case misplacement, and an
undeclared method is `point`, which claims nothing.

### 10.5 Width is a property of the measurement, not of its neighbours

When the manifest declares no width, it is inferred from each file's own
**median** sampling interval and applied to every row. The median rather than
the mode because the mode needs a rounding resolution chosen in advance, and
that choice is wrong for at least one real instrument either way: exact-mode
agreement is 100 % on a Picarro but only 12 % on a jittery 10 Hz GPS, while
both have a well-defined median.

Measured per file, and *before* the record is sorted — which is the only
point at which the per-file boundaries still exist. That ordering is safe
rather than merely convenient: 22 of the 1122 ICARTT files step backwards at
least once, 70 steps in all, spread across four unrelated products — a 1 s
PTR-MS, the 10 s and 50 s GPS logs, a met station and a drive Aeris — and in
**every one of the 22** the median of the positive intervals is identical
before and after sorting. A median over intervals is unmoved by a minority of
out-of-order rows for the same reason it is unmoved by a 23-day gap.

The rejected alternative — width as the distance to the next row — was
rejected on measurement, not on taste. Every sampling interval in the archive
was classified: all 1122 ICARTT files of `Data/2024` plus the 460 parquet
files of the 2026 aligned stage that lie outside its quarantine directories,
1582 files in all, read through TSARA's own readers. The rule is stated here
so the numbers can be re-derived: a file's cadence *c* is what TSARA itself
infers, the median positive interval; for each interval, *r* = Δt / *c*; it is
jitter when |*r* − 1| ≤ 0.25, a dropped row when *r* ≥ 1.75 and *r* is within
0.25 of an integer, and something else otherwise. Per-file fractions,
averaged over files:

| What the interval is | Mean fraction |
|---|---|
| Jitter around the file's one nominal cadence | 0.97724 |
| A dropped row (an integer multiple of the cadence) | 0.01113 |
| Genuinely something else | 0.01164 |

So 98.8 % of all intervals are explained by one cadence per file, and the
residual is not spread thinly across the archive but concentrated in a few
products: 137 of 1582 files exceed 1 % "something else", and **119 of those
137 are three sub-second 2026 instruments** — the LANL GPS at 21.7 %, whose
stamps are ≈0.1 s apart and heavily jittered, and the two LANL Aeris
directories. Restricted to the 1122 ICARTT files the residual is 0.585 %,
with 18 files above 1 %. The rule is therefore well founded for the gas
records this package exists to analyse, and its known weak case is a
sub-second product whose own timing is irregular. So:

> **A dropped row leaves a hole in the tiling. It never produces a wider
> cell.**

Under the rejected rule the widest single cell would have been **23.3 days** —
a 10 s nominal GPS record whose file spans a campaign — and 84 of the 1582
files have a gap exceeding 100× their cadence. Each stream therefore carries a
duty-cycle diagnostic, `tsara_cell_coverage`, the share of the record's extent
that cells actually cover: median 1.0001 archive-wide, with 78 files below
0.95 and one at 0.024.

Cadence is measured **per file**, not per instrument, and that is
load-bearing: some met records in the archive run at 1 s in one file and 5 s
in another, so a single instrument-wide cadence would give a whole file's
worth of rows the wrong width. It is measured in orchestration, the only place
the per-file boundaries still exist, and resolution runs before ordering
because the per-row width array is built in concatenation order.

A **declared** width gets one check the schema cannot perform, because a
manifest is validated before any file is read: if it exceeds the interval
actually measured between samples, the cells overlap and TSARA says so.
Measured on a 60 s record declared as 120 s cells, every adjacent pair
overlaps, the record reports 182 % coverage of itself, and binning anything
onto those cells counts each sample twice. It is a warning rather than a
refusal, since overlapping integrations are physically possible even though
none appear anywhere in the archive, and refusing would block a record TSARA
had merely misjudged. A width *narrower* than the spacing gets no warning at
all: that is the definition of a duty-cycled instrument, not a mistake.

The ladder is self-consistent, which is the strongest argument for it: the
instruments where cadence inference is *invalid* are precisely the ones that
declare their own boundaries. The iWAS canisters match their modal cadence on
between 3 % and 64 % of intervals and have coverage between 0.020 and 0.914 —
and they publish start, stop and mid columns. Inference never has to cover the
case it cannot do.

### 10.6 What is inferred, and what deliberately is not

**Label** is inferred from evidence a reader can justify. For ICARTT that is
the name of the independent variable. The specification says it is a start
time and most of the archive agrees, but two instruments publish `Time_Mid`
and one vendor writes `TIMESTAMP_UTC`, which is the analyzer's own resampling
grid rather than a specification start at all. So the *name* is read rather
than the specification trusted: a name saying `start` or `mid` justifies a
label, anything else yields `unknown` and a centred cell. Across all 1122
ICARTT files of the 2024 archive:

| Independent variable | Files | Inferred label |
|---|---|---|
| `Time_Start`, `starttime_UTC`, `StartTime_UTC`, `StartTime_seconds`, `iWAS_Start_UTC` | 700 | `start` |
| `Time_Mid` | 20 | `mid` |
| `TIMESTAMP_UTC`, `Time_UTC`, `IgorTime`, `TIMESTAMP`, `N46_Time_UTC` | 402 | none — centred |

So 64 % of the archive justifies a label from its own file and 36 % does not,
and the 36 % is not a long tail of oddities: it is five ordinary names that
simply do not say where the timestamp sits.
Assuming `start` everywhere because the specification says so would put a 30 s
error on every cell of the minute-average suite, in the direction nobody would
think to check. A hint is used only when every file of an instrument agrees;
picking a winner would put half of them half a cell out. What the file itself
suggested travels into the stream as `tsara_support_label_hint`, beside the
label actually used and its provenance, so a manifest overriding a file can be
seen to have done so.

A file that states every boundary needs no label inference at all: where its
index falls between the boundaries *is* the label. That is why an independent
variable named `Time_Mid` needs no special handling anywhere.

**Method is never inferred.** Scanning all 1122 ICARTT headers for three
independent signs of averaging:

| Evidence | Files | Instrument groups |
|---|---|---|
| Prose keyword declaring an average | 288 | 15 of 152 |
| A companion dispersion column | 0 | 0 |
| A companion sample-count column | 0 | 0 |
| Text saying "instantaneous" | 0 | 0 |

Prose is the *only* evidence that exists, and a regular expression over
English is not a basis for deciding whether a number may be treated as an
integral. Where a header does mention averaging it is surfaced as provenance a
human can read. Undeclared means `point`, marked `assumed`.

**Boundary columns are never guessed.** A file naming `Time_Stop` almost
certainly means it, and "almost certainly" applied to the wrong column
silently produces wrong cells for a whole campaign — exactly the failure this
phase exists to prevent. The manifest names them; candidates found in a file
are reported in `tsara_boundary_column_candidates` so a user can see what is
available. **169 of the 1122 files carry one**: `Time_Stop` on 123,
`StopTime_UTC` on 35 and `iWAS_Stop_UTC` on 11, concentrated in the stationary
suite (139 files), the mobile drives (30) and the Twin Otter (20). That is
15 % of the archive whose exact support is stated in the file and reachable
today by adding one line of YAML.

Because the list is read by a human deciding what to write in `start_column`
or `stop_column`, every entry has to be something they can actually write
there — and two kinds of entry were not:

- **Midpoint columns.** A midpoint fits neither field, so offering one invites
  the single manifest entry that is silently wrong: `stop_column: Time_Mid`
  halves every cell, and a halved cell looks exactly like a correct one.
  Excluding them costs nothing measurable: of the 84 files carrying a mid
  column, 64 carry a real stop column beside it and the other 20 are the files
  whose *independent variable* is itself `Time_Mid` — where the "candidate"
  offered was the file's own time axis, handed back to its owner.
- **TSARA's own reserved columns.** The list was read from the frame *after*
  the declared boundaries had been attached as `_tsara_time_start` /
  `_tsara_time_stop`, and the second of those matches any rule that looks for
  a stop. Every instrument that had already been configured correctly was
  therefore told, in its own provenance, that it might like to name a column
  TSARA invented. The list is now taken before they are attached, so it
  describes the file rather than the reader.

### 10.7 Clock offset is not support

Support says what interval a value describes. **Offset** says how the
instrument's clock relates to air-at-inlet time. They are separate concepts
and are kept apart deliberately, because on real data the offset is the larger
error: the 2026 campaign's own alignment stage applies whole-second lags of
1–6 s between analyzers, a −9 s inlet lag, and logger clock drift of up to
13.6 s, while label ambiguity on 1 s data is at most 0.5 s.

`InstrumentConfig.time_shift` is a signed duration **added** to every
timestamp and to both cell boundaries, so a shift moves a cell without
resizing it. A logger running 5 s fast takes `-5s`; a 9 s inlet lag also takes
`-9s`. It defaults to nothing applied, which is right for an archive already
corrected upstream, and the applied value is written to `tsara_time_shift` so
a double correction is visible rather than silent — the one failure mode
internal consistency cannot reveal, since both corrections are individually
right. Estimating a shift by cross-correlation is a future registered
estimator, not this field.

### 10.8 Uncertainty at a support

`DeclaredUncertainty.at_width` records the averaging interval a declared sigma
was quoted at, for the ordinary and easily-mishandled case of a spec-sheet
precision stated at 1 s being applied to 1-minute means.

**Ingestion records that declaration and acts on nothing.** The figures are
stored exactly as the manifest states them; the variable carries
`uncertainty_at_width` (the interval they describe) and
`uncertainty_at_width_ratio` (the median cell width divided by it — the $N$
below, not $N_{\mathrm{eff}}$), and a mismatch is warned about by name at
ingestion time.

That boundary is the same one drawn for the empirical noise estimator (§2.3)
and for the closure diagnostic, and it is worth stating why it is not the same
as a unit conversion. A unit conversion is a declared scale and offset: exact,
invertible, and assumption-free, so ingestion applies it. Moving a sigma from
one interval to another is not. It needs a decorrelation timescale, an AR(1)
model of how the error forgets itself (§3.4), and the assumption that
averaging is what produced the cell — three modelling choices, none of them
stated by any file. With $\rho_1 = e^{-w/\tau}$ for a quoted interval $w$ and
a cell holding $N = W/w$ of them, the §3.4 form gives $N_{\mathrm{eff}} =
N(1-\rho_1)/(1+\rho_1)$ clamped to $[1, N]$, and $\sigma_{\mathrm{cell}} =
\sigma/\sqrt{N_{\mathrm{eff}}}$. For a 1 s figure on 60 s cells:

| τ | N_eff | σ falls by | Naive √N would claim |
|---|---|---|---|
| 0 | 60.00 | 7.75× | 7.75× |
| 5 s | 5.98 | 2.45× | 7.75× |
| 20 s | 1.50 | 1.22× | 7.75× |
| 5 min | 1.00 | 1.00× | 7.75× |

At τ = 20 s the naive answer is over six times too confident; at τ = 5 min the
averaging buys nothing at all. So the correction cannot be skipped *or*
guessed, and τ is not a number most data owners have: it is not in any file,
no ICARTT or parquet product in the archive declares one, and establishing it
for an instrument means measuring it (§7 names that estimator as a future
avenue). A field that usually cannot be filled honestly must not be what
decides whether a stored number is right.

Three consequences follow, and they are why the arithmetic belongs to the
stage that consumes a sigma rather than to the stage that reads one:

1. **The invariant was never available.** Ingestion could rescale only when τ
   was declared, so "every sigma describes its own cell" was best-effort. A
   consumer had to read the recorded interval regardless, which is exactly
   what it must do now — with the difference that it is now true uniformly.
2. **One hop beats two.** A sigma quoted at 1 s, moved onto 60 s cells at
   ingestion and then onto a 5-minute pairing clock in §4, applies the AR(1)
   approximation twice, and the second hop needs the autocorrelation *between
   cell means*, which is not $e^{-W/\tau}$. Going once, from the quoted
   interval to the support actually being used, is both better arithmetic and
   easier to state.
3. **One implementation.** The same $N_{\mathrm{eff}}$ machinery is needed
   when §4 bins a fast stream onto a slow cell, when §5 rolls a window over
   cells, and when §7 weights a fit. A copy in ingestion would fork the
   load-bearing uncertainty maths before its main consumer was written.

Two facts a consumer needs are therefore recorded rather than resolved, both
per variable, both travelling into the saved product:

- `uncertainty_at_width` and `uncertainty_at_width_ratio` for the **random**
  component. Widths are per row, and a duty-cycled sampler's genuinely vary
  (§10.5), so the ratio is a median and the exact per-row widths stay in the
  bounds variable.
- `uncertainty_systematic_at_width` when a **systematic** component declares
  an interval. No stage may ever act on it: a systematic error is correlated
  across samples by definition and does not average down at all (§3.3), so no
  interval can change it. It is recorded because a manifest that states one
  has said something about its instrument, and a product should not be
  quieter than the manifest that produced it.

A consumer wanting a sigma at some support therefore reads three things — the
quoted interval, the decorrelation timescale, and the cell boundaries — and
resolves them in one step. What it must *also* read is `cell_methods`: the
$N_{\mathrm{eff}}$ correction assumes averaging produced the value, so it
applies to a `time: mean` cell and not to a `time: point` one (§10.3), whose
value averaged nothing whatever the cell's width. That check has no
implementation to live in yet, which is the point of writing it here.

### 10.9 The manufactured side, and what the harness catches

`TrueSupport` is the generator-side counterpart of the manifest's
`SupportSpec`, in the same relationship as `TrueUncertainty` to
`UncertaintySpec`, with `to_manifest_support()` converting between them so the
two schemas cannot drift. A `mean` instrument renders truth on a fine grid
*inside* each cell and averages it, rather than evaluating once and labelling
the result an average — so a 3 s plume inside a 60 s cell comes out diluted by
roughly twenty, exactly as a real minute mean would record it, and the answer
key stores the diluted peak the instrument could actually see. Noise is drawn
**at the cell**, not on the fine grid, so a declared `absolute` sigma keeps
meaning "the spread of the numbers this instrument publishes".

Events are located against each stream's **cell boundaries**, never against
the flattened fine grid. The grid looks sortable and is not: jitter is
permitted up to just under half the sampling interval, so full-width cells
centred on jittered stamps overlap and the grid descends. Measured on a 1 Hz
stream with 0.4 s jitter, 151 of 300 adjacent cells overlap. A binary search
over that grid mis-selects cells, by up to 0.8 % of a plume's peak in the
harshest configuration the schema permits. Cell starts are sorted whatever
the jitter, being a constant shift of an increasing clock, so they give a
valid search. The wider selection this implies costs nothing, because the
kernel returns exactly zero outside its support.

The averaging is validated against closed forms rather than a golden file: a
flat background averages exactly, a linear drift is exact in its increments
(the midpoint rule is exact for linear functions), the quadrature error falls
as the inverse square of the subsample count, and a narrow Gaussian of
amplitude $A$ and width $\sigma$ in a cell of width $W$ reaches
$A\sigma\sqrt{2\pi}/W$, matched to five decimal places.

`export_raw` writes an archive at any of the three rungs — per-row boundary
columns, a uniform declaration, or nothing — and the time column carries the
instant the label implies, so a start-labelled product writes start times.
Writing midpoints would hand ingestion the answer.

**How strong is the harness? Measured, by mutation**, the technique of §9.9.
Five realistic support bugs were injected and the round-trip file was run
alone against each: the label ignored, a boundary parsed on the wrong
timezone, the cell width doubled, the index left at the cell start, and the
clock correction applied with the wrong sign. **It caught three of five.**
Both misses had a cause worth fixing:

- The exporter only ever wrote UTC, so a reader ignoring the declared timezone
  had nothing to bite on. `export_raw` gained a `timezone`, and a round trip
  through a real zone now proves boundaries are parsed on the same convention
  as the axis they bound. This is the `raw_units` lesson of §9.9 recurring: a
  fixture that never exercises a path cannot test it.
- The width was asserted nowhere that *measures* it. Two of the three
  declarations take the width from the manifest or from columns; only the
  undeclared path measures a cadence, and that test checked the offset and the
  provenance but not the width.

With both fixed the harness catches **five of five**.

A second round, aimed at the paths that only matter when an instrument has
several files, scored **zero of four**: cadence measured per instrument
rather than per file, provenance reconciled the wrong way, a label hint used
when files disagree, and the zero-width repair disabled. The full suite
caught all four, so nothing was broken — but the round trip is the only test
that bears on whether ingestion is *correct* rather than self-consistent, and
it could not see any of them, because the exporter wrote one file per
instrument and never a degenerate cell. It now takes `split`, which writes an
instrument as several files with different sampling intervals, and
`zero_width_cells`. That lifts the score to **two of four**.

The remaining two need files of one instrument that *disagree* about their
support, and the only per-file variation TSARA models is the ICARTT
independent-variable name. They are therefore unreachable from a CSV-only
exporter, and that is the price of the decision above: a round trip through a
TSARA-written ICARTT would demonstrate the writer and reader agreeing with
each other rather than either matching FFI-1001.

A fourth round aimed at **assembly and the bundle** — where cells become the
CF representation and go to disk — put eight bugs to the suite and it caught
**six**. Both misses were in persistence: the bundle migration running
whatever the format version (found by hand, above), and `time_bnds` dropped
from the pinned time encoding, which no test could see because nothing ever
built bounds at a resolution other than nanoseconds. With tests for both, plus
one for the sigma `cell_methods` exclusion, the suite catches **eight of
eight**.

A fifth round aimed at the **uncertainty layer**, when it still moved a
declared sigma onto the cells: ten bugs in the $N_{\mathrm{eff}}$ arithmetic
and its plumbing — the correlation factor inverted, ρ's ratio swapped, the
root dropped, the clamp lowered, the count inverted, the scaling-up refusal
removed, quadrature turned into a sum, a rescaling claimed but not applied,
the cell widths never delivered, and the reported N_eff summarised by mean
rather than median. **The suite caught nine; the round trip caught none.**
The round trip's blindness was structural rather than an oversight, and worth
recording because it is the harness's one real limit here: the generator draws
noise at the delivered support, so `export_raw` cannot honestly declare that a
figure was quoted at a *finer* interval, and a truthful declaration at the
cell's own width exercises only the do-nothing case. Making that path testable
end to end means rendering noise on the fine grid and averaging it — the
alternative rejected in the design above. The arithmetic has since been
removed from ingestion entirely (§10.8), which retires eight of the ten
mutations rather than answering them; what remains is that ingestion records a
declaration and changes no number, which the suite covers.

A third round aimed at the **readers and the orchestrator** — the layer that
decides what interval each row describes and in what order the deciding
happens — put five bugs to the whole suite rather than the round trip alone:
a clock correction that moves the index but not the cell it names, sorting
before centring, a file's stated cells overridden by an assumed `start`
label, ICARTT trusting the specification instead of reading the independent
variable's name, and a cell whose stop precedes its start accepted. The round
trip caught **one of five** (it writes one label and one width, so most of
this layer is invisible to it); the suite caught **four**.

The miss was *sorting before centring* — the one ordering decision this stage
is built on, explained in a comment and enforced by nothing. Swapping the two
calls passed all 1072 tests. It now has a test: an instrument whose 100 s cell
starts before a 10 s cell and is centred after it, where the wrong order
leaves a non-monotonic stream. Three further mutations were added for the
defects this stage found by hand — the duplicate split counted on two
different axes, the candidate list read after TSARA attaches its own columns,
and a midpoint offered as a boundary. The suite now catches **eight of
eight**.

End to end on the real archive, the stationary Picarro suite (50,400 rows
across 35 days) ingests three ways to the same 60 s start-labelled cells at
coverage 1.000, by three independent routes that each say honestly how they
know: `reported` from the file's stop column, `declared` from the manifest,
and `inferred` from the `StartTime_UTC` column name. Four other real shapes
read correctly at the `reported` rung, including one worth naming: the
stationary POPS instrument declares **59 s cells on a 60 s cadence**, so the
cadence-inferred reading of it would have been a second too wide on every
row, in the direction that overlaps its neighbour.

### 10.10 Known limits, deliberately not modelled

- **Instrument response and cavity residence.** A laser analyzer's reading is
  a convolution of the truth with a response function, not a box-car mean.
  Modelling it is deconvolution, a future stage; `point` is the honest
  under-claim in the meantime.
- **Picarro sequential species and `_Sync` interpolation**, PTR-MS per-mass
  dwell, and non-uniform canister fill weighting. Assumed uniform where it
  matters, and recorded.
- **Overlapping cells.** Representable, but refused by the generator schema
  and unsupported in the effective-sample-count arithmetic. None were found
  anywhere in the archive.
- **Inclusive versus exclusive stop seconds.** One instrument declares a 59 s
  cell on a 60 s cadence; TSARA honours what the file says rather than
  rounding it.
- **Spatial support on mobile platforms.** A 15 s canister at 15 m/s covers
  about 225 m. **Measured since, on the 32 iWAS fills of the 2024-07-18
  drive against their own MetNav track: a median of 127 m and a maximum of
  307 m.** Cell position is the track at the cell midpoint; a
  `path_length_m` attribute is a future addition for clustering.
- **The declared-versus-empirical closure diagnostic** proposed during
  scoping — comparing a declared σ against `diff_mad` at the delivered support
  — is not built. The empirical estimator belongs to the analysis
  configuration, and ingestion computing it would mean reading a config this
  stage has no business reading (§9.6). It belongs to the phase that owns the
  noise estimator.

---

## 11. Alignment and pairing (Phase 4)

The stage that first *combines* values across time. Three operations live
here, and they need different amounts of trust — the test being whether an
operation needs a **model** or only a **declaration** (§10.8):

| operation | needs | where |
|---|---|---|
| averaging a fast stream onto a slow stream's cells | only the cells, which are already declared | §1.3, §11.2 |
| interpolating GPS and met onto another clock | a smoothness model, guarded by `max_interp_gap` | §1.2, §11.4 |
| moving a σ from its quoted interval onto a cell | τ and an AR(1) model | §3.4, §10.8 |

Nothing here detects events, computes a baseline or fits a slope; it hands
those phases honest pairs.

### 11.1 How each operation here is checked

This phase is the first that **combines** measurements: earlier stages apply
declared, exact, one-to-one maps (a unit conversion, a per-point σ from a
declared budget, a timestamp moved to its cell midpoint), and each of those is
recoverable from what the product records. Nothing here is. A binned value is
a weighted mean no instrument reported, interpolation assumes smoothness, the
τ correction assumes AR(1), and none of the three can be inverted back to the
values that went in.

That asymmetry sets the standard of proof. Five kinds of evidence are used,
because they fail differently:

| evidence | catches |
|---|---|
| a fixture small enough to check with a pencil | the code does something other than what the docstring says |
| an independent reimplementation, written from the definition and slow | the vectorized spelling is wrong |
| a closed form (an analytic plume integrated exactly) | the weighting scheme is subtly wrong |
| **Monte Carlo** | the formula is the *wrong formula* |
| mutation testing of the tests themselves | the tests would not have noticed |

The fourth is the one that matters, and it is the one an algebraic test
cannot supply: comparing algebra against algebra proves a formula was typed
correctly, never that it was the right formula. For an uncertainty the
experiment is direct — draw many realizations of an error with known
structure, average each, and measure how much the averages actually scatter.

Measured for §3, at 20 000 realizations (where the measurement itself is good
to about 0.5 %):

| case | observed scatter | predicted | disagreement |
|---|---|---|---|
| white noise, N = 60 | 0.2568 | 0.2582 (`independent`) | 0.6 % |
| AR(1), τ = 20 s, N = 60 | 1.3536 | 1.3501 (`ar1_neff`) | **0.3 %** |
| the same, other forms | 1.3536 | 1.6332 (`ar1_asymptotic`) | 17 % |
| the same, naive √N | 1.3536 | 0.2582 | 424 % |
| systematic, N = 100 | 0.02003 | 0.02000 (§3.3) | 0.2 % |

So the finite-*N* form is not merely the tidier algebra: it is the one that
matches what happens. The last two rows are the cost of the alternatives, and
the naive √N row is why §10.8 refuses to apply it when no τ is declared.

### 11.2 The joining operation

Everything TSARA does with more than one clock is one operation: **every value
is averaged onto target cells, weighted by how much of it falls inside each
one**. The two products of this phase differ only in what the target is.

| product | target cells | consumer |
|---|---|---|
| a species pair (§11.4) | the wider-supported member's own cells | §4 regression |
| a campaign matrix (§11.7) | a uniform grid | receptor modelling, continuous state |

They are not two designs. `tsara.align.binning.bin_streams_onto_cells` is the
operation; `pair_species` is that function with two variables selected and
incomplete rows dropped. Writing the pairwise form as its own implementation
was a design inversion caught during the phase, and it would have left two
implementations of one idea to drift apart.

**Deliberately variable-agnostic.** The binner does not know what a species
is. It takes whatever variables it is handed — raw concentrations now,
baselines and enhancements once §5 computes them, met, anything a later stage
invents — and puts them on the support asked for. TSARA is a loader and
transformer for sweeping analysis choices, so the joining block must not need
editing every time a new kind of variable appears upstream of it. There is
deliberately **no "PMF matrix" object**: a receptor-model matrix is a *call*
to this function with a chosen set of columns.

**What travels with a variable**, automatically, so a caller cannot forget:

* its uncertainty components, propagated through the *same* overlap weights
  that formed the value, random and systematic separately (§3);
* `n_source_<name>` and `coverage_<name>` — how many source cells contributed
  and how much of the target cell they covered, the two numbers that separate
  a well-determined value from a number that merely exists;
* everything the source declared about itself, plus `tsara_source_instrument`
  saying where it came from and `tsara_binned` saying whether this stage
  averaged it at all.

**Three behaviours are not left to callers.** A variable declaring
`circular: 1` is vector-averaged, never arithmetically (§11.5), and carries a
resultant length and dispersion instead of a sigma. A stream whose cells
already *are* the target passes through untouched — tested on the cells, not
on the instrument name, because averaging a cell onto itself is the identity
mathematically and not in floating point. Its *values* are untouched; its
companion columns are not a shortcut, and are exactly what the general path
would produce for one source cell covering its target — a count and coverage
of 1 where a value is present and 0 where it is masked, and for a direction a
resultant length of 1 and a dispersion of 0 (§11.5). Before the Phase-4
walkthrough the pass-through stamped a count and coverage of 1 on masked rows
too, and gave a direction no quality columns, so a product's columns and their
meaning depended on whether a stream happened to share the target's cells. And
a target cell with no contributing data stays `nan`: gases are binned, never
interpolated (§1.2).

**Column names.** A canonical name is kept as it is when only one selected
stream carries it. When two do — a campaign comparing two analyzers — both are
suffixed with their instrument rather than one silently winning. The spelling
therefore depends on the selection, which is why every column also records its
source instrument.

#### 11.2.1 The operation is symmetric in the code and must not be in use

Averaging a fast stream onto slow cells discards resolution the slow
instrument never had, which is honest. Evaluating a slow value on fast cells
invents no value — every row holds a number the instrument really reported —
but it invents **rows**, which is the interpolation rule (§1.2) restated for a
step function. Sixty rows from one 60 s mean enter a least-squares fit as sixty
independent measurements; the slope estimate survives and its standard error
shrinks by √60 ≈ 7.7, which is exactly the pseudo-replication that pairing on
the wider-supported clock exists to prevent (§1.3).

Both products of this phase already prevent it by choosing their target:
`pair_species` pairs on the wider-supported member, and `build_output_grid`
runs this same test against its cells before binning (§11.7). The primitive
they share refuses it too, and has to, because it is public and is the
documented route to a receptor-model matrix over a caller's own target cells.
Neither `coverage` nor `n_source` would have flagged it: a replicated row
reports coverage 1.0 and one contributing source cell, which is also what a
legitimately sparse canister measurement reports.

**The test is measured on the overlaps, not on a comparison of widths**, and
that choice is load-bearing. A width comparison needs a tolerance, because real
instruments disagree about their own nominal rate — the median cell widths of
the 2026 archive's three Aeris analyzers are 0.992 s, 0.993 s and 1.024 s
against a nominal 1 s — and a strict median-width rule refuses binning one onto
another over a difference that replicates nothing.

**The rule: a reading is replicated when the time it shares with the target
cells adds up to at least twice the width of the widest target cell it
touches** (`tsara.align.binning.measure_replication`).

| source cells | target cells | two cells' worth? | outcome |
|---|---|---|---|
| 60 s | 1 s | sixty | refused |
| 2 s, any phase | 1 s | two | refused |
| 1.9 s | 1 s | no | allowed; the sharing is counted (§11.7) |
| 1.023 s | 1 s | no | allowed |
| 10 s, half a period out of phase | 10 s | no | allowed; lossy for another reason (§11.9.1) |

**It replaced a rule with a phase hole.** Until the Phase-4 walkthrough the
test counted target cells lying *wholly* inside one source cell and refused at
two. It agreed with the table on every row but one: a perfectly regular 2 s
record offset from a 1 s grid by 0.3 s or 0.5 s wholly contains only one 1 s
cell, so it passed, and each of its readings fed two or three rows — while the
same record exactly in phase was refused. No stream in the permitted archive
reached it (the 2 s Wyoming Picarro sits exactly on whole seconds, and every
2026 lag correction is a whole number of seconds), but a fractional
`time_shift` would have. Twice a width is twice a width whatever the phase, so
the rule that replaced it has no such case, and it still needs no tolerance:
the sums are integer nanoseconds.

Below that line a reading can still feed two rows — a 1.9 s cell on a 1 s grid,
a 15 s canister fill across a minute boundary — and that is not refused but
counted, in `tsara_pairing_readings` (§11.4.1) and `tsara_grid_readings`
(§11.7), each with a warning when rows outnumber the readings behind them.

The escape route for the legitimate case is the one that already exists: a
smooth non-gas field wanted on a finer clock is *interpolated* under a gap
guard by `tsara.align.auxiliary` (§11.6), which refuses a variable declaring
`role: gas`.

#### 11.2.2 A companion column is not a variable

Four families of column exist only to qualify the column they are named after:
the two uncertainty components, `n_source_*`, `coverage_*`, and the resultant
length and dispersion an angular variable carries instead of a sigma. The
default selection excludes all four, through one predicate
(`tsara.core.naming.is_companion_name`) rather than a list per caller.

This is not tidiness. Three of the four families are produced *by this
function*, so a joined product that is joined again — which is what §5 does
with a baseline computed from one — would otherwise grow a
`coverage_coverage_ch4` on every pass: a column with no parent, no meaning, and
an arithmetic mean taken of a quantity whose own `cell_methods` says `sum`.
Asking about the name is the only test available, exactly as for the sigma
companions, since nothing else in a stream marks them and a stream may have
come from the generator, from ingestion, or from a file reloaded from disk.

Cell boundaries are excluded for a related reason: they describe the rows
rather than varying over them. They are usually a coordinate and therefore
invisible to a selection, but `stream_cells` deliberately accepts a stream that
carries them as a data variable, so the selection accepts that shape too. Any
variable that is not one value per cell is refused by name rather than reaching
the weighting as a shape mismatch.

### 11.3 Propagation

Implemented in `tsara.core.propagation`, specified in §3. The two components
never mix, the weights are the operation's rather than the estimator's (§3.2),
and every propagated σ carries the name of the form that produced it.

### 11.4 Pairing

Implemented in `tsara.align.pairing`, specified in §1.3. Two species, one
clock, real pairs only.

**Which clock.** The cells of whichever stream has the **wider support**, and
the other is averaged onto them by overlap. Never the reverse: a value may be
averaged onto a wider support, never split onto a narrower one. §1.3 records
why this stopped being "the slower instrument" — measured, the iWAS canisters
are about nine times slower than a 60 s stationary mean by rate (530 s against
60 s) and four times *narrower* by support (14.9 s against 60 s; medians over
all 261 fills of the ten 2024 drive days).

When the widths are equal, the clock is the member with **fewer measured
values where the two records overlap**, and if those tie too, the first
instrument by name (§11.4.1). No step looks at argument order, so which
species is numerator never changes which air is compared.

**Same-instrument species skip the binning entirely.** Several gases retrieved
from one spectrum already share a clock, and that is the commonest pair there
is. The fast path is not merely an optimization: averaging a cell onto itself
is the identity mathematically and *not* in floating point, so the general
path would perturb values that were never meant to change.

**Uncertainty, in order.** A declared figure quoted at a different interval
from the cells it sits on is first moved onto those cells (§10.8) — the
arithmetic ingestion deliberately refuses to do, done here at the point of
use, or refused again and labelled `unscaled` when no decorrelation timescale
was declared. Only then is the binned member's uncertainty propagated through
the *same* overlap weights that formed its value (§3), the two components
separately.

**Cell methods.** The binned species is a mean over its cell. The species
already on the clock was not averaged by this stage at all and keeps its own
stream's method, since stamping `time: mean` on a point sample would assert an
averaging that never happened. Counts are `time: sum`, CF's own word for
them. Coverage fractions and the sigma companions carry no cell method: a
coverage is a property of the cell rather than a statistic of the data inside
it, and a sigma describes the uncertainty *of* the cell's value, which differs
from a mean of sigmas by exactly √N_eff (§10.2).

**Coverage may slightly exceed 1**, and is left alone when it does. Fixed-width
cells centred on jittered timestamps overlap each other, so their overlaps
with one target cell can sum past its width — documented as benign in §10.2,
since the value is a weighted *mean* and the weights normalize. Clipping would
hide a real property of the source record behind a tidier number.

**What the product carries.** A paired series is an ordinary self-describing
`xarray.Dataset` with CF cells, not a bundle entry: it is a per-call
intermediate that Phase 6 will request per event, and the phase's saved
product is the output grid (§11.7). Its attributes:

| attribute | meaning |
|---|---|
| `tsara_pairing_clock` | instrument whose cells the pairs sit on |
| `tsara_pairing_clock_reason` | why that one: both median cell widths, and on a tie both measured-value counts |
| `tsara_pairing_min_coverage` | the guard applied (`PairingConfig.min_coverage`) |
| `tsara_pairing_cells_considered` | candidate cells before dropping |
| `tsara_pairing_cells_dropped` | how many produced no usable pair |
| `tsara_propagation_form` | registered form used for correlated random error (§3.4) |

and per variable:

| attribute | meaning |
|---|---|
| `tsara_source_instrument` | which stream this species came from |
| `tsara_binned` | 1 if averaged onto the clock, 0 if already on it |
| `tsara_sigma_at_support` | how a declared σ was moved onto its own cells, or `unscaled` |
| `tsara_pairing_readings` | distinct readings of this species behind the surviving pairs (§11.4.1) |

**How it is checked** (§11.1): a two-cell fixture whose paired values (1.5 and
5.5) can be worked out on paper; an independent O(N·M) reimplementation
written from the definition of §1.3, agreeing to 1e-12 on random cells with a
masked sample; two closed forms — the mean of a linear ramp over a cell equals
the ramp at the cell midpoint, and a species paired against a constant
multiple of itself returns that multiple exactly, which is the property that
would fail if the two members were averaged over different air; and invariants
needing no ground truth — binning a stream onto matching cells returns it, and
a cell with no partner data yields no pair rather than an interpolation. The
tie rule and the reading counts of §11.4.1 are pinned by fixtures that each
isolate one decision, and those tests were mutation-scored: nine plausible
defects injected one at a time (argument order deciding a tie, rows counted
instead of finite values, whole records counted instead of the shared span,
the name tie reversed, a zero-overlap or masked cell counted as a reading, a
binned member credited one reading per pair, the warning firing on equality,
the denser member preferred), and all nine caught. The first run caught eight;
the miss was the zero-overlap filter, which is reachable only when one source
cell nests inside another and the overlap search brackets a cell that does not
overlap, and a fixture of exactly that shape now pins it.

**And on real data.** The 2024-07-18 mobile-lab drive carries both members of
the canister case: on that day, 32 iWAS fills of median width 14.7 s at a
441 s cadence, and a 1 Hz Picarro. Pairing benzene against CH₄ chooses the canister as the
clock (14.7 s vs 1 s), keeps all 32 cells at coverage 1.000, and draws a
median of 16 one-second samples into each. Recomputing every pair with a
Python loop straight from §1.3 reproduces the paired values **exactly** — a
maximum difference of 0.000e+00 ppb over the 32 pairs.

The same comparison shows what the overlap weighting is worth. Scoring
instead against a plain unweighted mean of the samples whose *start* falls
inside the fill window — the obvious shortcut, which gives a partial edge cell
either full weight or none — the two disagree by up to **17.89 ppb** of CH₄.
On a ~1950 ppb background that is small; against the enhancements these
canisters exist to attribute it is not, and it is entirely an artefact of
which fraction of two edge samples counts.

#### 11.4.1 A tie in width, and the reading counted twice

"The wider-supported member" has no answer for two instruments with equal
cells, and the code as first written broke the tie by argument order. That is
harmless while equal-width cells coincide, since each cell is then its own
partner. It is not harmless when they are **out of phase and one member is
sparse**: every sparse reading then straddles two cells of the denser clock,
half-covers each, and appears in both pairs.

**On real data.** The 2024-07-18 mobile-lab drive has exactly that shape.
NOy-LIF writes `Time_Mid` and the Picarro writes `Time_Start`, both on the same
whole-second grid of 19,501 rows, so TSARA's cells for the two sit half a
second apart; NOy has a finite value in 19,498 rows and CO₂ in 8,448, a value
in every second row 69 % of the time and otherwise every third. Ingested with
TSARA and paired with the argument-order rule:

| call | clock | pairs | distinct CO₂ readings behind them |
|---|---|---|---|
| `pair_species(streams, "noy", "co2")` | LIF | 16,893 | 8,447 |
| `pair_species(streams, "co2", "noy")` | Picarro | 8,447 | 8,447 |

O₃ (9,751 finite values in the same 19,501 rows) against LIF did the same:
19,498 pairs in one order and 9,750 in the other.

**Why it matters, measured rather than argued.** For an ordinary least-squares
slope the two runs are algebraically identical: a reading that appears twice,
once beside each of two neighbouring values of the other species, pulls on
the line exactly as one appearance beside their mean does. What the copies
duplicate is the reading's *error*, which a fit counting the pairs as
independent then counts twice. The size of that cannot be read off the
algebra, so it was measured: the same two clocks at the same sparseness, a
smooth 600 s sinusoid for truth so that the two instruments see identical air
to well within noise, a true NOy/CO₂ slope of 8, and the noise redrawn 300
times per row. The naive least-squares standard error is compared with how
much the slope actually scatters across draws:

| σ on CO₂ (ppm) | σ on NOy (ppb) | naive SE ÷ real scatter, sparser member's clock | the same, denser member's clock |
|---|---|---|---|
| 0 | 1.0 | 0.96 | 1.00 |
| 0.15 | 1.0 | 1.02 | **0.82** |
| 0.30 | 0.5 | 1.03 | **0.74** |

The slope agrees to every printed digit on both clocks. On the sparser
member's cells the naive standard error describes the real scatter; on the
denser member's cells it is a fifth to a quarter too narrow as soon as the
duplicated species carries error of its own, bounded by 1/√2 when all of the
error sits there. Nothing in `coverage` or `n_source` shows it.

**How often the shape occurs.** Counted across all 1582 permitted files (the
1122 ICARTT files and the 460 parquet files outside the quarantine
directories): each file's cell width is its median positive sampling interval,
its label is inferred from the ICARTT independent variable's name (a name
saying nothing, and every parquet file, is `unknown` and therefore centred), its
phase is the modal cell start modulo that width to within 2 % of it, and files
are grouped into instruments by archive tree and date. Of the 3909 same-day
pairs of instruments with equal widths, **312 are half a cell apart at 1 s**
— 210 on the NOAA drives, 80 in SLC-SOS, 22 on the TwinOtter — and **all 2844
pairs in the 60 s stationary suite are in phase**, so that suite never meets
the case. No two 2026 instruments share an exact width.

**The rule.** On a tie in width, the member with fewer finite values sets the
clock. Each of its readings is then exactly one pair, and the denser member is
averaged across it, centred on the same interval. Counted over the span where
*both* records run, because a sparse analyzer logging all day beside a dense
one switched on for ten minutes has more readings in total and fewer in the
air they share; counted over the whole records rather than an event's
`interval`, so that one pair of instruments keeps one clock from event to
event. If the counts also tie, the first instrument by name decides. On the
drive above, both argument orders now give the Picarro clock and 8,447 pairs.

**What the rule does not settle, stated rather than rounded.** Two *dense*
instruments of equal width half a cell apart duplicate nothing on either
clock, but every reading of the binned member now contributes half of itself
to each of two *neighbouring* pairs, so adjacent pairs share an error. The
same experiment with no sparseness at all (every CO₂ row filled, 3599 pairs
on either clock):

| σ on CO₂ (ppm) | σ on NOy (ppb) | naive SE ÷ real scatter, CO₂ clock | the same, NOy clock |
|---|---|---|---|
| 0 | 1.0 | **0.70** | 0.99 |
| 0.15 | 1.0 | 0.86 | 0.81 |
| 0.30 | 0.5 | 0.99 | **0.72** |

Both tables are the executed output of `examples/notebooks/04_alignment_walkthrough.ipynb`
§9 (seeds 0–299), so they re-run from the repository.

No choice of clock is honest in general: whichever member is averaged carries
the shared error, and the standard error is too narrow by up to 1/√2 when that
member's error dominates. This is not rare. On the same 2024-07-18 drive the
iodide-CIMS species (finite in 92–97 % of rows) and the PTR-MS species (94–95 %)
are dense, start-labelled and half a cell from the mid-labelled LIF. The
tie rule still gives them a deterministic clock (the member with fewer finite
values, which there is the CIMS or PTR stream), and the reading count below
cannot flag them, because sharing a reading between two pairs is a
*correlation* between pairs rather than a shortfall of readings. Accounting for
it belongs to the regression, which can see the covariance the shared overlap
weights imply; it is recorded as an open question for Phase 7.

The same sharing moves a fitted slope, not only its error, when the signal
itself has structure at the scale of a cell. On a synthetic campaign of two
noise-free 60 s instruments half a cell apart, with plumes about as wide as
the cells and a true ratio of 0.25, a least-squares fit of tracer on CH₄ gives
**0.285** on the tracer's clock and **0.169** on the CH₄'s, while the
mass-weighted ratio is 0.251 on both. Which member gets smoothed decides which
way the slope is biased. The 60 s stationary suite is in phase throughout, so
the archive does not meet this at 60 s; at 1 s it would need plumes one or two
samples wide, which the drive records do contain (§9.5).

**The duplication a clock cannot remove.** A sparse member narrower than the
clock can still straddle its boundaries: a 15 s canister fill across a minute
boundary of a 60 s mean is one reading in two pairs whichever rule chooses
the clock. In a synthetic campaign of 327 fills against 60 s means, 84 fills
landed in two pairs, 20.5 % of the pairs. On real data the case is presently
empty: all 261 iWAS fills of the 2024 drive days, paired against the
mobile lab's own 60 s ground Picarro (each with its file's stop column naming
exact cells), yield **one** pair, because the ground record has no value while
the lab is driving. It is therefore recorded rather than prevented. Each
paired species carries `tsara_pairing_readings`, the number of distinct finite
readings behind the surviving pairs (a species already on the clock has one
per pair by construction), `PairedSpecies` exposes the same counts, and a
warning names the species when either count falls below the number of pairs.
That count, not the pair count, is a ceiling on a regression's N — a ceiling
and not an estimate, since readings shared between neighbouring pairs, as in
the dense case above, reduce the independent information without reducing the
count.

### 11.5 Circular statistics for angular variables

Supersedes the §1.5 stub. A direction is a point on a circle, so directions
are averaged as unit vectors, never arithmetically. Config:
`VariableConfig.circular`, valid only for `role: met`.

Measured on the ten 2024 mobile-lab drive days, 3337 sixty-second cells of
1 Hz wind direction: the arithmetic mean differs from the vector mean by more
than 45° in **26.7 %** of cells, with a median error of 7.0° and a maximum of
180.0°.

**What a binned direction carries.** Summing unit vectors gives two numbers,
not one: the mean direction, and the mean resultant length *R* — the length
of the average vector, 1 when every sample agrees and 0 when they cancel.
TSARA stores *R*, because every dispersion statistic in the circular
literature is a transform of it, so choosing between them is choosing a
presentation rather than an estimator.

The dispersion reported alongside it is the **exact circular standard
deviation**, $s = \sqrt{-2\ln R}$, exact for a wrapped normal (whose
resultant length is $e^{-\sigma^2/2}$) and the standard definition otherwise.
It is unbounded: as directions spread toward uniform it runs to infinity,
which is the honest description of a direction that has ceased to exist.

**Yamartino (1984) is deliberately not implemented.** It approximates the same
quantity in a single pass, for dataloggers that could not hold the sample
vectors in memory — not a constraint TSARA operates under. Measured on the
same drive data, the two agree to 0.16° in the median cell and diverge by up
to 66.5°, entirely in the tumbling cells where Yamartino saturates near 105°
and the exact form correctly does not. Anyone needing it has *R* and one line
of arithmetic.

**Why the dispersion is not decoration.** On a moving platform a 60 s mean
direction is usually not a well-determined quantity:

| resultant length | cells | share | exact sd, median |
|---|---|---|---|
| steady, *R* > 0.99 | 58 | 1.7 % | 7.1° |
| *R* 0.90–0.99 | 1140 | 34.2 % | 18.5° |
| *R* 0.50–0.90 | 1713 | 51.3 % | 40.6° |
| tumbling, *R* < 0.50 | 426 | 12.8 % | 80.5° |

Part of that spread is the van turning rather than the atmosphere — a
documented limit, not a correction TSARA applies.

**How far to trust R depends on how many readings made it.** *R* is a
statistic of a sample and is biased high when the sample is small: even
directions carrying no information at all cannot average to an *R* near zero
from a handful of readings, and the dispersion derived from *R* is understated
correspondingly. Measured, each figure the median (dispersion) or mean (*R*)
over 3000 draws:

| contributing readings *N* | mean *R*, directions uniformly random (truth 0) | reported dispersion, wrapped normal of true spread 40° |
|---|---|---|
| 2 | 0.64 | 19° |
| 5 | 0.40 | 34° |
| 10 | 0.28 | 37° |
| 15 | 0.23 | 38° |
| 60 | 0.11 | 40° |

For uniformly random directions the mean *R* tracks 1/√*N*. At the 60 readings
of a minute of 1 Hz wind the bias is a degree; at the ten of a 10 s analyzer
cell or the fifteen of a canister fill it is not negligible, and a cell with
two readings says almost nothing. TSARA does not correct for it: a
bias-corrected *R* assumes a distribution, and `n_source_<name>` travels with
every binned direction so the reader can see what the *R* rests on. The same
is true of a cell with one contributing reading, whose *R* is 1 by
construction — agreement of that reading with itself, not a steady wind.

**Unit vectors, not speed-weighted vectors.** Every reading counts equally
whatever the wind speed. The other established convention weights each
reading by its speed, and so reports the direction the air moved in on
average rather than the direction the vane usually pointed. Measured on 325
minute cells of the 2024-07-18 drive (`WindDir_calc_deg` and
`WindSpd_calc_m_s` from the mobile-lab MetNav file, cells with at least 30
readings), the two differ by a median 2.2°, by more than 8.4° in a tenth of
cells, by more than 20° in 1.5 %, and by 54.6° at worst; the difference grows as
the wind becomes variable (median 1.2° where *R* > 0.9, 8.2° where
*R* < 0.5). TSARA uses unit vectors because the join is variable-agnostic
(§11.2): speed-weighting would require the binner to know which speed
variable belongs to which direction, and a direction binned alone is the
case it must handle. A speed-weighted direction is recoverable by a caller
who bins the wind components `u` and `v` as ordinary scalars.

**The pass-through path carries the same columns.** A direction whose stream
already sits on the target cells is not re-averaged (§11.2), but it still
gains `<name>_resultant_length` (1 where a reading is present) and
`<name>_dispersion` (0), and carries no sigma, exactly as the general path
produces for one contributing reading. Until the Phase-4 walkthrough it gained
neither and kept its sigma, so a product's columns depended on whether a
stream happened to share the target's cells: the 2024 ground MetNav's 60 s
`WindDir_calc_deg` on a 60 s grid aligned to its cells — which the grid's own
phase warning recommends (§11.9.1) — had no *R*, and one grid period later it
did.

**No scientific threshold is applied.** A direction with *R* = 0.001 is
meaningless and is reported anyway, beside the *R* that says so; picking a
cut-off would put a magic number in the library where the judgement belongs
to the user. Guarding on *R* is left to the caller, exactly as the pairing
coverage guard is (§11.4).

**One numerical threshold is applied**, and it is a fact about float64 rather
than about wind. Directions that cancel mathematically do not cancel
numerically: `sin(180°)` is 1.22e-16, so a north/south pair leaves a residual
vector of length 6.1e-17 pointing due *east*, and `atan2` reports 90.000°
with complete confidence. Summed in three different orders the four compass
points give **129.60°, 153.43° and 132.19°** — a number that changes when its
inputs are reordered is not a measurement. So a resultant length at or below
$N\epsilon$, the rounding floor of the sum of *N* unit vectors that produced
it, is snapped to exactly zero and its direction reported as `nan`; *R* = 0,
a `nan` direction and an infinite dispersion then all say the same thing.
This is distinct from *R* = `nan`, which means nothing contributed at all.

A related defect in the same family: `-1e-17 % 360` is `360.0` in float64, so
the wrap can leave the half-open interval it documents and two readings either
side of north can average to 360.0 rather than 0.0. Angles landing on a full
turn are folded back to zero.

**Shared weighting.** `bin_circular_onto_cells` and `bin_onto_cells` both call
`tsara.core.support.overlap_pairs`, so a cell's wind direction and its methane
are averaged over exactly the same interval; a test compares the contributing
counts and coverage of the two paths.

**How it is checked** (§11.1): pencil-checkable fixtures (359° and 1° average
to 0°; the compass points cancel); the closed form above, over three decades
of σ, plus a 200 000-draw sample of a wrapped normal recovering its σ to 2 %;
an independent O(N·M) reimplementation written from the definition, agreeing
with the vectorized path to 1e-12; and on real data, the identity invariant —
binning a record onto its own cells returned it to within **5.7e-14 degrees**
over 19 470 samples, and the table above reproduces an independent script's
numbers exactly. The pass-through path is compared with the general path on
the same rows (the full cell array takes the pass-through, the same cells less
one do not), column by column; the ingestion wrap is pinned by the
magnetic-to-true case above. Eight defects were injected into those two
changes (no wrap, the wrap after QA/QC, the wrap applied to every variable, a
masked pass-through value counted or covered, the old pass-through columns,
*R* of 1 where a reading is masked, the wrong single-reading dispersion) and
all eight were caught.

### 11.6 Auxiliary fields, and the only interpolation TSARA performs

Implemented in `tsara.align.auxiliary`. §1.2 states the prohibition —
quantified species are never interpolated, only bin-averaged — and this
section is its single exception. Position and ambient meteorology vary
smoothly on sampling timescales, so evaluating them *between* samples is
physically justified where evaluating a concentration between samples is not.
This is the only module in TSARA that interpolates anything.

**Two guards, both refusals.**

*What.* A variable declaring `role: gas` is refused outright, and so is one
declaring **no role at all**. The second is deliberate: TSARA's own streams
always carry roles, so an absent one means the dataset came from elsewhere and
its smoothness is unknown. Admitting it on the assumption that someone would
have said otherwise is exactly how a prohibition erodes.

*How far.* A target instant whose bracketing samples are further apart than
`AlignmentConfig.max_interp_gap` gets `nan` rather than a bridged value, and
nothing is extrapolated past either end of a record. Both counts are reported
separately, because "there was no position here" and "the position here was
too uncertain to state" are different facts about a cell.

One case is deliberately exempt from the gap guard: a target landing **exactly
on** a source sample. That value was measured, not interpolated, so refusing it
would discard real data.

**A record sparser than its guard says so.** The exemption has a consequence
worth stating: when a record's own sampling interval is longer than the guard,
every gap is refused and only the cells whose midpoints happen to land exactly
on a sample keep a value — in the Phase-4 notebook, a 50 s GPS against a 10 s
guard positions 36 of 1800 one-second cells, one in fifty, and the result looks
like a join that worked. This is not hypothetical: 7 of the 13 Univ_Wyoming GPS
files in the 2024 archive (D06–D12) record a fix every 50 s, and at the default
guard they position **0.0 %** of 1 s gas cells whose midpoints fall between
seconds. Until the Phase-4 walkthrough the only record was an INFO log line,
which a notebook does not show. `interpolate_onto_cells` now logs a **warning**
when any target was refused *and* the source's median interval between finite
samples exceeds the guard, naming the interval, the guard and the counts. The
median rather than the mean, so that an ordinary 1 s record with one long
outage — refused and counted, correctly — does not trip it; and strictly
longer, since a gap exactly as long as the guard is bridged.

**Where a value is placed: the cell midpoint.** For a smooth field under a
linear model that is the representative instant, identical to the cell mean for
a straight constant-speed path and different only where the path curves.
Interpolation rather than binning is also what makes a *sparse* auxiliary
record usable at all: a fix every 10 s cannot fill 1 s cells by averaging,
since most cells contain no fix.

**What the guard costs.** `max_interp_gap` is not a technicality; it is how far
a platform may be allowed to stray from the straight chord between two fixes.
Measured on real driving: the 1 Hz MetNav track of all ten 2024 mobile-lab
drive days, scored only while the van was moving (ground speed above 2 m/s;
median 13.0 m/s, 90th percentile 18.7), thinned to one fix every *k* seconds,
interpolated linearly, and compared with the removed fixes that sit strictly
inside a thinned bracket of exactly *k* seconds:

| one fix every | positions scored | median error | p90 | p99 | max |
|---|---|---|---|---|---|
| 5 s | 117 253 | 0.9 m | 2.8 m | 6.1 m | 118 m |
| 10 s | 131 146 | 2.3 m | 9.5 m | 19.3 m | 133 m |
| 30 s | 139 222 | 13.5 m | 54.7 m | 92.7 m | 166 m |
| 50 s | 140 183 | 29.9 m | 108 m | 180 m | 326 m |
| 60 s | 140 335 | 39.5 m | 137 m | 225 m | 383 m |

So the 10 s default corresponds to about 10 m at the 90th percentile on urban
driving, and raising the guard to 60 s to use the 50 s Wyoming GPS accepts
positions about 110 m wrong at the 90th percentile and over 300 m at worst.
Whether that is acceptable depends on what the position is for — a
source-complex cluster radius is a different question from a street address —
which is why it is configuration rather than a constant. The error peaks
mid-way between fixes, where the van has had longest to turn away from the
chord (notebook 04 §11 plots the shape).

**Bin, or interpolate?** Interpolation evaluates a field at each cell's
midpoint. For a record *sparser* than the cells that is the only option: a fix
every 10 s cannot fill 1 s cells by averaging. For a record *denser* than the
cells it throws most of the record away and reports an instant for a value
that describes an interval, and the two can differ a great deal. Measured on
the 2024-07-18 drive's 1 Hz MetNav record, as the midpoint value against the
overlap-weighted cell mean (vector mean for direction) over the 32 iWAS fills
(exact cells from `iWAS_Stop_UTC`) and over epoch-aligned 60 s cells (325
where both are finite):

| field | 15 s fills: median / p90 / max | 60 s cells: median / p90 / max |
|---|---|---|
| wind direction | 7.9° / 30.1° / 59.6° | 13.1° / 39.5° / 159.7° |
| wind speed | 0.41 / 1.15 / 1.48 m/s | 0.56 / 1.31 / 3.28 m/s |
| air temperature | 0.04 / 0.10 / 0.28 °C | 0.10 / 0.29 / 0.62 °C |
| position, north | 0.4 / 3.4 / 4.7 m | 8.5 / 56.7 / 96.4 m |
| position, east | 0.1 / 3.7 / 7.3 m | 6.8 / 52.0 / 104.7 m |

The rule is therefore: **bin a field that is denser than the cells, interpolate
one that is sparser.** The output grid follows it — every selected variable,
met included, is binned (§11.7), and a binned direction carries the resultant
length that says how much it means. `attach_positions` interpolates because a
GPS record is often the sparser one and because, for a *position* on
canister-width cells, the instant costs a few metres against the
**127 m** median path covered during a fill.

**Circular fields are interpolated as unit vectors** (§11.5), so a direction
crossing the compass seam takes the short way — 359° to 1° passes through 0°,
not through 180°.

**The mobile position join.** Ingestion attaches a stationary site's single
position, which is exact and free, and deliberately leaves a moving platform's
track alone: putting it on a gas instrument's clock is interpolation, and its
guard lives in a config object ingestion has no business reading (§9.7). The
binding is recorded in stream attributes there and consumed here. The resulting
coordinates are named exactly as a stationary platform's are, so downstream
code reads position identically and only has to care about the shape.

The synthetic generator does **not** leave mobile gas streams in that shape: it
attaches positions to them directly, computed as linear interpolation of its
own GPS samples (`tsara.core.geodesy.positions_at`), with no binding. Two
consequences. Code exercised only on generated streams never calls
`attach_positions` and finds positions an ingested stream lacks — a
substitutability gap recorded for the phase that first consumes mobile
positions. And the generator cannot supply accuracy evidence for this join,
since its "true" positions are the same linear construction; the accuracy
evidence is the real-driving table above.

**Antimeridian: detected, not modelled.** Longitude is interpolated linearly,
which is right everywhere except across ±180°, where a step from 179.9 to
−179.9 would read as a journey most of the way round the planet. A track whose
longitudes span more than 180° is refused with a message saying so. The
alternative — treating longitude as circular always — would put a sine and
cosine round trip into every position on Earth to serve a case the target
archive does not contain, and a silently wrong position cannot be recovered
downstream.

**Spatial extent is not modelled.** Measured on the 32 iWAS fills of the
2024-07-18 drive, the van covers a median of **127 m** and up to **307 m**
during a single canister fill. TSARA reports the midpoint position and records
that limit (§10.10) rather than inventing a path length it has no information
for.

**Attributes** an interpolated coordinate carries:

| attribute | meaning |
|---|---|
| `tsara_interpolated_from` | the stream and variable the values came from |
| `tsara_max_interp_gap` | the guard that was applied |
| `tsara_interp_gap_masked` | targets refused because the gap was too long |
| `tsara_interp_outside_record` | targets beyond either end of the source |

**How it is checked** (§11.1): both refusals are asserted directly, since a
refusal that quietly stops refusing is invisible; a straight line interpolates
to itself exactly, which is also what would catch a timestamp-precision bug,
since epoch nanoseconds do not fit a float64 mantissa and converting them
directly quantises every timestamp onto a ~378 ns grid; and a direction across
the seam is checked against the short path with the non-circular case beside it
for contrast. The sparse-record warning is pinned from both sides — it fires
for a 50 s record against a 10 s guard, and stays silent for a guard the record
can meet, for one long outage in a dense record, and for a spacing exactly
equal to the guard.

The join is also checked **through real ingestion**: a generated mobile
campaign with a fix every 3 s and start-labelled 1 s gas cells is written as
raw files, ingested, and joined, and must reproduce the generator's positions
to 1e-9 degrees on every interpolated cell. That is consistency, not accuracy,
for the reason given above, but it is the only test on the path production
takes — the binding ingestion writes, the labels it reads, the midpoints it
recovers. Six defects were injected: three into the warning (equal spacing
warned, mean instead of median, no warning) were caught by the warning tests;
of three into the join, a target or source cell's start used in place of its
midpoint was caught by the hand-built fixtures and the round trip alike, and
ingestion binding latitude to the longitude column was caught **only** by the
round trip.

**And on real data.** The 2024-07-18 drive's MetNav track was attached to the
same 32 iWAS canister cells the pairing section uses: all 32 positioned, none
gap-masked, none outside the record, two landing exactly on a GPS second and
thirty interpolated between two. Wind direction interpolated circularly onto
the same cells stayed inside [0, 360) throughout.

One trap found while writing it, worth recording because it is the same family:
`pd.Timedelta("1ns").total_seconds()` is **0.0**, so validating a duration by
reading seconds off it rejects every sub-microsecond value as non-positive.
Positivity is tested by comparing `Timedelta` objects, matching
`tsara.config.base.validate_positive_timedelta` and TSARA's integer-nanosecond
convention everywhere else.

### 11.7 Output grid

Implemented in `tsara.align.grid`, specified in §1.4. A single uniform
`(time × variable)` cube, built **only** for the products that inherently need
one: the continuous rolling state, and the matrix a receptor model such as PMF
consumes. Baselines, detection and cross-species regression all run at native
rate and never see it (§1.1).

It is a thin layer over §11.2 — the only things it adds are *which cells* and
the rule that the period must respect the data going into it.

**There is deliberately no "PMF matrix" object.** A receptor-model matrix is
this function called with a chosen set of columns. Which columns is a
scientific decision — raw concentrations or the enhancements §5 will compute,
which species, whether met belongs in the same cube — and hard-coding any of it
would make the block need editing every time that decision changed.

#### The period rule

**A grid may not be so fine that one selected reading would cover two of its
cells.** A 60 s mean evaluated on 1 s cells is the same value repeated sixty
times: resolution the instrument never had, and sixty points where there is one
measurement. Every count downstream would then believe there were sixty. That
is the prohibition the interval model exists to enforce (§10), so it is an
error naming the offending instrument and a period that would work — longer
than half its widest reading — not a warning. The test is the binner's own
(§11.2.1), run against the grid's actual cells inside the requested window
before anything is binned, so the grid and the operation it calls cannot
disagree about the same data.

**It replaced a comparison of median widths**, found wanting in the Phase-4
walkthrough: "the period must be at least the widest selected cell". On a real
2026-01-19 LANL record the Aeris pico measures its cells at 1.023 s, so that
rule refused a one-second grid and said one measurement would be repeated
across several cells — while the binner accepted the same one-second cells
with 2.6 % more occupied rows than readings, and the next round period the
grid allowed was two seconds. The old rule also offered no protection the
overlap rule lacks.

Checked against the *selection* rather than against every stream the campaign
contains, and that is a real lever rather than a formality. Measured on the
2024-07-18 drive with five instruments loaded:

| request | outcome |
|---|---|
| 5 s over everything | refused — `iwas` has fills of about 15 s that day |
| 15 s over everything | 1301 cells |
| 1 s, canisters excluded | 19 501 cells |

The same run's 60 s matrix shows what the qualifying columns are for: the five
continuous instruments come back 100 % filled at a median of 60 contributing
samples per cell, the PTR benzene 97.5 %, and the two canister species 11.3 %
with a median of **zero**. A sparse instrument on a campaign grid is mostly
absent, which is honest, and is exactly why a two-species ratio uses a pair
clock (§11.4) rather than this product.

#### Readings that land in more than one row

What the period rule does not refuse, the grid records. A 15 s canister fill
that crosses a minute boundary contributes to both minutes, and each row
reports that fill's value, so the matrix holds one measurement in two rows and
a receptor model treating rows as independent observations counts it twice.
Measured over all ten 2024 drive days on a 60 s grid (iWAS fills with exact
cells from `iWAS_Stop_UTC`): **68 of 261 fills** land in two rows, and **320
rows** hold benzene from 261 fills. A fill crosses a boundary whenever it
starts in the last fifteen seconds of a minute, so about a quarter do.

Each gridded column therefore carries `tsara_grid_readings`, the number of
distinct finite readings behind it, and one warning names the columns whose
occupied rows outnumber their readings, worst first. One warning for the
whole grid rather than one per column, because a canister's VOCs share a
sampling pattern and would otherwise repeat the same sentence fifty times.
Like the pairing count (§11.4.1) it is a ceiling on independent rows, not an
estimate: readings shared at small weight between neighbouring rows lower the
independent information without lowering the count.

**How the rule and the record are checked.** The rule is pinned where it bites
hardest: a 2 s record refused on a 1 s grid at phases 0, 0.3 s and 0.5 s
(the case the old rule passed), a 1.9 s record allowed, mixed target widths
measured against the widest touched, a zero-width target cell ignored, and on
the grid a 1.023 s record accepted, a 60 s record refused at 30 s and accepted
at 31 s, and a wide instrument outside the window constraining nothing. The
record is pinned by a straddling fill (three rows, two readings, one warning),
a masked reading left uncounted, and ten canister columns producing one
warning that lists eight. Nine defects were injected — `>` for `>=` at exactly
two widths, the narrowest touched target instead of the widest, the old
whole-cell count, zero-width targets counted, the grid skipping its check,
masked readings counted, a warning when rows merely equal readings, every
column listed, and the widest-cell attribute taken as a median — and all nine
were caught, after the first run caught seven: no test had a masked value in a
gridded column, or cells of varying width, which are the only cases where
those two defects are visible.

**And on real data**, through the new code: the 2026-01-19 LANL pair now grids
at one second, with `ch4_pico` warned as 17 517 rows from 17 069 readings and
the ultra silent; the ten-day 60 s drive grid warns for benzene, 320 rows from
261 readings.

#### A uniform grid spans the gaps

The grid runs from the first selected cell to the last, so a campaign of
separate drives puts every hour between them into the matrix. Measured on the
Picarro CO₂, CH₄ and MetNav wind direction of the ten 2024 drive days: a 60 s
grid spans 29.5 days in 42 443 rows, 7.9 % of them holding any data; a 1 s grid
spans the same 29.5 days in **2 546 521 rows, 92.1 % empty**, built in 1.5 s
but holding 285 MB for three variables — about 95 MB per variable once its
count, coverage and angular columns are included, so a few dozen VOCs is
several gigabytes of mostly `nan`. For a receptor-model matrix, grid each drive
with `start` and `end`; the uniform span is for the continuous state, which
wants the gaps.

#### Where the cells fall

Abutting cells of exactly the requested period. With no explicit `start`, the
grid begins at the largest whole multiple of the period at or before the
earliest selected cell — anchored to the epoch rather than to whenever the
data happened to start, so that two runs over overlapping periods produce
cells that line up. Grids that never share a boundary could not have their
outputs compared at all.

#### Attributes

| attribute | meaning |
|---|---|
| `tsara_grid_freq` | the period, as requested |
| `tsara_grid_widest_source_cell_s` | the widest single selected cell, in seconds |
| `tsara_grid_variables` | which variables were selected, instrument-qualified |

and per variable, `tsara_grid_readings`: the distinct finite readings behind
the column's occupied rows.

The last is recorded because a reader cannot tell from the columns alone
whether a variable is absent because it was excluded or because it had no
data.

#### Persistence

`save_grid` writes `grid.nc` into a bundle directory and `load_grid` reads it
back. It does **not** touch `bundle.json`: that descriptor records which stage
created the bundle and what streams it wrote, and a grid is a different
stage's product arriving later, so editing it would make it say something its
writer never said. The grid carries its own provenance in its attributes
instead, which is what §1 asks of every saved output anyway — and loading
checks `tsara_stage`, because a stream and a grid are both netCDF files with a
time axis and reading one as the other would produce a plausible object with
the wrong meaning. Writing also refuses a grid whose bounds were destroyed
upstream, rather than producing a product claiming cells it does not have.

#### 11.7.1 Why there is no median binning option

`OutputGridConfig.bin_statistic` was specified in Phase 1 with values `mean`
and `median`, and removed in Phase 4 before it was ever implemented — Phase 4
would have been its first consumer.

The purpose of a median is robustness to sub-grid spikes. In this science a
sub-grid spike **is the plume**, which is the same argument that deleted the
QA/QC spike rule (§9.5): TSARA has no stage that can distinguish a sharp real
enhancement from a glitch, so it should not offer a statistic whose only job
is to suppress one.

Measured on the real 1 Hz mobile-lab CH₄, ten drive days, binned to 60 s cells
with the enhancement taken over a rolling 10-minute 5th-percentile baseline:

| | |
|---|---|
| enhanced cells (mean enhancement > 5 ppb) | 2329 |
| **enhancement mass the median discards** | **19.3 %** |
| cells whose plume fills less than half the minute | 49 (2.1 %) |
| …median loses, over those cells | 87.5 % |
| worst single cell | a 28 ppb enhancement reduced to zero |

The 19.3 % is the number that decides it. It is not noise, it is a *bias*, and
it scales with how sharp each species' plumes are relative to the cell — so
two species with different plume widths are biased by different amounts and
their ratio, computed from a median-binned PMF matrix, is wrong by the
difference. That is precisely the quantity TSARA exists to produce.

Two lesser costs, recorded because they were part of the decision: an
overlap-weighted median is a different algorithm from a weighted mean, so it
would be a second code path through binning *and* uncertainty propagation; and
the median standard-error inflation factor $\sqrt{\pi/2} \approx 1.253$ that
the deleted median-binning section specified is valid only for Gaussian noise
under equal weights,
which overlap weighting does not provide.

A caller who wants robustness has the tools: QA/QC `range` bounds and `flag`
columns act where the information about what is a glitch actually lives (§9.5).

### 11.8 The AR(1) model against generated ground truth

§3.4 closed half of the effective-sample-size question by measurement: the
finite-*N* form *solves* the AR(1) model correctly, and the large-*N*
approximation does not. It could not close the other half — whether the AR(1)
model describes real instrument error at all — because that is not a question
about algebra.

The generator can answer it. It injects error with a **declared** τ, and it
keeps the noise-free truth beside the observable. Bin both onto wide cells and
the difference between them *is* the error of the binned value: no baseline
estimate, no plume, nothing else in it. The scatter of that difference across
cells is what the uncertainty should have been.

Measured on six hours of 1 Hz data with a 5 ppb absolute random component,
binned to 60 s cells (358 cells, so the scatter itself is good to about 4 %):

| declared τ | form | observed scatter | reported σ |
|---|---|---|---|
| none | `independent` | 0.677 | 0.646 |
| 30 s | `ar1_neff` | 4.018 | 3.768 |
| 30 s | `ar1_asymptotic` | 4.018 | 5.000 |

Three things follow.

**The AR(1) model is describing the error.** With a 30 s timescale the binned
values scatter by 4.0 ppb, and the finite-*N* form predicts 3.8 — inside the
measurement's own precision. The model is not merely solved correctly; it is
approximately right about this error.

**The naive rule is not wrong at the margin, it is wrong by a factor of six.**
Independent averaging would have reported 0.646 ppb for the same cells. A
confidence interval built on it would be six times too narrow, which is why
§10.8 refuses to apply √N when no τ is declared rather than applying it
hopefully.

**The asymptotic form errs the other way, and saturates.** At N = 60 with
τ = 30 s it returns exactly N_eff = 1 — the whole minute worth one sample —
and so reports 5.000, overstating the real scatter by 24 %. Between the two
registered approximations the finite-*N* one is closer to the truth, which
is the empirical half of the §3.4 argument.

The residual disagreement is real and worth stating rather than rounding away:
the observed scatter implies N_eff ≈ 1.55 where `ar1_neff` computes 1.76. The
model is an approximation to a real error process, and this is the size of
that approximation on generated data whose τ is exactly known. A user whose
instrument does not decorrelate as an AR(1) process should expect no better.

**Which is also why estimating τ from data is a §7 avenue rather than a
field.** No file declares one, and this measurement shows what a wrong one
costs.

#### 11.8.1 A measured number without its denominator

Recorded because this is the third phase in which it has happened. The
canister figures above were first written as "14.7 s fills every 441 s",
measured from **one** drive day and quoted as though they described the
instrument. Re-run at the phase boundary across all ten 2024 drive days and
261 fills, the medians are **14.9 s and 530 s**, and the text was corrected
from "thirty times slower by rate" to "thirty-five".

Both of those were wrong, and the correction repeated the error. Each divided
the canister's cadence by the canister's *own* fill width (441 ÷ 14.7 = 30,
530 ÷ 14.9 = 36), which is a duty cycle, not a comparison with anything. The
claim is about a 60 s mean, against which the canister is about **seven** times
slower by rate on that one day (441 ÷ 60) and about **nine** across all ten
(530 ÷ 60). Found in the Phase-4 walkthrough (§11.4.1 came from the same
stage) and corrected wherever it was quoted.

The conclusion survives untouched, which is exactly why both errors were
invisible: rate and support still disagree, and the canister is still narrower
by support than a 60 s mean while being far slower by rate. What was wrong the
first time was the *provenance*. A number quoted without saying what it was
measured over cannot be checked by anyone, including its author a week later.
What was wrong the second time is the complement: re-measuring the inputs does
not re-check the arithmetic that consumes them, so a stated ratio should show
its numerator and denominator.

Every archive-derived number in §11 has now been re-run and states its sample.
The day-specific checks say which day; the instrument-level claims say how many
fills over how many days.

### 11.9 The acceptance criterion

One source drives two species. One is measured every second, the other once a
minute, and the ratio between them is fixed and known. If pairing averages
both members over the same air, that ratio comes back; if it averages them
over *different* air — an off-by-one cell, a label read as a midpoint when it
was a start, an overlap weight applied to the wrong partner — the ratio
drifts, and nothing else in the suite would say so.

| quantity | result |
|---|---|
| mass-weighted ratio, paired | 0.250000034 against a truth of 0.25 |
| per-cell ratio, median relative error | 8 × 10⁻⁶ |
| worst per-cell error, fully covered | 3 × 10⁻⁴, on the smallest enhancement |

The per-cell residual is the generator's, not the aligner's: it is the
midpoint-rule quadrature the slow instrument uses to average truth over its
own 60 s cells, sixty times coarser than the fast instrument's. It appears as
a *relative* error only where the enhancement is a couple of ppb, which is the
signature of a fixed absolute error rather than a ratio bias.

**Coverage was scored against the same truth.** Exactly one cell of 120 is
partly covered — the fast instrument's record begins inside the slow
instrument's first cell — and it is the only cell whose ratio is wrong, by
10 %. The correlation between one-minus-coverage and absolute ratio error is
**1.000**. The number that qualifies a pair does what it claims.

#### 11.9.1 A grid out of phase with its source blends two cells

Found by running the same check through the grid. The slow instrument's cells
are centred on its timestamps, so they sit half a cell off an epoch-anchored
grid of the *same* period. Every grid value is then a weighted mean of two
adjacent source cells — an honest average, but a smoothed one — and the
recovered ratio moves from **0.250000034** to **0.249940526**. Aligning the
grid's start to that instrument's own cell boundaries restores it exactly.

Nothing is silently wrong: `n_source` reports 2 instead of 1, and `grid_cells`
warns when it detects a source whose cells match the period but not its phase.
It is a warning rather than a refusal because blending is sometimes
unavoidable, and which instrument a grid should be in phase with is the user's
choice, not TSARA's.

---

## References

- JCGM 100:2008. *Evaluation of measurement data — Guide to the expression of
  uncertainty in measurement (GUM).* (Correlated-error propagation, §3.)
- York, D., Evensen, N. M., Martínez, M. L., & Delgado, J. D. B. (2004).
  Unified equations for the slope, intercept, and standard errors of the best
  straight line. *American Journal of Physics*, 72(3), 367–375.
- Cantrell, C. A. (2008). Technical Note: Review of methods for linear
  least-squares fitting of data and application to atmospheric chemistry
  problems. *Atmospheric Chemistry and Physics*, 8, 5477–5487.
- Wu, C., & Yu, J. Z. (2018). Evaluation of linear regression techniques for
  atmospheric applications: the importance of appropriate weighting.
  *Atmospheric Measurement Techniques*, 11, 1233–1250.
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics.* Wiley.
  (Circular mean, mean resultant length and circular standard deviation, §11.5.)
- Yamartino, R. J. (1984). A comparison of several "single-pass" estimators
  of the standard deviation of wind direction. *Journal of Climate and
  Applied Meteorology*, 23, 1362–1366.
