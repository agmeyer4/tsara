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
self-describe: xarray/parquet attributes record the package version, the
resolved configuration, the algorithm names used, and the uncertainty
provenance of every interval (see §2.4).

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
| Ratio regression | **pairing clock** (§1.3), per event/window | the slower stream's **cells**, with the faster stream averaged onto them by overlap |
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

1. The **pairing clock** is the *slower* of the two instruments' own **cells**,
   restricted to the event/window.
2. The faster stream is averaged onto those cells **weighted by overlap**
   (`tsara.core.support.bin_onto_cells`), with uncertainty propagated per §3,
   the count of contributing samples recorded, and the fraction of each cell
   actually covered recorded alongside it. Before Phase 3.5 the cell was
   *assumed* to be the slow instrument's sampling period centred on its
   timestamp; now it is read from the stream's own boundaries, so a canister
   that integrated for 14.1 s is paired over exactly 14.1 s (§10).
3. Cells with `n_native = 0` for either species are dropped — a pair is never
   fabricated.

Consequences: every regression point contains at least one real measurement
of *each* species, and the regression sample size N equals the number of real
pairs. This eliminates interpolation pseudo-replication (interpolated points
posing as independent samples and silently inflating degrees of freedom).

Different species pairs may therefore be paired on different clocks (each pair
uses its own slower member). Each ratio is an independent slope estimate with
its own honest CI; no shared clock across pairs is required for the ratios to
be comparable as estimates.

### 1.4 Output grid

A single uniform master grid — the familiar `(time × species)` cube — is
constructed **only** for the continuous rolling state and the PMF export
matrix, which inherently require one. Construction is binning-only
(`bin_statistic`), with per-cell propagated uncertainties and `n_native`
counts carried alongside values. Grid cells carry CF boundaries like any
other stream, and `cell_methods = "time: mean (interval: <native>)"` records
the resolution of the data that went into them (§10.2). Validation requires the grid period to be
≥ the slowest stream's native period; cells with no native samples are NaN
with `n_native = 0`, never interpolated. Config: `OutputGridConfig`
(`tsara.config.analysis`) — deliberately not named "the grid" or paired with
the aux-interpolation guard, since neither streams nor cross-species pairing
(§1.3) use it; it exists solely for this output boundary.

### 1.5 Circular statistics for angular variables **[stub — Phase 4]**

Wind direction and other angular quantities are averaged as unit vectors
(never arithmetically). Dispersion via the Yamartino (1984) single-pass
estimator or exact circular standard deviation — to be specified in Phase 4.

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
one-minute product; §10.8 sets out when TSARA will move it onto the stream's
own cells and the three cases in which it refuses to. Omitting a component means "not
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

Provenance is recorded **per component**, because real manifests mix modes
freely (the shipped example pairs a *reported* random component with a
*declared* systematic one). The species-level label is then `mixed` when the
two components disagree. Two further component values are needed to keep §2.3
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

For a random component with declared decorrelation timescale τ, v1 models the
error autocorrelation as AR(1)-like, $\rho(\Delta t) = e^{-|\Delta t|/\tau}$,
and applies the standard effective-sample-size correction with lag-1
correlation $\rho_1 = e^{-\Delta t/\tau}$:

$$N_{\mathrm{eff}} = N\,\frac{1-\rho_1}{1+\rho_1}, \qquad
N_{\mathrm{eff}} \in [1, N],$$

so $\sigma_{\bar{x}}^2 \approx \sigma^2 / N_{\mathrm{eff}}$. The exact double
sum (§3.1) remains available as a slower, assumption-free alternative; τ → 0
recovers §3.2 and τ → ∞ is handled by declaring the component systematic
instead. *(Exact N_eff form is an open flag — see CLAUDE.md §5.)*

This form has one implemented consumer so far: moving a declared σ from the
interval it was quoted at onto the cells it actually describes (§10.8), where
$N$ is the number of quoted intervals per cell. Worked numbers there show why
the naive $\sqrt{N}$ shortcut is not merely imprecise but confidently wrong.

### 3.5 Median binning

When `bin_statistic: median` is selected, the standard error of the median is
inflated relative to the mean by $\sqrt{\pi/2} \approx 1.253$ for Gaussian
noise; the propagation applies this factor to the random component. (Median
binning trades this efficiency loss for robustness to sub-grid spikes.)

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
| apply QA/QC | bounds are written in the units the author thinks in |
| resolve uncertainty | `absolute` is declared in canonical units (§2.2) |

`range` bounds are physical statements in canonical units — the shipped
example converts ppm→ppb and then bounds in ppb, so masking before
conversion would compare ppb bounds against ppm numbers and reject the whole
record. `flag` reads a separate instrument status column, which is never a
converted quantity.

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
period twice, and the remedy is in the manifest's path templates. The split
is counted per file *before* concatenation, which makes it exact rather than
heuristic: after concatenation the two are indistinguishable.

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
  variable, including the `sigma_rand_`/`sigma_sys_` companions — a random
  error describes the same cell as the value it belongs to.

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
averaging had happened. Instrument response and cavity residence are
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
`_width_source`, `_method_source` — because the three are established
independently. A stationary analyzer whose file carries a stop column and
whose manifest declares `method: mean` is honestly reported / reported /
declared, and one label per stream could not say that. Reconciling several
files takes the **weakest** value rather than the majority, so one
undocumented file cannot hide inside a well-documented instrument.

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

The rejected alternative — width as the distance to the next row — was
rejected on measurement, not on taste. Classifying every sampling interval
across 303 files spanning the whole 2024 archive and the 2026 aligned stage,
for continuous instruments:

| What the interval is | Mean fraction |
|---|---|
| Jitter around the file's one nominal cadence | 0.99218 |
| A dropped row (an integer multiple of the cadence) | 0.00570 |
| Genuinely something else | 0.00211 |

Only 9 of 299 files have more than 1 % "other". So one width per file
describes the data, and:

> **A dropped row leaves a hole in the tiling. It never produces a wider
> cell.**

Under the rejected rule the widest single cell in that sample would have been
**23.3 days** — a 10 s nominal GPS record whose file spans a campaign. 33 of
299 files have a gap exceeding 100× their cadence. Each stream therefore
carries a duty-cycle diagnostic, `tsara_cell_coverage`, the share of the
record's extent that cells actually cover: median 1.0000 across the sample,
with 32 of 299 files below 0.95 and one at 0.050.

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
label, anything else yields `unknown` and a centred cell. Sampled across the
archive, 18 of 29 files infer `start` and the rest fall back to centred.
Assuming `start` everywhere because the specification says so would put a 30 s
error on every cell of the minute-average suite, in the direction nobody would
think to check. A hint is used only when every file of an instrument agrees;
picking a winner would put half of them half a cell out.

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
available.

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

**Rescaling happens only when `decorrelation_timescale` is declared.** How
much averaging helps depends entirely on how quickly the error forgets itself,
and nothing in a file reveals that. With $\rho_1 = e^{-w/\tau}$ for a quoted
interval $w$ and a cell holding $N = W/w$ of them, the §3.4 form gives
$N_{\mathrm{eff}} = N(1-\rho_1)/(1+\rho_1)$ clamped to $[1, N]$, and
$\sigma_{\mathrm{cell}} = \sigma/\sqrt{N_{\mathrm{eff}}}$. For a 1 s figure on
60 s cells:

| τ | N_eff | σ falls by | Naive √N would claim |
|---|---|---|---|
| 0 | 60.00 | 7.75× | 7.75× |
| 5 s | 5.98 | 2.45× | 7.75× |
| 20 s | 1.50 | 1.22× | 7.75× |
| 5 min | 1.00 | 1.00× | 7.75× |

At τ = 20 s the naive answer is over six times too confident; at τ = 5 min the
averaging buys nothing at all. Guessing a timescale in order to apply a
correction would be worse than applying none, so three cases are refused, each
recorded in `uncertainty_at_width_status` rather than only logged:

1. **No decorrelation timescale** — the correction is unknowable.
2. **Cells finer than the quoted interval** — recovering noise below that
   interval assumes the error is white down there, a claim about the
   instrument that no manifest has made.
3. **A systematic component quoted at an interval** — a systematic error is
   correlated across samples by definition and does not average down at all
   (§3.3), so the interval cannot change it.

Widths are per row, so a sampler whose fills vary is corrected cell by cell.

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

End to end on the real archive, the stationary Picarro suite (50,400 rows
across 35 days) ingests three ways to the same 60 s start-labelled cells at
coverage 1.000, by three independent routes that each say honestly how they
know: `reported` from the file's stop column, `declared` from the manifest,
and `inferred` from the `StartTime_UTC` column name.

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
  about 225 m. Cell position is the track at the cell midpoint; a
  `path_length_m` attribute is a future addition for clustering.
- **The declared-versus-empirical closure diagnostic** proposed during
  scoping — comparing a declared σ against `diff_mad` at the delivered support
  — is not built. The empirical estimator belongs to the analysis
  configuration, and ingestion computing it would mean reading a config this
  stage has no business reading (§9.6). It belongs to the phase that owns the
  noise estimator.

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
- Yamartino, R. J. (1984). A comparison of several "single-pass" estimators
  of the standard deviation of wind direction. *Journal of Climate and
  Applied Meteorology*, 23, 1362–1366.
