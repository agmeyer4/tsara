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
instrument's own native timestamps. Species measured by the same instrument
share a clock and live in the same Dataset (so per-instrument operations remain
vectorized over species). No resampling of any kind happens at ingestion.

The pipeline defers any change of clock to the last possible moment:

| Stage | Clock used |
|---|---|
| QA/QC, unit conversion | native |
| Rolling baseline, enhancement Δ | native (time-based windows) |
| Plume detection | native → events are time **intervals** |
| Ratio regression | **pairing clock** (§1.3), per event/window |
| Continuous rolling state, PMF matrix | **output grid** (§1.4) |

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

A concentration inside a plume is not a smooth field; linearly interpolating
it invents structure exactly where the science happens. Platform position and
ambient met vary smoothly on sampling timescales, so interpolation onto gas
timestamps is physically justified there — but never across gaps longer than
`max_interp_gap` (config: `AlignmentConfig.max_interp_gap`, `tsara.config.analysis`).

### 1.3 Pairing for regression (fast → slow, never the reverse)

To regress species *y* against species *x* within an event or rolling window,
when they come from instruments with different rates:

1. The **pairing clock** is the native timestamp set of the *slower* of the
   two instruments, restricted to the event/window.
2. The faster stream is bin-averaged into cells centered on those timestamps
   (cell width = the slow instrument's sampling period), with uncertainty
   propagated per §3 and the count of native samples per cell,
   `n_native`, recorded.
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
counts carried alongside values. Validation requires the grid period to be
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
carrying τ for the random component (§3.4). Omitting a component means "not
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
