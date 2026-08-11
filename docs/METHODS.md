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
