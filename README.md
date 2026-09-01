# TSARA — Time Series and Ratio Analyses

TSARA turns raw, multi-rate trace gas timeseries — from fixed sites and from
vehicles — into **enhancement ratios with defensible uncertainties**. It reads a
campaign's archive as described by a YAML manifest, keeps every instrument on its
own clock, computes rolling baselines, detects plume events, and fits ratios
between species with the measurement error in *both* axes carried through to the
answer. The output is both a catalog of discrete plume events and a continuous
rolling state, ready for source fingerprinting and downstream receptor modeling
(e.g. PMF).

The mathematics, the rationale, and the alternatives that were rejected all live
in [`docs/METHODS.md`](docs/METHODS.md). It is written as the authoritative
methods document, not as an afterthought: every algorithm TSARA ships has a
section there before it has a caller.

## Status

**Alpha — phases 1–3 of a 10-phase roadmap are complete.** The package is built
one phase per review cycle, and only what is listed as done below exists.

| Phase | | |
|---|---|---|
| 1 | Configuration layer (Pydantic schemas, YAML loader, logging) | ✅ done |
| 2 | Synthetic data generator with ground-truth plumes *and* ground-truth error | ✅ done |
| 3 | Ingestion (reader registry, CSV / ICARTT / Parquet, crawler, QA/QC, units, uncertainty) | ✅ done |
| 4 | Alignment & pairing (native-rate pairing, error propagation, circular stats, output grid) | planned |
| 5 | Baselines + continuous rolling state | planned |
| 6 | Plume detection + nested-event bookkeeping | planned |
| 7 | Regression (OLS / York / ODR), combined UQ, stability cube | planned |
| 8 | Smoothing + spatiotemporal source complexes | planned |
| 9 | `Pipeline` class + `tsara` CLI | planned |
| 10 | Docs + tutorial | planned |

So today TSARA can **manufacture a campaign with a known answer key, and read a
real one into analysis-ready streams**. It cannot yet compute baselines, ratios,
or the stability cube; there is no CLI yet (phase 9).

## Install

Python 3.11 or newer. Not on PyPI — install the checkout in editable mode:

```bash
git clone https://github.com/agmeyer4/tsara.git
cd tsara
pip install -e ".[dev,viz]"
```

The extras are deliberately split. `dev` is the test and notebook toolchain
(pytest, ruff, mypy, nbconvert); `viz` is matplotlib alone, kept out of the
runtime dependencies so a headless batch run on a cluster never has to install a
plotting stack.

## Quickstart

TSARA ships a synthetic generator, so the whole ingestion path runs with no data
of your own. From the repository root:

```python
from tsara import load_manifest, load_synthetic, setup_logging
from tsara.ingest import ingest_campaign, save_streams
from tsara.synthetic import export_raw, generate

setup_logging()  # a library logs nothing until an application asks it to

# 1. Manufacture a campaign whose answer key you already have.
dataset = generate(load_synthetic("examples/configs/synthetic_example.yaml"))
print(sorted(dataset.streams), len(dataset.ground_truth.events), "events")

# 2. Write it out as raw CSV plus the manifest that describes those files.
manifest_path = export_raw(dataset, "demo_campaign")

# 3. Read it back the way a real archive is read.
streams = ingest_campaign(load_manifest(manifest_path))
for name in streams:
    print(name, dict(streams[name].sizes), list(streams[name].data_vars))

# 4. Checkpoint the stage product.
save_streams(streams, "demo_bundle")
```

```
['aeris', 'gps', 'met', 'picarro'] 170 events
aeris {'time': 42977} ['ch4', 'sigma_rand_ch4', 'sigma_sys_ch4', 'c2h6', ...]
picarro {'time': 10800} ['co2', 'sigma_rand_co2', 'sigma_sys_co2']
met {'time': 2160} ['wind_dir', 'wind_speed']
gps {'time': 21600} ['latitude', 'longitude']
```

Four instruments, four different sampling rates, four `time` axes — that is the
point, not an accident (see *Streams stay at native rate*, below).

## Reading your own campaign

Ingestion is driven entirely by a manifest, so adding an instrument is a YAML
edit rather than a code change:

```yaml
name: uu_rooftop_2025
base_path: ../data/raw
platform:
  kind: stationary          # or `mobile`, with a `gps_instrument`
  latitude: 40.7649
  longitude: -111.8421
instruments:
  picarro:
    loader:
      format: csv           # csv | icartt | parquet
      path_template: "picarro/%Y/%m/%d/*.dat"
      time: {column: EPOCH_TIME, format: unix, timezone: UTC}
    variables:
      ch4:
        column: CH4_dry
        role: gas
        units: ppm
        convert: {from_unit: ppm, to_unit: ppb, scale: 1000.0}
        uncertainty:
          random: {mode: declared, absolute: 0.5, relative: 0.001}
```

Then `ingest_campaign(load_manifest("manifest.yaml"))`. Fuller, commented
examples ship in [`examples/configs/`](examples/configs/):

| file | shows |
|---|---|
| `manifest_stationary_example.yaml` | fixed site, one static coordinate |
| `manifest_mobile_example.yaml` | vehicle, GPS instrument, systematic uncertainty, reported per-point error |
| `manifest_multiformat_example.yaml` | one campaign mixing CSV, ICARTT and Parquet, several directory layouts |
| `synthetic_example.yaml`, `synthetic_bootstrap.yaml` | generating data, parametrically or bootstrapped from a real record's residuals |
| `analysis_example.yaml` | the analysis side: baseline sweeps, detection, regression, clustering |

## The ideas the API is shaped around

**Streams stay at native rate.** A 10 Hz analyzer is not resampled onto a 1 Hz
GPS clock at ingestion, or ever, on the gas side. Interpolating a gas creates
samples that were never measured and would inflate a regression's `N` with
pseudo-replicates. Streams are paired late, per event, on the slower clock, with
the faster stream bin-averaged onto it. Smooth auxiliary fields (GPS, met) *may*
be interpolated, under an explicit maximum-gap guard. (`METHODS.md` §1.1)

**Uncertainty is first class, and never assumed.** Every variable can declare a
two-component budget: a `random` part that averages down and a `systematic` part
that does not, each either declared in the manifest or read from a per-point
error column the instrument reports. Components propagate separately — quadrature
for random, weighted mean of sigmas for systematic — and every value carries a
provenance label (`declared`, `reported`, `empirical`, `zero`, `unknown`), so a
budget nobody stated can never be mistaken for a budget that is zero.
(`METHODS.md` §2)

**One reader seam.** A reader's entire job is `(path, loader config) → RawTable`:
a frame on UTC nanosecond timestamps, columns still named as the raw file names
them. Everything after that — units, QA/QC, uncertainty, assembly — is written
once and is format-independent. New formats register themselves by name
(`@register_reader("csv")`), the same pattern used for swappable noise and
regression estimators. (`METHODS.md` §9.1)

**Files are read as they actually are.** Real archives are not
specification-compliant, so the ICARTT reader settles disagreements by measuring
the data rather than trusting the header, masks both sentinel families
(missing *and* below/above detection limit) while keeping the counts, and
reconciles what a campaign's files say about themselves by joining
disagreements rather than picking a winner. (`METHODS.md` §9.2)

**Every stage saves itself.** Each phase ships persistence for the products it
introduces, so a long run can be inspected in a notebook, resumed after a crash,
and audited later. A bundle is a plain directory: `bundle.json`, the resolved
`manifest.yaml`, and one netCDF per stream under `streams/`.

## Repository layout

```
src/tsara/
  config/      Pydantic schemas + YAML loader (manifest, analysis, synthetic)
  core/        Shared primitives: exceptions, logging, timebase, geodesy, naming
  ingest/      Readers, crawler, QA/QC, units, uncertainty, campaign, bundles
  synthetic/   Ground-truth data generation, profiling, raw-file export
docs/METHODS.md   The methods document: mathematics, rationale, rejected options
examples/configs/    Commented YAML for every schema
examples/notebooks/  Executed walkthroughs, outputs committed
tests/               pytest suite, 100% line + branch coverage enforced
```

## Notebooks

Both are committed with their outputs, so they read on GitHub without being run,
and neither needs any real data:

- [`01_synthetic_data_walkthrough.ipynb`](examples/notebooks/01_synthetic_data_walkthrough.ipynb)
  — what the generator makes, and what "ground truth" means for error as well as
  for plumes.
- [`02_ingestion_walkthrough.ipynb`](examples/notebooks/02_ingestion_walkthrough.ipynb)
  — a campaign built, exported as raw files, and read back: path templates,
  awkward ICARTT, unit conversion, QA/QC, uncertainty provenance, bundles.

## Development

```bash
pytest                      # suite; the 100% line+branch floor is enforced in config
ruff check . && ruff format --check .
mypy --strict src tests
```

A few house rules worth knowing before contributing: every algorithm gets a
`docs/METHODS.md` section in the phase that introduces it; every stage product
gets save/load in that same phase; swappable estimators are registered by name
and every registered name has a methods section. Tests that enumerate their own
subjects by hand are treated as bugs — discover the subjects and guard against
the discovery returning nothing.

## License

MIT — see [LICENSE](LICENSE).
