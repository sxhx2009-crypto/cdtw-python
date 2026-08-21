"""Does this CDTW implementation actually behave like CDTW on real data?

Buchin, Nusser and Wong motivate CDTW by saying it is robust to the sampling
rate, unlike DTW.  That claim is falsifiable on real signals, so this script
measures it directly and contrasts it with DTW.

Real data cannot establish mathematical correctness -- no dataset carries a
ground-truth CDTW value, and that question is settled instead by the closed
form derivation, the independent grid solver and the path re-integration in
``validation/``.  What real data can do is check that the *defining* property
of a curve measure, invariance to how the curve was sampled, actually shows up,
and map where it stops holding.

Four measurements:

1. characterization  -- how much of a real signal is turning points, and how
   large the total variation is next to the amplitude.
2. grid convergence  -- whether ``grid_size`` has converged on real windows.
3. sampling rate     -- CDTW against DTW under upsampling, decimation and
   irregular sampling.
4. noise control     -- a synthetic sweep isolating noise as the cause of the
   decimation behaviour, plus the same sweep by smoothing real data.

Run from the repository root::

    python experiments/sampling_rate_experiment.py

Needs network access on the first run; the downloads are cached under
``experiments/_cache`` afterwards.  Results are written to
``experiments/sampling_rate_results.json``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdtw import _as_curve, cdtw_distance  # noqa: E402

CACHE = Path(__file__).resolve().parent / "_cache"
RESULTS = Path(__file__).resolve().parent / "sampling_rate_results.json"
SEED = 20260820
WINDOW = 256
GRID = 64  # the pilot below shows real windows are converged well before this

SOURCES = {
    # Real electricity-transformer sensor readings, hourly.  "OT" is oil
    # temperature.
    "ett_OT": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
        "ETT-small/ETTh1.csv",
        "OT",
    ),
    # Daily minimum temperatures, Melbourne.
    "temps": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "daily-min-temperatures.csv",
        "Temp",
    ),
    # Monthly sunspot counts.
    "sunspots": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "monthly-sunspots.csv",
        None,  # resolved below: the one column that is not the date
    ),
}

# Disjoint windows chosen once, before any result was looked at.
WINDOWS = {"ett_OT": (1000, 5000), "temps": (500, 2000), "sunspots": (300, 1500)}


def api_url(raw: str) -> str:
    """Rewrite a raw.githubusercontent URL as a GitHub contents API URL.

    Different host, so it still works in networks where raw.githubusercontent
    is blocked -- which happened while this experiment was being written.
    """

    owner, repo, ref, path = raw.removeprefix(
        "https://raw.githubusercontent.com/"
    ).split("/", 3)
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "curl/8", "Accept": "application/vnd.github.raw"}
    )
    return urllib.request.urlopen(request, timeout=90).read()


def load(name: str) -> tuple[np.ndarray, str, str, str] | None:
    """Return the series, the URL used, the file sha256 and the series sha256.

    Returns ``None`` when every source is unreachable.  A blocked or
    rate-limited upstream should degrade the experiment, not abort it: the
    conclusions rest on agreement across several independent signals, so any
    two of them still carry the result.
    """

    url, column = SOURCES[name]
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{name}.csv"
    used = url
    if cached.exists() and cached.stat().st_size > 0:
        payload = cached.read_bytes()
        used = f"{url} (cached)"
    else:
        payload, last = None, None
        for candidate in (url, api_url(url)):
            for attempt in range(3):
                try:
                    payload = fetch(candidate)
                    used = candidate
                    break
                except Exception as error:  # pragma: no cover - network dependent
                    last = error
                    time.sleep(2.0 * (attempt + 1))
            if payload is not None:
                break
        if payload is None:  # pragma: no cover - network dependent
            print(f"  SKIP {name}: unreachable ({str(last)[:60]})")
            return None
        cached.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8", "replace"))))
    if column is None:
        column = [key for key in rows[0] if not key.lower().startswith(("date", "month"))][0]
    values = np.array([float(row[column]) for row in rows], dtype=np.float64)
    # The parsed numbers are what the results actually depend on, so hash them
    # too: CSV formatting can change upstream without changing the series.
    series_digest = hashlib.sha256(values.tobytes()).hexdigest()
    return values, used, digest, series_digest


def dtw(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Textbook DTW: the raw sum, and the sum divided by the warping length.

    Used only as a contrast, so it is deliberately the plain formulation with
    no window and no step-pattern weighting.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.size, b.size
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        ai = a[i - 1]
        previous, current = cost[i - 1], cost[i]
        previous_steps, current_steps = steps[i - 1], steps[i]
        for j in range(1, m + 1):
            local = abs(ai - b[j - 1])
            up, left, diagonal = previous[j], current[j - 1], previous[j - 1]
            if diagonal <= up and diagonal <= left:
                best, taken = diagonal, previous_steps[j - 1]
            elif up <= left:
                best, taken = up, previous_steps[j]
            else:
                best, taken = left, current_steps[j - 1]
            current[j] = local + best
            current_steps[j] = taken + 1.0
    return float(cost[n, m]), float(cost[n, m] / steps[n, m])


def cdtw_of(a: np.ndarray, b: np.ndarray) -> float:
    return cdtw_distance(a, b, grid_size=GRID, memory_limit_mib=None)


def upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """More samples along the *same* polygonal curve (linear interpolation)."""

    source = np.arange(x.size, dtype=np.float64)
    target = np.linspace(0.0, x.size - 1, (x.size - 1) * factor + 1)
    return np.interp(target, source, x)


def decimate(x: np.ndarray, k: int) -> np.ndarray:
    """Fewer samples: a genuinely lower sampling rate, which loses detail."""

    return x[::k].copy()


def jitter(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same number of samples, irregular sample times."""

    interior = np.sort(rng.uniform(0.0, x.size - 1, x.size - 2))
    times = np.concatenate(([0.0], interior, [x.size - 1.0]))
    return np.interp(times, np.arange(x.size, dtype=np.float64), x)


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    return np.convolve(x, np.ones(width) / width, mode="valid")


def describe(series: np.ndarray) -> dict:
    curve = _as_curve(series, "series")
    variation = float(np.sum(np.abs(np.diff(series))))
    amplitude = float(series.max() - series.min())
    return {
        "samples": int(series.size),
        "vertices": int(curve.vertices.size),
        "vertex_fraction": round(curve.vertices.size / series.size, 4),
        "total_variation": round(variation, 4),
        "amplitude": round(amplitude, 4),
        "variation_over_amplitude": round(variation / max(amplitude, 1e-12), 2),
    }


def main() -> int:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    results: dict = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": SEED,
        "window_length": WINDOW,
        "grid_size": GRID,
        "datasets": {},
        "grid_convergence": [],
        "sampling_rate": {},
        "noise_control": [],
        "smoothing_sweep": [],
    }

    series = {}
    print("=== datasets ===")
    for name in SOURCES:
        loaded = load(name)
        if loaded is None:
            continue
        data, used, digest, series_digest = loaded
        series[name] = data
        i0, i1 = WINDOWS[name]
        results["datasets"][name] = {
            "url": SOURCES[name][0],
            "fetched_from": used,
            "file_sha256": digest,
            "series_sha256": series_digest,
            "length": int(data.size),
            "windows": [i0, i1],
            "window_a": describe(data[i0 : i0 + WINDOW]),
            "window_b": describe(data[i1 : i1 + WINDOW]),
        }
        a = results["datasets"][name]["window_a"]
        print(
            f"  {name:10s} n={data.size:6d}  window vertices {a['vertices']:4d}/{WINDOW}"
            f"  variation/amplitude {a['variation_over_amplitude']:6.1f}"
        )

    if len(series) < 2:
        raise SystemExit(
            f"only {len(series)} dataset(s) reachable; the comparison needs at "
            "least two.  Retry when the network is available."
        )
    results["datasets_used"] = sorted(series)

    pilot = "ett_OT" if "ett_OT" in series else sorted(series)[0]
    results["grid_convergence_dataset"] = pilot
    print(f"\n=== grid convergence on a real window ({pilot}) ===")
    p0, p1 = WINDOWS[pilot]
    a = series[pilot][p0 : p0 + WINDOW]
    b = series[pilot][p1 : p1 + WINDOW]
    previous = None
    for grid in (16, 32, 64, 128, 256, 512):
        value = cdtw_distance(a, b, grid_size=grid, memory_limit_mib=None)
        change = None if previous is None else abs(value - previous) / value
        results["grid_convergence"].append(
            {"grid_size": grid, "distance": value, "relative_change": change}
        )
        print(
            f"  grid={grid:4d}  {value:14.6f}"
            + ("" if change is None else f"   change {change:.2e}")
        )
        previous = value

    print("\n=== sampling rate: CDTW vs DTW ===")
    for name in sorted(series):
        i0, i1 = WINDOWS[name]
        a = series[name][i0 : i0 + WINDOW]
        b = series[name][i1 : i1 + WINDOW]
        base_c = cdtw_of(a, b)
        base_raw, base_norm = dtw(a, b)
        rows = []
        transforms = (
            [(f"upsample x{f}", upsample(a, f), upsample(b, f)) for f in (2, 3, 4)]
            + [(f"decimate 1/{k}", decimate(a, k), decimate(b, k)) for k in (2, 3, 5)]
            + [("irregular sampling", jitter(a, rng), jitter(b, rng))]
        )
        print(f"\n  {name}   reference CDTW={base_c:.4f}  DTW={base_raw:.4f}")
        print(f"    {'transform':22s} {'samples':>8s} {'CDTW':>10s} {'DTW':>10s} {'DTW/len':>10s}")
        for label, a2, b2 in transforms:
            value = cdtw_of(a2, b2)
            raw, norm = dtw(a2, b2)
            row = {
                "transform": label,
                "samples": int(a2.size),
                "cdtw_ratio": value / base_c,
                "dtw_ratio": raw / base_raw,
                "dtw_normalized_ratio": norm / base_norm,
            }
            rows.append(row)
            print(
                f"    {label:22s} {a2.size:8d} {row['cdtw_ratio']:10.6f}"
                f" {row['dtw_ratio']:10.4f} {row['dtw_normalized_ratio']:10.4f}"
            )
        upsamples = [r for r in rows if r["transform"].startswith("upsample")]
        summary = {
            "cdtw_max_deviation": max(abs(r["cdtw_ratio"] - 1.0) for r in upsamples),
            "dtw_max_deviation": max(abs(r["dtw_ratio"] - 1.0) for r in upsamples),
            "dtw_normalized_max_deviation": max(
                abs(r["dtw_normalized_ratio"] - 1.0) for r in upsamples
            ),
        }
        results["sampling_rate"][name] = {
            "reference": {"cdtw": base_c, "dtw": base_raw, "dtw_normalized": base_norm},
            "rows": rows,
            "upsample_summary": summary,
        }
        print(
            f"    upsampling only -- CDTW {summary['cdtw_max_deviation']:.2e},"
            f" DTW {summary['dtw_max_deviation']:.2e},"
            f" DTW/len {summary['dtw_normalized_max_deviation']:.2e}"
        )

    print("\n=== noise control: is noise what breaks decimation? ===")
    length = 512
    t = np.linspace(0.0, 6.0 * np.pi, length)
    clean_a, clean_b = np.sin(t), 1.3 * np.sin(t + 0.7)
    control = np.random.default_rng(1)
    print(f"  {'sigma':>7s} {'vertices':>10s} {'var/amp':>8s} {'1/2':>9s} {'1/3':>9s} {'1/5':>9s}")
    for sigma in (0.0, 0.001, 0.01, 0.05, 0.2):
        a = clean_a + sigma * control.normal(0.0, 1.0, length)
        b = clean_b + sigma * control.normal(0.0, 1.0, length)
        base = cdtw_of(a, b)
        ratios = [cdtw_of(decimate(a, k), decimate(b, k)) / base for k in (2, 3, 5)]
        info = describe(a)
        results["noise_control"].append(
            {
                "sigma": sigma,
                "vertices": info["vertices"],
                "variation_over_amplitude": info["variation_over_amplitude"],
                "decimation_ratios": {"1/2": ratios[0], "1/3": ratios[1], "1/5": ratios[2]},
            }
        )
        print(
            f"  {sigma:7.3f} {info['vertices']:6d}/{length:<3d} "
            f"{info['variation_over_amplitude']:8.1f} "
            + " ".join(f"{r:9.4f}" for r in ratios)
        )

    smooth_on = "temps" if "temps" in series else sorted(series)[0]
    results["smoothing_dataset"] = smooth_on
    print(f"\n=== the same sweep by smoothing real data ({smooth_on}) ===")
    s0, s1 = WINDOWS[smooth_on]
    raw = series[smooth_on]
    a0, b0 = raw[s0 : s0 + 512], raw[s1 : s1 + 512]
    print(f"  {'width':>7s} {'vertices':>10s} {'var/amp':>8s} {'1/2':>9s} {'1/3':>9s} {'1/5':>9s}")
    for width in (1, 3, 7, 15, 31, 63):
        a, b = smooth(a0, width), smooth(b0, width)
        base = cdtw_of(a, b)
        ratios = [cdtw_of(decimate(a, k), decimate(b, k)) / base for k in (2, 3, 5)]
        info = describe(a)
        results["smoothing_sweep"].append(
            {
                "smoothing_width": width,
                "vertices": info["vertices"],
                "samples": info["samples"],
                "variation_over_amplitude": info["variation_over_amplitude"],
                "decimation_ratios": {"1/2": ratios[0], "1/3": ratios[1], "1/5": ratios[2]},
            }
        )
        print(
            f"  {width:7d} {info['vertices']:6d}/{info['samples']:<3d} "
            f"{info['variation_over_amplitude']:8.1f} "
            + " ".join(f"{r:9.4f}" for r in ratios)
        )

    results["elapsed_seconds"] = time.perf_counter() - started
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {RESULTS.name}  ({results['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
