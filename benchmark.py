#!/usr/bin/env python3
"""
DATA COMPASS — Pipeline Benchmark
Compares v2 vs v1 classification on every file in sample_documents/.

Usage:
  python benchmark.py                          # all files, both pipelines
  python benchmark.py --file sample_documents/adhar1.1.jpeg
  python benchmark.py --v2-only
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed.  Run:  pip install requests tabulate")
    sys.exit(1)

try:
    from tabulate import tabulate  # noqa: F401  (used in summary only)
except ImportError:
    print("ERROR: 'tabulate' not installed.  Run:  pip install requests tabulate")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE       = "http://localhost:8000"
SAMPLE_DIR     = Path(__file__).parent / "sample_documents"
RESULTS_JSON   = Path(__file__).parent / "benchmark_results.json"
RESULTS_CSV    = Path(__file__).parent / "benchmark_results.csv"
SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

STEP_KEYS = [
    ("file_read_ms",               "file_read"),
    ("page_splitting_ms",          "page_splitting"),
    ("metadata_fingerprinting_ms", "metadata"),
    ("phash_computation_ms",       "phash"),
    ("boundary_detection_ms",      "boundary_detection"),
    ("heading_confirmation_ms",    "heading_confirm"),
    ("industry_routing_ms",        "industry_routing"),
    ("hypothesis_loading_ms",      "hypothesis_loading"),
    ("siglip2_classification_ms",  "siglip2"),
    ("supabase_logging_ms",        "supabase_logging"),
]

WIDTH = 60


# ── Formatting helpers ────────────────────────────────────────────────────────

def _bar(value_ms: float, max_ms: float, width: int = 10) -> str:
    if max_ms == 0:
        return "░" * width
    filled = round((value_ms / max_ms) * width)
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / total)}%" if total else "0%"


def _trunc(s: str, n: int = 80) -> str:
    return (s[:n] + "…") if len(s) > n else s


def _sep(char: str = "─", width: int = WIDTH) -> str:
    return char * width


def _header(title: str, char: str = "═", width: int = WIDTH) -> str:
    return f"\n{char * width}\n{title}\n{char * width}"


# ── Server check ──────────────────────────────────────────────────────────────

def _check_server() -> None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        print(f"\nERROR: Cannot reach server at {API_BASE}.\n"
              f"  Make sure it is running:  source venv/bin/activate && uvicorn app.main:app --port 8000\n"
              f"  Detail: {exc}\n")
        sys.exit(1)


# ── API calls ─────────────────────────────────────────────────────────────────

def _post_v2(filepath: Path) -> tuple[dict | None, float, str | None]:
    """Returns (data, elapsed_s, error_msg)."""
    try:
        with open(filepath, "rb") as fh:
            t0 = time.perf_counter()
            r = requests.post(
                f"{API_BASE}/v2/classify",
                files={"file": (filepath.name, fh)},
                headers={"X-Benchmark-Timing": "true"},
                timeout=120,
            )
            elapsed = round(time.perf_counter() - t0, 3)
        data = r.json()
        if not r.ok or "error" in data:
            return None, elapsed, data.get("error", f"HTTP {r.status_code}")
        return data, elapsed, None
    except Exception as exc:
        return None, 0.0, str(exc)


def _post_v1(filepath: Path) -> tuple[dict | None, float, str | None]:
    try:
        with open(filepath, "rb") as fh:
            t0 = time.perf_counter()
            r = requests.post(
                f"{API_BASE}/classify",
                files={"file": (filepath.name, fh)},
                timeout=120,
            )
            elapsed = round(time.perf_counter() - t0, 3)
        data = r.json()
        if not r.ok or "error" in data:
            return None, elapsed, data.get("error", f"HTTP {r.status_code}")
        return data, elapsed, None
    except Exception as exc:
        return None, 0.0, str(exc)


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_v2(data: dict) -> list[dict]:
    """Returns one dict per detected document."""
    out = []
    for r in data.get("results", []):
        cls = r.get("classification", {})
        rt  = r.get("routing", {})
        out.append({
            "document_index":    r.get("document_index"),
            "page_range":        r.get("page_range"),
            "predicted_class":   cls.get("predicted_class"),
            "confidence":        cls.get("confidence"),
            "confidence_band":   cls.get("confidence_band"),
            "reasoning":         _trunc(cls.get("reasoning") or "", 80),
            "industry_1":        rt.get("industry_1"),
            "routing_confidence":rt.get("confidence"),
            "router_path":       rt.get("router_path"),
            "hypothesis_scope":  rt.get("hypothesis_scope"),
        })
    return out


def _extract_v1(data: dict) -> dict:
    page_res = data.get("page_results") or []
    return {
        "predicted_label":  data.get("predicted_label"),
        "confidence":       data.get("confidence"),
        "confidence_band":  data.get("confidence_band"),
        "status":           data.get("status"),
        "page_results_cnt": len(page_res),
    }


# ── Per-file terminal rendering ───────────────────────────────────────────────

def _render_timing_block(timing: dict) -> list[str]:
    if not timing:
        return []
    total_ms = timing.get("total_pipeline_ms", 1)
    step_vals = [(label, timing.get(key, 0.0)) for key, label in STEP_KEYS]
    max_ms = max((v for _, v in step_vals), default=1)
    slowest_label = max(step_vals, key=lambda x: x[1])[0]

    lines = ["│   STEP BREAKDOWN:"]
    for label, ms in step_vals:
        bar    = _bar(ms, max_ms)
        note   = "  ← BOTTLENECK" if label == slowest_label else ""
        lines.append(f"│   {label:<22}: {ms:>6.1f}ms  {bar}{note}")
    lines.append("│   " + "─" * 44)
    lines.append(f"│   {'TOTAL PIPELINE':<22}: {total_ms:>6.1f}ms")
    return lines


def _render_file_block(
    filename: str,
    v2_data: dict | None, v2_elapsed: float, v2_err: str | None,
    v1_data: dict | None, v1_elapsed: float, v1_err: str | None,
    v2_only: bool,
) -> None:
    print(f"\nFILE: {filename}")
    print(_sep())

    # ── V2 block ──────────────────────────────────────────────────────────────
    if v2_err:
        print(f"V2  │ ERROR: {v2_err}")
    else:
        docs = v2_data.get("documents_detected", 0)
        timing = v2_data.get("pipeline_timing", {})
        total_ms = timing.get("total_pipeline_ms")
        time_str = f"{total_ms:.1f}ms" if total_ms is not None else f"{v2_elapsed:.2f}s"

        docs_list = _extract_v2(v2_data)
        first = docs_list[0] if docs_list else {}

        conf_pct = f"{round((first.get('confidence') or 0) * 100, 1)}%"
        print(f"V2  │ docs: {docs} │ "
              f"class: {first.get('predicted_class','?'):<22} │ "
              f"conf: {conf_pct} {first.get('confidence_band','?')} │ "
              f"TOTAL: {time_str}")
        print(f"    │ industry: {str(first.get('industry_1') or 'Unknown'):<28} │ "
              f"scope: {first.get('hypothesis_scope','?')}")
        print(f"    │ reasoning: {first.get('reasoning','')}")

        if timing:
            for line in _render_timing_block(timing):
                print(f"    {line}")

        if docs > 1:
            for d in docs_list[1:]:
                conf_pct2 = f"{round((d.get('confidence') or 0) * 100, 1)}%"
                print(f"    │ doc {d['document_index']} pages {d['page_range']}: "
                      f"{d.get('predicted_class','?')} {conf_pct2} {d.get('confidence_band','?')}")

    if v2_only:
        print(_sep())
        return

    # ── V1 block ──────────────────────────────────────────────────────────────
    if v1_err:
        print(f"V1  │ ERROR: {v1_err}")
    else:
        v1 = _extract_v1(v1_data)
        conf_str = f"{v1['confidence']}%" if v1["confidence"] is not None else "?"
        print(f"V1  │ class: {str(v1['predicted_label']):<28} │ "
              f"conf: {conf_str} {v1.get('confidence_band','?')} │ "
              f"time: {v1_elapsed:.2f}s")
        print(f"    │ status: {v1.get('status','?')}")

    # ── Agreement ──────────────────────────────────────────────────────────────
    if not v2_err and not v1_err:
        v2_class = (_extract_v2(v2_data)[0].get("predicted_class") or "").lower() if v2_data else ""
        v1_class = (v1_data.get("predicted_label") or "").lower() if v1_data else ""
        agree = v2_class == v1_class
        tag = "✅ AGREE" if agree else f"❌ DISAGREE  (v1={v1_class}  v2={v2_class})"
        print(f"MATCH: {tag}")

    print(_sep())


# ── Summary ───────────────────────────────────────────────────────────────────

def _render_summary(records: list[dict], v2_only: bool) -> None:
    print(_header("BENCHMARK SUMMARY"))

    total = len(records)
    v2_times  = [r["v2_elapsed"] for r in records if r.get("v2_elapsed") and not r.get("v2_error")]
    v1_times  = [r["v1_elapsed"] for r in records if r.get("v1_elapsed") and not r.get("v1_error") and not v2_only]

    v2_avg = round(sum(v2_times) / len(v2_times), 2) if v2_times else 0
    v1_avg = round(sum(v1_times) / len(v1_times), 2) if v1_times else 0

    print(f"\nTotal files tested:        {total}")
    print(f"V2 avg processing time:    {v2_avg:.2f}s")
    if not v2_only:
        print(f"V1 avg processing time:    {v1_avg:.2f}s")

    if not v2_only:
        agreements = [r for r in records if r.get("agree") is True]
        disagrees  = [r for r in records if r.get("agree") is False]
        agree_pct  = _pct(len(agreements), total)
        print(f"\nAGREEMENT RATE:            {agree_pct}  (V1 and V2 same class)")
        print(f"DISAGREEMENTS:             {len(disagrees)} files")

    # Confidence distributions
    def _conf_dist(band_list: list[str], label: str) -> None:
        h = sum(1 for b in band_list if b == "HIGH")
        m = sum(1 for b in band_list if b == "MEDIUM")
        lo = sum(1 for b in band_list if b == "LOW")
        n = len(band_list) or 1
        print(f"\n{label} confidence distribution:")
        print(f"  HIGH:   {h:3d} files ({_pct(h, n)})")
        print(f"  MEDIUM: {m:3d} files ({_pct(m, n)})")
        print(f"  LOW:    {lo:3d} files ({_pct(lo, n)})")

    v2_bands = [r["v2_confidence_band"] for r in records if r.get("v2_confidence_band")]
    _conf_dist(v2_bands, "V2")
    if not v2_only:
        v1_bands = [r["v1_confidence_band"] for r in records if r.get("v1_confidence_band")]
        _conf_dist(v1_bands, "V1")

    # Hypothesis scope
    scopes = [r.get("v2_hypothesis_scope") for r in records if r.get("v2_hypothesis_scope")]
    n_scoped   = sum(1 for s in scopes if s == "industry_scoped")
    n_fallback = sum(1 for s in scopes if s == "full_set_fallback")
    n_sc = len(scopes) or 1
    print(f"\nV2 hypothesis scope:")
    print(f"  industry_scoped:    {n_scoped:3d} files ({_pct(n_scoped, n_sc)})")
    print(f"  full_set_fallback:  {n_fallback:3d} files ({_pct(n_fallback, n_sc)})")

    # Step timing averages
    all_timings = [r["v2_pipeline_timing"] for r in records if r.get("v2_pipeline_timing")]
    if all_timings:
        print(f"\nSTEP TIMING AVERAGES (v2):")
        header_row = f"{'Step':<24}  {'Avg(ms)':>8}  {'Max(ms)':>8}  {'% of Total':>10}"
        print(header_row)
        print("─" * 56)

        total_avgs = {}
        for key, label in STEP_KEYS:
            vals = [t.get(key, 0.0) for t in all_timings]
            avg_ms = round(sum(vals) / len(vals), 1)
            max_ms = round(max(vals), 1)
            total_avgs[label] = avg_ms

        grand_avg = sum(total_avgs.values()) or 1
        slowest_step = max(total_avgs, key=lambda k: total_avgs[k])

        for key, label in STEP_KEYS:
            vals = [t.get(key, 0.0) for t in all_timings]
            avg_ms = round(sum(vals) / len(vals), 1)
            max_ms = round(max(vals), 1)
            pct_str = f"{round(100 * avg_ms / grand_avg, 1)}%"
            note = "   ← AVG BOTTLENECK" if label == slowest_step else ""
            print(f"{label:<24}  {avg_ms:>8.1f}  {max_ms:>8.1f}  {pct_str:>10}{note}")

        total_avg_all = [t.get("total_pipeline_ms", 0.0) for t in all_timings]
        t_avg = round(sum(total_avg_all) / len(total_avg_all), 1)
        t_max = round(max(total_avg_all), 1)
        print("─" * 56)
        print(f"{'TOTAL':<24}  {t_avg:>8.1f}  {t_max:>8.1f}  {'100.0%':>10}")

    # Disagreement details
    if not v2_only:
        disagrees = [r for r in records if r.get("agree") is False]
        if disagrees:
            print(f"\nDISAGREEMENT DETAILS:")
            rows = []
            for r in disagrees:
                rows.append([
                    r["filename"][:30],
                    r.get("v1_predicted_label", "?"),
                    r.get("v2_predicted_class", "?"),
                    f"{r.get('v1_confidence','?')}%",
                    f"{round((r.get('v2_confidence') or 0)*100, 1)}%",
                ])
            print(tabulate(rows,
                           headers=["File", "V1 Class", "V2 Class", "V1 Conf", "V2 Conf"],
                           tablefmt="simple"))

    print("═" * WIDTH)


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_json(records: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nJSON saved → {RESULTS_JSON}")


def _save_csv(records: list[dict]) -> None:
    if not records:
        return
    step_col_keys = [key for key, _ in STEP_KEYS] + ["total_pipeline_ms"]
    fieldnames = [
        "filename", "v2_elapsed_s", "v1_elapsed_s",
        "v2_docs_detected", "v2_predicted_class", "v2_confidence", "v2_confidence_band",
        "v2_industry_1", "v2_routing_confidence", "v2_router_path", "v2_hypothesis_scope",
        "v1_predicted_label", "v1_confidence", "v1_confidence_band", "v1_status",
        "agree",
    ] + step_col_keys

    rows = []
    for r in records:
        timing = r.get("v2_pipeline_timing") or {}
        row = {
            "filename":             r.get("filename"),
            "v2_elapsed_s":         r.get("v2_elapsed"),
            "v1_elapsed_s":         r.get("v1_elapsed"),
            "v2_docs_detected":     r.get("v2_docs_detected"),
            "v2_predicted_class":   r.get("v2_predicted_class"),
            "v2_confidence":        r.get("v2_confidence"),
            "v2_confidence_band":   r.get("v2_confidence_band"),
            "v2_industry_1":        r.get("v2_industry_1"),
            "v2_routing_confidence":r.get("v2_routing_confidence"),
            "v2_router_path":       r.get("v2_router_path"),
            "v2_hypothesis_scope":  r.get("v2_hypothesis_scope"),
            "v1_predicted_label":   r.get("v1_predicted_label"),
            "v1_confidence":        r.get("v1_confidence"),
            "v1_confidence_band":   r.get("v1_confidence_band"),
            "v1_status":            r.get("v1_status"),
            "agree":                r.get("agree"),
        }
        for key in step_col_keys:
            row[key] = timing.get(key, "")
        rows.append(row)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {RESULTS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Data Compass pipeline benchmark")
    parser.add_argument("--file",    help="Test a single file instead of all sample_documents/")
    parser.add_argument("--v2-only", action="store_true", help="Skip v1 comparison")
    args = parser.parse_args()

    _check_server()

    # Collect files
    if args.file:
        files = [Path(args.file)]
        if not files[0].exists():
            print(f"ERROR: file not found: {files[0]}")
            sys.exit(1)
    else:
        if not SAMPLE_DIR.exists():
            print(f"ERROR: sample_documents/ directory not found at {SAMPLE_DIR}")
            sys.exit(1)
        files = sorted(
            f for f in SAMPLE_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
        if not files:
            print(f"No supported files found in {SAMPLE_DIR}")
            sys.exit(0)

    print(_header("DATA COMPASS — PIPELINE BENCHMARK"))

    records: list[dict] = []

    for filepath in files:
        filename = filepath.name

        # V2
        v2_data, v2_elapsed, v2_err = _post_v2(filepath)

        # V1
        if args.v2_only:
            v1_data, v1_elapsed, v1_err = None, 0.0, None
        else:
            v1_data, v1_elapsed, v1_err = _post_v1(filepath)

        # Render
        _render_file_block(
            filename,
            v2_data, v2_elapsed, v2_err,
            v1_data, v1_elapsed, v1_err,
            args.v2_only,
        )

        # Build record
        v2_docs   = _extract_v2(v2_data) if v2_data else []
        first_v2  = v2_docs[0] if v2_docs else {}
        v1_ex     = _extract_v1(v1_data) if v1_data else {}

        v2_class = (first_v2.get("predicted_class") or "").lower()
        v1_class = (v1_ex.get("predicted_label") or "").lower()
        agree    = (v2_class == v1_class) if (v2_data and v1_data) else None

        record = {
            "filename":              filename,
            "v2_elapsed":            v2_elapsed,
            "v1_elapsed":            v1_elapsed if not args.v2_only else None,
            "v2_error":              v2_err,
            "v1_error":              v1_err if not args.v2_only else None,
            "v2_docs_detected":      v2_data.get("documents_detected") if v2_data else None,
            "v2_predicted_class":    first_v2.get("predicted_class"),
            "v2_confidence":         first_v2.get("confidence"),
            "v2_confidence_band":    first_v2.get("confidence_band"),
            "v2_industry_1":         first_v2.get("industry_1"),
            "v2_routing_confidence": first_v2.get("routing_confidence"),
            "v2_router_path":        first_v2.get("router_path"),
            "v2_hypothesis_scope":   first_v2.get("hypothesis_scope"),
            "v2_pipeline_timing":    v2_data.get("pipeline_timing") if v2_data else None,
            "v1_predicted_label":    v1_ex.get("predicted_label"),
            "v1_confidence":         v1_ex.get("confidence"),
            "v1_confidence_band":    v1_ex.get("confidence_band"),
            "v1_status":             v1_ex.get("status"),
            "agree":                 agree,
            "v2_all_docs":           v2_docs,
        }
        records.append(record)

    _render_summary(records, args.v2_only)
    _save_json(records)
    _save_csv(records)


if __name__ == "__main__":
    main()
