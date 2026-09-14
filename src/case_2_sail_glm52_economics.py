"""Reproduce the GLM-5.2 slide using total (input + output) throughput.

Standard library only; all scenario inputs come from data/assumptions.csv.
"""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_assumptions(path=ROOT / "data" / "assumptions.csv"):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {}
    for row in rows:
        key, value = row["parameter"], float(row["value"])
        if key in values or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid or duplicate assumption: {key}")
        values[key] = value
    return values


def calculate(a, gpu_price=None, utilization=None):
    """Utilization is realized billable throughput / benchmark throughput.

    GPU rental is paid for the full hour, including idle time.
    """
    price = a["gpu_price_usd_h"] if gpu_price is None else gpu_price
    u = a["utilization"] if utilization is None else utilization
    if not math.isfinite(price) or price < 0 or not 0 < u <= 1:
        raise ValueError("Require finite GPU price >= 0 and 0 < utilization <= 1")
    n = a["gpu_count"]
    if n != int(n) or n <= 0:
        raise ValueError("GPU count must be a positive integer")
    input_tokens, output_tokens = a["input_tokens"], a["output_tokens"]
    request_revenue = (input_tokens * a["input_price_usd_mtok"]
                       + output_tokens * a["output_price_usd_mtok"]) / 1_000_000
    rps = a["total_throughput_tok_s"] * u / (input_tokens + output_tokens)
    revenue = request_revenue * rps * 3600
    cost = n * price
    return {
        "request_revenue_usd": request_revenue,
        "requests_s": rps,
        "requests_h": rps * 3600,
        "input_tokens_h": rps * 3600 * input_tokens,
        "output_tokens_h": rps * 3600 * output_tokens,
        "revenue_node_usd_h": revenue,
        "gpu_cost_node_usd_h": cost,
        "gpu_contribution_usd_h": revenue - cost,
        "gpu_margin_pct": (revenue - cost) / revenue * 100,
        "break_even_gpu_usd_h": revenue / n,
        "utilization": u,
    }


def sensitivity(a):
    return [{"gpu_price_usd_h": a[key], **calculate(a, gpu_price=a[key])}
            for key in ("sensitivity_low", "gpu_price_usd_h",
                        "sensitivity_high", "sensitivity_rounded_break_even")]


def check_slide(a):
    """Regression against externally supplied slide values, at its precision."""
    r = calculate(a)
    expected = {"revenue_node_usd_h": (1, 51.5),
                "gpu_cost_node_usd_h": (1, 21.0),
                "gpu_margin_pct": (0, 59),
                "break_even_gpu_usd_h": (2, 6.43)}
    for key, (digits, target) in expected.items():
        if round(r[key], digits) != target:
            raise AssertionError(f"Slide mismatch: {key}={r[key]}, expected ~{target}")
    margins = [round(row["gpu_margin_pct"]) for row in sensitivity(a)]
    if margins != [69, 59, 9, 0]:
        raise AssertionError(f"Sensitivity mismatch: {margins}")
    zero = calculate(a, gpu_price=r["break_even_gpu_usd_h"])
    if not math.isclose(zero["gpu_margin_pct"], 0, abs_tol=1e-10):
        raise AssertionError("Exact break-even does not yield zero margin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assumptions", type=Path, default=ROOT / "data/assumptions.csv")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    parser.add_argument("--check-slide", action="store_true")
    args = parser.parse_args()
    a = load_assumptions(args.assumptions)
    if args.check_slide:
        check_slide(a)
    result = {"base": calculate(a), "sensitivity": sensitivity(a)}
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print("GLM-5.2 / Sail — fixed slide scenario; USD")
    print("Throughput: total input + output tokens; utilization: 100% of benchmark")
    for key, value in result["base"].items():
        print(f"{key:28s} {value:.6f}")
    print("\nGPU USD/h | margin, exact % | slide %")
    for row in result["sensitivity"]:
        print(f"{row['gpu_price_usd_h']:9.2f} | {row['gpu_margin_pct']:15.6f} | {row['gpu_margin_pct']:.0f}%")
    if args.check_slide:
        print("\nPASS: all slide values match at the displayed precision.")


if __name__ == "__main__":
    main()
