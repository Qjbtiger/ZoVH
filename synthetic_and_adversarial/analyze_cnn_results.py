import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Expected CSV header based on training logger
EXPECTED_HEADER = [
    "Function",
    "Dimension",
    "Seed",
    "Method",
    "Sample_Index",
    "Sample_RandomSeed",
    "Frobenius_Error",
]


def read_per_method_csvs(input_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for fp in sorted(input_dir.glob("*.csv")):
        try:
            with fp.open("r", newline="") as f:
                reader = csv.DictReader(f)
                # Validate header has required fields
                if not set(EXPECTED_HEADER).issubset(reader.fieldnames or {}):
                    continue
                for r in reader:
                    rows.append(r)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
    return rows


ess_key = ("Function", "Method")


def aggregate_mean_std(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    # Group by (Function, Method)
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        try:
            func = r["Function"].strip()
            method = r["Method"].strip()
            err = float(r["Frobenius_Error"])
        except (KeyError, ValueError):
            continue
        key = (func, method)
        buckets.setdefault(key, []).append(err)

    summary: List[Dict[str, object]] = []
    for (func, method), errs in sorted(buckets.items()):
        arr = np.asarray(errs, dtype=np.float64)
        mean = float(np.mean(arr)) if arr.size > 0 else float("nan")
        std = float(np.std(arr, ddof=0)) if arr.size > 0 else float("nan")
        summary.append(
            {
                "Function": func,
                "Method": method,
                "Count": int(arr.size),
                "Mean_Frobenius_Error": mean,
                "Std_Frobenius_Error": std,
            }
        )
    return summary


def write_summary_csv(summary: List[Dict[str, object]], output_fp: Path) -> None:
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    with output_fp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Function",
                "Method",
                "Count",
                "Mean_Frobenius_Error",
                "Std_Frobenius_Error",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def print_table(summary: List[Dict[str, object]]) -> None:
    if not summary:
        print("No data to summarize.")
        return
    # Compute column widths
    headers = [
        "Function",
        "Method",
        "Count",
        "Mean_Frobenius_Error",
        "Std_Frobenius_Error",
    ]
    rows = [headers]
    for r in summary:
        rows.append(
            [
                str(r["Function"]),
                str(r["Method"]),
                str(r["Count"]),
                f"{r['Mean_Frobenius_Error']:.6e}",
                f"{r['Std_Frobenius_Error']:.6e}",
            ]
        )
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def fmt_row(vals: List[str]) -> str:
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt_row(rows[0]))
    print(sep)
    for r in rows[1:]:
        print(fmt_row(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate mean/std Frobenius errors per layer (Function) and Method from NEW result cnn/*.csv"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "NEW result cnn"),
        help="Directory containing per-method CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "NEW result cnn" / "summary_by_layer_and_method.csv"),
        help="Path to write the aggregated summary CSV",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_fp = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rows = read_per_method_csvs(input_dir)
    summary = aggregate_mean_std(rows)
    write_summary_csv(summary, output_fp)
    print(f"Wrote summary to: {output_fp}")
    print_table(summary)
