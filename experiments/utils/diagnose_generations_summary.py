#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_generations_summary.py

Compact, per-function summary of average generations by strategy and
relative Δ% for Diversity vs Adaptive and Random.

- Input: _final_results_merged.csv (same schema as before)
- Output: prints a readable table and saves ga_generations_summary.csv
- Flags any function where Δ% is negative (Diversity uses MORE generations)

Usage:
    python diagnose_generations_summary.py --input _final_results_merged.csv
"""

import argparse
import pandas as pd
import numpy as np

def rel_change(base, other):
    """Relative improvement in % (positive means 'other' needs fewer generations)."""
    if base == 0 or pd.isna(base) or pd.isna(other):
        return np.nan
    return 100.0 * (base - other) / base

def main():
    parser = argparse.ArgumentParser(description="Diagnose per-function generation deltas (Diversity vs baselines).")
    parser.add_argument("--input", default="_final_results_merged.csv", help="Path to the merged CSV file.")
    parser.add_argument("--output", default="ga_generations_summary.csv", help="Where to save the summary CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["experiment", "strategy", "avg_generations"]
    if any(col not in df.columns for col in required):
        raise ValueError(f"CSV is missing required columns: {required}")

    # Average generations per function & strategy
    pivot = (
        df.groupby(["experiment", "strategy"])["avg_generations"]
          .mean()
          .reset_index()
          .pivot(index="experiment", columns="strategy", values="avg_generations")
    )

    rows = []
    for func, row in pivot.iterrows():
        # Skip if a strategy is missing for this function
        if not {"adaptive mutation", "diversity mutation", "random mutation"}.issubset(row.index):
            continue
        adaptive = row["adaptive mutation"]
        diversity = row["diversity mutation"]
        random_ = row["random mutation"]

        d_vs_adapt = rel_change(adaptive, diversity)
        d_vs_rand  = rel_change(random_, diversity)

        flag_adapt = "⚠️" if (pd.notna(d_vs_adapt) and d_vs_adapt < 0) else ""
        flag_rand  = "⚠️" if (pd.notna(d_vs_rand) and d_vs_rand < 0) else ""

        rows.append({
            "Function": func,
            "Adaptive (avg gens)": adaptive,
            "Diversity (avg gens)": diversity,
            "Random (avg gens)": random_,
            "Δ% vs Adaptive": d_vs_adapt,
            "Δ% vs Random": d_vs_rand,
            "Flag vs Adaptive": flag_adapt,
            "Flag vs Random": flag_rand
        })

    out = pd.DataFrame(rows).sort_values("Function")

    # Pretty print
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:,.2f}".format)

    print("\n=== Diversity Mutation Generations Summary (per function) ===\n")
    printable = out.copy()
    printable["Δ% vs Adaptive"] = printable["Δ% vs Adaptive"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "NA")
    printable["Δ% vs Random"]   = printable["Δ% vs Random"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "NA")
    print(printable.to_string(index=False))

    # Save CSV
    out.to_csv(args.output, index=False)
    print(f"\n✅ Saved summary to '{args.output}'")
    neg_any = (out["Δ% vs Adaptive"].lt(0).fillna(False) | out["Δ% vs Random"].lt(0).fillna(False)).sum()
    if neg_any > 0:
        print(f"⚠️ {neg_any} function(s) show negative Δ% (Diversity slower) against at least one baseline.")
    else:
        print("✅ No functions show negative Δ% — Diversity is faster across the board for generations.")

if __name__ == "__main__":
    main()
