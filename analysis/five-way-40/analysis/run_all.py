"""Regenerate every table, figure, and the PDF report in results/ from data/campaign/, in order.

Usage (from this folder): python run_all.py
"""
import importlib
import time

STEPS = ["load", "a0_audit", "a1_h17", "a2_h18", "a3_ratios", "a4_h0", "a5_h13", "a6_h8", "a7_order",
         "a8_variance", "a9_low_quantile", "a10_ladder", "a11_estimands", "a12_calibration", "a13_replication",
         "a14_verdicts", "make_pdf"]

if __name__ == "__main__":
    for step in STEPS:
        t0 = time.time()
        print(f"=== {step}", flush=True)
        importlib.import_module(step).main()
        print(f"=== {step} done in {time.time() - t0:.0f} s", flush=True)
