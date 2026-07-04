"""Single entry point: rebuild every result and figure for the paper.

Runs the drivers in dependency order (arm_learned exports sweep_representation used by the
foundation arms; compare_strong needs the ECGFounder loader; figures reads every CSV). Each
driver writes its CSVs to results/ and figures to figures/. See README.md for the full
driver -> result -> manuscript-element map.

Run from this code/ directory:
    python3 paper.py            # rebuild everything (near-instant once the caches under PTBXL_DIR exist)

The four figures land in figures/; copy them into the manuscript figures dir before rebuilding the PDF.

Notes:
  - Set PTBXL_DIR / MITBIH_DIR in core/config.py first, and download PTB-XL 1.0.3 + MIT-BIH (see README.md "Data").
  - Re-runs are near-instant once the cohort, feature, and frozen-embedding caches exist under PTBXL_DIR;
    the first run streams PTB-XL records and embeds the foundation models.
  - native500_check.py (the ECGFounder native-500 Hz consistency check) is NOT part of this pipeline: it
    re-streams records at 500 Hz from PhysioNet (rate-limited) and prints to stdout. Run it on its own.
"""
import runpy
import sys

DRIVERS = [
    "benchmark",            # PTB-XL raw + hand-crafted sweep -> results_raw / baselines / summary
    "arm_learned",          # conv-AE (deep_ae) arm           -> results_learned (also exports sweep_representation)
    "arm_foundation",       # HuBERT-ECG frozen arm           -> results_foundation
    "compare_strong",       # ECGFounder head-to-head         -> results_compare_strong (Table 1 foundation rows)
    "fed_cluster",          # federated k-means IID / non-IID -> results_federated (Table 2)
    "mitbih_bench",         # MIT-BIH N-vs-V beat morphology  -> results_mitbih (Table 3)
    "run_analyses", # significance / K-sweep / RF ceiling / imbalance (stdout)
    "figures",              # Figs 1-4 + printed tables
]


def main(drivers=DRIVERS):
    for name in drivers:
        print("\n" + "=" * 78 + f"\n[{name}]\n" + "=" * 78)
        runpy.run_module(f"experiments.{name}", run_name="__main__")
    print("\nDone. Results in results/, figures in figures/ (copy the 4 PDFs to the manuscript figures dir).")


if __name__ == "__main__":
    main(sys.argv[1:] or DRIVERS)
