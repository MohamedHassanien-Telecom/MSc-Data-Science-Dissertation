"""Phase IV entry point: figures, tables and the collapse report (src/visualise.py).

Run from the notebook, or from the command line:  python3 -m phases.phase4_analysis
"""
import json
from pathlib import Path
from src.visualise import phase4

def run_phase4(p2=None, p3=None, out_root=None, cfg=None):
    """Run Phase IV on the Phase II aggregate (p2) and the Phase III result (p3).
    Both are read from out_root (default OUTPUT_DIR/run_0) when not given; without
    phase3.json the compression figures are skipped."""
    if out_root is None and cfg is not None:
        out_root = Path(cfg["OUTPUT_DIR"]) / "run_0"
    elif out_root is None:
        raise ValueError("Either 'out_root' or 'cfg' must be provided.")

    if p2 is None:
        p2_aggregated_path = out_root / "phase2_aggregated.json"
        if not p2_aggregated_path.exists():
            raise FileNotFoundError(f"{p2_aggregated_path} not found. Run Phase II first.")
        print(f"[Phase IV] Loading Phase II results from {p2_aggregated_path}")
        with open(p2_aggregated_path, "r") as f:
            p2 = json.load(f)

    if p3 is None:
        p3_path = out_root / "phase3.json"
        if p3_path.exists():
            with open(p3_path, "r") as f:
                p3 = json.load(f)
        else:
            print("[Phase IV] Phase III results not found; running without compression plots.")
            p3 = {"by_cr": {}, "snr_by_cr": {}}

    report = phase4(p2, p3, out_root, cfg)
    return report

if __name__ == "__main__":
    from src.config import CONFIG
    run_phase4(out_root=Path(CONFIG["OUTPUT_DIR"]) / "run_0", cfg=CONFIG)
