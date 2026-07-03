from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


SUITE = "voc662_8sc_officialc3_rerun01_trainsemi_wandb_hf_20260701"
REPO_ID = "tanprodium/augseg-boundarymix-v2v3-runs"
REPO_TYPE = "model"

SAVE_ROOT = Path("exp_boundary_mix_v2_v3/reruns") / SUITE


def discover_methods() -> list[str]:
    methods: set[str] = set()

    if SAVE_ROOT.exists():
        for save_dir in SAVE_ROOT.glob("*_r101_c321_bs8x1_gbs8"):
            if save_dir.is_dir():
                method = save_dir.name.replace("_r101_c321_bs8x1_gbs8", "")
                methods.add(method)

    return sorted(methods)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    print("HF_TOKEN_PRESENT =", bool(token))

    api = HfApi(token=token)
    files = set(api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))

    methods = discover_methods()

    print()
    print("HF STANDARD CHECK")
    print("-" * 140)
    print(f"{'method':60s} {'expected_latest':65s} {'exists'}")
    print("-" * 140)

    ok = 0
    missing = 0

    for method in methods:
        expected = f"{SUITE}/{method}/latest.tar.gz"
        exists = expected in files

        if exists:
            ok += 1
        else:
            missing += 1

        print(f"{method:60s} {expected:65s} {exists}")

    print("-" * 140)
    print(f"OK={ok} MISSING={missing}")

    print()
    print("POSSIBLE WRONG PATHS")
    print("-" * 140)

    suite_files = sorted(f for f in files if f.startswith(SUITE + "/"))
    wrong = []

    for f in suite_files:
        if not f.endswith("latest.tar.gz"):
            continue

        parts = f.split("/")
        if len(parts) != 3:
            wrong.append(f)
            continue

        suite, method, filename = parts
        if suite != SUITE or filename != "latest.tar.gz":
            wrong.append(f)

    if wrong:
        for f in wrong:
            print("[WRONG?]", f)
    else:
        print("No suspicious latest.tar.gz path under suite.")

    print()
    print("ALL LATEST UNDER SUITE")
    print("-" * 140)
    for f in suite_files:
        if f.endswith("latest.tar.gz"):
            print(f)


if __name__ == "__main__":
    main()
