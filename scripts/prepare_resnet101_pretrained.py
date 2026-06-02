#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augseg.models.resnet import resnet101


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="pretrained/resnet101.pth")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained file: {path}")

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)!r}")

    model = resnet101(
        pretrained=False,
        zero_init_residual=True,
        multi_grid=True,
        replace_stride_with_dilation=[False, False, True],
        sync_bn=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"path: {path.resolve()}")
    print(f"state_dict_keys: {len(state)}")
    print(f"missing_keys: {len(missing)}")
    for key in missing[:50]:
        print(f"  missing: {key}")
    if len(missing) > 50:
        print(f"  ... {len(missing) - 50} more missing keys")
    print(f"unexpected_keys: {len(unexpected)}")
    for key in unexpected[:50]:
        print(f"  unexpected: {key}")
    if len(unexpected) > 50:
        print(f"  ... {len(unexpected) - 50} more unexpected keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
