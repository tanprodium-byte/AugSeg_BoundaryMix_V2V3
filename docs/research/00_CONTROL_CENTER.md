# Research Control Center

## Repository state

Repo: /home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
Branch: feature/auto-run-12-methods-gpu-suite
Base commit: f8d0ac8
Known untracked path: artifacts/slide_assets/
Do not modify: artifacts/slide_assets/

## Active method

M01 — c3_csl_official_guided_cutmix_no_bcr

## Current phase

DESIGN_GATE

## Current owner

01 — Research HQ

## Current objective

Tạo một ablation giữ nguyên toàn bộ C3 official guided CutMix,
nhưng loại bỏ BCR/V3-d2.

## PROVEN

- C3 dùng baseline adaptive sample confidence ở bước đầu.
- CSL reliability chọn target box ở bước second-stage copy.
- C3 không dùng CSL để weight CE.
- C3 không dùng CSL để thay sample confidence.
- BCR được cộng như auxiliary loss riêng.
- M01 có thể triển khai bằng config mới, không cần sửa mixer.

## UNVERIFIED

- Config M01 chưa được tạo.
- Snapshot path chưa được kiểm tra.
- HF path chưa được kiểm tra.
- Registry cho ba method chưa được tạo.
- Config comparison test chưa chạy.

## Next exact action

Research HQ tạo và phê duyệt Experiment Contract cho M01.

## Forbidden actions

- Không sửa train_semi.py cho M01.
- Không sửa util/boundary_mix.py cho M01.
- Không bắt đầu M02.
- Không bắt đầu M03.
- Không chạy training dài.