# AGENTS.md — BoundaryMix V2/V3 Implementation Rules

## Project context

This repository is based on AugSeg and currently contains BoundaryMix V1 experiments for semi-supervised semantic segmentation on PASCAL VOC 662 labels.

The new research direction is artificial CutMix seam/boundary reliability.

Implement only the following:
1. V2: Boundary-Affected Component Weighting only.
2. V3: JS Boundary Compatibility only.
3. V2 + V3: Component Weighting + JS Boundary Compatibility.

Do not implement class-aware CutMix.
Do not add unrelated SSL tricks.
Do not change the AugSeg training pipeline unnecessarily.

## Critical separation rules

- V2 must not implicitly include V3.
- V3 must not implicitly include V2.
- When boundary_component.enabled=false and boundary_compatibility.enabled=false, the loss path must behave like the existing A1/neutral path or baseline path depending on config.
- V3 standalone must use q_C = 1 and must not use component gate.
- Only V2+V3 may use q_C from V2 to gate BCR.

## Implementation requirements

- Add config flags for all new behavior.
- Keep A0/A1/A4 backward-compatible.
- Teacher probabilities, pseudo-labels, component weights, and gates used only as weights/gates must be detached or computed in no_grad.
- Clamp probabilities with epsilon before JS/KL to avoid NaN.
- Log debug statistics for V2 and V3.
- Add sanity checks or a small test script.
- Do not claim performance improvements without training results.

## Preferred structure

Add separate modules if possible:

- utils/boundary_component.py
- utils/boundary_compatibility.py

Preferred functions:

compute_component_weights(...)
compute_js_boundary_compatibility_loss(...)

## Required configs

Add configs for:

- V2 only
- V3-d1
- V3-d2
- V3-d3
- V2+V3-best template

## Validation before final answer

Before finishing, report:
- files changed
- exact config files added
- how disabled modules preserve baseline/A1 behavior
- how to run a smoke test
- any limitations or untested parts
