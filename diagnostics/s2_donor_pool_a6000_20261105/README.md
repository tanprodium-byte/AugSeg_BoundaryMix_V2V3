# Controlled S2 donor-pool diagnostic: portable A6000 package

This package prepares an allocation diagnostic, not training. No real-model validation or replay
was run while preparing this port on RTX5090. CPU synthetic/mocks do not constitute replay results.
Do not modify or bootstrap the existing four-profile production deployment or environment.

## Locked design

Teacher `legacy5090_33aad027a10538589c67daa0`, teacher state from best epoch70 (checkpoint epoch71),
expected validation mIoU **75.20909874915435%**, absolute tolerance **0.0001pp**. If it does not match,
stop; do not change checkpoint, tolerance, validation recipe or precision to force a match.

FP32, no autocast/TF32, crop321, batch16,1024 receivers,64 shared batches. Frozen eval teacher,
input-RGB autograd only for the historical saliency probe; parameters, gradients and BN buffers
must not change. No optimizer, EMA, model training, new module, W&B run or HF upload.
Native-resolution validation batch1 precedes replay; C321/B16 applies to replay.

Six conditions: three locked nested donor pools VOC662/1323/2646 × baseline/S2. Pool lists are
external inputs. Receiver/donor plans and256 augmented-input batch checksums are unchanged.
Receiver GT is observational only and cannot enter pseudo labels, gate, saliency or selection.
Each pool uses1024 donor slots, uniform with replacement, with534/741/880 unique donors.

Baseline retains both AugSeg mixing stages and the original confidence–entropy gate. S2 retains
one-stage exact-component-mask relocation, saliency×area score, temperature0.2 and rectangle fallback
on zero eligible components. Selector exceptions abort rather than silently changing the recipe.
The helpers are29 byte-identical historical Git-object exports at
`3cf5e66c4479532de813e4236649fbeea8e27421`; the training entrypoint is evidence, never imported.
`scientific_equivalence.json` lists unchanged instrumentation AST functions and path-only plan changes.
`operational.diff` records the port; `source_origins.json` preserves original checksums.
Historical CPU binding is not an A6000 runtime attestation. Float32 EMA BN counters are restored
with strict `assign=True`, preserving exact teacher bytes rather than silently casting to int64.

## External inputs: transfer separately, never Git

`inputs.json` specifies source paths, exact target paths, sizes and SHA256. Do not reconstruct lists.

- Dataset root: `/home/islabworker2/minhhb/tanvt/augseg_inputs/VOC2012`.
  Required5616 files:2808 JPEGImages/ID.jpg +2808 SegmentationClassAug/ID.png, including all1449
  validation images and the1359 scheduled unique donors. `input_payload_hashes.csv` contains only
  relative paths, sizes and SHA256; no images/masks. Existing full VOC2012 can be reused if checks pass.
- Labeled lists: `augseg_inputs/splits/pascal/{662,1323,2646}/labeled.txt`;
  validation: `augseg_inputs/splits/pascal/val.txt`. Counts/checksums are checked, then nesting,
  validation disjointness and frozen-plan membership. Unlabeled training lists are not needed here.
- Checkpoint: `augseg_diagnostic_inputs/s2_donor_pool/ckpt_best.pth`,715095449 bytes,
  SHA256 `5edd26a05f65bf1010409306c0da18f88f290f0a4e46eb0ce89787cfff1d07c0`.
  No pretrained download is necessary: strict full teacher state overwrites the constructor.

A6000 input presence and free resources are **not verified by this preparation**. Transfer missing
files separately over the already configured private route. Never use public HF/Git for these inputs.
Hashes are diagnostic preparation/transfer identities, not historical training input attestation.
`transfer_inputs.py` below creates/verifies a separately stored private archive only when explicitly
requested; it never connects to another host, installs inputs or calls a model.

## Commands on A6000

Use a separate clone for this diagnostic branch, under `/home/islabworker2/minhhb/tanvt`.
Pin the **full commit reported after remote verification**, not a moving branch. No production checkout changes.

```bash
cd /home/islabworker2/minhhb/tanvt
git clone --single-branch --branch migration/s2-donor-pool-a6000-20261105 \
  https://github.com/tanprodium-byte/AugSeg_BoundaryMix_V2V3.git AugSeg_S2_donor_diagnostic
cd AugSeg_S2_donor_diagnostic
git checkout --detach <VERIFIED_FULL_COMMIT>
export AUGSEG_PYTHON=/home/islabworker2/minhhb/tanvt/envs/augseg-bs16/bin/python
DIAG=diagnostics/s2_donor_pool_a6000_20261105
bash "$DIAG/run.sh" check-package
bash "$DIAG/run.sh" check-deps
# After external inputs are installed at exact paths:
bash "$DIAG/run.sh" check-inputs
# prepare refuses an existing output directory; never overwrite or auto-create a new attempt.
bash "$DIAG/run.sh" prepare
bash "$DIAG/run.sh" test
bash "$DIAG/run.sh" preflight
# Only on the target host, after all checks pass and the user elects to run:
bash "$DIAG/run.sh" run
# Independent report/paired deltas/ZIP, never starts inference/training:
bash "$DIAG/run.sh" analyze
```

Output: `/home/islabworker2/minhhb/tanvt/augseg_diagnostics/s2_donor_pool_replay_a6000_20261105/`.
The code checkout remains immutable. No services or queues are installed or changed. The foreground
run needs a persistent terminal/session if desired; this package does not enqueue or auto-start anything.
GPU0 guard≥16384MiB free and output filesystem≥32GiB free are retained; other compute processes may
exist. These are preflight thresholds, not a guarantee that FP32 B16 will fit. No auto batch/AMP change,
kill, resume or retry. `diagnostic.lock` prevents concurrent replay; completed outputs are not rerun;
partial observations block automatic replay. Analysis errors cannot restart the model.

Tests run in a fresh temporary directory using synthetic tensors/toy modules/mocks only. They never
load the designated checkpoint or run real validation. On A6000 after explicit prepare they record a
package-bound CPU prerequisite receipt in the external output directory. Real actions enforce target
host/user and package/input/CPU/resource guards before any checkpoint restore or CUDA setup.

## Measurements and permitted interpretation

Per-batch/per-class raw counts: valid GT-origin actually surviving mixing, active correct/wrong pseudo
(q≥0.95), valid filtered, GT-ignore, pseudo reference-ignore active/filtered; all counts conserve BHW.
Histograms distinguish reference class from assigned target class. Mass uses full BHW; conditional
coverage/error denominators are explicit, zero denominator is NA. Intermediate baseline donors can
already contain labeled GT, and provenance/reference transport includes both mixing stages.
Selected/candidate components record class, area, saliency score, selection probability and entropy;
zero-component fallback is separate from selector exception. Input hashes and teacher parameter/buffer
hashes are checked throughout. No per-image saliency caching changes the batch-mean gradient contract.

Read all21 classes and paired S2−baseline deltas per pool, then difference-of-deltas across pools.
Reduction in wrong pseudo mass must be read with loss of correct pseudo mass and increase in GT mass.
Removing supervision of displaced receiver content is not correcting a prediction for that same content.
Bootstrap uses paired receiver batches, not independent pixels or training seeds. This measures donor
pool effects on supervision allocation with a fixed teacher, not historical masks, repetition effects,
learning utility, causation of the full mIoU gain or robustness over training seeds.

Historical source whitespace/CRLF is deliberately retained to preserve pinned bytes; syntax and
SHA256 checks cover those exports. Whitespace diff checks apply to new operational files, excluding
the immutable historical source export and generated unified diff. `.gitattributes` prevents EOL conversion.

Private input archive creation (source host only, outside every Git checkout):
```bash
python -B diagnostics/s2_donor_pool_a6000_20261105/transfer_inputs.py --output <NEW_PRIVATE_TRANSFER_DIRECTORY>
```
The archive contains only the5621 required input files (5616 images/masks,4 lists,1 checkpoint),
with payload_manifest.json, SHA256SUMS and archive_receipt.json alongside. Transfer them privately,
verify `sha256sum -c SHA256SUMS` at the receiver, then extract to a NEW private staging directory,
not over production files. Map relative `augseg_inputs/...` and `augseg_diagnostic_inputs/...` to the
target root, reuse existing files only after checksum comparison, and run check-inputs. No automatic
network transfer, extraction, overwrite or installation occurs in this package.
