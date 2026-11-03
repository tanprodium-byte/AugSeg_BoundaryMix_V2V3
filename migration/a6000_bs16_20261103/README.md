# BS16 A6000 migration — prepared package, not runtime attestation

Target root is `/home/islabworker2/minhhb/tanvt` (NOT tantv). Do not touch foodvlm_framework. No queue/unit is installed or activated by this package.

## Sources and scientific contract

`historical/` is immutable exported evidence; `sources/baseline`, `sources/s1`, and `sources/s2` are three separate complete runtime source trees. All scientific/runtime Python source bytes equal their respective exports. Parent S2 remains `9047b1a235541917d77f133401a2547651e7c0e1`; `historical/git_exports/` supplies immutable content and SHA256 verification without publishing unrelated local Git history. This directory is a content export, not a Git commit object.

VOC662/R101/DeepLabV3+/C321, physical16 labeled+16 unlabeled, world1, seed2, fresh80, loader620,49600 updates. No accumulation/AMP/algorithm optimization. S1 keeps both AugSeg stages, teacher saliency computation, K8 pool, beta sampling, sample gate and donor/transport. Random-K8 changes only multinomial weights to uniform. S2 keeps its original one-stage exact-mask relocation, eligibility/fallback/RNG and saliency computation; only component score is area/(HW). Historical reviewed pins are unchanged. New package checks attest preparation, not runtime on A6000.

The new A6000 baseline is the same-hardware comparator for the cohort's overall policies. The historical S2 saliency×area control remains RTX5090: this four-run cohort does not create an A6000 saliency×area S2 control, so it alone cannot isolate that coefficient across hardware. No extra run is registered.

## Fetch and check exact commit

Clone this branch into `/home/islabworker2/minhhb/tanvt/AugSeg_BoundaryMix_V2V3_legacy_c321_rerun01`, detach to the full migration commit SHA returned by deployment, and record `git rev-parse HEAD`. Do not run from default branch.

Before an environment exists, inspect `SHA256.json`, `registry.json`, `origins.json`, `inputs.json`. `check.sh` runs CPU synthetic/mocks only; it cannot train or write remote resources.

## Environment (explicit, separate; not run by bootstrap)

Create a dedicated Python3.11 environment at `/home/islabworker2/minhhb/tanvt/envs/augseg-bs16`. Do not install into another project's environment. Set `AUGSEG_PYTHON` to its absolute interpreter path.

```bash
python3.11 -m venv /home/islabworker2/minhhb/tanvt/envs/augseg-bs16
export AUGSEG_PYTHON=/home/islabworker2/minhhb/tanvt/envs/augseg-bs16/bin/python
"$AUGSEG_PYTHON" -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$AUGSEG_PYTHON" -m pip install -r migration/a6000_bs16_20261103/runtime/requirements-a6000.txt
bash migration/a6000_bs16_20261103/check.sh
```

The saved requirements are run-associated package records, current installed metadata is separately labeled, and historically loaded CUDA/cuDNN/TF32 remain UNKNOWN. W&B cudaVersion is not CUDA runtime. No claim of bitwise equivalence across GPUs. Direct imported dependencies are pinned from source-host metadata; compare with the separately preserved saved inventories. Target installation and loaded-library compatibility remain unverified. Source sets cuDNN deterministic/benchmark policy via its unchanged seed helper.

## Inputs and bootstrap

Supply dataset/splits/pretrained separately at the exact destinations in `inputs.json`; nothing downloads automatically. Full dataset hash is UNKNOWN; recorded pretrained hash comes from reviewed setup, not rehashed during export. Preflight rehashes all split/pretrained inputs and verifies every referenced path before launch.

```bash
bash migration/a6000_bs16_20261103/run.sh bootstrap
```

Bootstrap copies source trees/configs to `<root>/augseg_deployments/` and creates source-attestation files there; it never creates production executions or remote runs. It refuses differing existing files. The original RTX5090 source pins are preserved under historical; portable checks use the complete SHA256 package manifest and immutable control evidence instead of159 machine-local protected paths.

Configure credentials outside Git at `<root>/augseg_auth/credentials.env`, owned by the caller with mode600 (or set AUGSEG_AUTH_FILE). Use existing account credentials; never paste them into configs or logs. `WANDB_API_KEY` and an existing HF_TOKEN/HUGGINGFACE_TOKEN/HUGGING_FACE_HUB_TOKEN are read only from that injection. Online mode is mandatory, with bounded read retries and no offline fallback.

## Explicit production commands, in registered order (NOT executed during preparation)

For each method below, first `run.sh preflight --method METHOD`, then `run.sh launch --method METHOD`, then `run.sh verify-done --method METHOD`:

1. baseline_augseg
2. s1_two_stage_saliency_k8
3. s1_two_stage_random_k8
4. s2_mask_relocated_area_only

Launch is synchronous through training and separate finalization. The next profile requires predecessor raw completion plus remotely verified receipt. Preflight checks A6000 GPU0 free>=16384MiB and disk>=32GiB; other users' compute is allowed, own locks/freshness/collision checks remain. This guard is not a VRAM guarantee. No automatic resume, new attempt or training retry.

If training succeeds but finalization fails, use only `run.sh finalize --method METHOD`; it has no training path. Inspect existing remote HF outcome first if an upload intent exists. A matching existing remote artifact can be verified without uploading; an uncertain intent with no remote path remains held. No placeholder/test remote run or upload is part of bootstrap/checks. W&B run IDs, outputs and HF paths are independent of RTX5090. Targets remain the existing W&B project/HF repository; A6000 is explicit in groups/tags/prefixes.

## Evidence and readiness limits

`evidence/controls/` contains derived, allowlisted CSV/log/metadata records, not production state. `evidence/derivations.json` records original hashes and transformations. No checkpoint/model tensors, dataset, pretrained, bundles, W&B directories, credentials, Codex state, locks or markers are included. Sources of package versions are labeled in runtime/inventory.json.

CPU checks establish scientific byte/config invariants and operation state-machine behavior, not A6000 environment/resources/remote write permission or training success. Before smoke/production, verify target inputs, interpreter/dependencies, source checksum, fresh remote identities, authentication and resources. Runtime logging/upload success remains untested until actual authorized execution.

The unmodified baseline logger may report git_commit=unknown outside a Git worktree. Its parent SHA, complete scientific source hashes, portable source pin (also in W&B config), migration commit and package manifest in provenance are authoritative; do not replace unknown with a historical clean-worktree claim.

Historical/source exports retain their original whitespace because hashes are immutable. Scoped .gitattributes disables whitespace lint only for these exports/diffs; new operational Python/shell/docs keep normal diff checks. Scientific byte comparisons remain strict.
