# Root layout policy

This repository root should stay small and readable.

## Keep in root

- `train_semi.py`: main training entrypoint.
- `single_run.sh`: manual single-config launcher. Keep it in root because it resolves `ROOT` from its own location.
- `AGENTS.md`: short rules for Codex/agents.
- `.gitignore`: Git ignore rules.
- Main code/config folders: `augseg/`, `configs/`, `exps/`, `tools/`, `util/`, `scheduler/`, `scripts/`.

## Do not create new runner scripts in root

New automation runner scripts should go under:

- `ops/runners/active/<SUITE_ID>/<SERVER_NAME>/`

Old runner scripts should go under:

- `ops/runners/archive/<OLD_SUITE>/<SERVER_NAME>/`

## Docs

- `docs/policies/`: operating rules and scheduler policy.
- `docs/status/`: setup/status notes.
- `docs/audits/`: server/system audits.
- `docs/*.md`: method specs, usage notes, and project documentation.

## Artifacts

Generated reports, W&B exports, and other non-source outputs should go under:

- `artifacts/`

## Runtime outputs

Training outputs, temporary configs, logs, checkpoints, and status files should stay under:

- `runs/`
- `wandb/`
- `tmp/`

## Important rule

Do not move any script that is currently referenced by an active systemd service.

Before moving runner scripts, check:

```bash
systemctl --user status <service-name> --no-pager
systemctl --user cat <service-name>