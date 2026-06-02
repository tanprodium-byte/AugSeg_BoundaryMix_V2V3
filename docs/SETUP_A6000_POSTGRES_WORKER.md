# A6000 Postgres Worker Setup

This worker does not need network routing to supermaster. It only needs outbound access to Cloud Postgres and HuggingFace.

## 1. Get the repo onto A6000

Use either:

```bash
git clone <repo-url> AugSeg_BoundaryMix_V2V3
```

or copy the current repo from supermaster, including the generated `scheduler/`, `scripts/`, `docs/`, and `.codex_smoke/real_configs/` files.

## 2. Create the environment

Create or reuse the AugSeg conda environment:

```bash
conda create -n augseg-bm python=3.11
conda activate augseg-bm
python -m pip install -r requirements.txt
python -m pip install "psycopg[binary]"
```

Install any CUDA/PyTorch packages required by the host GPU and the existing AugSeg training setup.

## 3. Dataset and pretrained paths

Set up read-only symlinks or mounts for datasets and pretrained weights so the copied configs resolve the same paths used by the repo.

Recommended:

```bash
ln -s /path/to/VOCdevkit ./data/VOCdevkit
ln -s /path/to/pretrained ./pretrained
```

Do not modify the original dataset or pretrained directories from the worker.

## 4. HuggingFace auth

Authenticate the machine:

```bash
huggingface-cli login
python scripts/check_hf_access.py
```

Do not paste or print the token in logs.

## 5. Cloud Postgres secret

Create a private env file:

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets
cat > ~/.secrets/augseg_scheduler.env <<'EOF'
export AUGSEG_SCHEDULER_DB_URL='postgresql://...'
EOF
chmod 600 ~/.secrets/augseg_scheduler.env
```

Do not commit this file and do not hardcode the URL into systemd services.

## 6. Smoke tests

Source the secret, then run:

```bash
source ~/.secrets/augseg_scheduler.env
python scheduler/status_postgres.py
python scripts/check_hf_access.py
bash scripts/check_gpu_free.sh
```

Only start the worker when the GPU is free.

## 7. User systemd service

Create `~/.config/systemd/user/augseg-worker-a6000-postgres.service`:

```ini
[Unit]
Description=AugSeg BoundaryMix A6000 Postgres Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc 'source ~/.secrets/augseg_scheduler.env && conda run -n augseg-bm python scheduler/worker.py --backend postgres --server-name islab-server3 --gpu-id 0 --loop --sleep-sec 60'
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

Install and start:

```bash
systemctl --user daemon-reload
systemctl --user enable augseg-worker-a6000-postgres.service
systemctl --user start augseg-worker-a6000-postgres.service
systemctl --user status augseg-worker-a6000-postgres.service
```

For logs:

```bash
journalctl --user -u augseg-worker-a6000-postgres.service -f
```
