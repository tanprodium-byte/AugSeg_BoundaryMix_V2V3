# A6000 Systemd Worker Setup

This is the production-lite worker setup for `islab-server3` with an A6000 GPU. The worker calls the coordinator API running on supermaster and never writes `scheduler/state.sqlite` directly.

## 1. Clone Repo

```bash
git clone <repo-url> /home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
cd /home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
git checkout boundarymix-v2-v3
```

Use the same branch/commit family as supermaster.

## 2. Create Conda Env

Create or reuse the `augseg-bm` environment with the same CUDA/PyTorch stack used on supermaster.

```bash
conda create -n augseg-bm python=3.11 -y
conda activate augseg-bm
pip install -r requirements.txt
pip install huggingface_hub
```

If this repo does not have a complete `requirements.txt`, mirror the package versions from the supermaster `augseg-bm` environment.

## 3. Dataset And Pretrained Links

Set up dataset, split, and pretrained paths as symlinks only. Treat shared source paths as read-only.

Expected repo-visible paths:

```text
data/VOC2012
data/splitsall/pascal_u2pl/662/labeled.txt
data/splitsall/pascal_u2pl/662/unlabeled.txt
data/splitsall/pascal_u2pl/val.txt
pretrained/resnet101.pth
```

Do not chmod, chown, delete, or modify shared dataset/pretrained files.

## 4. Hugging Face Auth

Log in interactively on `islab-server3`. Do not put the token in systemd service files.

```bash
hf auth login
python scripts/check_hf_access.py
```

## 5. CUDA Test

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print((x @ x).mean().item())
PY
```

## 6. Coordinator Health

Replace `<supermaster-ip>` with the reachable supermaster IP or hostname.

```bash
curl http://<supermaster-ip>:8787/health
```

Expected response:

```json
{"ok": true}
```

## 7. Create User Service

Create `~/.config/systemd/user/augseg-worker-a6000.service`:

```ini
[Unit]
Description=AugSeg BoundaryMix A6000 Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3
ExecStart=/bin/bash -lc 'conda run -n augseg-bm python scheduler/worker.py --server-name islab-server3 --gpu-id 0 --coordinator-url http://<supermaster-ip>:8787 --loop --sleep-sec 60'
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

Then load and start:

```bash
systemctl --user daemon-reload
systemctl --user enable augseg-worker-a6000.service
systemctl --user start augseg-worker-a6000.service
systemctl --user status augseg-worker-a6000.service --no-pager
journalctl --user -u augseg-worker-a6000.service -f
```

The worker loop checks GPU memory first. If the GPU is busy, it sleeps and tries again; it only requests a job and launches a segment when the GPU is under the configured free threshold.
