from __future__ import annotations

from pathlib import Path
import socket
import yaml


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "scheduler/state.sqlite"
HF_SETTINGS_PATH = ROOT / "scheduler/hf_settings.yaml"
REAL_CONFIG_ROOT = ROOT / ".codex_smoke/real_configs/voc5_single_gpu_gbs8"
RUN_ROOT = ROOT / ".scheduler_runs"

VOC5_CONFIG_IDS = [
    "v2_component_weighting",
    "v2_v3_best_template",
    "v3_js_bcr_d1",
    "v3_js_bcr_d2",
    "v3_js_bcr_d3",
]


def load_hf_settings() -> dict:
    return yaml.safe_load(HF_SETTINGS_PATH.read_text())


def default_worker_id(server_name: str | None, gpu_id: int) -> str:
    server = server_name or socket.gethostname()
    return f"{server}:gpu{gpu_id}"
