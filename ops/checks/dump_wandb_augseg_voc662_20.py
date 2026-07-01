#!/usr/bin/env python3
import csv
import datetime
import json
import os
import subprocess
from pathlib import Path


ENTITY = "tanprodium-uit"
DEFAULT_PROJECT = "augseg-voc662"
LIMIT = 20
OUT_DIR = Path("wandb_exports")


def read_proc_env(pid: str) -> dict:
    env_path = Path(f"/proc/{pid}/environ")
    if not env_path.exists():
        return {}

    raw = env_path.read_bytes()
    items = raw.split(b"\0")
    env = {}

    for item in items:
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode(errors="ignore")] = value.decode(errors="ignore")

    return env


def find_augseg_runner_pid() -> str:
    out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)

    for line in out.splitlines():
        if "run_voc8_sc_5090_segments_wandb.sh" in line:
            parts = line.strip().split(maxsplit=1)
            if parts:
                return parts[0]

    return ""


def setup_wandb_env() -> tuple[str, str]:
    """
    Ưu tiên dùng WANDB_API_KEY nếu shell hiện tại đã có.
    Nếu chưa có, lấy key từ process runner AugSeg đang chạy.
    Không in key ra màn hình.
    """
    if os.environ.get("WANDB_API_KEY"):
        project = os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
        return "current_shell", project

    pid = find_augseg_runner_pid()
    if not pid:
        raise RuntimeError(
            "Không tìm thấy process run_voc8_sc_5090_segments_wandb.sh. "
            "Cần runner đang chạy, hoặc bạn phải export WANDB_API_KEY trước."
        )

    runner_env = read_proc_env(pid)
    api_key = runner_env.get("WANDB_API_KEY", "")
    project = runner_env.get("WANDB_PROJECT", DEFAULT_PROJECT)

    if not api_key:
        raise RuntimeError(
            f"Tìm thấy runner PID={pid}, nhưng không thấy WANDB_API_KEY trong environment."
        )

    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_PROJECT"] = project

    return f"runner_pid_{pid}", project


def jsonable(value):
    """
    Convert object sang dạng ghi được JSON.
    Tránh dùng hasattr(value, "item") vì W&B Summary object có thể raise KeyError.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    try:
        return value.item()
    except Exception:
        pass

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def pick(d: dict, keys: list[str], default=""):
    if not isinstance(d, dict):
        return default

    for key in keys:
        value = d.get(key)
        if value not in (None, ""):
            return value

    return default


def short(value, n=180):
    value = jsonable(value)

    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)

    text = text.replace("\n", "\\n")

    if len(text) > n:
        return text[:n] + "..."

    return text


def method_hint(name: str) -> str:
    n = name.lower()

    hints = [
        ("s1_saliency_box_direct_cutmix", "S1: saliency box direct CutMix, paste cùng tọa độ."),
        ("s1_saliency_box_relocated_cutmix", "S1 relocated: saliency box, paste sang vị trí random target."),
        ("s1_saliency_box_adaptive_relocated_cutmix", "S1 adaptive relocated: relocated CutMix + confidence gate."),
        ("s2_saliency_component_mask_direct_cutmix", "S2: saliency component-mask direct CutMix."),
        ("s3_saliency_component_mask_direct_plus_v3_d2", "S3: saliency component-mask direct CutMix + V3/BCR d2."),
        ("c1_csl_official_reliability_replace_confidence", "C1: official-like CSL reliability thay confidence."),
        ("c2_csl_official_reliable_mask_perturbation", "C2: official-like CSL reliable-mask perturbation."),
        ("c3_csl_official_guided_cutmix_plus_v3_d2", "C3 official: CSL-guided CutMix + V3/BCR d2."),
        ("c3_csl_guided_cutmix_plus_v3_d2", "C3 proxy/cũ: CSL-guided CutMix dùng proxy, không phải official PCOS."),
        ("baseline", "Baseline run."),
        ("v2", "BoundaryMix/V2-related run."),
        ("v3", "V3/BCR-related run."),
    ]

    for key, hint in hints:
        if key in n:
            return hint

    return "Không nhận diện tự động từ tên run; xem selected_config bên dưới."


def selected_config(cfg: dict) -> dict:
    """
    Lấy phần config quan trọng để đọc nhanh nội dung phương pháp.
    Full config vẫn lưu trong JSON.
    """
    if not isinstance(cfg, dict):
        return {}

    out = {}

    top_keys = [
        "name",
        "seed",
        "epochs",
        "batch_size",
        "crop_size",
        "dataset",
        "nclass",
        "conf_thresh",
        "backbone",
        "lr",
        "lr_multi",
        "criterion",
    ]

    for key in top_keys:
        if key in cfg:
            out[key] = cfg[key]

    sections = [
        "wandb",
        "saver",
        "saliency_cutmix",
        "boundary_mix",
        "boundary_component",
        "boundary_compatibility",
        "csl",
        "csl_cutmix",
        "checkpoint",
    ]

    for section in sections:
        if section in cfg:
            out[section] = cfg[section]

    return out

def safe_run_attr_dict(run, attr_name: str) -> dict:
    """
    Đọc run.config / run.summary / run.metadata an toàn.
    Nếu W&B API bị busy hoặc timeout thì không làm chết toàn bộ script.
    """
    try:
        value = getattr(run, attr_name)
        if value is None:
            return {}
        return dict(value)
    except Exception as e:
        return {
            "_wandb_read_error": f"{type(e).__name__}: {e}",
            "_wandb_attr": attr_name,
        }

def main():
    source, project = setup_wandb_env()

    print(f"W&B key source: {source}")
    print(f"W&B target: {ENTITY}/{project}")
    print("API key is loaded but will not be printed.")
    print()

    import wandb

    OUT_DIR.mkdir(exist_ok=True)

    out_md = OUT_DIR / "latest20_augseg_voc662.md"
    out_csv = OUT_DIR / "latest20_augseg_voc662.csv"
    out_json = OUT_DIR / "latest20_augseg_voc662.json"

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{project}", order="-created_at"))[:LIMIT]

    print(f"Found latest {len(runs)} runs.")
    print()

    items = []

    for idx, run in enumerate(runs, 1):
        cfg = safe_run_attr_dict(run, "config")
        summary = safe_run_attr_dict(run, "summary")
        metadata = safe_run_attr_dict(run, "metadata")

        epoch = pick(summary, ["epoch", "EPOCH", "_step", "iter", "iteration"])
        val_miou = pick(
            summary,
            ["val/mIoU", "VAL/MIOU", "val_mIoU", "val/miou_STU", "eval/mIoU", "mIoU"],
        )
        best_miou = pick(
            summary,
            ["val/best_mIoU", "VAL/BEST_M", "val/best_mIoU_STU", "best_mIoU", "best/miou", "best_miou"],
        )

        method_name = cfg.get("name") or (cfg.get("wandb") or {}).get("name") or run.name
        hint = method_hint(str(method_name))

        item = {
            "idx": idx,
            "run_name": run.name,
            "method_name": method_name,
            "method_hint": hint,
            "id": run.id,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "epoch_or_step": jsonable(epoch),
            "val_mIoU": jsonable(val_miou),
            "best_mIoU": jsonable(best_miou),
            "group": getattr(run, "group", None),
            "job_type": getattr(run, "job_type", None),
            "tags": list(getattr(run, "tags", []) or []),
            "program": metadata.get("program", ""),
            "codePath": metadata.get("codePath", ""),
            "args": jsonable(metadata.get("args", "")),
            "selected_config": jsonable(selected_config(cfg)),
            "full_config": jsonable(cfg),
            "summary": jsonable(summary),
        }

        items.append(item)

        print(f"{idx:02d}. {run.name}")
        print(f"    method={method_name}")
        print(f"    state={run.state} | epoch/step={short(epoch)} | best={short(best_miou)}")
        print(f"    hint={hint}")
        print(f"    url={run.url}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "run_name",
                "method_name",
                "method_hint",
                "state",
                "created_at",
                "url",
                "epoch_or_step",
                "val_mIoU",
                "best_mIoU",
                "program",
                "args",
            ]
        )

        for item in items:
            writer.writerow(
                [
                    item["idx"],
                    item["run_name"],
                    item["method_name"],
                    item["method_hint"],
                    item["state"],
                    item["created_at"],
                    item["url"],
                    short(item["epoch_or_step"]),
                    short(item["val_mIoU"]),
                    short(item["best_mIoU"]),
                    short(item["program"]),
                    short(item["args"], 500),
                ]
            )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Latest {len(items)} W&B runs in `{ENTITY}/{project}`\n\n")
        f.write(f"Generated at: `{datetime.datetime.now().isoformat(timespec='seconds')}`\n\n")

        for item in items:
            f.write(f"## {item['idx']:02d}. {item['run_name']}\n\n")
            f.write(f"- **method_name:** `{item['method_name']}`\n")
            f.write(f"- **method_hint:** {item['method_hint']}\n")
            f.write(f"- **state:** `{item['state']}`\n")
            f.write(f"- **created_at:** `{item['created_at']}`\n")
            f.write(f"- **url:** {item['url']}\n")
            f.write(f"- **epoch_or_step:** `{short(item['epoch_or_step'])}`\n")
            f.write(f"- **val_mIoU:** `{short(item['val_mIoU'])}`\n")
            f.write(f"- **best_mIoU:** `{short(item['best_mIoU'])}`\n")
            f.write(f"- **program:** `{short(item['program'])}`\n")
            f.write(f"- **args:** `{short(item['args'], 700)}`\n\n")

            f.write("### Selected config\n\n")
            f.write("```json\n")
            f.write(json.dumps(item["selected_config"], ensure_ascii=False, indent=2, sort_keys=True))
            f.write("\n```\n\n")

    print()
    print("Saved:")
    print(f"  {out_md}")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()