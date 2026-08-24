import argparse
import yaml
import os
import os.path as osp
import posixpath
import pprint
import re

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import time
import numpy as np
import pandas as pd
import torch.nn.functional as F

from augseg.utils.dist_helper import setup_distributed
from augseg.utils.utils import set_random_seed, setup_default_logging
from augseg.models.model_helper import ModelBuilder
from augseg.utils.loss_helper import get_criterion
from augseg.dataset.builder import get_loader
from augseg.utils.lr_helper import get_optimizer, get_scheduler
from augseg.utils.utils import AverageMeter, intersectionAndUnion
from augseg.dataset.augs_ALIA import cut_mix_label_adaptive
from augseg.utils.loss_helper import compute_unsupervised_loss_by_threshold
from util.boundary_mix import (
    _rand_bbox,
    boundary_mix_debug_stats,
    compute_c4_direct_mix_stats,
    cut_mix_label_adaptive_c4_direct_labeled,
    cut_mix_label_adaptive_with_mask,
    thresholded_boundary_mix_loss,
)
from util.boundary_component import compute_component_weights
from util.boundary_compatibility import compute_js_boundary_compatibility_loss
from util.csl_cutmix import get_csl_guided_boxes
from util.csl_official import (
    apply_csl_reliable_mask_perturbation,
    compute_csl_official_selection,
)
from util.csl_reliability import apply_csl_random_reliable_mask, compute_csl_reliability
from util.saliency_cutmix import (
    get_saliency_component_guided_boxes,
    get_saliency_component_guided_masks,
    get_saliency_guided_boxes,
)
from util.u1_saliency_u2u import (
    NUM_CANDIDATES as U1_NUM_CANDIDATES,
    TEMPERATURE as U1_TEMPERATURE,
    U1Diagnostics,
    U1_RNG_POLICY_VERSION,
    aggregate_diagnostics as aggregate_u1_diagnostics,
    apply_u1_saliency_u2u,
    synchronized_failure_check as u1_synchronized_failure_check,
)
from tools.visualize_boundary_mix_debug import save_boundary_mix_debug
from util.run_logging import (
    get_or_create_run_id,
    append_csv_row,
    trim_csv_rows_by_epoch,
    migrate_legacy_csv_if_missing,
    default_log_paths,
    get_git_commit,
    write_json,
)
from util.checkpointing import (
    save_checkpoint,
    load_checkpoint_if_available,
    copy_config_to_save_path,
)
from util.hf_auto import (
    maybe_download_hf_bundle,
    maybe_upload_hf_bundle,
)

# ---------------- Aa op mapping + strength normalize (0..1) ----------------
AA_OPS = {
    1: "identity",
    2: "autocontrast",
    3: "equalize",
    4: "blur",
    5: "contrast",
    6: "brightness",
    7: "color",
    8: "sharpness",
    9: "posterize",
    10: "solarize",
    11: "hue",
}


def select_unlabeled_mix_branch(
    u1_enabled, use_cutmix, trigger_prob, u2_enabled=False, u3_enabled=False, u4_enabled=False
):
    """Select U1 or the unchanged legacy CutMix trigger path.

    Keeping this boundary small makes the disabled-path RNG contract directly
    testable: U1 consumes no legacy trigger draw, while disabled U1 consumes the
    same single global NumPy draw as the pre-U1 path.
    """
    if sum(bool(enabled) for enabled in (u1_enabled, u2_enabled, u3_enabled, u4_enabled)) > 1:
        raise ValueError("U1, U2, U3, and U4 cannot be enabled simultaneously")
    if u1_enabled:
        return "u1", 1, 1
    if u2_enabled:
        return "u2", 1, 1
    if u3_enabled:
        return "u3", 1, 1
    if u4_enabled:
        return "u4", 1, 1
    rnd = np.random.uniform(0, 1)
    triggered = int(rnd < trigger_prob)
    return "legacy", triggered, int(triggered and use_cutmix)


def is_u_saliency_mix_branch(mix_branch):
    """Return whether the U1–U4 helper has completed the iteration's mixing."""
    return mix_branch in ("u1", "u2", "u3", "u4")

def aa_strength(k_id: int, t: float) -> float:
    """
    Chuẩn hoá intensity về [0,1] (0=nhẹ, 1=mạnh).
    Giải quyết vấn đề: thang đo khác nhau + đảo chiều (solarize/posterize).
    """
    import math
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return float("nan")

    # blur sigma: càng lớn càng mạnh
    if k_id == 4:
        return float((t - 0.1) / (2.0 - 0.1))

    # contrast/brightness/color/sharpness: factor (<=1) càng nhỏ càng mạnh
    if k_id in (5, 6, 7, 8):
        vmin, vmax = 0.05, 0.95
        v = max(vmin, min(vmax, float(t)))
        return float((1.0 - v) / (1.0 - vmin))

    # posterize bits: bits càng thấp càng mạnh
    if k_id == 9:
        b = int(round(float(t)))
        b = max(1, min(8, b))
        return float((8 - b) / 7.0)

    # solarize threshold: threshold càng thấp càng mạnh
    if k_id == 10:
        thr = int(round(float(t)))
        thr = max(1, min(256, thr))
        return float((256 - thr) / 255.0)

    # hue: |hue| càng lớn càng mạnh
    if k_id == 11:
        v = max(-0.5, min(0.5, float(t)))
        return float(abs(v) / 0.5)

    return float("nan")

def read_run_id_from_ckpt(ckpt_path, fallback_run_id):
    """
    Đọc run_id từ checkpoint.
    Nếu checkpoint chưa có run_id thì dùng fallback_run_id.
    """
    print(f"[DEBUG] loading ckpt: {ckpt_path}", flush=True)

    if not os.path.exists(ckpt_path):
        return fallback_run_id

    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[DEBUG] torch.load finished in {time.time() - t0:.2f}s", flush=True)

    return ckpt.get("run_id", fallback_run_id)
    
def trim_iter_csv_for_resume(train_iter_csv, last_epoch, logger = None):
    """
    Xóa các dòng iter log của epoch chưa hoàn tất để tránh duplicate khi resume.

    Quy ước:
    - nếu last_epoch = 7
    - nghĩa là checkpoint hiện tại tương ứng trạng thái sau epoch 6
    - vậy chỉ giữ meta/epoch < 7
    """
    if train_iter_csv is None:
        return
    
    if not os.path.exists(train_iter_csv):
        return

    try: 
        df = pd.read_csv(train_iter_csv)

        if "meta/epoch" not in df.columns:
            if logger is not None:
                logger.info(
                    f"[resume-trim] bỏ qua trim vì file không có cột meta/epoch: {train_iter_csv}"
                )
            return
            
        old_len = len(df)

        df = df[df["meta/epoch"] < last_epoch].copy()

        new_len = len(df)
        removed = old_len - new_len
        
        df.to_csv(train_iter_csv, index = False)

        if logger is not None:
            logger.info(
                f"[resume-trim] file={train_iter_csv}, last_epoch={last_epoch}, "
                f"removed_rows={removed}, remaining_rows={new_len}"
            )
    except Exception as e:
        if logger is not None:
            logger.info(f"[resume-trim] lỗi khi trim iter csv: {e}")

def build_iter_log_columns():
    cols = [
        # meta
        "meta/log_type",
        "meta/epoch",
        "meta/iter_in_epoch",
        "meta/global_iter",

        # core training
        "iter/sup_loss",
        "iter/uns_loss",
        "iter/pseudo_high_ratio",
        "iter/lr",

        # validation
        "val/model",
        "val/class_id",
        "val/loss",
        "val/miou",
        "val/class_iou",
        "val/best_miou",

        # Ar
        "ar/triggered",
        "ar/applied",
        "ar/area_ratio_est",

        # U
        "u/entropy_mean",
        "u/maxprob_mean",
        "u/maxprob_p10",
        "u/maxprob_p50",
        "u/maxprob_p90",
        "u/pseudo_ratio_mean",

        # Aa global
        "aa/ops_per_image",
    ]

    cols.extend([
        "bcr/num_pairs_candidate",
        "bcr/num_pairs_sampled",
        "bcr/num_pairs_active",
        "bcr/num_pairs_same",
        "bcr/num_pairs_diff",
        "bcr/num_pairs_uncertain",
        "bcr/mean_s_sem",
        "bcr/mean_s_sem_same",
        "bcr/mean_s_sem_diff",
        "bcr/mean_s_S",
        "bcr/mean_s_S_same",
        "bcr/mean_s_S_diff",
        "bcr/mean_r_ab",
        "bcr/loss_bcr",
        "bcr/loss_same",
        "bcr/loss_diff",
        "bcr/use_component_gate",
        "bcr/component_gate_mode",
        "bcr/mean_q_pair_a",
        "bcr/mean_q_pair_b",
        "bcr/mean_q_pair_product",
        "bcr/mean_r_before_component_gate",
        "bcr/mean_r_after_component_gate",
        "bcr/mean_abs_sS_minus_sSem_same",
        "bcr/mean_s_T",
        "bcr/mean_s_T_same",
        "bcr/mean_s_T_diff",
        "bcr/mean_teacher_feature_gate_same",
        "bcr/mean_teacher_feature_gate_diff",
        "bcr/mean_abs_sS_minus_sT_active",
        "bcr/affinity_temperature",
        "bcr/mean_A_affinity",
        "bcr/mean_A_same",
        "bcr/mean_A_diff",
        "bcr/mean_y_rel",
        "bcr/loss_affinity",
        "v2/num_affected_components",
        "v2/num_affected_pixels",
        "v2/mean_visible_ratio",
        "v2/mean_visible_area",
        "v2/mean_component_confidence",
        "v2/mean_q_C",
        "v2/min_q_C",
        "v2/max_q_C",
        "v2/loss_mix_v2",
        "saliency/enabled",
        "saliency/mode",
        "saliency/num_candidates",
        "saliency/temperature",
        "saliency/score_selected",
        "saliency/score_candidate_mean",
        "saliency/score_candidate_max",
        "saliency/score_candidate_min",
        "saliency/score_candidate_std",
        "saliency/prob_selected",
        "saliency/prob_max",
        "saliency/selection_entropy",
        "saliency/fallback_ratio",
        "saliency/source_is_labeled_ratio",
        "saliency/num_components",
        "saliency/num_valid_components",
        "saliency/selected_component_class",
        "saliency/selected_component_area",
        "saliency/selected_component_score",
        "saliency/component_score_mean",
        "saliency/component_score_max",
        "saliency/component_box_area",
        "relocation_draw_count",
        "relocation_zero_count",
        "relocation_nonzero_count",
        "relocation_success_count",
        "relocation_zero_only_count",
        "relocation_random_zero_count",
        "relocation_empty_mask_count",
        "relocation_valid_translation_sum",
        "relocation_expected_zero_sum",
        "relocation_displacement_magnitude_sum",
        "relocation_source_destination_iou_sum",
        "relocation_zero_rate",
        "relocation_nonzero_rate",
        "relocation_success_rate",
        "relocation_zero_only_rate",
        "relocation_random_zero_rate",
        "relocation_mean_valid_translation_count",
        "relocation_expected_zero_rate",
        "relocation_mean_displacement_magnitude",
        "relocation_mean_source_destination_iou",
        "saliency_selector_exception_batch",
        "saliency/mix1_score_selected",
        "saliency/mix1_score_candidate_mean",
        "saliency/mix1_selection_entropy",
        "saliency/mix2_score_selected",
        "saliency/mix2_score_candidate_mean",
        "saliency/mix2_selection_entropy",
        "csl/enabled",
        "csl/mode",
        "csl/use_csl_for_ce_weight",
        "csl/use_csl_for_mix_confidence",
        "csl/use_csl_for_cutmix",
        "csl/random_mask_reliable",
        "csl/perturb_input",
        "csl/mean_reliability",
        "csl/reliability_std",
        "csl/reliability_min",
        "csl/reliability_max",
        "csl/confidence_mean",
        "csl/entropy_mean",
        "csl/margin_mean",
        "csl/residual_variance_mean",
        "csl/official_weight_valid_mean",
        "csl/reliable_ratio",
        "csl/sample_reliability_mean",
        "csl/mask_prob",
        "csl/masked_ratio",
        "csl/raw_reliability_mean",
        "csl/effective_weight_mean",
        "csl/perturb_reliable_ratio",
        "csl_ce/weight_mean",
        "csl_ce/weight_sum",
        "csl_cutmix/enabled",
        "csl_cutmix/score_selected",
        "csl_cutmix/score_candidate_mean",
        "csl_cutmix/score_candidate_max",
        "csl_cutmix/score_candidate_min",
        "csl_cutmix/score_candidate_std",
        "csl_cutmix/prob_selected",
        "csl_cutmix/prob_max",
        "csl_cutmix/selection_entropy",
        "csl_cutmix/target_reliability_selected",
        "csl_cutmix/fallback_ratio",
        "c4/gate_attempted_count",
        "c4/gate_pass_count",
        "c4/gate_pass_ratio",
        "c4/mixed_sample_count",
        "c4/mixed_sample_ratio",
        "c4/pasted_pixel_count",
        "c4/pasted_pixel_ratio",
        "c4/selected_box_area_mean",
        "c4/selected_box_area_ratio_mean",
        "c4/valid_labeled_pasted_pixel_count",
        "c4/valid_labeled_pixel_ratio",
        "c4/ignore_labeled_pasted_pixel_count",
        "c4/ignore_labeled_pixel_ratio",
        "cuda/max_memory_allocated",
        "cuda/max_memory_reserved",
    ])

    for kid in range(1, 12):
        op_name = AA_OPS.get(kid, f"op{kid}")
        cols.extend([
            f"aa/op_rate/{op_name}",
            f"aa/count/{op_name}",
            f"aa/pct_mean/{op_name}",
            f"aa/pct_max/{op_name}",
            f"aa/t_mean/{op_name}",
            f"aa/t_std/{op_name}",
        ])

    return cols


ITER_LOG_COLUMNS = build_iter_log_columns()


def _safe_path_token(value):
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return token.strip("_") or "unknown"

def _sanitize_wandb_id(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]

def _get_backbone_token(cfg):
    encoder_type = cfg.get("net", {}).get("encoder", {}).get("type", "")
    if encoder_type == "augseg.models.resnet.resnet101":
        return "r101"
    if encoder_type == "augseg.models.resnet.resnet50":
        return "r50"

    short_name = str(encoder_type).split(".")[-1]
    return _safe_path_token(short_name)


def _get_crop_token(cfg):
    crop_size = cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size", [])
    if not isinstance(crop_size, (list, tuple)) or len(crop_size) != 2:
        return "cunknown"

    h, w = int(crop_size[0]), int(crop_size[1])
    if h == w:
        return f"c{h}"
    return f"c{h}x{w}"


def _get_runtime_profile_token(cfg, world_size):
    per_gpu_batch = int(cfg.get("dataset", {}).get("train", {}).get("batch_size", 1))
    world_size = int(world_size)
    global_batch = per_gpu_batch * world_size
    return (
        f"{_get_backbone_token(cfg)}_{_get_crop_token(cfg)}_"
        f"bs{per_gpu_batch}x{world_size}_gbs{global_batch}"
    )


def _append_profile_to_hf_path(path_in_repo, profile):
    if not path_in_repo:
        return path_in_repo

    path_parts = [part for part in str(path_in_repo).split("/") if part]
    if profile in path_parts:
        return path_in_repo

    repo_dir, filename = posixpath.split(str(path_in_repo))
    if not filename:
        return posixpath.join(repo_dir, profile)
    return posixpath.join(repo_dir, profile, filename)


def _append_profile_to_snapshot_dir(snapshot_dir, profile):
    if not snapshot_dir:
        return snapshot_dir

    normalized_parts = [part for part in osp.normpath(str(snapshot_dir)).split(os.sep) if part]
    if profile in normalized_parts:
        return snapshot_dir

    parent, basename = osp.split(str(snapshot_dir).rstrip(os.sep))
    if basename == profile or basename.endswith(f"_{profile}"):
        return snapshot_dir
    profiled_basename = f"{basename}_{profile}" if basename else profile
    return osp.join(parent, profiled_basename) if parent else profiled_basename


def apply_runtime_profile_paths(cfg, world_size):
    profile = _get_runtime_profile_token(cfg, world_size)

    cfg.setdefault("saver", {})
    cfg.setdefault("hf", {})

    if bool(cfg["saver"].get("auto_profile_dir", False)):
        cfg["saver"]["snapshot_dir"] = _append_profile_to_snapshot_dir(
            cfg["saver"].get("snapshot_dir", ""),
            profile,
        )

    if bool(cfg["hf"].get("auto_profile_path", False)):
        cfg["hf"]["path_in_repo"] = _append_profile_to_hf_path(
            cfg["hf"].get("path_in_repo", ""),
            profile,
        )

    return profile


def make_default_iter_log_dict(
    sup_loss,
    uns_loss,
    pseudo_high_ratio,
    lr,
    epoch,
    step,
    global_iter,
    ar_triggered,
    ar_applied,
    ar_area_ratio_est,
    u_entropy_mean,
    u_maxprob_mean,
    u_maxprob_p10,
    u_maxprob_p50,
    u_maxprob_p90,
    u_pseudo_ratio_mean,
):
    log_dict = {
        # meta
        "meta/log_type": "train_iter",
        "meta/epoch": int(epoch),
        "meta/iter_in_epoch": int(step),
        "meta/global_iter": int(global_iter),

        # core training
        "iter/sup_loss": float(sup_loss),
        "iter/uns_loss": float(uns_loss),
        "iter/pseudo_high_ratio": float(pseudo_high_ratio),
        "iter/lr": float(lr),

        # validation default
        "val/model": "",
        "val/class_id": -1,
        "val/loss": float("nan"),
        "val/miou": float("nan"),
        "val/class_iou": float("nan"),
        "val/best_miou": float("nan"),

        # Ar
        "ar/triggered": int(ar_triggered),
        "ar/applied": int(ar_applied),
        "ar/area_ratio_est": float(ar_area_ratio_est),

        # U
        "u/entropy_mean": float(u_entropy_mean),
        "u/maxprob_mean": float(u_maxprob_mean),
        "u/maxprob_p10": float(u_maxprob_p10),
        "u/maxprob_p50": float(u_maxprob_p50),
        "u/maxprob_p90": float(u_maxprob_p90),
        "u/pseudo_ratio_mean": float(u_pseudo_ratio_mean),

        # Aa global default
        "aa/ops_per_image": 0.0,
    }

    for kid in range(1, 12):
        op_name = AA_OPS.get(kid, f"op{kid}")
        log_dict[f"aa/op_rate/{op_name}"] = 0.0
        log_dict[f"aa/count/{op_name}"] = 0
        log_dict[f"aa/pct_mean/{op_name}"] = -1.0
        log_dict[f"aa/pct_max/{op_name}"] = -1.0
        log_dict[f"aa/t_mean/{op_name}"] = -1.0
        log_dict[f"aa/t_std/{op_name}"] = -1.0

    return log_dict

def main(in_args):
    args = in_args
    if args.seed is not None:
        # print("set random seed to", args.seed)
        set_random_seed(args.seed, deterministic=True)
        # set_random_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    rank, word_size = setup_distributed(port=args.port)

    # ✅ đặt ở đây
    if rank != 0:
        os.environ["WANDB_MODE"] = "disabled"
    
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    ###########################
    # 1. output settings
    ###########################
    cfg["exp_path"] = osp.dirname(args.config)
    cfg.setdefault("saver", {})
    cfg["saver"].setdefault("auto_profile_dir", False)
    cfg.setdefault("hf", {})
    cfg["hf"].setdefault("enabled", False)
    cfg["hf"].setdefault("repo_type", "model")
    cfg["hf"].setdefault("auto_download", True)
    cfg["hf"].setdefault("auto_upload", True)
    cfg["hf"].setdefault("upload_every_epoch", True)
    cfg["hf"].setdefault("keep_only_latest", True)
    cfg["hf"].setdefault("bundle_name", "latest.tar.gz")
    cfg["hf"].setdefault("auto_profile_path", False)
    cfg["hf"].setdefault("squash_after_upload", False)

    runtime_profile = apply_runtime_profile_paths(cfg, word_size)

    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]

    cfg.setdefault("run", {})
    cfg["run"].setdefault("suite_id", "")
    cfg["run"].setdefault("name", os.path.basename(os.path.normpath(cfg["save_path"])) or "run")

    cfg.setdefault("wandb", {})
    cfg["wandb"].setdefault("enable", False)
    cfg["wandb"].setdefault("project", "augseg-voc662")
    cfg["wandb"].setdefault("entity", None)
    cfg["wandb"].setdefault("group", cfg["run"].get("suite_id") or None)
    cfg["wandb"].setdefault("name", cfg["run"].get("name") or None)
    cfg["wandb"].setdefault("resume", "allow")
    cfg["wandb"].setdefault("tags", [])
    cfg["run"].setdefault("log_every", cfg.get("wandb", {}).get("log_every", 50))

    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"].setdefault("auto_resume", True)
    cfg["checkpoint"].setdefault("save_latest", True)
    cfg["checkpoint"].setdefault("save_best", True)

    cfg.setdefault("boundary_mix", {})
    cfg["boundary_mix"].setdefault("enabled", False)
    cfg["boundary_mix"].setdefault("mode", "fixed")
    cfg["boundary_mix"].setdefault("kernel_size", 5)
    cfg["boundary_mix"].setdefault("gamma_in", 0.7)
    cfg["boundary_mix"].setdefault("gamma_out", 0.3)
    cfg["boundary_mix"].setdefault("use_confidence", True)
    cfg["boundary_mix"].setdefault("normalize_weight", True)
    cfg["boundary_mix"].setdefault("debug", False)
    cfg["boundary_mix"].setdefault("debug_first_batches", 3)
    cfg["boundary_mix"].setdefault("vis_debug", False)
    cfg["boundary_mix"].setdefault("vis_dir", "exp_boundary_debug_vis")
    cfg["boundary_mix"].setdefault("vis_max_batches", 2)

    cfg.setdefault("boundary_component", {})
    cfg["boundary_component"].setdefault("enabled", False)
    cfg["boundary_component"].setdefault("connectivity", 8)
    cfg["boundary_component"].setdefault("apply_to", "target_only")
    cfg["boundary_component"].setdefault("foreground_only", True)
    cfg["boundary_component"].setdefault("tau_visible_low", 0.2)
    cfg["boundary_component"].setdefault("tau_visible_high", 0.6)
    cfg["boundary_component"].setdefault("area_min", 64)
    cfg["boundary_component"].setdefault("area_max", 512)
    cfg["boundary_component"].setdefault("use_mean_confidence_in_q", True)
    cfg["boundary_component"].setdefault("base_pixel_weight", "one")
    cfg["boundary_component"].setdefault("weight_mode", "soft")
    cfg["boundary_component"].setdefault("force_q_one", False)
    cfg["boundary_component"].setdefault("eps", 1e-6)
    cfg["boundary_component"].setdefault("debug_log", False)

    cfg.setdefault("boundary_compatibility", {})
    cfg["boundary_compatibility"].setdefault("enabled", False)
    cfg["boundary_compatibility"].setdefault("semantic_metric", "js")
    cfg["boundary_compatibility"].setdefault("band_width", 3)
    cfg["boundary_compatibility"].setdefault("pair_mode", "radius")
    cfg["boundary_compatibility"].setdefault("pair_radius", 1)
    cfg["boundary_compatibility"].setdefault("topk", 5)
    cfg["boundary_compatibility"].setdefault("max_pairs_per_image", 2048)
    cfg["boundary_compatibility"].setdefault("tau_same", 0.8)
    cfg["boundary_compatibility"].setdefault("tau_diff", 0.3)
    cfg["boundary_compatibility"].setdefault("margin", 0.4)
    cfg["boundary_compatibility"].setdefault("lambda_bcr", 0.01)
    cfg["boundary_compatibility"].setdefault("use_confidence_gate", True)
    cfg["boundary_compatibility"].setdefault("use_component_gate", False)
    cfg["boundary_compatibility"].setdefault("component_gate_mode", "direct")
    cfg["boundary_compatibility"].setdefault("component_gate_alpha", 0.5)
    cfg["boundary_compatibility"].setdefault("component_gate_threshold", 0.1)
    cfg["boundary_compatibility"].setdefault("same_loss_mode", "hard_one")
    cfg["boundary_compatibility"].setdefault("relation_mode", "base_margin")
    cfg["boundary_compatibility"].setdefault("use_teacher_features", False)
    cfg["boundary_compatibility"].setdefault("teacher_feature_detach", True)
    cfg["boundary_compatibility"].setdefault("teacher_feature_source", "mixed")
    cfg["boundary_compatibility"].setdefault("affinity_target", "hard")
    cfg["boundary_compatibility"].setdefault("affinity_temperature", 0.2)
    cfg["boundary_compatibility"].setdefault("feature_layer", "decoder")
    cfg["boundary_compatibility"].setdefault("detach_teacher_distribution", True)
    cfg["boundary_compatibility"].setdefault("detach_gate", True)
    cfg["boundary_compatibility"].setdefault("eps", 1e-6)
    cfg["boundary_compatibility"].setdefault("debug_log", False)

    cfg.setdefault("saliency_cutmix", {})
    cfg["saliency_cutmix"].setdefault("enabled", False)
    cfg["saliency_cutmix"].setdefault("mode", "box")
    cfg["saliency_cutmix"].setdefault("saliency_mode", "grad")
    cfg["saliency_cutmix"].setdefault("saliency_model", "teacher")
    cfg["saliency_cutmix"].setdefault("saliency_loss", "supervised_ce")
    cfg["saliency_cutmix"].setdefault("detach_saliency", True)
    cfg["saliency_cutmix"].setdefault("num_candidates", 8)
    cfg["saliency_cutmix"].setdefault("selection", "softmax")
    cfg["saliency_cutmix"].setdefault("temperature", 0.2)
    cfg["saliency_cutmix"].setdefault("apply_to", "labeled_source_only")
    cfg["saliency_cutmix"].setdefault("fallback", "random_box")
    cfg["saliency_cutmix"].setdefault("direct_labeled_mix", False)
    cfg["saliency_cutmix"].setdefault("direct_paste_policy", "same_coordinate")
    cfg["saliency_cutmix"].setdefault("direct_confidence_gate", False)
    cfg["saliency_cutmix"].setdefault("paste_mode", "box")
    cfg["saliency_cutmix"].setdefault("component_source", "labeled_gt")
    cfg["saliency_cutmix"].setdefault("connectivity", 8)
    cfg["saliency_cutmix"].setdefault("foreground_only", True)
    cfg["saliency_cutmix"].setdefault("ignore_label", cfg.get("dataset", {}).get("ignore_label", 255))
    cfg["saliency_cutmix"].setdefault("min_component_area", 64)
    cfg["saliency_cutmix"].setdefault("max_component_area", 20000)
    cfg["saliency_cutmix"].setdefault("component_score", "mean_saliency_area")
    cfg["saliency_cutmix"].setdefault("box_expand_ratio", 1.2)
    cfg["saliency_cutmix"].setdefault("use_saliency_for_loss_weight", False)
    cfg["saliency_cutmix"].setdefault("use_saliency_conf_product", False)
    cfg["saliency_cutmix"].setdefault("debug_log", False)
    cfg["saliency_cutmix"].setdefault("eps", 1e-6)

    cfg.setdefault("u1_saliency_u2u", {})
    cfg["u1_saliency_u2u"].setdefault("enabled", False)
    cfg["u1_saliency_u2u"].setdefault("num_candidates", U1_NUM_CANDIDATES)
    cfg["u1_saliency_u2u"].setdefault("temperature", U1_TEMPERATURE)
    cfg["u1_saliency_u2u"].setdefault("rng_policy", U1_RNG_POLICY_VERSION)
    cfg["u1_saliency_u2u"].setdefault("debug_log", True)

    cfg.setdefault("u2_cross_view_saliency_u2u", {})
    cfg["u2_cross_view_saliency_u2u"].setdefault("enabled", False)
    cfg["u2_cross_view_saliency_u2u"].setdefault("num_candidates", U1_NUM_CANDIDATES)
    cfg["u2_cross_view_saliency_u2u"].setdefault("temperature", U1_TEMPERATURE)
    cfg["u2_cross_view_saliency_u2u"].setdefault("rng_policy", U1_RNG_POLICY_VERSION)
    cfg["u2_cross_view_saliency_u2u"].setdefault("debug_log", True)

    cfg.setdefault("u3_confidence_filtered_cross_view_saliency_u2u", {})
    cfg["u3_confidence_filtered_cross_view_saliency_u2u"].setdefault("enabled", False)
    cfg["u3_confidence_filtered_cross_view_saliency_u2u"].setdefault("num_candidates", U1_NUM_CANDIDATES)
    cfg["u3_confidence_filtered_cross_view_saliency_u2u"].setdefault("temperature", U1_TEMPERATURE)
    cfg["u3_confidence_filtered_cross_view_saliency_u2u"].setdefault("rng_policy", U1_RNG_POLICY_VERSION)
    cfg["u3_confidence_filtered_cross_view_saliency_u2u"].setdefault("debug_log", True)

    cfg.setdefault("u4_confidence_filtered_self_pseudo_saliency_u2u", {})
    cfg["u4_confidence_filtered_self_pseudo_saliency_u2u"].setdefault("enabled", False)
    cfg["u4_confidence_filtered_self_pseudo_saliency_u2u"].setdefault("num_candidates", U1_NUM_CANDIDATES)
    cfg["u4_confidence_filtered_self_pseudo_saliency_u2u"].setdefault("temperature", U1_TEMPERATURE)
    cfg["u4_confidence_filtered_self_pseudo_saliency_u2u"].setdefault("rng_policy", U1_RNG_POLICY_VERSION)
    cfg["u4_confidence_filtered_self_pseudo_saliency_u2u"].setdefault("debug_log", True)

    cfg.setdefault("csl", {})
    cfg["csl"].setdefault("enabled", False)
    cfg["csl"].setdefault("mode", "disabled")
    cfg["csl"].setdefault("reliability_mode", "entropy_margin")
    cfg["csl"].setdefault("output", "soft_weight")
    cfg["csl"].setdefault("detach_reliability", True)
    cfg["csl"].setdefault("use_csl_for_ce_weight", False)
    cfg["csl"].setdefault("use_csl_for_mix_confidence", False)
    cfg["csl"].setdefault("use_csl_for_cutmix", False)
    cfg["csl"].setdefault("random_mask_reliable", False)
    cfg["csl"].setdefault("perturb_input", False)
    cfg["csl"].setdefault("mask_prob", 0.3)
    cfg["csl"].setdefault("mask_mode", "zero")
    cfg["csl"].setdefault("block_size", 1)
    cfg["csl"].setdefault("cover_ratio", 1.0)
    cfg["csl"].setdefault("alpha", 8.0)
    cfg["csl"].setdefault("mask_labeled_pixels", False)
    cfg["csl"].setdefault("debug_log", False)
    cfg["csl"].setdefault("eps", 1e-6)

    cfg.setdefault("csl_cutmix", {})
    cfg["csl_cutmix"].setdefault("enabled", False)
    cfg["csl_cutmix"].setdefault("mode", "box")
    cfg["csl_cutmix"].setdefault("target_policy", "low_reliability")
    cfg["csl_cutmix"].setdefault("source_policy", "labeled_source")
    cfg["csl_cutmix"].setdefault("num_candidates", 8)
    cfg["csl_cutmix"].setdefault("selection", "softmax")
    cfg["csl_cutmix"].setdefault("temperature", 0.2)
    cfg["csl_cutmix"].setdefault("apply_to", "unlabeled_target_only")
    cfg["csl_cutmix"].setdefault("fallback", "random_box")
    cfg["csl_cutmix"].setdefault("debug_log", False)

    cfg.setdefault("fixed_size_csl_destination", {})
    cfg["fixed_size_csl_destination"].setdefault("enabled", False)
    cfg["fixed_size_csl_destination"].setdefault("num_candidates", 8)
    cfg["fixed_size_csl_destination"].setdefault("selection", "softmax")
    cfg["fixed_size_csl_destination"].setdefault("temperature", 0.2)
    cfg["fixed_size_csl_destination"].setdefault("target_policy", "low_reliability")

    if rank == 0:
        crop_size = cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size", [])
        per_gpu_batch = int(cfg.get("dataset", {}).get("train", {}).get("batch_size", 1))
        global_batch = per_gpu_batch * int(word_size)
        print(
            f"[profile] runtime_profile={runtime_profile} crop={crop_size} "
            f"per_gpu_batch={per_gpu_batch} world_size={word_size} global_batch={global_batch}",
            flush=True,
        )
        print(f"[profile] snapshot_dir={cfg['saver'].get('snapshot_dir')}", flush=True)
        print(f"[profile] save_path={cfg['save_path']}", flush=True)
        print(f"[profile] hf.path_in_repo={cfg.get('hf', {}).get('path_in_repo')}", flush=True)
    
    if not os.path.exists(cfg["log_path"]) and rank == 0:
        os.makedirs(cfg["log_path"])
    if not osp.exists(cfg["save_path"]) and rank == 0:
        os.makedirs(cfg["save_path"])

    if rank == 0:
        maybe_download_hf_bundle(
            cfg=cfg,
            save_path=cfg["save_path"],
            skip_if_ckpt_exists=True,
        )
        copy_config_to_save_path(args.config, cfg["save_path"])
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
    else:
        logger, curr_timestr = None, ""

    if rank == 0:
        suite_id = str(cfg.get("run", {}).get("suite_id", "") or "").strip()
        run_name = str(cfg.get("run", {}).get("name", "") or "").strip()

        if suite_id and run_name:
            run_id = _sanitize_wandb_id(f"{suite_id}__{run_name}")
            with open(os.path.join(cfg["save_path"], "run_id.txt"), "w", encoding="utf-8") as f:
                f.write(str(run_id) + "\n")
        else:
            run_id = get_or_create_run_id(cfg["save_path"], run_name or "run")
    else:
        run_id = ""

    run_id_list = [run_id]
    dist.broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]

    # Canonical run-state paths inside save_path.
    log_paths = default_log_paths(cfg["save_path"])

    if rank == 0:
        # Legacy epoch summary vẫn giữ nguyên ở log_path.
        csv_path = os.path.join(
            cfg["log_path"],
            f"seg_{run_id}_stat.csv",
        )

        # Detailed iteration diagnostics giờ dùng canonical path:
        # save_path/iter_metrics.csv
        train_iter_csv = str(log_paths["iter_csv"])
    else:
        csv_path = None
        train_iter_csv = None

    # tensorboard: hiện tại cứ để theo launch mới cho an toàn
    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        logger.info(f"[log] run_id = {run_id}")
        if flag_use_tb:
            tb_logger = SummaryWriter(osp.join(cfg["log_path"], "events_seg", curr_timestr))
        else:
            tb_logger = None
    else:
        tb_logger = None

    # ---------------- W&B (wandb) ----------------
    wandb_run = None
    use_wandb = cfg.get("wandb", {}).get("enable", False)

    if use_wandb:
        if rank != 0:
            # DDP: chỉ rank0 log, rank khác tắt để khỏi spam nhiều runs
            os.environ["WANDB_MODE"] = "disabled"
        else:
            import wandb

            suite_id = str(cfg.get("run", {}).get("suite_id", "") or "").strip()
            run_name = str(cfg.get("run", {}).get("name", "") or "").strip()
            wandb_cfg = cfg.get("wandb", {}) or {}

            wandb_group = wandb_cfg.get("group") or suite_id or None
            wandb_name = wandb_cfg.get("name") or run_name or run_id
            wandb_resume = wandb_cfg.get("resume", "allow")
            wandb_tags = _as_list(wandb_cfg.get("tags"))

            if suite_id and suite_id not in wandb_tags:
                wandb_tags.append(suite_id)

            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "augseg-voc662"),
                entity=wandb_cfg.get("entity", None),
                group=wandb_group,
                name=wandb_name,
                id=run_id,
                resume=wandb_resume,
                tags=wandb_tags,
                config=cfg,
                dir=cfg["log_path"],
                settings=wandb.Settings(start_method="thread"),
            )

    # make sure all folders and csv handler are correctly created on rank ==0.
    dist.barrier(device_ids=[local_rank])

    ###########################
    # 2. prepare model 1
    ###########################
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    modules_head = [model.decoder]
    if cfg["net"].get("aux_loss", False):
        modules_head.append(model.auxor)
    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)

    ###########################
    # 3. data
    ###########################
    sup_loss_fn = get_criterion(cfg)
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=args.seed)

    ##############################
    # 4. optimizer & scheduler
    ##############################
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )
    optimizer = get_optimizer(params_list, cfg_optim)

    ###########################
    # 5. prepare model more
    ###########################
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model -- freeze training
    model_teacher = ModelBuilder(cfg["net"])
    if cfg["net"].get("sync_bn", True):
        model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
    model_teacher.cuda(local_rank)
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    for p in model_teacher.parameters():
        p.requires_grad = False

    # initialize teacher model -- not neccesary if using warmup
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data
        
    ######################################
    # 6. resume
    ######################################
    last_epoch = 0
    best_prec = 0
    best_epoch = -1
    best_prec_stu = 0
    best_epoch_stu = -1

    # auto_resume > pretrain
    last_epoch, best_prec, ckpt_run_id, ckpt_path = load_checkpoint_if_available(
        save_path=cfg["save_path"],
        model=model,
        teacher_model=model_teacher,
        optimizer=optimizer,
        auto_resume=bool(cfg.get("checkpoint", {}).get("auto_resume", True)),
        filename="ckpt.pth",
        map_location="cuda:%d" % local_rank,
        strict=True,
    )

    if ckpt_run_id:
        run_id = ckpt_run_id

        if rank == 0:
            with open(
                os.path.join(cfg["save_path"], "run_id.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(run_id) + "\n")

            # Legacy epoch-summary filename vẫn phụ thuộc run_id.
            csv_path = os.path.join(
                cfg["log_path"],
                f"seg_{run_id}_stat.csv",
            )

            # Không đổi train_iter_csv ở đây.
            # Nó luôn là save_path/iter_metrics.csv.

    if rank == 0 and ckpt_path is not None:
        logger.info(
            "Resumed checkpoint from %s, start_epoch=%d, best_miou=%.4f"
            % (str(ckpt_path), last_epoch, best_prec)
        )

        # Các run cũ từng lưu detailed diagnostics ở:
        # log_path/train_iter_<run_id>.csv
        legacy_train_iter_csv = os.path.join(
            cfg["log_path"],
            f"train_iter_{run_id}.csv",
        )

        # Chỉ migrate khi iter_metrics.csv chưa tồn tại.
        if migrate_legacy_csv_if_missing(
            legacy_train_iter_csv,
            train_iter_csv,
        ):
            logger.info(
                "[log-migrate] copied legacy iter CSV %s -> %s",
                legacy_train_iter_csv,
                train_iter_csv,
            )

        # Nếu lần trước crash giữa epoch N thì checkpoint vẫn yêu cầu
        # resume từ epoch N. Xóa diagnostics dở dang của epoch N trở đi.
        trim_iter_csv_for_resume(
            train_iter_csv,
            last_epoch,
            logger,
        )

        # Legacy epoch summary.
        trim_csv_rows_by_epoch(
            csv_path,
            keep_epoch_lt=last_epoch,
        )

        # Canonical epoch-level history.
        trim_csv_rows_by_epoch(
            log_paths["epoch_csv"],
            keep_epoch_lt=last_epoch,
        )

    dist.barrier(device_ids=[local_rank])

    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer, start_epoch=last_epoch
    )

    ######################################
    # 7. training loop
    ######################################
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        res_loss_sup, res_loss_unsup = train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            cfg,
            wandb_run,   # <-- thêm dòng này
            train_iter_csv=train_iter_csv,
            runtime_seed=int(args.seed),
        )

        # Validation and store checkpoint
        if "cityscapes" in cfg["dataset"].get("type", "pascal"):
            if epoch % 10 == 0 or epoch > (cfg_trainer["epochs"]-50):
                if cfg_trainer.get("evaluate_student", True):
                    val_stu = validate_citys(model, val_loader, epoch, logger, cfg, sup_loss_fn)
                    prec_stu = val_stu["miou"]
                else:
                    val_stu = None
                    prec_stu = -1000.0

                val_tea = validate_citys(model_teacher, val_loader, epoch, logger, cfg, sup_loss_fn)
                prec_tea = val_tea["miou"]
                prec = prec_tea
            else:
                val_stu = None
                val_tea = None
                prec_stu = -1000.0
                prec_tea = -1000.0
                prec = prec_tea
        else:
            if cfg_trainer.get("evaluate_student", True):
                val_stu = validate(model, val_loader, epoch, logger, cfg, sup_loss_fn)
                prec_stu = val_stu["miou"]
            else:
                val_stu = None
                prec_stu = -1000.0

            val_tea = validate(model_teacher, val_loader, epoch, logger, cfg, sup_loss_fn)
            prec_tea = val_tea["miou"]
            prec = prec_tea

        if rank == 0:
            if prec_stu > best_prec_stu:
                best_prec_stu = prec_stu
                best_epoch_stu = epoch

            is_best = prec > best_prec
            if is_best:
                best_prec = prec
                best_epoch = epoch

            # save statistics
            tmp_results = {
                        'loss_lb': res_loss_sup,
                        'loss_ub': res_loss_unsup,
                        'miou_stu': prec_stu,
                        'miou_tea': prec_tea,
                        "best": best_prec,
                        "best-stu":best_prec_stu}
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch+1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=False, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')

            global_step = int((epoch + 1) * len(train_loader_sup) - 1)

            # ---------------- val_epoch: student ----------------
            if (train_iter_csv is not None) and (val_stu is not None):
                row = {c: np.nan for c in ITER_LOG_COLUMNS}
                row["meta/log_type"] = "val_epoch"
                row["meta/epoch"] = int(epoch)
                row["meta/iter_in_epoch"] = -1
                row["meta/global_iter"] = int(global_step)

                row["val/model"] = "student"
                row["val/class_id"] = -1
                row["val/loss"] = float(val_stu["loss"])
                row["val/miou"] = float(val_stu["miou"])
                row["val/class_iou"] = np.nan
                row["val/best_miou"] = float(best_prec_stu)

                df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                if os.path.exists(train_iter_csv):
                    df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(train_iter_csv, index=False)

            # ---------------- val_epoch: teacher ----------------
            if (train_iter_csv is not None) and (val_tea is not None):
                row = {c: np.nan for c in ITER_LOG_COLUMNS}
                row["meta/log_type"] = "val_epoch"
                row["meta/epoch"] = int(epoch)
                row["meta/iter_in_epoch"] = -1
                row["meta/global_iter"] = int(global_step)

                row["val/model"] = "teacher"
                row["val/class_id"] = -1
                row["val/loss"] = float(val_tea["loss"])
                row["val/miou"] = float(val_tea["miou"])
                row["val/class_iou"] = np.nan
                row["val/best_miou"] = float(best_prec)

                df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                if os.path.exists(train_iter_csv):
                    df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(train_iter_csv, index=False)

            # ---------------- val_class: student ----------------
            if (train_iter_csv is not None) and (val_stu is not None):
                for class_id, class_iou in enumerate(val_stu["iou_class"]):
                    row = {c: np.nan for c in ITER_LOG_COLUMNS}
                    row["meta/log_type"] = "val_class"
                    row["meta/epoch"] = int(epoch)
                    row["meta/iter_in_epoch"] = -1
                    row["meta/global_iter"] = int(global_step)

                    row["val/model"] = "student"
                    row["val/class_id"] = int(class_id)
                    row["val/miou"] = float(val_stu["miou"])
                    row["val/class_iou"] = float(class_iou)
                    row["val/best_miou"] = float(best_prec_stu)

                    df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                    if os.path.exists(train_iter_csv):
                        df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(train_iter_csv, index=False)

            # ---------------- val_class: teacher ----------------
            if (train_iter_csv is not None) and (val_tea is not None):
                for class_id, class_iou in enumerate(val_tea["iou_class"]):
                    row = {c: np.nan for c in ITER_LOG_COLUMNS}
                    row["meta/log_type"] = "val_class"
                    row["meta/epoch"] = int(epoch)
                    row["meta/iter_in_epoch"] = -1
                    row["meta/global_iter"] = int(global_step)

                    row["val/model"] = "teacher"
                    row["val/class_id"] = int(class_id)
                    row["val/loss"] = np.nan
                    row["val/miou"] = float(val_tea["miou"])
                    row["val/class_iou"] = float(class_iou)
                    row["val/best_miou"] = float(best_prec)

                    df_row = pd.DataFrame([row], columns=ITER_LOG_COLUMNS)
                    if os.path.exists(train_iter_csv):
                        df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(train_iter_csv, index=False)
            
            logger.info(" <<Test>> - Epoch: {}.  MIoU: {:.2f}/{:.2f}.  \033[34mBest-STU:{:.2f}/{}  \033[31mBest-EMA: {:.2f}/{}\033[0m".format(epoch, 
                prec_stu * 100, prec_tea * 100, best_prec_stu * 100, best_epoch_stu, best_prec * 100, best_epoch))
            if tb_logger is not None:
                tb_logger.add_scalar("mIoU val", prec, epoch)

            manifest = {
                "epoch": int(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_lb": float(res_loss_sup),
                "loss_ub": float(res_loss_unsup),
                "miou_stu": float(prec_stu),
                "miou_tea": float(prec_tea),
                "best_miou": float(best_prec),
                "best_miou_stu": float(best_prec_stu),
                "best_epoch": int(best_epoch),
                "best_epoch_stu": int(best_epoch_stu),
                "is_best": bool(is_best),
                "run_name": str(cfg.get("run", {}).get("name", "run")),
                "run_id": str(run_id),
                "config_path": args.config,
                "save_path": cfg["save_path"],
                "world_size": word_size,
                "git_commit": get_git_commit("."),
            }
            write_json(log_paths["manifest"], manifest)
            append_csv_row(log_paths["epoch_csv"], manifest)

            save_checkpoint(
                save_path=cfg["save_path"],
                model=model,
                teacher_model=model_teacher,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_miou=best_prec,
                run_id=run_id,
                cfg=cfg,
                args=args,
                is_best=is_best,
                save_latest=bool(cfg.get("checkpoint", {}).get("save_latest", True)),
                save_best=bool(cfg.get("checkpoint", {}).get("save_best", True)),
                extra={
                    "manifest": manifest,
                    "best_miou_stu": best_prec_stu,
                    "best_epoch": best_epoch,
                    "best_epoch_stu": best_epoch_stu,
                },
            )

            if bool(cfg.get("hf", {}).get("enabled", False)) and bool(cfg.get("hf", {}).get("upload_every_epoch", True)):
                maybe_upload_hf_bundle(
                    cfg=cfg,
                    save_path=cfg["save_path"],
                    config_path=args.config,
                    manifest=manifest,
                )

            if (rank == 0) and (wandb_run is not None):
                global_step = int((epoch + 1) * len(train_loader_sup) - 1)
                wandb_run.log({
                    "val/miou_stu": float(prec_stu),
                    "val/miou_tea": float(prec_tea),
                    "epoch/loss_sup": float(res_loss_sup),
                    "epoch/loss_unsup": float(res_loss_unsup),
                    "meta/epoch": int(epoch + 1),
                }, step=global_step)
            
    if (rank == 0) and (wandb_run is not None):
        wandb_run.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()





def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    cfg,
    wandb_run=None,   # <-- thêm
    train_iter_csv=None,
    runtime_seed=0,
):

    local_rank = torch.cuda.current_device()
    ema_decay_origin = cfg["net"]["ema_decay"]
    rank, world_size = dist.get_rank(), dist.get_world_size()
    log_every_value = cfg.get("run", {}).get("log_every", None)
    if log_every_value is None:
        log_every_value = cfg.get("wandb", {}).get("log_every", 50)
    log_every = max(1, int(log_every_value))
    flag_extra_weak = cfg["trainer"]["unsupervised"].get("flag_extra_weak", False)
    boundary_mix_cfg = cfg.get("boundary_mix", {})
    boundary_mix_enabled = bool(boundary_mix_cfg.get("enabled", False))
    boundary_mix_debug_enabled = bool(boundary_mix_cfg.get("debug", False))
    boundary_mix_vis_enabled = bool(boundary_mix_cfg.get("vis_debug", False))
    boundary_mix_vis_saved = 0
    boundary_component_cfg = cfg.get("boundary_component", {})
    boundary_component_enabled = bool(boundary_component_cfg.get("enabled", False))
    boundary_component_debug_enabled = bool(boundary_component_cfg.get("debug_log", False))
    boundary_compatibility_cfg = cfg.get("boundary_compatibility", {})
    boundary_compatibility_lambda = float(boundary_compatibility_cfg.get("lambda_bcr", 0.0))
    boundary_compatibility_enabled = (
        bool(boundary_compatibility_cfg.get("enabled", False))
        and boundary_compatibility_lambda != 0.0
    )
    boundary_compatibility_debug_enabled = bool(boundary_compatibility_cfg.get("debug_log", False))
    saliency_cutmix_cfg = cfg.get("saliency_cutmix", {})
    saliency_cutmix_enabled = bool(saliency_cutmix_cfg.get("enabled", False))
    saliency_cutmix_debug_enabled = bool(saliency_cutmix_cfg.get("debug_log", False))
    u1_cfg = cfg.get("u1_saliency_u2u", {})
    u1_enabled = bool(u1_cfg.get("enabled", False))
    u1_debug_enabled = bool(u1_cfg.get("debug_log", True))
    u2_cfg = cfg.get("u2_cross_view_saliency_u2u", {})
    u2_enabled = bool(u2_cfg.get("enabled", False))
    u2_debug_enabled = bool(u2_cfg.get("debug_log", True))
    u3_cfg = cfg.get("u3_confidence_filtered_cross_view_saliency_u2u", {})
    u3_enabled = bool(u3_cfg.get("enabled", False))
    u3_debug_enabled = bool(u3_cfg.get("debug_log", True))
    u4_cfg = cfg.get("u4_confidence_filtered_self_pseudo_saliency_u2u", {})
    u4_enabled = bool(u4_cfg.get("enabled", False))
    u4_debug_enabled = bool(u4_cfg.get("debug_log", True))
    s2_relocated_policy_active = (
        saliency_cutmix_cfg.get("direct_paste_policy")
        == "component_mask_random_valid_destination"
    )
    csl_cfg = cfg.get("csl", {})
    csl_enabled = bool(csl_cfg.get("enabled", False))
    csl_mode = csl_cfg.get("mode", "disabled")
    csl_reliability_mode = csl_cfg.get("reliability_mode", "entropy_margin")
    csl_official_enabled = csl_enabled and (
        csl_mode in ("official_reliability_replace_confidence", "official_reliable_mask_perturbation")
        or csl_reliability_mode == "official_pcos"
    )
    csl_use_ce_weight = csl_enabled and bool(csl_cfg.get("use_csl_for_ce_weight", False))
    csl_use_mix_confidence = csl_enabled and bool(csl_cfg.get("use_csl_for_mix_confidence", False))
    csl_use_cutmix = csl_enabled and bool(csl_cfg.get("use_csl_for_cutmix", False))
    csl_perturb_input = csl_official_enabled and bool(csl_cfg.get("perturb_input", False))
    csl_debug_enabled = bool(csl_cfg.get("debug_log", False))
    csl_cutmix_cfg = cfg.get("csl_cutmix", {})
    csl_cutmix_enabled = csl_use_cutmix and bool(csl_cutmix_cfg.get("enabled", False))
    fixed_size_csl_destination_cfg = cfg.get("fixed_size_csl_destination", {})
    fixed_size_csl_destination_enabled = bool(fixed_size_csl_destination_cfg.get("enabled", False))
    s1_adaptive_fixed_size_csl_enabled = (
        fixed_size_csl_destination_enabled
        and csl_enabled
        and csl_mode == "s1_adaptive_relocated_official_fixed_size_cutmix"
        and csl_reliability_mode == "official_pcos"
    )
    csl_official_guided_cutmix_enabled = (
        csl_cutmix_enabled
        and csl_mode == "official_guided_cutmix"
        and csl_reliability_mode == "official_pcos"
    )
    csl_c4_direct_labeled_enabled = (
        csl_cutmix_enabled
        and csl_mode == "official_direct_labeled_guided_cutmix_plus_ce_weight"
        and csl_reliability_mode == "official_pcos"
        and csl_use_mix_confidence
        and csl_use_ce_weight
    )
    if s2_relocated_policy_active:
        valid_s2_relocated = (
            saliency_cutmix_enabled
            and saliency_cutmix_cfg.get("mode") == "component_mask"
            and bool(saliency_cutmix_cfg.get("direct_labeled_mix", False))
            and not bool(saliency_cutmix_cfg.get("direct_confidence_gate", False))
            and not csl_enabled
            and not bool(csl_cutmix_cfg.get("enabled", False))
            and not boundary_mix_enabled
            and not boundary_component_enabled
            and not boundary_compatibility_enabled
            and not csl_use_ce_weight
        )
        if not valid_s2_relocated:
            raise ValueError(
                "component_mask_random_valid_destination requires isolated S2 component-mask direct mixing "
                "with confidence gate, CSL, CSL CutMix, BoundaryMix, component weighting, and BCR disabled"
            )
    if csl_mode == "official_guided_cutmix" and not csl_official_guided_cutmix_enabled:
        raise ValueError(
            "csl.mode='official_guided_cutmix' requires reliability_mode='official_pcos', "
            "use_csl_for_cutmix=true, and csl_cutmix.enabled=true"
        )
    if csl_mode == "official_direct_labeled_guided_cutmix_plus_ce_weight" and not csl_c4_direct_labeled_enabled:
        raise ValueError(
            "C4 requires reliability_mode='official_pcos', use_csl_for_cutmix=true, "
            "use_csl_for_mix_confidence=true, use_csl_for_ce_weight=true, and csl_cutmix.enabled=true"
        )
    if (
        fixed_size_csl_destination_enabled
        or saliency_cutmix_cfg.get("direct_paste_policy") == "csl_official_fixed_size_target"
    ) and csl_mode != "s1_adaptive_relocated_official_fixed_size_cutmix":
        raise ValueError(
            "fixed-size official CSL destination selection is reserved for "
            "csl.mode='s1_adaptive_relocated_official_fixed_size_cutmix'"
        )
    if csl_mode == "s1_adaptive_relocated_official_fixed_size_cutmix":
        valid_s1_fixed_size_csl = (
            s1_adaptive_fixed_size_csl_enabled
            and saliency_cutmix_enabled
            and saliency_cutmix_cfg.get("mode", "box") == "box"
            and bool(saliency_cutmix_cfg.get("direct_labeled_mix", False))
            and saliency_cutmix_cfg.get("direct_paste_policy") == "csl_official_fixed_size_target"
            and bool(saliency_cutmix_cfg.get("direct_confidence_gate", False))
            and not csl_use_mix_confidence
            and not csl_use_ce_weight
            and not csl_use_cutmix
            and not csl_perturb_input
            and not boundary_mix_enabled
            and not boundary_component_enabled
            and not boundary_compatibility_enabled
            and int(fixed_size_csl_destination_cfg.get("num_candidates", 8)) == 8
            and fixed_size_csl_destination_cfg.get("selection", "softmax") == "softmax"
            and float(fixed_size_csl_destination_cfg.get("temperature", 0.2)) == 0.2
            and fixed_size_csl_destination_cfg.get("target_policy", "low_reliability") == "low_reliability"
        )
        if not valid_s1_fixed_size_csl:
            raise ValueError(
                "s1_adaptive_relocated_official_fixed_size_cutmix requires S1 box direct mixing, "
                "the adaptive confidence gate, fixed-size CSL K=8 low-reliability softmax selection "
                "at temperature 0.2, official PCOS reliability, and all CSL confidence/loss, "
                "perturbation, BoundaryMix, component, BCR, C3, and C4 paths disabled"
            )
    csl_cutmix_debug_enabled = bool(csl_cutmix_cfg.get("debug_log", False))
    if sum(bool(enabled) for enabled in (u1_enabled, u2_enabled, u3_enabled, u4_enabled)) > 1:
        raise ValueError("U1, U2, U3, and U4 cannot be enabled simultaneously")
    if u1_enabled or u2_enabled or u3_enabled or u4_enabled:
        method_name = "U1" if u1_enabled else ("U2" if u2_enabled else ("U3" if u3_enabled else "U4"))
        method_cfg = u1_cfg if u1_enabled else (u2_cfg if u2_enabled else (u3_cfg if u3_enabled else u4_cfg))
        crop_size = cfg.get("dataset", {}).get("train", {}).get("crop", {}).get("size")
        incompatible = {
            "saliency_cutmix": saliency_cutmix_enabled,
            "fixed_size_csl_destination": fixed_size_csl_destination_enabled,
            "csl": csl_enabled,
            "csl_cutmix": csl_cutmix_enabled,
            "boundary_mix": boundary_mix_enabled,
            "boundary_component": boundary_component_enabled,
            "boundary_compatibility": boundary_compatibility_enabled,
        }
        active_incompatible = [name for name, active in incompatible.items() if active]
        if active_incompatible:
            raise ValueError("%s is isolated and incompatible with: %s" % (method_name, ", ".join(active_incompatible)))
        if list(crop_size or []) != [321, 321]:
            raise ValueError("%s Phase B supports crop [321,321] only" % method_name)
        if int(method_cfg.get("num_candidates")) != U1_NUM_CANDIDATES:
            raise ValueError("%s requires exactly eight candidates" % method_name)
        if float(method_cfg.get("temperature")) != U1_TEMPERATURE:
            raise ValueError("%s requires temperature 0.2" % method_name)
        if method_cfg.get("rng_policy") != U1_RNG_POLICY_VERSION:
            raise ValueError("%s requires rng_policy=u1_rng_policy_v1" % method_name)
    model.train()
    
    # data loader
    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(loader_u), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, mixmatch!"

    # metric indicators
    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)
    meter_high_pseudo_ratio = AverageMeter(20)
    
    # print freq 8 times for a epoch
    print_freq = len(loader_u) // 8 # 8 for semi 4 for sup
    print_freq_lst = [i * print_freq for i in range(1,8)]
    print_freq_lst.append(len(loader_u) -1)

    # start iterations
    model.train()
    model_teacher.eval()
    u1_diagnostics_accumulator = U1Diagnostics()
    for step in range(len(loader_l)):
        batch_start = time.time()
        # --------- init per-iter stats for wandb (tránh dính iter trước) ---------
        ar_triggered = 0
        ar_applied = 0
        ar_area_ratio_est = float("nan")

        u_entropy_mean = float("nan")
        u_maxprob_mean = float("nan")
        u_maxprob_p10 = float("nan")
        u_maxprob_p50 = float("nan")
        u_maxprob_p90 = float("nan")
        u_pseudo_ratio_mean = float("nan")
        bcr_loss = None
        bcr_stats = None
        component_stats = None
        component_loss_value = float("nan")
        saliency_stats = None
        csl_stats = None
        csl_mask_stats = None
        csl_cutmix_stats = None
        c4_stats = None
        csl_reliability_u = None
        csl_weight_u = None
        csl_reliable_mask_u = None
        saliency_selector_exception_batch = 0
        relocation_diagnostics = None
        destination_kwargs = {}
        if s2_relocated_policy_active:
            relocation_diagnostics = {
                "relocation_draw_count": 0,
                "relocation_zero_count": 0,
                "relocation_nonzero_count": 0,
                "relocation_success_count": 0,
                "relocation_zero_only_count": 0,
                "relocation_random_zero_count": 0,
                "relocation_empty_mask_count": 0,
                "relocation_valid_translation_sum": 0,
                "relocation_expected_zero_sum": 0.0,
                "relocation_displacement_magnitude_sum": 0.0,
                "relocation_source_destination_iou_sum": 0.0,
            }
            destination_kwargs = {
                "destination_context": {
                    "base_seed": runtime_seed,
                    "rank": rank,
                    "epoch": epoch,
                    "iteration": epoch * len(loader_l) + step,
                },
                "destination_diagnostics": relocation_diagnostics,
            }

        i_iter = epoch * len(loader_l) + step # total iters till now
        # log schedule (đúng y hệt block W&B phía dưới)
        do_log_now = (rank == 0) and (
            (i_iter % log_every == 0) or (step == len(loader_l) - 1)
        )
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step() # lr is updated at the iteration level

        name, image_l, label_l = next(loader_l_iter)
        image_l = image_l.cuda(local_rank, non_blocking=True)
        label_l = label_l.cuda(local_rank, non_blocking=True)

        num_classes = cfg["net"]["num_classes"]
        ignore = cfg["dataset"]["ignore_label"]

        bad = (label_l != ignore) & ((label_l < 0) | (label_l >= num_classes))
        if bad.any() and rank == 0:
            print("❌ BAD sample id:", name)
            print("unique bad values:", torch.unique(label_l[bad])[:50].tolist())
            print("label min/max:", label_l.min().item(), label_l.max().item())
            raise RuntimeError("Out-of-range labels in supervised mask")

        batch_u = next(loader_u_iter)

        # batch_u có thể là:
        # - cũ: (idx, weak, strong, label)
        # - mới: (idx, weak, strong, label, k_ids, t_vals)
        if len(batch_u) == 4:
            _, image_u_weak, image_u_aug, label_u_ignore = batch_u
            k_ids, t_vals = None, None
        else:
            _, image_u_weak, image_u_aug, label_u_ignore, k_ids, t_vals = batch_u

        image_u_weak = image_u_weak.cuda(local_rank, non_blocking=True)
        image_u_aug  = image_u_aug.cuda(local_rank, non_blocking=True)
        label_u_ignore = label_u_ignore.cuda(local_rank, non_blocking=True)
    
        
        # start the training
        if epoch < cfg["trainer"].get("sup_only_epoch", 0):
            # forward
            pred, aux = model(image_l)
            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                sup_loss = sup_loss_fn([pred, aux], label_l)
                del aux
            else:
                sup_loss = sup_loss_fn(pred, label_l)
                del pred

            # no unlabeled data during the warmup period
            unsup_loss = torch.tensor(0.0).cuda()
            pseduo_high_ratio = torch.tensor(0.0).cuda()

        else:
            # 1. generate pseudo labels
            p_threshold = cfg["trainer"]["unsupervised"].get("threshold", 0.95)
            teacher_probs_u_aug = None
            with torch.no_grad():
                model_teacher.eval()
                pred_u, _ = model_teacher(image_u_weak.detach())
                pred_u = F.softmax(pred_u, dim=1)
                if boundary_compatibility_enabled:
                    teacher_probs_u_aug = pred_u.detach().clone()
                # obtain pseudos
                logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)
                if csl_enabled:
                    if csl_official_enabled:
                        official_ignore = label_u_ignore if label_u_ignore.shape == label_u_aug.shape else None
                        csl_selection = compute_csl_official_selection(
                            pred_u,
                            ignore_mask=official_ignore,
                            alpha=float(csl_cfg.get("alpha", 8.0)),
                            eps=float(csl_cfg.get("eps", 1e-8)),
                        )
                        csl_reliability_u = csl_selection["weight"]
                        csl_weight_u = csl_selection["weight"]
                        csl_reliable_mask_u = csl_selection["reliable_mask"]
                        csl_stats = csl_selection.get("stats", {})
                        if csl_use_mix_confidence:
                            confidence = csl_selection["sample_reliability"].cpu().numpy().tolist()
                    else:
                        csl_reliability_u, csl_stats = compute_csl_reliability(
                            pred_u,
                            cfg=csl_cfg,
                        )
                        csl_weight_u = csl_reliability_u
                        if csl_use_ce_weight and bool(csl_cfg.get("random_mask_reliable", False)):
                            csl_weight_u, csl_mask_stats = apply_csl_random_reliable_mask(
                                csl_reliability_u,
                                mask_prob=float(csl_cfg.get("mask_prob", 0.3)),
                                training=model.training,
                            )
                
                # obtain confidence
                entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
                entropy /= np.log(cfg["net"]["num_classes"])
                if not (csl_official_enabled and csl_use_mix_confidence):
                    confidence = 1.0 - entropy
                    confidence = confidence * logits_u_aug
                    confidence = confidence.mean(dim=[1,2])  # 1*C
                    confidence = confidence.cpu().numpy().tolist()
                # effect stats: entropy mean (teacher on weak)
                u_entropy_mean = float(entropy.detach().mean().item())

                # confidence = logits_u_aug.ge(p_threshold).float().mean(dim=[1,2]).cpu().numpy().tolist()
                del pred_u
            model.train()
            
            # 2. apply cutmix (Ar) + log flags
            use_cutmix = cfg["trainer"]["unsupervised"].get("use_cutmix", False)
            trigger_prob = cfg["trainer"]["unsupervised"].get("use_cutmix_trigger_prob", 1.0)

            mix_branch, ar_triggered, ar_applied = select_unlabeled_mix_branch(
                u1_enabled, use_cutmix, trigger_prob, u2_enabled=u2_enabled,
                u3_enabled=u3_enabled, u4_enabled=u4_enabled
            )

            mix_source_mask = None
            target_component_label = None
            target_component_confidence = None
            if ar_applied:
                # estimate area ratio CHỈ để log -> chỉ tính khi do_log_now
                label_u_before = None
                if do_log_now:
                    label_u_before = label_u_aug.clone()                    

                saliency_labeled_boxes = None
                saliency_labeled_masks = None
                csl_target_boxes = None
                if is_u_saliency_mix_branch(mix_branch):
                    saliency_probe_rgb=image_u_aug if mix_branch == "u2" else None
                    if mix_branch == "u3":
                        saliency_probe_rgb = image_u_aug
                    confidence_filtered_probe = mix_branch == "u3"
                    confidence_threshold=p_threshold if mix_branch == "u3" else None
                    ignore_label=ignore if mix_branch == "u3" else None
                    if mix_branch == "u4":
                        confidence_filtered_probe = True
                        confidence_threshold = p_threshold
                        ignore_label = ignore
                    image_u_aug, label_u_aug, logits_u_aug, u1_step_diagnostics = apply_u1_saliency_u2u(
                        teacher=model_teacher,
                        weak_rgb=image_u_weak,
                        strong_rgb=image_u_aug,
                        hard_pseudo=label_u_aug.detach(),
                        confidence=logits_u_aug.detach(),
                        base_seed=runtime_seed,
                        rank=rank,
                        epoch=epoch,
                        step=step,
                        absolute_global_iteration=i_iter,
                        saliency_probe_rgb=saliency_probe_rgb,
                        confidence_filtered_probe=confidence_filtered_probe,
                        confidence_threshold=confidence_threshold,
                        ignore_label=ignore_label,
                    )
                    u1_diagnostics_accumulator.add_(u1_step_diagnostics)
                if saliency_cutmix_enabled:
                    try:
                        saliency_mode = saliency_cutmix_cfg.get("mode", "box")
                        if saliency_mode == "box":
                            saliency_labeled_boxes, saliency_stats = get_saliency_guided_boxes(
                                model_teacher,
                                image_l,
                                label_l,
                                _rand_bbox,
                                num_candidates=int(saliency_cutmix_cfg.get("num_candidates", 8)),
                                temperature=float(saliency_cutmix_cfg.get("temperature", 0.2)),
                                ignore_index=int(saliency_cutmix_cfg.get("ignore_label", cfg["dataset"].get("ignore_label", 255))),
                                eps=float(saliency_cutmix_cfg.get("eps", 1e-6)),
                                lam_sampler=lambda: np.random.beta(8, 2),
                            )
                        elif saliency_mode == "component_box":
                            if saliency_cutmix_cfg.get("component_source", "labeled_gt") != "labeled_gt":
                                raise ValueError("saliency_cutmix.component_source currently supports 'labeled_gt' only")
                            saliency_labeled_boxes, saliency_stats = get_saliency_component_guided_boxes(
                                model_teacher,
                                image_l,
                                label_l,
                                _rand_bbox,
                                temperature=float(saliency_cutmix_cfg.get("temperature", 0.2)),
                                ignore_index=int(saliency_cutmix_cfg.get("ignore_label", cfg["dataset"].get("ignore_label", 255))),
                                eps=float(saliency_cutmix_cfg.get("eps", 1e-6)),
                                lam_sampler=lambda: np.random.beta(8, 2),
                                connectivity=int(saliency_cutmix_cfg.get("connectivity", 8)),
                                foreground_only=bool(saliency_cutmix_cfg.get("foreground_only", True)),
                                min_component_area=int(saliency_cutmix_cfg.get("min_component_area", 64)),
                                max_component_area=int(saliency_cutmix_cfg.get("max_component_area", 20000)),
                                box_expand_ratio=float(saliency_cutmix_cfg.get("box_expand_ratio", 1.2)),
                            )
                        elif saliency_mode == "component_mask":
                            if saliency_cutmix_cfg.get("component_source", "labeled_gt") != "labeled_gt":
                                raise ValueError("saliency_cutmix.component_source currently supports 'labeled_gt' only")
                            saliency_labeled_masks, saliency_stats = get_saliency_component_guided_masks(
                                model_teacher,
                                image_l,
                                label_l,
                                _rand_bbox,
                                temperature=float(saliency_cutmix_cfg.get("temperature", 0.2)),
                                ignore_index=int(saliency_cutmix_cfg.get("ignore_label", cfg["dataset"].get("ignore_label", 255))),
                                eps=float(saliency_cutmix_cfg.get("eps", 1e-6)),
                                lam_sampler=lambda: np.random.beta(8, 2),
                                connectivity=int(saliency_cutmix_cfg.get("connectivity", 8)),
                                foreground_only=bool(saliency_cutmix_cfg.get("foreground_only", True)),
                                min_component_area=int(saliency_cutmix_cfg.get("min_component_area", 64)),
                                max_component_area=int(saliency_cutmix_cfg.get("max_component_area", 20000)),
                            )
                        else:
                            raise ValueError("Unsupported saliency_cutmix.mode: %s" % saliency_mode)
                    except Exception as exc:
                        saliency_selector_exception_batch = 1
                        saliency_labeled_boxes = None
                        saliency_labeled_masks = None
                        saliency_stats = {
                            "saliency/fallback_ratio": 1.0,
                            "saliency/source_is_labeled_ratio": 1.0,
                        }
                        if rank == 0 and saliency_cutmix_debug_enabled:
                            logger.info(
                                "[saliency_cutmix] fallback=random_box epoch=%d step=%d global_iter=%d error=%s"
                                % (epoch, step, i_iter, str(exc))
                            )
                if csl_cutmix_enabled:
                    try:
                        if csl_official_guided_cutmix_enabled and csl_reliability_u is None:
                            raise ValueError("official guided CutMix requires official PCOS reliability map")
                        if csl_c4_direct_labeled_enabled:
                            csl_target_boxes, csl_cutmix_stats = get_csl_guided_boxes(
                                csl_reliability_u,
                                _rand_bbox,
                                num_candidates=int(csl_cutmix_cfg.get("num_candidates", 8)),
                                temperature=float(csl_cutmix_cfg.get("temperature", 0.2)),
                                policy=csl_cutmix_cfg.get("target_policy", "low_reliability"),
                                strict=True,
                            )
                        else:
                            csl_target_boxes, csl_cutmix_stats = get_csl_guided_boxes(
                                csl_reliability_u,
                                _rand_bbox,
                                num_candidates=int(csl_cutmix_cfg.get("num_candidates", 8)),
                                temperature=float(csl_cutmix_cfg.get("temperature", 0.2)),
                                policy=csl_cutmix_cfg.get("target_policy", "low_reliability"),
                            )
                    except Exception as exc:
                        if csl_c4_direct_labeled_enabled:
                            raise RuntimeError(
                                f"C4 strict target-box selection failed at epoch={epoch}, "
                                f"step={step}, global_iter={i_iter}"
                            ) from exc
                        csl_target_boxes = None
                        csl_cutmix_stats = {"csl_cutmix/fallback_ratio": 1.0}
                        if rank == 0 and csl_cutmix_debug_enabled:
                            logger.info(
                                "[csl_cutmix] fallback=random_box epoch=%d step=%d global_iter=%d error=%s"
                                % (epoch, step, i_iter, str(exc))
                            )

                if is_u_saliency_mix_branch(mix_branch):
                    # U1–U4 already produced aligned strong RGB/pseudo/confidence tensors.
                    pass
                elif (
                    boundary_mix_enabled
                    or boundary_component_enabled
                    or boundary_compatibility_enabled
                    or saliency_cutmix_enabled
                    or csl_use_ce_weight
                    or csl_cutmix_enabled
                ):
                    if csl_c4_direct_labeled_enabled:
                        # C4 uses [row1, col1, row2, col2] coordinates end-to-end.
                        c4_target_boxes = csl_target_boxes
                        (
                            image_u_aug,
                            label_u_aug,
                            logits_u_aug,
                            mix_source_mask,
                            csl_weight_u,
                        ) = cut_mix_label_adaptive_c4_direct_labeled(
                            image_u_aug,
                            label_u_aug,
                            logits_u_aug,
                            image_l,
                            label_l,
                            confidence,
                            c4_target_boxes,
                            unlabeled_weight=csl_weight_u,
                            ignore_index=ignore,
                        )
                        c4_stats = compute_c4_direct_mix_stats(
                            mix_source_mask,
                            label_u_aug,
                            c4_target_boxes,
                            ignore_index=ignore,
                        )
                        if (
                            rank == 0
                            and csl_cutmix_debug_enabled
                            and (
                                step < int(csl_cutmix_cfg.get("debug_first_batches", 3))
                                or do_log_now
                            )
                        ):
                            logger.info(
                                "[c4] epoch=%d step=%d global_iter=%d gate=%d/%d "
                                "mixed=%d/%d pasted=%.6f valid=%.6f ignore=%.6f"
                                % (
                                    epoch,
                                    step,
                                    i_iter,
                                    c4_stats["c4/gate_pass_count"],
                                    c4_stats["c4/gate_attempted_count"],
                                    c4_stats["c4/mixed_sample_count"],
                                    c4_stats["c4/gate_attempted_count"],
                                    c4_stats["c4/pasted_pixel_ratio"],
                                    c4_stats["c4/valid_labeled_pixel_ratio"],
                                    c4_stats["c4/ignore_labeled_pixel_ratio"],
                                )
                            )
                    elif boundary_component_enabled:
                        mixed_result = cut_mix_label_adaptive_with_mask(
                            image_u_aug, label_u_aug, logits_u_aug,
                            image_l, label_l, confidence,
                            return_target_metadata=True,
                            unlabeled_probs=teacher_probs_u_aug,
                            unlabeled_weight=csl_weight_u if csl_use_ce_weight else None,
                            labeled_boxes=saliency_labeled_boxes,
                            labeled_masks=saliency_labeled_masks,
                            direct_labeled_mix=bool(saliency_cutmix_cfg.get("direct_labeled_mix", False)),
                            direct_paste_policy=saliency_cutmix_cfg.get("direct_paste_policy", "same_coordinate"),
                            direct_confidence_gate=bool(saliency_cutmix_cfg.get("direct_confidence_gate", False)),
                            target_boxes=csl_target_boxes,
                            csl_destination_reliability=csl_reliability_u if s1_adaptive_fixed_size_csl_enabled else None,
                            csl_destination_num_candidates=int(fixed_size_csl_destination_cfg.get("num_candidates", 8)),
                            csl_destination_temperature=float(fixed_size_csl_destination_cfg.get("temperature", 0.2)),
                            csl_destination_policy=fixed_size_csl_destination_cfg.get("target_policy", "low_reliability"),
                            **destination_kwargs,
                        )
                        if boundary_compatibility_enabled and csl_use_ce_weight:
                            (
                                image_u_aug,
                                label_u_aug,
                                logits_u_aug,
                                mix_source_mask,
                                target_component_label,
                                target_component_confidence,
                                teacher_probs_u_aug,
                                csl_weight_u,
                            ) = mixed_result
                        elif boundary_compatibility_enabled:
                            (
                                image_u_aug,
                                label_u_aug,
                                logits_u_aug,
                                mix_source_mask,
                                target_component_label,
                                target_component_confidence,
                                teacher_probs_u_aug,
                            ) = mixed_result
                        elif csl_use_ce_weight:
                            (
                                image_u_aug,
                                label_u_aug,
                                logits_u_aug,
                                mix_source_mask,
                                target_component_label,
                                target_component_confidence,
                                csl_weight_u,
                            ) = mixed_result
                        else:
                            (
                                image_u_aug,
                                label_u_aug,
                                logits_u_aug,
                                mix_source_mask,
                                target_component_label,
                                target_component_confidence,
                            ) = mixed_result
                    else:
                        mixed_result = cut_mix_label_adaptive_with_mask(
                            image_u_aug, label_u_aug, logits_u_aug,
                            image_l, label_l, confidence,
                            unlabeled_probs=teacher_probs_u_aug,
                            unlabeled_weight=csl_weight_u if csl_use_ce_weight else None,
                            labeled_boxes=saliency_labeled_boxes,
                            labeled_masks=saliency_labeled_masks,
                            direct_labeled_mix=bool(saliency_cutmix_cfg.get("direct_labeled_mix", False)),
                            direct_paste_policy=saliency_cutmix_cfg.get("direct_paste_policy", "same_coordinate"),
                            direct_confidence_gate=bool(saliency_cutmix_cfg.get("direct_confidence_gate", False)),
                            target_boxes=csl_target_boxes,
                            csl_destination_reliability=csl_reliability_u if s1_adaptive_fixed_size_csl_enabled else None,
                            csl_destination_num_candidates=int(fixed_size_csl_destination_cfg.get("num_candidates", 8)),
                            csl_destination_temperature=float(fixed_size_csl_destination_cfg.get("temperature", 0.2)),
                            csl_destination_policy=fixed_size_csl_destination_cfg.get("target_policy", "low_reliability"),
                            **destination_kwargs,
                        )
                        if boundary_compatibility_enabled and csl_use_ce_weight:
                            image_u_aug, label_u_aug, logits_u_aug, mix_source_mask, teacher_probs_u_aug, csl_weight_u = mixed_result
                        elif boundary_compatibility_enabled:
                            image_u_aug, label_u_aug, logits_u_aug, mix_source_mask, teacher_probs_u_aug = mixed_result
                        elif csl_use_ce_weight:
                            image_u_aug, label_u_aug, logits_u_aug, mix_source_mask, csl_weight_u = mixed_result
                        else:
                            image_u_aug, label_u_aug, logits_u_aug, mix_source_mask = mixed_result
                elif cfg["trainer"]["unsupervised"].get("use_cutmix_adaptive", False):                                    
                    image_u_aug, label_u_aug, logits_u_aug = cut_mix_label_adaptive(
                        image_u_aug, label_u_aug, logits_u_aug,
                        image_l, label_l, confidence
                    )
                else:
                    image_u_aug, label_u_aug, logits_u_aug = cut_mix_label_adaptive(
                        image_u_aug, label_u_aug, logits_u_aug,
                        image_l, label_l, confidence
                    )

                u1_log_boundary = (i_iter % log_every == 0) or (step == len(loader_l) - 1)
                if (u1_enabled or u2_enabled or u3_enabled or u4_enabled) and u1_log_boundary:
                    u1_aggregated = aggregate_u1_diagnostics(
                        u1_diagnostics_accumulator,
                        device=image_u_aug.device,
                    )
                    if u4_enabled:
                        u1_aggregated = {
                            (key.replace("u3/", "u4/", 1) if key.startswith("u3/") else key): value
                            for key, value in u1_aggregated.items()
                        }
                    method_debug_enabled = (
                        u1_debug_enabled if u1_enabled else (
                            u2_debug_enabled if u2_enabled else (u3_debug_enabled if u3_enabled else u4_debug_enabled)
                        )
                    )
                    if rank == 0 and method_debug_enabled:
                        logger.info(
                            "[%s] epoch=%d step=%d global_iter=%d %s",
                            "u1_saliency_u2u" if u1_enabled else (
                                "u2_cross_view_saliency_u2u" if u2_enabled
                                else (
                                    "u3_confidence_filtered_cross_view_saliency_u2u" if u3_enabled
                                    else "u4_confidence_filtered_self_pseudo_saliency_u2u"
                                )
                            ),
                            epoch,
                            step,
                            i_iter,
                            " ".join("%s=%s" % (key, value) for key, value in sorted(u1_aggregated.items())),
                        )
                    u1_diagnostics_accumulator.reset_()

                if label_u_before is not None:
                    ar_area_ratio_est = (label_u_aug != label_u_before).float().mean().item()
                    del label_u_before

                if rank == 0 and saliency_cutmix_enabled and saliency_cutmix_debug_enabled and saliency_stats is not None and do_log_now:
                    logger.info(
                        "[saliency_cutmix] epoch=%d step=%d global_iter=%d "
                        "selected=%.6f candidate_mean=%.6f candidate_min=%.6f candidate_max=%.6f "
                        "candidate_std=%.6f prob_selected=%.6f prob_max=%.6f entropy=%.6f fallback=%.3f "
                        "components=%.3f valid_components=%.3f selected_class=%.3f selected_area=%.3f "
                        "selected_component_score=%.6f component_score_mean=%.6f component_score_max=%.6f component_box_area=%.3f"
                        % (
                            epoch,
                            step,
                            i_iter,
                            saliency_stats.get("saliency/score_selected", float("nan")),
                            saliency_stats.get("saliency/score_candidate_mean", float("nan")),
                            saliency_stats.get("saliency/score_candidate_min", float("nan")),
                            saliency_stats.get("saliency/score_candidate_max", float("nan")),
                            saliency_stats.get("saliency/score_candidate_std", float("nan")),
                            saliency_stats.get("saliency/prob_selected", float("nan")),
                            saliency_stats.get("saliency/prob_max", float("nan")),
                            saliency_stats.get("saliency/selection_entropy", float("nan")),
                            saliency_stats.get("saliency/fallback_ratio", float("nan")),
                            saliency_stats.get("saliency/num_components", float("nan")),
                            saliency_stats.get("saliency/num_valid_components", float("nan")),
                            saliency_stats.get("saliency/selected_component_class", float("nan")),
                            saliency_stats.get("saliency/selected_component_area", float("nan")),
                            saliency_stats.get("saliency/selected_component_score", float("nan")),
                            saliency_stats.get("saliency/component_score_mean", float("nan")),
                            saliency_stats.get("saliency/component_score_max", float("nan")),
                            saliency_stats.get("saliency/component_box_area", float("nan")),
                        )
                    )

            debug_mixed_image = None
            if (
                rank == 0
                and boundary_mix_enabled
                and boundary_mix_vis_enabled
                and mix_source_mask is not None
                and boundary_mix_vis_saved < int(boundary_mix_cfg.get("vis_max_batches", 2))
            ):
                debug_mixed_image = image_u_aug.detach().clone()


            # effect stats: maxprob quantiles + pseudo ratio (sau cutmix nếu có)
            # CHỈ tính khi thật sự log để tránh chậm (torch.quantile khá tốn)
            if do_log_now:
                mp = logits_u_aug.detach()           # [B,H,W] max prob
                mp_flat = mp.reshape(-1)

                u_maxprob_mean = float(mp_flat.mean().item())
                q = torch.quantile(
                    mp_flat,
                    torch.tensor([0.1, 0.5, 0.9], device=mp.device)
                )
                u_maxprob_p10 = float(q[0].item())
                u_maxprob_p50 = float(q[1].item())
                u_maxprob_p90 = float(q[2].item())

                u_pseudo_ratio_mean = float((mp >= p_threshold).float().mean().item())

            if csl_perturb_input and (csl_weight_u is not None or csl_reliable_mask_u is not None):
                if csl_weight_u is not None:
                    reliable_for_perturb = torch.isclose(
                        csl_weight_u.detach(),
                        torch.ones_like(csl_weight_u.detach()),
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    reliable_for_perturb = reliable_for_perturb & label_u_aug.detach().ne(ignore)
                else:
                    reliable_for_perturb = csl_reliable_mask_u
                if mix_source_mask is not None and not bool(csl_cfg.get("mask_labeled_pixels", False)):
                    reliable_for_perturb = reliable_for_perturb & ~mix_source_mask.detach().bool()
                image_u_aug, _, perturb_stats = apply_csl_reliable_mask_perturbation(
                    image_u_aug,
                    reliable_for_perturb,
                    mask_prob=float(csl_cfg.get("mask_prob", 0.3)),
                    block_size=int(csl_cfg.get("block_size", 1)),
                    cover_ratio=float(csl_cfg.get("cover_ratio", 1.0)),
                    mode=csl_cfg.get("mask_mode", "zero"),
                )
                if csl_mask_stats is None:
                    csl_mask_stats = {}
                csl_mask_stats.update(perturb_stats)
                if rank == 0 and csl_debug_enabled and (step < int(csl_cfg.get("debug_first_batches", 3)) or do_log_now):
                    logger.info(
                        "[csl_official] epoch=%d step=%d global_iter=%d perturb_reliable_ratio=%.6f masked_ratio=%.6f"
                        % (
                            epoch,
                            step,
                            i_iter,
                            csl_mask_stats.get("csl/perturb_reliable_ratio", float("nan")),
                            csl_mask_stats.get("csl/masked_ratio", float("nan")),
                        )
                    )



            # 3. forward concate labeled + unlabeld into student networks
            num_labeled = len(image_l)
            decoder_features_u_strong = None
            teacher_features_u_strong = None
            bcr_relation_mode = boundary_compatibility_cfg.get("relation_mode", "base_margin")
            bcr_use_teacher_features = bool(boundary_compatibility_cfg.get("use_teacher_features", False)) or (
                bcr_relation_mode in ("teacher_feature_gate", "teacher_relation_consistency")
            )
            if boundary_compatibility_enabled and bcr_use_teacher_features:
                if boundary_compatibility_cfg.get("teacher_feature_source", "mixed") != "mixed":
                    raise ValueError("boundary_compatibility.teacher_feature_source currently supports 'mixed' only")
                with torch.no_grad():
                    model_teacher.eval()
                    teacher_out = model_teacher(image_u_aug.detach(), return_features=True)
                    if not isinstance(teacher_out, (tuple, list)) or len(teacher_out) < 3:
                        raise ValueError("model_teacher(..., return_features=True) must return logits, aux, features")
                    teacher_feature_dict = teacher_out[2]
                    if not isinstance(teacher_feature_dict, dict) or "decoder" not in teacher_feature_dict:
                        raise ValueError("teacher feature output must contain decoder features")
                    teacher_features_u_strong = teacher_feature_dict["decoder"].detach()
                    del teacher_out, teacher_feature_dict
                model.train()
            if flag_extra_weak:
                if boundary_compatibility_enabled:
                    pred_all, aux_all, feature_all = model(
                        torch.cat((image_l, image_u_weak, image_u_aug), dim=0),
                        return_features=True,
                    )
                    decoder_features_all = feature_all.get("decoder")
                    _, decoder_features_u_strong = decoder_features_all[num_labeled:].chunk(2)
                    del feature_all, decoder_features_all
                else:
                    pred_all, aux_all = model(torch.cat((image_l, image_u_weak, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                _, pred_u_strong = pred_all[num_labeled:].chunk(2)
                del pred_all
            else:
                if boundary_compatibility_enabled:
                    pred_all, aux_all, feature_all = model(
                        torch.cat((image_l, image_u_aug), dim=0),
                        return_features=True,
                    )
                    decoder_features_u_strong = feature_all.get("decoder")[num_labeled:]
                    del feature_all
                else:
                    pred_all, aux_all = model(torch.cat((image_l, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                pred_u_strong = pred_all[num_labeled:]
                del pred_all

            # 4. supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = aux_all[:num_labeled]
                sup_loss = sup_loss_fn([pred_l, aux], label_l)
                del aux_all, aux
            else:
                sup_loss = sup_loss_fn(pred_l, label_l)

            # 5. unsupervised loss
            bcr_loss = pred_u_strong.sum() * 0.0
            bcr_stats = None
            component_stats = None
            component_loss_value = float("nan")
            component_weight_for_bcr = None
            if boundary_component_enabled and mix_source_mask is not None:
                component_eps = float(boundary_component_cfg.get("eps", 1e-6))
                if target_component_label is None:
                    target_component_label = label_u_aug
                if target_component_confidence is None:
                    target_component_confidence = logits_u_aug

                component_weight, component_stats = compute_component_weights(
                    target_component_label.detach(),
                    target_component_confidence.detach(),
                    mix_source_mask.detach(),
                    ignore_index=ignore,
                    num_classes=cfg["net"]["num_classes"],
                    connectivity=int(boundary_component_cfg.get("connectivity", 8)),
                    apply_to=boundary_component_cfg.get("apply_to", "target_only"),
                    foreground_only=bool(boundary_component_cfg.get("foreground_only", True)),
                    tau_visible_low=float(boundary_component_cfg.get("tau_visible_low", 0.2)),
                    tau_visible_high=float(boundary_component_cfg.get("tau_visible_high", 0.6)),
                    area_min=float(boundary_component_cfg.get("area_min", 64)),
                    area_max=float(boundary_component_cfg.get("area_max", 512)),
                    use_mean_confidence_in_q=bool(boundary_component_cfg.get("use_mean_confidence_in_q", True)),
                    base_pixel_weight=boundary_component_cfg.get("base_pixel_weight", "one"),
                    weight_mode=boundary_component_cfg.get("weight_mode", "soft"),
                    force_q_one=bool(boundary_component_cfg.get("force_q_one", False)),
                    eps=component_eps,
                )
                valid = logits_u_aug.detach().ge(p_threshold).bool() & label_u_aug.detach().ne(ignore).bool()
                component_target = label_u_aug.detach().clone()
                component_target[~valid] = ignore
                component_weight = component_weight.to(device=pred_u_strong.device, dtype=pred_u_strong.dtype)
                component_weight_for_bcr = component_weight.detach()
                component_weight = component_weight * valid.to(device=pred_u_strong.device, dtype=pred_u_strong.dtype)
                weighted_loss_denominator = component_weight.sum().clamp_min(component_eps)
                component_ce = F.cross_entropy(
                    pred_u_strong,
                    component_target,
                    ignore_index=ignore,
                    reduction="none",
                )
                unsup_loss = (component_ce * component_weight).sum() / weighted_loss_denominator
                component_loss_value = float(unsup_loss.detach().item())
                pseduo_high_ratio = valid.float().mean()

                do_component_debug = (
                    rank == 0
                    and boundary_component_debug_enabled
                    and (
                        step < int(boundary_component_cfg.get("debug_first_batches", 3))
                        or do_log_now
                    )
                )
                if do_component_debug:
                    per_class_count = component_stats.get("per_class_affected_component_count", {})
                    per_class_q = component_stats.get("per_class_mean_q_C", {})
                    logger.info(
                        "[boundary_component] epoch=%d step=%d global_iter=%d "
                        "num_affected_components=%d num_affected_pixels=%d "
                        "visible_ratio_mean=%.4f visible_ratio_std=%.4f visible_ratio_min=%.4f visible_ratio_max=%.4f "
                        "visible_area_mean=%.4f visible_area_std=%.4f visible_area_min=%.4f visible_area_max=%.4f "
                        "mean_confidence_R_C=%.4f q_C_mean=%.4f q_C_std=%.4f q_C_min=%.4f q_C_max=%.4f "
                        "weighted_loss_denominator=%.4f per_class_count=%s per_class_mean_q=%s"
                        % (
                            epoch,
                            step,
                            i_iter,
                            component_stats["num_affected_components"],
                            component_stats["num_affected_pixels"],
                            component_stats["visible_ratio_mean"],
                            component_stats["visible_ratio_std"],
                            component_stats["visible_ratio_min"],
                            component_stats["visible_ratio_max"],
                            component_stats["visible_area_mean"],
                            component_stats["visible_area_std"],
                            component_stats["visible_area_min"],
                            component_stats["visible_area_max"],
                            component_stats["mean_confidence_R_C"],
                            component_stats["q_C_mean"],
                            component_stats["q_C_std"],
                            component_stats["q_C_min"],
                            component_stats["q_C_max"],
                            float(weighted_loss_denominator.detach().item()),
                            per_class_count,
                            per_class_q,
                        )
                    )
            elif csl_use_ce_weight and csl_weight_u is not None:
                csl_eps = float(csl_cfg.get("eps", 1e-6))
                csl_target = label_u_aug.detach().clone()
                csl_valid = csl_target.ne(ignore)
                csl_target[~csl_valid] = ignore
                csl_weight = csl_weight_u.detach().to(device=pred_u_strong.device, dtype=pred_u_strong.dtype)
                csl_weight = csl_weight * csl_valid.to(device=pred_u_strong.device, dtype=pred_u_strong.dtype)
                csl_ce = F.cross_entropy(
                    pred_u_strong,
                    csl_target,
                    ignore_index=ignore,
                    reduction="none",
                )
                weighted_loss_denominator = csl_weight.sum().clamp_min(csl_eps)
                unsup_loss = (csl_ce * csl_weight).sum() / weighted_loss_denominator
                pseduo_high_ratio = (csl_weight > 0).float().mean()
                if csl_stats is None:
                    csl_stats = {}
                csl_stats["csl_ce/weight_mean"] = float(csl_weight.detach().mean().item())
                csl_stats["csl_ce/weight_sum"] = float(csl_weight.detach().sum().item())

                if rank == 0 and csl_debug_enabled and (step < int(csl_cfg.get("debug_first_batches", 3)) or do_log_now):
                    logger.info(
                        "[csl] epoch=%d step=%d global_iter=%d weight_mean=%.6f weight_sum=%.3f"
                        % (
                            epoch,
                            step,
                            i_iter,
                            csl_stats["csl_ce/weight_mean"],
                            csl_stats["csl_ce/weight_sum"],
                        )
                    )
            elif boundary_mix_enabled and mix_source_mask is not None:
                boundary_mix_kernel_size = int(boundary_mix_cfg.get("kernel_size", 5))
                boundary_mix_gamma_in = float(boundary_mix_cfg.get("gamma_in", 0.7))
                boundary_mix_gamma_out = float(boundary_mix_cfg.get("gamma_out", 0.3))
                boundary_mix_use_confidence = bool(boundary_mix_cfg.get("use_confidence", True))
                boundary_mix_normalize_weight = bool(boundary_mix_cfg.get("normalize_weight", True))
                unsup_loss, pseduo_high_ratio = thresholded_boundary_mix_loss(
                    pred_u_strong,
                    label_u_aug.detach(),
                    logits_u_aug.detach(),
                    mix_source_mask.detach(),
                    thresh=p_threshold,
                    ignore_index=ignore,
                    kernel_size=boundary_mix_kernel_size,
                    gamma_in=boundary_mix_gamma_in,
                    gamma_out=boundary_mix_gamma_out,
                    use_confidence=boundary_mix_use_confidence,
                    normalize_weight=boundary_mix_normalize_weight,
                )

                do_boundary_debug = (
                    rank == 0
                    and boundary_mix_debug_enabled
                    and (
                        step < int(boundary_mix_cfg.get("debug_first_batches", 3))
                        or do_log_now
                    )
                )
                if do_boundary_debug:
                    bm_stats = boundary_mix_debug_stats(
                        mix_source_mask.detach(),
                        logits_u_aug.detach(),
                        kernel_size=boundary_mix_kernel_size,
                        gamma_in=boundary_mix_gamma_in,
                        gamma_out=boundary_mix_gamma_out,
                        use_confidence=boundary_mix_use_confidence,
                        normalize_weight=boundary_mix_normalize_weight,
                    )
                    logger.info(
                        "[boundary_mix] epoch=%d step=%d global_iter=%d "
                        "weight_mean=%.4f weight_min=%.4f weight_max=%.4f "
                        "B_in_pixels=%d B_out_pixels=%d "
                        "target_exterior_conf_mean=%.4f target_outer_boundary_conf_mean=%.4f"
                        % (
                            epoch,
                            step,
                            i_iter,
                            bm_stats["weight_mean"],
                            bm_stats["weight_min"],
                            bm_stats["weight_max"],
                            bm_stats["b_in_pixels"],
                            bm_stats["b_out_pixels"],
                            bm_stats["target_exterior_conf_mean"],
                            bm_stats["target_outer_boundary_conf_mean"],
                        )
                    )

                    if (
                        boundary_mix_vis_enabled
                        and debug_mixed_image is not None
                        and boundary_mix_vis_saved < int(boundary_mix_cfg.get("vis_max_batches", 2))
                    ):
                        prefix = "epoch%03d_iter%06d" % (epoch, i_iter)
                        saved_files = save_boundary_mix_debug(
                            boundary_mix_cfg.get("vis_dir", "exp_boundary_debug_vis"),
                            mix_mask=mix_source_mask.detach(),
                            confidence=logits_u_aug.detach(),
                            mixed_image=debug_mixed_image,
                            mixed_label=label_u_aug.detach(),
                            prediction=pred_u_strong.detach(),
                            mean=cfg["dataset"].get("mean"),
                            std=cfg["dataset"].get("std"),
                            prefix=prefix,
                            kernel_size=boundary_mix_kernel_size,
                            gamma_in=boundary_mix_gamma_in,
                            gamma_out=boundary_mix_gamma_out,
                            use_confidence=boundary_mix_use_confidence,
                            normalize_weight=boundary_mix_normalize_weight,
                            max_items=1,
                        )
                        boundary_mix_vis_saved += 1
                        logger.info(
                            "[boundary_mix] saved %d debug visualization PNGs under %s"
                            % (len(saved_files), boundary_mix_cfg.get("vis_dir", "exp_boundary_debug_vis"))
                        )
            else:
                unsup_loss, pseduo_high_ratio = compute_unsupervised_loss_by_threshold(
                            pred_u_strong, label_u_aug.detach(),
                            logits_u_aug.detach(), thresh=p_threshold)
            unsup_loss *= cfg["trainer"]["unsupervised"].get("loss_weight", 1.0)

            if boundary_compatibility_enabled and mix_source_mask is not None:
                if boundary_compatibility_cfg.get("semantic_metric", "js") != "js":
                    raise ValueError("boundary_compatibility.semantic_metric currently supports 'js' only")
                if boundary_compatibility_cfg.get("feature_layer", "decoder") != "decoder":
                    raise ValueError("boundary_compatibility.feature_layer currently supports 'decoder' only")
                use_component_gate = (
                    boundary_component_enabled
                    and bool(boundary_compatibility_cfg.get("use_component_gate", False))
                )
                bcr_loss, bcr_stats = compute_js_boundary_compatibility_loss(
                    decoder_features_u_strong,
                    label_u_aug.detach(),
                    teacher_probs_u_aug.detach() if teacher_probs_u_aug is not None else None,
                    logits_u_aug.detach(),
                    mix_source_mask.detach(),
                    component_weight_for_bcr if use_component_gate else None,
                    teacher_features_u_strong,
                    boundary_compatibility_cfg,
                    num_classes=cfg["net"]["num_classes"],
                    ignore_index=ignore,
                    band_width=int(boundary_compatibility_cfg.get("band_width", 3)),
                    pair_mode=boundary_compatibility_cfg.get("pair_mode", "radius"),
                    pair_radius=int(boundary_compatibility_cfg.get("pair_radius", 1)),
                    topk=int(boundary_compatibility_cfg.get("topk", 5)),
                    max_pairs_per_image=int(boundary_compatibility_cfg.get("max_pairs_per_image", 2048)),
                    tau_same=float(boundary_compatibility_cfg.get("tau_same", 0.8)),
                    tau_diff=float(boundary_compatibility_cfg.get("tau_diff", 0.3)),
                    margin=float(boundary_compatibility_cfg.get("margin", 0.4)),
                    use_confidence_gate=bool(boundary_compatibility_cfg.get("use_confidence_gate", True)),
                    use_component_gate=use_component_gate,
                    component_gate_mode=boundary_compatibility_cfg.get("component_gate_mode", "direct"),
                    component_gate_alpha=float(boundary_compatibility_cfg.get("component_gate_alpha", 0.5)),
                    component_gate_threshold=float(boundary_compatibility_cfg.get("component_gate_threshold", 0.1)),
                    same_loss_mode=boundary_compatibility_cfg.get("same_loss_mode", "hard_one"),
                    relation_mode=boundary_compatibility_cfg.get("relation_mode", "base_margin"),
                    use_teacher_features=bool(boundary_compatibility_cfg.get("use_teacher_features", False)),
                    teacher_feature_detach=bool(boundary_compatibility_cfg.get("teacher_feature_detach", True)),
                    affinity_target=boundary_compatibility_cfg.get("affinity_target", "hard"),
                    affinity_temperature=float(boundary_compatibility_cfg.get("affinity_temperature", 0.2)),
                    detach_teacher_distribution=bool(boundary_compatibility_cfg.get("detach_teacher_distribution", True)),
                    detach_gate=bool(boundary_compatibility_cfg.get("detach_gate", True)),
                    eps=float(boundary_compatibility_cfg.get("eps", 1e-6)),
                )
                do_bcr_debug = (
                    rank == 0
                    and boundary_compatibility_debug_enabled
                    and (
                        step < int(boundary_compatibility_cfg.get("debug_first_batches", 3))
                        or do_log_now
                    )
                )
                if do_bcr_debug:
                    log_items = []
                    for key in sorted(bcr_stats):
                        value = bcr_stats[key]
                        if isinstance(value, float):
                            log_items.append("%s=%.6f" % (key, value))
                        else:
                            log_items.append("%s=%s" % (key, value))
                    logger.info(
                        "[boundary_compatibility] epoch=%d step=%d global_iter=%d %s"
                        % (epoch, step, i_iter, " ".join(log_items))
                    )
            del pred_l, pred_u_strong, label_u_aug, logits_u_aug, decoder_features_u_strong, teacher_features_u_strong

        loss = sup_loss + unsup_loss
        if bcr_loss is not None:
            loss = loss + boundary_compatibility_lambda * bcr_loss
        if u1_enabled or u2_enabled or u3_enabled or u4_enabled:
            u1_synchronized_failure_check(
                not bool(torch.isfinite(loss).item()),
                "nonfinite_integrated_student_loss",
                device=loss.device,
                rank=rank,
                epoch=epoch,
                step=step,
                iteration=i_iter,
            )

        # update student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        with torch.no_grad():
            if epoch > cfg["trainer"].get("sup_only_epoch", 0):
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 0)
                        + 1
                    ),
                    ema_decay_origin,
                )
            else:
                ema_decay = 0.0
            # update weight
            for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            # update bn
            for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                # buffer_eval.data = buffer_train.data

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item() / world_size)

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item() / world_size)

        reduced_pseudo_high_ratio = pseduo_high_ratio.clone().detach()
        dist.all_reduce(reduced_pseudo_high_ratio)
        meter_high_pseudo_ratio.update(reduced_pseudo_high_ratio.item() / world_size)

        # 12. print log information
        batch_end = time.time()
        batch_times.update(batch_end - batch_start)
        # if i_iter % 10 == 0 and rank == 0:
        if step in print_freq_lst and rank == 0:
            logger.info(
                "Epoch/Iter [{}:{:3}/{:3}].  "
                "Sup:{sup_loss.val:.3f}({sup_loss.avg:.3f})  "
                "Uns:{uns_loss.val:.3f}({uns_loss.avg:.3f})  "
                "Pseudo:{high_ratio.val:.3f}({high_ratio.avg:.3f})  "
                "Time:{batch_time.avg:.2f}  "
                "LR:{lr.val:.5f}".format(
                    cfg["trainer"]["epochs"], epoch, step,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    high_ratio=meter_high_pseudo_ratio,
                    batch_time=batch_times,
                    lr=learning_rates,
                )
            )

            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
                tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
                tb_logger.add_scalar("High ratio", meter_high_pseudo_ratio.avg, i_iter)

        # ---------- Iter log: CSV luôn ghi, W&B là tùy chọn ----------
        if rank == 0:
            do_log = (i_iter % log_every == 0) or (step == len(loader_l) - 1)
            if do_log:
                log_dict = make_default_iter_log_dict(
                    sup_loss=sup_losses.val,
                    uns_loss=uns_losses.val,
                    pseudo_high_ratio=meter_high_pseudo_ratio.val,
                    lr=learning_rates.val,
                    epoch=epoch,
                    step=step,
                    global_iter=i_iter,
                    ar_triggered=ar_triggered,
                    ar_applied=ar_applied,
                    ar_area_ratio_est=ar_area_ratio_est,
                    u_entropy_mean=u_entropy_mean,
                    u_maxprob_mean=u_maxprob_mean,
                    u_maxprob_p10=u_maxprob_p10,
                    u_maxprob_p50=u_maxprob_p50,
                    u_maxprob_p90=u_maxprob_p90,
                    u_pseudo_ratio_mean=u_pseudo_ratio_mean,
                )
                # -------- Aa (photometric) --------
                if (k_ids is not None) and (t_vals is not None):
                    k_np = k_ids.detach().cpu().numpy()
                    t_np = t_vals.detach().cpu().numpy()

                    log_dict["aa/ops_per_image"] = float((k_np > 0).sum(axis=1).mean())
                    total_ops = max(int((k_np > 0).sum()), 1)

                    AA_DEFAULT_PCT = {
                        "identity": 0.0,
                        "autocontrast": 100.0,
                        "equalize": 100.0,
                    }

                    for kid in range(1, 12):
                        op_name = AA_OPS.get(kid, f"op{kid}")
                        m = (k_np == kid)
                        cnt = int(m.sum())

                        log_dict[f"aa/op_rate/{op_name}"] = cnt / total_ops
                        log_dict[f"aa/count/{op_name}"] = cnt

                        if cnt <= 0:
                            continue

                        tv = t_np[m]
                        tv = tv[np.isfinite(tv)]

                        if tv.size > 0:
                            sv = np.array([aa_strength(kid, float(x)) for x in tv], dtype=np.float32)
                            sv = sv[np.isfinite(sv)]

                            if sv.size > 0:
                                sv_pct = sv * 100.0
                                log_dict[f"aa/pct_mean/{op_name}"] = float(sv_pct.mean())
                                log_dict[f"aa/pct_max/{op_name}"] = float(sv_pct.max())

                            log_dict[f"aa/t_mean/{op_name}"] = float(tv.mean())
                            log_dict[f"aa/t_std/{op_name}"] = float(tv.std())

                if component_stats is not None:
                    log_dict.update({
                        "v2/num_affected_components": component_stats.get("num_affected_components", np.nan),
                        "v2/num_affected_pixels": component_stats.get("num_affected_pixels", np.nan),
                        "v2/mean_visible_ratio": component_stats.get("visible_ratio_mean", np.nan),
                        "v2/mean_visible_area": component_stats.get("visible_area_mean", np.nan),
                        "v2/mean_component_confidence": component_stats.get("mean_confidence_R_C", np.nan),
                        "v2/mean_q_C": component_stats.get("q_C_mean", np.nan),
                        "v2/min_q_C": component_stats.get("q_C_min", np.nan),
                        "v2/max_q_C": component_stats.get("q_C_max", np.nan),
                        "v2/loss_mix_v2": component_loss_value,
                    })

                if bcr_stats is not None:
                    for key, value in bcr_stats.items():
                        if key in ITER_LOG_COLUMNS:
                            log_dict[key] = value

                if saliency_cutmix_enabled:
                    saliency_mode_value = {
                        "box": 1.0,
                        "component_box": 2.0,
                    }.get(saliency_cutmix_cfg.get("mode", "box"), -1.0)
                    log_dict.update({
                        "saliency/enabled": 1.0,
                        "saliency/mode": saliency_mode_value,
                        "saliency/num_candidates": float(saliency_cutmix_cfg.get("num_candidates", 8)),
                        "saliency/temperature": float(saliency_cutmix_cfg.get("temperature", 0.2)),
                    })
                    if saliency_stats is not None:
                        for key, value in saliency_stats.items():
                            if key in ITER_LOG_COLUMNS:
                                log_dict[key] = value
                        log_dict["saliency/mix1_score_selected"] = saliency_stats.get("saliency/score_selected", np.nan)
                        log_dict["saliency/mix1_score_candidate_mean"] = saliency_stats.get("saliency/score_candidate_mean", np.nan)
                        log_dict["saliency/mix1_selection_entropy"] = saliency_stats.get("saliency/selection_entropy", np.nan)
                        log_dict["saliency/mix2_score_selected"] = np.nan
                        log_dict["saliency/mix2_score_candidate_mean"] = np.nan
                        log_dict["saliency/mix2_selection_entropy"] = np.nan
                else:
                    log_dict["saliency/enabled"] = 0.0
                    log_dict["saliency/mode"] = 0.0

                if s2_relocated_policy_active:
                    relocation_draw_count = float(relocation_diagnostics["relocation_draw_count"])
                    relocation_denominator = relocation_draw_count if relocation_draw_count > 0 else 1.0
                    log_dict.update(relocation_diagnostics)
                    log_dict.update({
                        "relocation_zero_rate": float(relocation_diagnostics["relocation_zero_count"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_nonzero_rate": float(relocation_diagnostics["relocation_nonzero_count"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_success_rate": float(relocation_diagnostics["relocation_success_count"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_zero_only_rate": float(relocation_diagnostics["relocation_zero_only_count"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_random_zero_rate": float(relocation_diagnostics["relocation_random_zero_count"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_mean_valid_translation_count": float(relocation_diagnostics["relocation_valid_translation_sum"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_expected_zero_rate": float(relocation_diagnostics["relocation_expected_zero_sum"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_mean_displacement_magnitude": float(relocation_diagnostics["relocation_displacement_magnitude_sum"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                        "relocation_mean_source_destination_iou": float(relocation_diagnostics["relocation_source_destination_iou_sum"]) / relocation_denominator if relocation_draw_count > 0 else 0.0,
                    })
                log_dict["saliency_selector_exception_batch"] = int(saliency_selector_exception_batch)

                if csl_enabled:
                    csl_mode_value = {
                        "pseudo_selection": 1.0,
                        "random_reliable_masking": 2.0,
                        "guided_cutmix": 3.0,
                        "official_reliability_replace_confidence": 4.0,
                        "official_reliable_mask_perturbation": 5.0,
                        "official_direct_labeled_guided_cutmix_plus_ce_weight": 6.0,
                    }.get(csl_cfg.get("mode", "disabled"), -1.0)
                    log_dict.update({
                        "csl/enabled": 1.0,
                        "csl/mode": csl_mode_value,
                        "csl/use_csl_for_ce_weight": float(csl_use_ce_weight),
                        "csl/use_csl_for_mix_confidence": float(csl_use_mix_confidence),
                        "csl/use_csl_for_cutmix": float(csl_use_cutmix),
                        "csl/random_mask_reliable": float(bool(csl_cfg.get("random_mask_reliable", False))),
                        "csl/perturb_input": float(csl_perturb_input),
                    })
                    for stats_dict in (csl_stats, csl_mask_stats):
                        if stats_dict is None:
                            continue
                        for key, value in stats_dict.items():
                            if key in ITER_LOG_COLUMNS:
                                log_dict[key] = value
                else:
                    log_dict["csl/enabled"] = 0.0
                    log_dict["csl/mode"] = 0.0

                if csl_cutmix_enabled:
                    log_dict["csl_cutmix/enabled"] = 1.0
                    if csl_cutmix_stats is not None:
                        for key, value in csl_cutmix_stats.items():
                            if key in ITER_LOG_COLUMNS:
                                log_dict[key] = value
                else:
                    log_dict["csl_cutmix/enabled"] = 0.0

                if c4_stats is not None:
                    for key, value in c4_stats.items():
                        if key in ITER_LOG_COLUMNS:
                            log_dict[key] = value

                if torch.cuda.is_available():
                    log_dict["cuda/max_memory_allocated"] = float(torch.cuda.max_memory_allocated(local_rank))
                    log_dict["cuda/max_memory_reserved"] = float(torch.cuda.max_memory_reserved(local_rank))

                # W&B optional
                if wandb_run is not None:
                    wandb_run.log(log_dict, step=int(i_iter))

                # CSV always on if path is provided
                if train_iter_csv is not None:
                    df_row = pd.DataFrame([log_dict], columns=ITER_LOG_COLUMNS)
                    if os.path.exists(train_iter_csv):
                        df_row.to_csv(train_iter_csv, mode="a", header=False, index=False)
                    else:
                        df_row.to_csv(train_iter_csv, index=False)

    
    return sup_losses.avg, uns_losses.avg


def validate(
    model,
    data_loader,
    epoch,
    logger,
    cfg,
    sup_loss_fn,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    loss_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            pred, aux = model(images)

            if "aux_loss" in cfg["net"].keys():
                val_loss = sup_loss_fn([pred, aux], labels)
            else:
                val_loss = sup_loss_fn(pred, labels)

        reduced_val_loss = val_loss.detach().clone()
        dist.all_reduce(reduced_val_loss)
        loss_meter.update(reduced_val_loss.item() / world_size)

        output = pred.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))

    return {
        "loss": float(loss_meter.avg),
        "miou": float(mIoU),
        "iou_class": iou_class.astype(np.float64),
    }

def validate_citys(
    model,
    data_loader,
    epoch,
    logger,
    cfg,
    sup_loss_fn,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    num_classes = cfg["net"]["num_classes"]
    ignore_label = cfg["dataset"]["ignore_label"]
    if cfg["dataset"]["val"].get("crop", False):
        crop_size, _ = cfg["dataset"]["val"]["crop"].get("size", [800, 800])
    else:
        crop_size = 800

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            final = torch.zeros(batch_size, num_classes, h, w).cuda()
            row = 0
            while row < h:
                col = 0
                while col < w:
                    pred, _ = model(
                        images[:, :, row:min(h, row + crop_size), col:min(w, col + crop_size)]
                    )
                    final[:, :, row:min(h, row + crop_size), col:min(w, col + crop_size)] += pred.softmax(dim=1)
                    col += int(crop_size * 2 / 3)
                row += int(crop_size * 2 / 3)

            labels_cuda = labels.cuda()
            if "aux_loss" in cfg["net"].keys():
                val_loss = sup_loss_fn([final, final], labels_cuda)
            else:
                val_loss = sup_loss_fn(final, labels_cuda)

            reduced_val_loss = val_loss.detach().clone()
            dist.all_reduce(reduced_val_loss)
            loss_meter.update(reduced_val_loss.item() / world_size)

            output = final.argmax(dim=1).cpu().numpy()
            target_origin = labels.numpy()

        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))

    return {
        "loss": float(loss_meter.avg),
        "miou": float(mIoU),
        "iou_class": iou_class.astype(np.float64),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
