# Latest 20 W&B runs in `tanprodium-uit/augseg-voc662`

Generated at: `2026-06-30T10:50:33`

## 01. s3_saliency_component_mask_direct_plus_v3_d2

- **method_name:** `s3_saliency_component_mask_direct_plus_v3_d2`
- **method_hint:** S3: saliency component-mask direct CutMix + V3/BCR d2.
- **state:** `running`
- **created_at:** `2026-06-30T06:04:25Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__s3_saliency_component_mask_direct_plus_v3_d2
- **epoch_or_step:** `8`
- **val_mIoU:** `71.13`
- **best_mIoU:** `71.13`
- **program:** `/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--log-dir", "runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80", "--status-dir", "runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "supermaster:gpu0", "--server-name", "supermaster", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume", "-...`

### Selected config

```json
{}
```

## 02. s2_saliency_component_mask_direct_cutmix

- **method_name:** `s2_saliency_component_mask_direct_cutmix`
- **method_hint:** S2: saliency component-mask direct CutMix.
- **state:** `running`
- **created_at:** `2026-06-30T05:24:13Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__s2_saliency_component_mask_direct_cutmix
- **epoch_or_step:** `16`
- **val_mIoU:** `72.15`
- **best_mIoU:** `72.15`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--log-dir", "runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80", "--status-dir", "runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume"...`

### Selected config

```json
{}
```

## 03. s1_saliency_box_relocated_cutmix

- **method_name:** `s1_saliency_box_relocated_cutmix`
- **method_hint:** S1 relocated: saliency box, paste sang vị trí random target.
- **state:** `finished`
- **created_at:** `2026-06-30T05:00:36Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__s1_saliency_box_relocated_cutmix
- **epoch_or_step:** `19`
- **val_mIoU:** `71.86`
- **best_mIoU:** `72.17`
- **program:** `/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--log-dir", "runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80", "--status-dir", "runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "supermaster:gpu0", "--server-name", "supermaster", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume", "-...`

### Selected config

```json
{}
```

## 04. s1_saliency_box_direct_cutmix

- **method_name:** `s1_saliency_box_direct_cutmix`
- **method_hint:** S1: saliency box direct CutMix, paste cùng tọa độ.
- **state:** `finished`
- **created_at:** `2026-06-30T03:53:29Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__s1_saliency_box_direct_cutmix
- **epoch_or_step:** `19`
- **val_mIoU:** `71.76`
- **best_mIoU:** `71.97`
- **program:** ``
- **args:** ``

### Selected config

```json
{}
```

## 05. s1_saliency_box_adaptive_relocated_cutmix

- **method_name:** `s1_saliency_box_adaptive_relocated_cutmix`
- **method_hint:** S1 adaptive relocated: relocated CutMix + confidence gate.
- **state:** `finished`
- **created_at:** `2026-06-30T02:46:29Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__s1_saliency_box_adaptive_relocated_cutmix
- **epoch_or_step:** ``
- **val_mIoU:** ``
- **best_mIoU:** ``
- **program:** `/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--log-dir", "runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80", "--status-dir", "runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "supermaster:gpu0", "--server-name", "supermaster", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume", "-...`

### Selected config

```json
{}
```

## 06. c3_official_guided_cutmix_plus_v3_d2

- **method_name:** `c3_official_guided_cutmix_plus_v3_d2`
- **method_hint:** V3/BCR-related run.
- **state:** `finished`
- **created_at:** `2026-06-30T01:18:47Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__c3_csl_official_guided_cutmix_plus_v3_d2
- **epoch_or_step:** ``
- **val_mIoU:** ``
- **best_mIoU:** ``
- **program:** `/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--log-dir", "runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80", "--status-dir", "runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "supermaster:gpu0", "--server-name", "supermaster", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume", "-...`

### Selected config

```json
{}
```

## 07. c2_official_reliable_mask_perturbation

- **method_name:** `c2_official_reliable_mask_perturbation`
- **method_hint:** Không nhận diện tự động từ tên run; xem selected_config bên dưới.
- **state:** `finished`
- **created_at:** `2026-06-30T00:50:22Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__c2_csl_official_reliable_mask_perturbation
- **epoch_or_step:** `19`
- **val_mIoU:** `71.13`
- **best_mIoU:** `71.53`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--log-dir", "runs/suite_logs/voc662_8_sc_methods_official_c3_20_40_60_80", "--status-dir", "runs/suite_status/voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume"...`

### Selected config

```json
{}
```

## 08. c1_official_reliability_replace_confidence

- **method_name:** `c1_official_reliability_replace_confidence`
- **method_hint:** Không nhận diện tự động từ tên run; xem selected_config bên dưới.
- **state:** `finished`
- **created_at:** `2026-06-29T19:21:29Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_8_sc_methods_official_c3_20_40_60_80__c1_csl_official_reliability_replace_confidence
- **epoch_or_step:** `19`
- **val_mIoU:** `72.6`
- **best_mIoU:** `72.77`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_8_sc_methods_official_c3_20_40_60_80.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_8_sc_methods_official_c3_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--sleep-sec", "30", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 09. v3_d2_teacher_feature_gate

- **method_name:** `v3_d2_teacher_feature_gate`
- **method_hint:** V3/BCR-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:10:51Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__v3_d2_teacher_feature_gate
- **epoch_or_step:** `79`
- **val_mIoU:** `72.12`
- **best_mIoU:** `72.3`
- **program:** `/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "supermaster:gpu0", "--server-name", "supermaster", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 10. v23_d2_soft_same_target

- **method_name:** `v23_d2_soft_same_target`
- **method_hint:** BoundaryMix/V2-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:10:46Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__v23_d2_soft_same_target
- **epoch_or_step:** `54`
- **val_mIoU:** `72.41`
- **best_mIoU:** `72.63`
- **program:** ``
- **args:** ``

### Selected config

```json
{}
```

## 11. s3_saliency_component_box_plus_v3_d2

- **method_name:** `s3_saliency_component_box_plus_v3_d2`
- **method_hint:** V3/BCR-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:10:40Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__s3_saliency_component_box_plus_v3_d2
- **epoch_or_step:** `59`
- **val_mIoU:** `72.29`
- **best_mIoU:** `72.39`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 12. s2_saliency_component_box_cutmix

- **method_name:** `s2_saliency_component_box_cutmix`
- **method_hint:** Không nhận diện tự động từ tên run; xem selected_config bên dưới.
- **state:** `crashed`
- **created_at:** `2026-06-19T10:10:37Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__s2_saliency_component_box_cutmix
- **epoch_or_step:** `33`
- **val_mIoU:** `73.21`
- **best_mIoU:** `73.4`
- **program:** ``
- **args:** ``

### Selected config

```json
{}
```

## 13. s1_saliency_box_cutmix

- **method_name:** `s1_saliency_box_cutmix`
- **method_hint:** Không nhận diện tự động từ tên run; xem selected_config bên dưới.
- **state:** `finished`
- **created_at:** `2026-06-19T10:10:33Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__s1_saliency_box_cutmix
- **epoch_or_step:** `59`
- **val_mIoU:** `72.15`
- **best_mIoU:** `72.29`
- **program:** ``
- **args:** ``

### Selected config

```json
{}
```

## 14. v3_d2_teacher_relation_consistency

- **method_name:** `v3_d2_teacher_relation_consistency`
- **method_hint:** V3/BCR-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:50Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__v3_d2_teacher_relation_consistency
- **epoch_or_step:** `79`
- **val_mIoU:** `72.13`
- **best_mIoU:** `72.29`
- **program:** ``
- **args:** ``

### Selected config

```json
{}
```

## 15. v3_d2_affinity_bce

- **method_name:** `v3_d2_affinity_bce`
- **method_hint:** V3/BCR-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:46Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__v3_d2_affinity_bce
- **epoch_or_step:** `79`
- **val_mIoU:** `72.25`
- **best_mIoU:** `72.44`
- **program:** ``
- **args:** ``

### Selected config

```json
{}
```

## 16. v23_d2_soft_qc_gate_a05

- **method_name:** `v23_d2_soft_qc_gate_a05`
- **method_hint:** BoundaryMix/V2-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:42Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__v23_d2_soft_qc_gate_a05
- **epoch_or_step:** ``
- **val_mIoU:** ``
- **best_mIoU:** ``
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 17. v23_d2_no_qc_gate

- **method_name:** `v23_d2_no_qc_gate`
- **method_hint:** BoundaryMix/V2-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:37Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__v23_d2_no_qc_gate
- **epoch_or_step:** `79`
- **val_mIoU:** `72.73`
- **best_mIoU:** `72.83`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 18. c3_entropy_margin_guided_cutmix_plus_v3_d2

- **method_name:** `c3_entropy_margin_guided_cutmix_plus_v3_d2`
- **method_hint:** V3/BCR-related run.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:32Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__c3_csl_guided_cutmix_plus_v3_d2
- **epoch_or_step:** `79`
- **val_mIoU:** `72.35`
- **best_mIoU:** `72.43`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 19. c2_entropy_margin_random_reliable_masking

- **method_name:** `c2_entropy_margin_random_reliable_masking`
- **method_hint:** Không nhận diện tự động từ tên run; xem selected_config bên dưới.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:29Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__c2_csl_random_reliable_masking
- **epoch_or_step:** `59`
- **val_mIoU:** `72.68`
- **best_mIoU:** `72.83`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

## 20. c1_entropy_margin_reliability_weighting

- **method_name:** `c1_entropy_margin_reliability_weighting`
- **method_hint:** Không nhận diện tự động từ tên run; xem selected_config bên dưới.
- **state:** `finished`
- **created_at:** `2026-06-19T10:06:25Z`
- **url:** https://wandb.ai/tanprodium-uit/augseg-voc662/runs/voc662_12_methods_segments_20_40_60_80__c1_csl_pseudo_selection
- **epoch_or_step:** `79`
- **val_mIoU:** `73.13`
- **best_mIoU:** `73.46`
- **program:** `/home/islabworker3/tantv/AugSeg_BoundaryMix_V2V3/tools/run_experiment_suite.py`
- **args:** `["--registry", "configs/experiment_registry_voc662_12_methods.yaml", "--mode", "full", "--schedule-mode", "segments", "--epoch-targets", "20,40,60,80", "--suite-name", "voc662_12_methods_segments_20_40_60_80", "--queue-backend", "postgres", "--db-url-env", "AUGSEG_SCHEDULER_DB_URL", "--worker-id", "islab-server3:gpu0", "--server-name", "islab-server3", "--gpu", "0", "--nproc-per-node", "1", "--min-free-mb", "17000", "--poll-sec", "60", "--launcher", "python-module", "--loop", "--resume", "--retry-failed", "--max-retries", "1"]`

### Selected config

```json
{}
```

