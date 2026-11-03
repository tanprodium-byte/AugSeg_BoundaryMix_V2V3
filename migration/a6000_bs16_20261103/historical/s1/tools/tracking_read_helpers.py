"""Read-only tracking checks reused verbatim from the protected legacy finalizer."""
import os
import math
ENTITY="tanprodium-uit"
PROJECT="AugsegLegacyC321Rerun01-RTX5090"
def require(condition,message):
    if not condition:raise ValueError(message)

def validate_wandb_completion(run, cfg, run_id, rows, iterations):
    """Unsampled, unfiltered-key history; canonical mIoUs are fractions, not percent.

    atol=1e-7 accommodates CSV/JSON float serialization (<0.00001 percentage
    points). Diagnostic legacy NaNs are not scientific validation metrics.
    """
    require(run.id == run_id and list(run.path) == [ENTITY, PROJECT, run_id], "W&B exact run identity mismatch")
    require(run.state == "finished" and run.name == cfg["wandb"]["name"], "W&B run is not finished with expected name")
    remote = run.config
    for group in ("dataset", "trainer", "net", "hf"):
        require(remote.get(group) == cfg[group], f"W&B config/profile mismatch: {group}")
    require(remote.get("name") == cfg["name"] and remote.get("run", {}).get("name") == cfg["run"]["name"], "W&B config identity mismatch")
    require(remote.get("wandb", {}).get("resume") == "never", "W&B resume policy mismatch")
    final_step = 80 * iterations - 1
    require(int(run.summary.get("_step", -1)) == final_step and int(run.summary.get("meta/global_iter", -1)) == final_step,
            "W&B final step incomplete")
    require(int(run.summary.get("meta/epoch", -1)) == 80, "W&B final epoch incomplete")
    expected = {(e + 1) * iterations - 1: e for e in range(80)}
    validations = {}
    # Do not use history() (sampled), nor keys=[...] (drops sparse records).
    for record in run.scan_history(min_step=0, max_step=final_step + 1, page_size=1000):
        keys = ("val/miou_stu", "val/miou_tea")
        step = int(record.get("_step", -1))
        if step not in expected:
            # Legacy scan_history returns a rectangular schema: absent metrics
            # appear as None on train_iter rows. These are not validations.
            require(all(record.get(k) is None for k in keys), "W&B unexpected validation step")
            continue
        require(step not in validations, "W&B duplicate validation step")
        require(int(record.get("meta/epoch", -1)) == expected[step] + 1 and int(record.get("meta/global_iter", -1)) == step,
                "W&B validation epoch/global iteration mismatch")
        values = []
        for key in keys:
            raw = record.get(key)
            value = float(raw) if raw is not None else float("nan")
            require(math.isfinite(value) and 0 <= value <= 1, f"W&B missing/nonfinite/scale mismatch: {key}")
            values.append(value)
        validations[step] = values
    require(set(validations) == set(expected), "W&B validation coverage is not exactly 80 epochs for both models")
    require(len(rows) == 80 and [int(r["epoch"]) for r in rows] == list(range(80)), "local validation coverage mismatch")
    for step, epoch in expected.items():
        for index, model in enumerate(("stu", "tea")):
            local = float(rows[epoch][f"miou_{model}"])
            require(math.isfinite(local) and 0 <= local <= 1, "local mIoU nonfinite/scale mismatch")
            require(math.isclose(validations[step][index], local, rel_tol=0, abs_tol=1e-7), "local/W&B validation metric mismatch")
    for model in ("stu", "tea"):
        value = float(run.summary.get(f"val/miou_{model}", float("nan")))
        require(math.isfinite(value) and math.isclose(value, float(rows[-1][f"miou_{model}"]), rel_tol=0, abs_tol=1e-7), "W&B final metric mismatch")
    return {"epochs": 80, "iterations_per_epoch": iterations, "final_step": final_step,
            "student_teacher_validation_coverage": 80, "metric_scale": "fraction_0_1", "absolute_tolerance": 1e-7}

def verify_wandb_fresh(cfg, run_id, api):
    """A successful GraphQL project/run:null is genuine absence, not SDK text."""
    import requests
    key = os.environ.get("WANDB_API_KEY")
    require(bool(key), "W&B credential unavailable for exact freshness query")
    response = requests.post(
        os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/") + "/graphql",
        auth=("api", key), timeout=30,
        json={"query": "query($entity:String!,$project:String!,$run:String!){project(name:$project,entityName:$entity){name run(name:$run){name}}}",
              "variables": {"entity": ENTITY, "project": PROJECT, "run": run_id}})
    response.raise_for_status()
    data = response.json()
    require(not data.get("errors"), "W&B exact freshness API error; absence UNKNOWN")
    project = data.get("data", {}).get("project")
    require(isinstance(project, dict) and "run" in project, "W&B project unreadable; absence UNKNOWN")
    require(project["run"] is None, "W&B exact run-ID collision")
    require(not list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": cfg["wandb"]["name"]})), "W&B display-name collision")

