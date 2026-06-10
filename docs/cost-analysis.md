# Cost Analysis: Root Detection MLOps Application

This report covers the financial cost of running the root-detection
training + inference pipeline, the trade-offs made between performance
and budget, the concrete code that implements those trade-offs, and the
monitoring strategy used to keep costs predictable.

---

## 1. Architecture summary

| Component | Where it runs | Always on? | Resource footprint |
|---|---|---|---|
| Training pipeline (sweep → eval → register → deploy) | Azure ML compute = `lambda-0` (Arc-connected Kubernetes cluster, lab-owned hardware) | Triggered weekly + on-demand | 1 GPU pod per trial, ~30 min – 3 h per pipeline |
| Inference endpoint (`root-detection-endpoint`) | Azure ML compute = `lambda-0` (Kubernetes online endpoint) | 24/7 | 1 CPU pod, fixed `instance_count=1` |
| Combine job (root + anchor → serving artifact) | `lambda-0` | One-off per new model version | ~5 min CPU job |
| Workspace + data assets + ACR | Azure (East/West Europe region) | 24/7 | Storage + registry baseline |
| Backend service (FastAPI) | Developer laptop / lab VM | When running locally | Negligible |

The single biggest decision driving cost is that **all compute runs on
`lambda-0`, the lab's Kubernetes cluster, instead of Azure managed VMs**.
This is not free (it consumes the lab's CPU/RAM/GPU capacity), but it
incurs **zero Azure billing** for compute. Azure costs are limited to
the workspace's storage, container registry, and small management
overhead.

---

## 2. Compute cost estimate

### 2.1 Inference (online endpoint)

| Configuration | Where | Hourly cost | Monthly (24/7) | Per-request | Notes |
|---|---|---|---|---|---|
| **Current: K8s on lambda-0, `instance_type=defaultinstancetype` (CPU)** | Local hardware | €0 | **€0** | ~45-50 s | After flip-merge disabled (see §5.1) |
| Hypothetical: Azure Managed `Standard_DS3_v2` (4 vCPU CPU) | Azure | €0.196/h | ~€143 | ~85-90 s | Original setup before lambda-0 migration |
| Hypothetical: Azure Managed `Standard_NC6s_v3` (1× V100 GPU) | Azure | €2.51/h | ~€1830 | ~5-10 s | Never deployed; see §3.1 |

The endpoint pod is **always on**, regardless of request volume
(see §6.1 for why). On lambda-0 this consumes lab resources permanently
but is fully covered by sunk-cost campus infrastructure. **Azure cost
savings vs. the original managed deployment: ~€143/month, ~€1700/year.**

### 2.2 Training pipeline

| Configuration | Where | Hourly cost | Per-run cost | Yearly cost (weekly + ~10 ad-hoc/year) |
|---|---|---|---|---|
| **Current: K8s on lambda-0 GPU pod** | Local hardware | €0 | €0 | **€0** |
| Hypothetical: Azure Managed `Standard_NC6s_v3` (V100) | Azure | €2.51/h | ~€7.50 (3 sweep trials × 1 h) | ~€470 |
| Hypothetical: Azure Managed `Standard_NC24ads_A100_v4` (A100) | Azure | €4.10/h | ~€8 (~2 h total) | ~€500 |

A weekly retrain pipeline doing batch-size sweep (3 trials) is ~3 GPU
hours per run. **Azure cost savings vs. equivalent managed GPU
compute: ~€500/year.**

### 2.3 Combine job

Submitted once per new model version (so typically once per training
pipeline run). Runs ~5 min on CPU. On lambda-0 the cost is negligible
(seconds of CPU time). On equivalent managed compute it would be
~€0.02/run.

---

## 3. Storage and other Azure costs

These are the only line items that cannot be moved off Azure to the
lab.

| Item | Pricing | Estimated monthly cost |
|---|---|---|
| **Workspace blob storage**: data assets (`cv5_train`, `cv5_val`, `cv5_test`, `cv5_patches:N`) and pipeline artifacts. Approx 5 GB sources + 5-10 GB patches per registered version | €0.02/GB-month (LRS Hot) | ~€0.30 to €1.00 |
| **Azure Container Registry**: Docker images for the inference environment (~3 GB compressed) and training environment (~5 GB compressed) | €5/month (Basic tier) | ~€5.00 |
| **Azure ML metadata + job logs** | Free tier covers small teams | €0 |
| **Azure Arc connected K8s** | Free for clusters ≤ 6 vCPU | €0 |
| **Outbound bandwidth**: model artifact downloads, container image pulls | €0.08/GB after first 5 GB free | ~€1.00 |

**Total monthly Azure spend: ~€7-10.** This is what the workspace
costs even if we send zero requests and never train. Effectively a
"keep the lights on" fee.

### 3.1 Data asset versioning storage growth

Every run of `submit_preprocessing_pipeline.py` registers a new
version of `cv5_patches` (1 → 2 → 3 …). Each version holds ~30-50k
PNG patches, ~3-5 GB of storage. **If left unchecked, registering
weekly for a year adds ~250 GB ≈ €5/month per year of accrual.** See
§6.3 for the cleanup strategy.

---

## 4. Total cost summary

| Scenario | Compute | Storage + Azure overhead | **Total monthly** |
|---|---|---|---|
| **Current architecture (lambda-0 + minimal Azure)** | €0 | ~€7-10 | **~€7-10** |
| Original architecture (Azure Managed VMs for endpoint + training) | ~€150 endpoint + ~€40 amortised training | ~€7-10 | **~€200** |
| Worst case (GPU endpoint + Managed training, no lambda-0) | ~€1800 endpoint + ~€40 training | ~€7-10 | **~€1850** |

**Switching to lambda-0 for compute reduced monthly Azure spend from
~€200 to ~€8, a 96% reduction.**

---

## 5. Performance vs. cost trade-offs

### 5.1 Inference flip-merge disabled

[`scripts/score.py:208-241`](../scripts/score.py#L208-L241) defines
`_install_horizontal_flip_merge()`, which runs the root and anchor
models **twice** per request (once on the original image, once on a
horizontally flipped copy) and merges via elementwise max to
symmetrize a spatial bias.

| | Flip-merge ON (`DISABLE_FLIP_MERGE=0`) | Flip-merge OFF (`DISABLE_FLIP_MERGE=1`, current) |
|---|---|---|
| Latency / request | ~90 s | ~45-50 s |
| CPU work | 2× model passes | 1× model pass |
| Quality | Marginally better on early models | Same on current model (bias fixed by `extract_petri_dish` cropping) |
| Decision | n/a | **Disabled** (see [`deploy_endpoint.py:299-308`](../scripts/deploy_endpoint.py#L299-L308)) |

Halving inference time effectively halves the CPU cost of every
request. Quality regression was checked by comparing the output mask
for `test_image_02.png` before and after: identical 5/5 plant
detections with similar confidences.

```python
# scripts/deploy_endpoint.py: env var that controls this
environment_variables={
    "MODEL_VERSION": f"{ROOT_MODEL_NAME}-v{serving_version}",
    "DISABLE_DISH_CROP": "0",
    # Horizontal flip-merge in score.py runs the root + anchor models
    # TWICE per request. Disabling halves latency for no quality regression.
    "DISABLE_FLIP_MERGE": "1",
},
```

### 5.2 Inference on CPU instead of GPU

A GPU inference pod would cut per-request time from ~45 s to ~5 s but
would reserve an entire GPU 24/7, either an Azure billing line of
~€1800/month or a permanent block on the lab's shared GPU pool.

| Throughput | CPU pod | GPU pod (always-on) |
|---|---|---|
| 10 req/day | 7 min CPU work / day, ~99.95% idle | Whole GPU reserved, 99.97% idle |
| 1000 req/day | 12 hours work, viable | 1.4 hours work, viable |

For sprint-scale traffic (<100 req/day) the GPU's throughput advantage
is moot. CPU is the right choice.

### 5.3 Training on GPU

In contrast, the training pipeline benefits massively from GPU. A
typical sweep trial on CPU would take days; on `lambda-0`'s GPU it
takes ~30-90 min. The asymmetry between training (continuous, GPU-bound
math) and inference (sporadic, single-image) is the reason they use
different compute types.

### 5.4 Always-on endpoint vs. on-demand

The endpoint pod runs continuously. The alternative (delete the
deployment when not in use and redeploy before each demo) would save
~1 vCPU + 2 GB RAM on lambda-0 between sprints but costs ~10 min of
redeploy time per reactivation. Trade-off accepted in favour of
always-available demos during sprint review.

### 5.5 Single instance, no autoscaling

`INSTANCE_COUNT = 1` in [`deploy_endpoint.py:90`](../scripts/deploy_endpoint.py#L90).
The endpoint runs exactly one pod regardless of load. If traffic
genuinely scaled to 100+ requests/minute, latency would queue up
(`max_concurrent_requests_per_instance=1` in
[`deploy_endpoint.py:307`](../scripts/deploy_endpoint.py#L307) means
synchronous serving). For the current request volume this is fine;
autoscaling would only burn more lab resources for no real benefit.

---

## 6. CPU and GPU resource management: implementation

This section walks through the actual code that controls what hardware
each pipeline step uses and how much of it.

### 6.1 Inference uses a CPU instance type

[`scripts/deploy_endpoint.py:89`](../scripts/deploy_endpoint.py#L89):

```python
# Kubernetes "instance type": a named CPU/memory/GPU profile defined on the
# K8s cluster's azureml extension (NOT an Azure VM SKU). The cluster admin
# defines these. "defaultinstancetype" exists on every Azure ML K8s extension
# install. If lambda-0 has a custom instance type with more RAM/CPU set up
# (e.g. "cpu-4", "gpu"), change this constant.
INSTANCE_TYPE = "defaultinstancetype"
INSTANCE_COUNT = 1
```

This name maps inside lambda-0 to a CPU-only resource request
(typically ~1-2 vCPU, 2-4 GB RAM). Importantly, it **does not** request
GPU resources, so the inference pod is scheduled on a CPU node of the
cluster, leaving GPU nodes free for training.

### 6.2 Training uses a GPU instance type

[`scripts/submit_pipeline.py:42`](../scripts/submit_pipeline.py#L42):

```python
COMPUTE_NAME = "lambda-0"
INSTANCE_TYPE = "gpu"  # K8s instance type that requests 1 GPU
```

The `"gpu"` instance type is a separate K8s profile that requests
exactly one GPU plus the surrounding CPU/RAM. The sweep step requests
this profile for each batch-size trial:

```python
# scripts/submit_pipeline.py:194-195 (build_pipeline)
sweep_step.compute = COMPUTE_NAME
sweep_step.resources = JobResourceConfiguration(instance_type=INSTANCE_TYPE)
```

`max_concurrent_trials=1` in
[`submit_pipeline.py:191`](../scripts/submit_pipeline.py#L191) prevents
the sweep from grabbing multiple GPUs simultaneously, which would
either starve other team members or fail to schedule.

### 6.3 Storage growth control

`get_next_version()` in
[`scripts/preprocess_dataset.py:72-89`](../scripts/preprocess_dataset.py#L72-L89)
auto-increments the data asset version each run:

```python
def get_next_version(client: MLClient, name: str) -> str:
    try:
        existing = list(client.data.list(name=name))
    except Exception:
        return "1"
    if not existing:
        return "1"
    max_v = 0
    for d in existing:
        if d.version and d.version.isdigit():
            max_v = max(max_v, int(d.version))
    return str(max_v + 1) if max_v > 0 else "1"
```

This is good for **traceability** but bad for **storage cost** if left
running indefinitely. Cleanup strategy: archive versions older than 4
weeks via `az ml data archive` (does not delete blob storage but removes
the asset from listings; full delete needs a separate
`az ml data delete`).

### 6.4 Combine-job caching avoids unnecessary work

[`scripts/deploy_endpoint.py:185-189`](../scripts/deploy_endpoint.py#L185-L189):

```python
try:
    existing = client.models.get(name=SERVING_MODEL_NAME, version=root_version)
    print(f"Serving model exists    : {SERVING_MODEL_NAME} v{existing.version} (skip combine)")
    return existing.version
except Exception:
    pass
```

If the combined root+anchor artifact already exists for the requested
model version, the deploy script skips the 5-min combine job entirely.
This saves ~5 min × 50 redeploys/year = ~4 hours of lambda-0 CPU per
year.

### 6.5 Probe timing tuned to avoid unnecessary restarts

Inference startup loads two .keras models (~60 s combined). Default K8s
probe timing (1 s timeout, 3 failures → restart) would kill the pod
mid-startup. Setting:

```python
# scripts/deploy_endpoint.py:316-317
liveness_probe=ProbeSettings(failure_threshold=30, timeout=2, period=10, initial_delay=10),
readiness_probe=ProbeSettings(failure_threshold=30, timeout=10, period=10, initial_delay=10),
```

…gives the pod ~5 minutes to come up cleanly. Without this, pods would
crash-loop, K8s would restart them indefinitely, and the cluster would
do nothing but burn CPU pulling and re-initialising containers.

---

## 7. Monitoring strategy for cost control

### 7.1 Pod-level resource usage

```powershell
kubectl top pods -n azureml -l ml.azure.com/endpoint-name=root-detection-endpoint
```

Shows live CPU + memory consumption of the endpoint pod. Used to
verify the pod isn't running hot (a sign of memory leak or runaway
inference). Baseline: ~150 m CPU when idle, ~1500 m CPU during a
request, ~2 GB RAM steady.

### 7.2 Per-request latency in container logs

[`scripts/get_logs.py`](../scripts/get_logs.py) pulls the endpoint's
container logs through the Azure ML SDK. The per-request DIAGNOSTIC
lines from `score.py` include processing time, anchor counts, per-zone
pixel activations, and the "fixed-threshold" decision. Sudden growth
in processing time would indicate either a model regression or a
resource bottleneck, both cost-relevant.

### 7.3 Azure ML Studio metrics

For training pipeline runs, each child sweep trial's Metrics tab shows
per-epoch `loss`, `val_loss`, `f1_score`, `val_f1_score`. The
`MlflowEpochMetricsCallback` in
[`train_root_detection.py`](../scripts/train_root_detection.py)
logs these explicitly so they render as line charts. This lets us
detect a regression early (e.g. F1 stagnating across epochs → kill the
job, save compute hours).

### 7.4 Azure Cost Management portal

```powershell
# Project the month's spend across the buas-y2 resource group
az consumption usage list \
  --start-date 2026-06-01 --end-date 2026-06-30 \
  --query "[?contains(instanceName,'cv5')].{day:usageStart, sku:meterName, eur:pretaxCost}" \
  --output table
```

Run weekly during the sprint to confirm the Azure spend doesn't
unexpectedly spike (e.g. someone accidentally re-creates a Managed
endpoint instead of a Kubernetes one).

### 7.5 Conditional auto-deploy from the pipeline

[`scripts/submit_pipeline.py`](../scripts/submit_pipeline.py) wires a
deploy step that only redeploys the endpoint *if a new model version
was registered this run*. The check is in
[`scripts/deploy_endpoint.py:343-352`](../scripts/deploy_endpoint.py#L343-L352):

```python
if args.registered_version_file:
    pipeline_version = resolve_root_version_from_pipeline(args.registered_version_file)
    if pipeline_version is None:
        return    # ← cheap no-op, no redeploy
    args.root_version = pipeline_version
```

This prevents weekly pipelines that didn't beat the F1 threshold from
spinning up new K8s pods unnecessarily. **Saves ~5-10 min of cluster
churn on weeks the model didn't improve**, and avoids the brief
double-pod resource use during a rolling update.

### 7.6 Scaling strategy (future-proofing)

The current deployment runs `INSTANCE_COUNT = 1`. If demand grew, the
two scaling levers available are:

1. **Vertical**: bump `INSTANCE_TYPE` to a richer K8s profile
   (e.g. `cpu-4` with 4 vCPU / 8 GB RAM instead of `defaultinstancetype`).
   ~3-4× throughput per pod, no extra Azure cost.
2. **Horizontal**: increase `INSTANCE_COUNT` to 2+ and let Azure ML's
   `azureml-fe` router load-balance. Doubles the lab resources used,
   adds redundancy.

Both can be enabled with a single-line change in `deploy_endpoint.py`
and a redeploy. Neither has been enabled by default because current
traffic doesn't justify it.

---

## 8. Summary

| Lever | Action taken | Savings |
|---|---|---|
| Compute target | Switched endpoint + training from Azure Managed VMs to lambda-0 K8s | ~€2200/year |
| Inference latency | Disabled flip-merge after dish-cropping fix made it redundant | 50% reduction in CPU work per request |
| Inference SKU | CPU instead of GPU for endpoint | Avoided GPU reservation, freed cluster headroom |
| Combine job | Cached by serving-model version | ~4 h/year of lambda-0 CPU |
| Probe tuning | Generous failure thresholds | Prevented crash-loop CPU waste |
| Deploy step in pipeline | Skips when no new model registered | Avoided unnecessary K8s churn |

**Final monthly cost: ~€7-10**, dominated by ACR (Docker registry)
and blob storage, both unavoidable for an Azure ML workspace. Compute
is fully covered by the lab-owned `lambda-0` cluster.

The same architecture would cost ~€200/month with everything on Azure
Managed compute. The chosen design represents a **96 % cost reduction**
without giving up any of the MLOps capabilities (versioning, scheduling,
rolling deploys, blue/green slots, lineage tracking).
