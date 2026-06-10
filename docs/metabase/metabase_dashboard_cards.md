# Metabase Dashboard — Build Sheet

Native SQL questions for the CV5 monitoring dashboard, against the **cv5-db** connection.
Each card: suggested name, visualization type, and SQL.

**Workflow per card:** `+ New → SQL query` → select **cv5-db** → paste SQL → **Run** →
pick the visualization → **Save** (use the name below). Pin all saved cards to one
dashboard at the end.

**Notes / verify first**
- Plant counts use `num_detected` (actual detections), **not** `total_plants` (fixed
  slot capacity, always ~5). `total_plants` is used only as the denominator in the
  detection-rate card.
- The jsonb extraction is wrapped in a `jsonb_typeof = 'number'` guard so a malformed
  `output_summary` can never error the aggregate at query time.
- Avoid the jsonb `?` existence operator in Metabase native SQL — it clashes with the
  JDBC `?` placeholder. Use `(labels ->> 'epoch') IS NOT NULL` instead.
- Verify literals against your data: `SELECT DISTINCT status FROM annotations;` and
  `SELECT DISTINCT status FROM model_versions;` — these queries assume `'annotated'`
  and `'deployed'`.

---

## Section 1 — Operational (from `inference_audit_log`)

### 1.1 Total requests — *Number*
```sql
SELECT count(*) AS total_requests
FROM inference_audit_log;
```

### 1.2 Failed requests — *Number*
```sql
SELECT count(*) FILTER (WHERE error IS NOT NULL) AS failed_requests
FROM inference_audit_log;
```

### 1.3 Error rate % — *Number* (or Gauge)
```sql
SELECT round(
         100.0 * count(*) FILTER (WHERE error IS NOT NULL)
         / nullif(count(*), 0), 2
       ) AS error_rate_pct
FROM inference_audit_log;
```

### 1.4 Latency — three separate *Number* cards

Predictions run on CPU, so latency is often tens of thousands of ms. Each card formats
its own value as a text string: `>= 1000` shows seconds, else milliseconds. Output is
text (e.g. `12.34 s` / `847.0 ms`), which renders fine as a single-value KPI tile but
can't be used in numeric charts or with Metabase's own unit formatting.

#### 1.4a Avg latency — *Number*
```sql
SELECT CASE
         WHEN avg(latency_ms) >= 1000
         THEN round((avg(latency_ms) / 1000.0)::numeric, 2) || ' s'
         ELSE round(avg(latency_ms)::numeric, 1) || ' ms'
       END AS avg_latency
FROM inference_audit_log;
```

#### 1.4b p50 latency — *Number*
```sql
SELECT CASE
         WHEN percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) >= 1000
         THEN round((percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) / 1000.0)::numeric, 2) || ' s'
         ELSE round(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)::numeric, 1) || ' ms'
       END AS p50_latency
FROM inference_audit_log;
```

#### 1.4c p95 latency — *Number*
```sql
SELECT CASE
         WHEN percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) >= 1000
         THEN round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) / 1000.0)::numeric, 2) || ' s'
         ELSE round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 1) || ' ms'
       END AS p95_latency
FROM inference_audit_log;
```

### 1.5 Requests over time — *Bar* or *Line*
```sql
SELECT date_trunc('day', logged_at) AS day,
       count(*)                      AS requests
FROM inference_audit_log
GROUP BY 1
ORDER BY 1;
```

### 1.6 Avg latency over time (s) — *Line*
```sql
SELECT date_trunc('hour', logged_at) AS bucket,
       round((avg(latency_ms) / 1000.0)::numeric, 2) AS avg_s
FROM inference_audit_log
GROUP BY 1
ORDER BY 1;
```
> In seconds for a consistent axis unit. Name the card "Avg latency over time (s)".

### 1.7 Per-model breakdown — *Table*
```sql
SELECT
  coalesce(model_name, 'unknown') AS model_name,
  count(*)                        AS requests,
  round(100.0 * count(*) FILTER (WHERE error IS NOT NULL)
        / nullif(count(*), 0), 2) AS error_rate_pct,
  round(avg(latency_ms)::numeric, 1) AS avg_latency_ms,
  coalesce(sum(
    CASE WHEN jsonb_typeof(output_summary -> 'num_detected') = 'number'
         THEN (output_summary ->> 'num_detected')::int ELSE 0 END
  ) FILTER (WHERE error IS NULL), 0) AS plants_detected
FROM inference_audit_log
GROUP BY 1
ORDER BY requests DESC;
```

---

## Section 2 — Business / plant detection (from `inference_audit_log`)

### 2.1 Total plants detected — *Number*
```sql
SELECT coalesce(sum(
         CASE WHEN jsonb_typeof(output_summary -> 'num_detected') = 'number'
              THEN (output_summary ->> 'num_detected')::int ELSE 0 END
       ), 0) AS total_plants_detected
FROM inference_audit_log
WHERE error IS NULL;
```

### 2.2 Avg plants per request — *Number*
```sql
SELECT round(
         coalesce(sum(
           CASE WHEN jsonb_typeof(output_summary -> 'num_detected') = 'number'
                THEN (output_summary ->> 'num_detected')::int ELSE 0 END
         ), 0)::numeric / nullif(count(*), 0), 2
       ) AS avg_plants_per_request
FROM inference_audit_log
WHERE error IS NULL;
```

### 2.3 Detection rate % (detected / available slots) — *Number* or *Gauge*
```sql
SELECT round(
         100.0 * sum(
           CASE WHEN jsonb_typeof(output_summary -> 'num_detected') = 'number'
                THEN (output_summary ->> 'num_detected')::int ELSE 0 END
         )::numeric
         / nullif(sum(
           CASE WHEN jsonb_typeof(output_summary -> 'total_plants') = 'number'
                THEN (output_summary ->> 'total_plants')::int ELSE 0 END
         ), 0), 2
       ) AS detection_rate_pct
FROM inference_audit_log
WHERE error IS NULL;
```

### 2.4 Plants detected over time — *Line* or *Bar*
```sql
SELECT date_trunc('day', logged_at) AS day,
       coalesce(sum(
         CASE WHEN jsonb_typeof(output_summary -> 'num_detected') = 'number'
              THEN (output_summary ->> 'num_detected')::int ELSE 0 END
       ), 0) AS plants_detected
FROM inference_audit_log
WHERE error IS NULL
GROUP BY 1
ORDER BY 1;
```

### 2.5 Sample distribution by species — *Bar* or *Pie* (from `annotations`)
```sql
SELECT species,
       count(*) AS samples
FROM annotations
WHERE species IS NOT NULL
  AND status = 'annotated'      -- verify literal
GROUP BY species
ORDER BY samples DESC;
```

---

## Section 3 — Model quality (from `monitoring_metrics`)

### 3.1 Latest validation F1 (%) — *Number* or *Gauge*
```sql
SELECT round((metric_value * 100)::numeric, 2) AS f1_validation_pct
FROM monitoring_metrics
WHERE metric_name = 'f1_validation'
ORDER BY recorded_at DESC
LIMIT 1;
```
> Scope to the deployed model instead (matches the API): add
> `AND model_version_id = (SELECT id FROM model_versions WHERE status = 'deployed' LIMIT 1)`.

### 3.2 Latest throughput (img/s) — *Number*
```sql
SELECT round(metric_value::numeric, 2) AS throughput_img_per_s
FROM monitoring_metrics
WHERE metric_name = 'throughput_img_per_s'
ORDER BY recorded_at DESC
LIMIT 1;
```

### 3.3 F1 over training epochs — *Line*
```sql
SELECT (labels ->> 'epoch')::int::text       AS epoch,
       round((metric_value * 100)::numeric, 2) AS f1_score
FROM monitoring_metrics
WHERE metric_name = 'f1_validation'
  AND (labels ->> 'epoch') IS NOT NULL
  AND model_version_id = (SELECT id FROM model_versions WHERE status = 'deployed' LIMIT 1)
ORDER BY (labels ->> 'epoch')::int;
```
> `epoch` cast to text so Metabase treats it as a category (X-axis), not a measure.
> `model_version_id` scoping prevents Metabase summing same-epoch rows across models.
> f1 scaled ×100 to show as a percentage (e.g. 27.0, not 0.27).
> To compare models instead, drop the `model_version_id` filter, add
> `coalesce(labels ->> 'run_id', model_version_id::text) AS run` and set it as the
> chart's series breakout.

---

## Section 4 — System resources (from `inference_audit_log`)

Resource columns added per the system-metrics ticket (all numeric, nullable):
`cpu_pct_system`, `cpu_pct_process`, `cpu_count`, `ram_used_mb`, `ram_available_mb`,
`gpu_pct`, `vram_used_mb`, `disk_free_mb`. Every card filters `IS NOT NULL` on its
metric so CPU-only rows (null GPU/VRAM) don't get averaged in as zeros. Cards stay blank
until inferences run through the updated logging path.

### 4.1 CPU utilization (system %) — *Number* or *Gauge*
```sql
SELECT round(avg(cpu_pct_system)::numeric, 1) AS cpu_pct_system
FROM inference_audit_log
WHERE cpu_pct_system IS NOT NULL;
```

### 4.2 RAM used (GB) — *Number*
```sql
SELECT round((avg(ram_used_mb) / 1024.0)::numeric, 2) AS ram_used_gb
FROM inference_audit_log
WHERE ram_used_mb IS NOT NULL;
```
> Pair with `ram_available_mb` for a used/total gauge if wanted.

### 4.3 GPU utilization (%) — *Number* or *Gauge*
```sql
SELECT round(avg(gpu_pct)::numeric, 1) AS gpu_pct
FROM inference_audit_log
WHERE gpu_pct IS NOT NULL;
```
> No rows on CPU-only data — card shows blank, the honest signal that no GPU was used.

### 4.4 VRAM used (GB) — *Number*
```sql
SELECT round((avg(vram_used_mb) / 1024.0)::numeric, 2) AS vram_used_gb
FROM inference_audit_log
WHERE vram_used_mb IS NOT NULL;
```

### 4.5 Disk free (GB) — *Number*
```sql
SELECT round((min(disk_free_mb) / 1024.0)::numeric, 2) AS disk_free_gb
FROM inference_audit_log
WHERE disk_free_mb IS NOT NULL;
```
> `min` not `avg` — for free space the worst (lowest) point is what matters.

### 4.6 CPU & RAM over time — *Line* (two series)
```sql
SELECT date_trunc('hour', logged_at)                  AS bucket,
       round(avg(cpu_pct_system)::numeric, 1)         AS cpu_pct,
       round((avg(ram_used_mb) / 1024.0)::numeric, 2) AS ram_used_gb
FROM inference_audit_log
WHERE cpu_pct_system IS NOT NULL
GROUP BY 1
ORDER BY 1;
```
> Two units (% vs GB) — in **Settings → Data** put `ram_used_gb` on a right Y-axis.

### 4.7 Latency vs CPU load — *Scatter*
```sql
SELECT round((latency_ms / 1000.0)::numeric, 2) AS latency_s,
       cpu_pct_system
FROM inference_audit_log
WHERE error IS NULL
  AND latency_ms IS NOT NULL
  AND cpu_pct_system IS NOT NULL;
```
> Tests the ticket's hypothesis that resource pressure drives high CPU latency — an
> up-and-to-the-right cloud confirms correlation.

---

## Optional — global dashboard date filter

To mirror the endpoint's `since` parameter, add an optional field/variable filter to the
time-based cards (1.5, 1.6, 2.4, 4.6) and wire a single dashboard date filter to all of them:

```sql
-- example: append to a WHERE clause
[[ AND logged_at >= {{since}} ]]
```
Mark `{{since}}` as a **Field Filter** (type: Date) or a plain optional Date variable, then
on the dashboard add a Date filter and connect it to each card's `since`.