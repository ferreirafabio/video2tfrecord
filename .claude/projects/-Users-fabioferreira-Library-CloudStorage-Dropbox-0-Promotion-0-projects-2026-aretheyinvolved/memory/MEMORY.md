# Project Memory

## Infrastructure

### Servers
- **Hetzner**: Production server (web app, PostgreSQL database)
  - **URL**: https://aretheyinvolved.com (Cloudflare DNS/SSL)
  - SSH: `ssh -i ~/.ssh/aretheyinvolved root@128.140.41.232`
  - Deploy: `./scripts/deploy/deploy.sh` (one-command with validation)
  - **Stack**: nginx (port 80) → Next.js (port 3000) → PostgreSQL
- **KISLURM**: Compute cluster with GPUs for heavy lifting (LLM extraction, OCR)
  - SSH: `ssh kis3bat1.rz.ki.privat` (ProxyJump via aadlogin configured in ~/.ssh/config)
  - **NEVER** use `ssh kisbat1` — the Host entry has commented-out ProxyJump
  - **DB Connectivity**: KISLURM CANNOT directly reach Hetzner (128.140.41.232). Must tunnel through aadlogin (university network). Create tunnel: `ssh -f -N -L 45432:128.140.41.232:5432 -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes aadlogin`. DB scripts then connect to `localhost:45432` on KISLURM. See `finalize_dataset.py` for automation.

### Dropbox Backups
- **App folder**: `Apps/aretheyinvolved/`
- **DB backups**: `backups/db/` (7-day retention, 3 AM nightly)
- **Documents**: `documents/{dataset}/`
- **Scripts**: `scripts/backup/`
- **Full docs**: `docs/BACKUP.md`

## Scripts Directory Structure

```
scripts/
├── backup/             # Dropbox backups
├── db/                 # Database ops + migrations/
│   └── db_connection.py  # Shared DB connection + bulk_update() helper
├── deploy/             # deploy.sh
├── download/           # Data acquisition
├── extraction/         # NER + name extraction
├── monitoring/         # Consolidated: monitor.py (unified CLI)
├── shared/             # Shared constants (ROLE_PRIORITY, etc.)
└── slurm/              # KISLURM jobs + pipeline_monitor.sh (orchestrator)
```

## Role Classifications (5 roles, priority order)
1. `sender` - Sender/author (highest priority)
2. `recipient` - Recipient (includes CC)
3. `passenger` - Flight log passenger
4. `mentioned` - Named in document text
5. `other` - Role unclear (lowest priority)

## Name Extraction Pipeline v2 (6-Stage)

See CLAUDE.md for full details. Key points:
- Stage 0: Deterministic cleaning (same-length)
- Stage 1: XLM-R NER (high recall)
- Stage 1.5: LLM Recovery (gated for noisy docs)
- Stage 2: LLM Classifier
- Stage 3: Hard Validator (no hallucinations)
- Stage 4: LLM Repair (OCR fixes)

### Post-Processing: JSON-Level Cleaning
`scripts/extraction/clean_names.py` reads raw `_names.json` → writes `_names_clean.json`.
- `process_directory(dir)` — cleans ALL files in a directory (for standalone use)
- `process_files(file_list)` — cleans specific files only (for per-worker use in SLURM)
- **Auto-integrated into `generate_summaries.py`**: Each worker cleans only its chunk before generating summaries. No separate step needed.
- All downstream scripts (`update_database.py`, `generate_summaries.py`) prefer `_names_clean.json`.

### Key Scripts
- `scripts/extraction/extract_names_v2.py` - v2 pipeline CLI
- `scripts/slurm/slurm_extract_v2.sh` - SLURM array job
- `scripts/extraction/validate_extractions.py` - Verify no hallucinations
- `scripts/extraction/clean_names.py` - JSON post-processing (clean, filter, dedup)
- `scripts/extraction/generate_summaries.py` - LLM summaries with grounded names
- `scripts/db/full_import.py` - Single-command pipeline (8 steps, nothing forgotten)
- `scripts/db/collect_metadata.py` - Collect file sizes/page counts (run on KISLURM, import JSON to DB)

### Dropbox App Credentials
- **App key**: `556v1afekl4nhxe` (old app with documents)
- App folder: `Apps/aretheyinvolved/`
- **IMPORTANT**: New app (`c8eubnlur6rv6qi`) has a SEPARATE folder — documents are in the OLD app only

## Bulk DB Imports (COPY + Temp Table Pattern)
- `db_connection.py` provides `bulk_update()` — COPY into temp table + UPDATE JOIN. ~1000x faster than per-row UPDATEs over SSH tunnel.
- **All 5 import scripts use it**: `import_r2_urls.py`, `import_dropbox_urls.py`, `import_gdrive_urls.py`, `import_summaries.py`, `collect_metadata.py`
- `get_db_connection()` has `statement_timeout=300s` + `connect_timeout=10s` to prevent zombie connections
- `collect_metadata.py` supports `--force` to overwrite existing values (default: COALESCE preserves existing)
- **Performance**: 33K rows in ~2s (was ~41 min per script). Full pipeline: 7.5s total.
- **NEVER use `execute_values()` for >1K rows over remote connections** — generates huge SQL strings that timeout. Use COPY instead.
- See `memory/lessons.md` "Bulk Update via COPY" section for details.

## Name Deduplication
- `name_entities` table groups related names (UNIQUE on `entity_name`, no confidence/verified cols)
- `group_names.py` has two algorithms: trigram grouping + single-token last-name merging
- **Batch SQL**: ~600 individual calls → ~6 (batch entity INSERT+UPDATE, bulk co-occurrence pre-compute)
- Run `scripts/db/group_names.py --apply` after new data import
- See `memory/NAME_DEDUPLICATION.md` for algorithm details

## Frontend Security
- **API auth**: Set `API_KEYS=key1,key2` in `.env`
- **Rate limiting**: 100 req/min per API key
- **URL validation**: `getSafeUrl()` blocks `javascript:` URLs

## Deployment
**Use**: `./scripts/deploy/deploy.sh`

The script:
1. Runs type-check locally
2. Builds locally to verify
3. Rsyncs to server
4. Builds and deploys container
5. Runs health checks

**If API returns 0 names**: Database password mismatch. See CLAUDE.md for fix.

### CRITICAL: DB Password Shell Escaping
**NEVER** pass DB password via shell env vars (`DB_PASSWORD='...' python`). The password has `!` and `@` which break bash.
**ALWAYS** use Python wrapper: `python3 -c "import os, subprocess; os.environ['DB_PASSWORD']='w@7bGMivi7eEzHWu9!R8R'; subprocess.run(...)"`

## Monitoring
- `scripts/monitoring/monitor.py` — unified CLI (9 scripts → 1)
  - Subcommands: progress, jobs, diagnose, email, all
  - `monitor_cron.sh` — cron wrapper (every minute, diagnose every 5 min)
  - **Email**: Uses `smtplib` via `smtp.informatik.uni-freiburg.de` (port 25, no auth). HTML `<pre>` for monospace tables. Recipients must be `@informatik.uni-freiburg.de`. Nullmailer silently fails — don't use.
- `scripts/monitoring/measure_efficiency.py` — compare pipeline performance across versions
- Keep `scripts/slurm/pipeline_monitor.sh` separate (orchestrator, not monitor)

## Summary Generation (Cross-Document Batching, v5)
- `scripts/extraction/generate_summaries.py` — MAP→MULTI with cross-doc batching via vLLM (PROMPT_VERSION="v5")
- **vLLM v0.15.1**, Qwen3-32B-AWQ on KISLURM — 3-5x speedup over transformers
- **Per-worker clean_names**: Each SLURM worker auto-runs `clean_names` on only its chunk of files before generating summaries. No need to run `clean_names.py` separately before summaries. Uses `process_files()` (not `process_directory()`).
- **Cross-doc batching**: 8 docs → batch MAP (chunks of 64, `MAX_BATCH_TOKENS=96_000`) → route back → batch MULTI
- **Hierarchical reduce**: 97+ page docs split into groups, reduce each, then final reduce
- **Per-item retry**: Failed prompts retried individually (max 2), not whole-batch
- **House-Oversight**: Single-page OCR'd JPGs — SINGLE-only (no MAP)
- **SLURM**: `sbatch --array=0-N scripts/slurm/slurm_summaries.sh [dataset]` (80G mem)

## Multi-Tier NER Pipeline
- **RTX 2080** (Tier 1): Stage 0 (deterministic clean) + Stage 1 (XLM-R NER). Output: `_ner.json`. Partition: `alldlc_gpu-rtx2080`, 12G mem.
- **H200** (Tier 2): Stages 1.5-4 (LLM recovery + classify + validate + repair). Input: `_ner.json`. Output: `_names.json`. Partition: `alldlc2_gpu-h200`, 24G mem.
- **CLI flags**: `--ner-only` (Tier 1), `--from-ner` (Tier 2), `--files-from` (file list), `--ner-batch-size` (default 32)
- **TIER2_SKIP_GATE**: Skip Tier 2 if `clean_text_length < 50 AND person_spans == 0`. Escape hatch: never skip if `clean_text_length >= 200`.
- **SLURM scripts**: `slurm_ner_tier1.sh` (RTX 2080), `slurm_ner_tier2.sh` (H200), `cpu_extract_text.sh`, `cpu_postprocess.sh`
- **RTX 2080 limitations**: No bf16, no OCR (LightOnOCR can't run). XLM-R uses fp16 via `torch_dtype` param.
- **Hardware capability contract**: RTX 2080 = ONLY XLM-R NER + CPU work. H200/A100 = REQUIRED for Qwen 32B, vLLM, OCR. Runtime guards: `shared_model._validate_gpu()` (compute >= 8.0), SLURM scripts check at startup.
- **Rerun script**: `scripts/slurm/rerun_from_list.sh` — two-phase (NER-only → from-ner --force --inter-batch-size 8) from file list
- **Tests**: `tests/test_classifier_truncation_fix.py` — 53 tests for token budget, batch splitting, JSON repair, diagnostics, GPU validation. All CPU-only, 0.06s.

### Classifier Output Truncation Fix (2026-02-09)
- `TOKENS_PER_SPAN_OUTPUT=50` (was 30), `MAX_OUTPUT_TOKENS=4096` (was 2048). Removed `indent=2` from prompt JSON. Added `_try_repair_json()`.

### NER Manifest (2026-02-09)
- `_ner_manifest.jsonl` — index for instant lookup of NER stats. Builder: `discover_pending.py build-manifest`. Query: `discover_pending.py query --min-spans 25`.

### Tier 1 Optimization (2026-02-08)
- `discover_pending.py` replaces bash skip-gate. fp16/batch32, `--files-from` flag. 850K files in ~25 min.

## Dataset Stats
- **DOJ**: 17,664 text files (87% pdf_text/CPU, 11% GPU). 53% >500 chars, 23% <50 chars (photos).
- **House-Oversight**: 33,228 text files (100% GPU OCR). ~90% >500 chars.
- **DS1-DS12**: All verified (except DS9 — unavailable). 346,551 raw PDFs on KISLURM.
- **DS10**: 504,031 actual files (ID range EFTA01262782-EFTA02205654, 79GB zip). Fully extracted 2026-02-07.
- **DS11**: 331,661 actual files (ID range EFTA02205655-EFTA02730264, 26GB zip). Fully extracted.
- **DS10/DS11 strategy**: DS11 first (smaller, validates pipeline), then DS10. Stagger: DS10 CPU+RTX2080 while DS11 H200 runs.
- **H200 capacity**: 4 nodes x 8 GPUs = 32 H200 GPUs total (partition `alldlc2_gpu-h200`, nodes dlc2gpu18-21). **Cap: 24 workers max** (user-imposed). RTX 2080 unlimited.
- **Adding workers**: Submit a new array job — `discover_pending.py` finds remaining files idempotently. Not elastic; each job discovers files independently at startup. Small duplicate work possible but no corruption (atomic writes).

## DOJ URL Maintenance
Links break when DOJ changes URLs. Migration: `scripts/db/migrations/fix_doj_urls.sql`

## House Oversight Google Drive
- **Public folder**: `https://drive.google.com/drive/folders/1TrGxDGQLDLZu1vvvZDBAh-e7wN3y6Hoz`
- **Scripts**: `scripts/db/build_gdrive_mapping.py`, `scripts/db/import_gdrive_urls.py`

## Future Tasks
- **Improve clean_names.py garbage filter**: Misses repeated-letter patterns, common English words, truncated fragments, preposition+name, place/org names. Worst: EFTA00007157 (101 clean, ~30 garbage). Re-run after fixing.
- **House-Oversight R2 upload + Dropbox backup**: Use `finalize_dataset.py --upload-r2 --upload-dropbox --apply` for single-command flow. Or run individual scripts on KISLURM.
- **Frontend expandable grouped view** for name entities

## Inter-Document Batching (Tier 2)
- `generate_batch()` in `shared_model.py` — left-padded batched generation, OOM fallback
- `classify_spans_batch()` / `repair_names_batch()` — inter-doc batching for Stages 2 & 4
- **Generation-budget bucketing**: Sort prompts by `max_new_tokens`, chunk into groups. Each chunk's budget = max within chunk. Prevents 10-span docs sharing batch with 80-span outliers.
- **CLI**: `--inter-batch-size 8` (default 1 = backward compat)
- **Validation**: batch=1 vs batch=8 = 100% match (5 files). batch=8 vs golden = 99.7% (1000 files, 3 mismatches are improvements).
- **Throughput**: ~200 files/min with batch=8 on H200 (vs ~0.5/min batch=1). Caveat: 97% of first-1000 files skip-gated; LLM-processed: ~8 files/min.
- **Golden comparison**: Split into Golden-A (semantic: normalized_name+role) and Golden-B (full: +start,end,original_text). Golden-A gates the exit code; Golden-B is informational.
- **HF generate() limitation**: ALL sequences run until longest EOS or max_new_tokens. Bucketing by output budget prevents waste.
- **Stderr buffering on NFS**: Python logging appears frozen for minutes. Jobs are processing even when stderr doesn't update.

## Key Learnings (see memory/lessons.md for full details)
- **Always use full_import.py**: Never run `update_database.py` alone
- **Shell escaping**: Never pass DB password via shell env vars; use Python `os.environ`
- **KISLURM→Hetzner tunnel**: `ssh -f -N -L 45432:128.140.41.232:5432 aadlogin`. `.venv/bin/python3` has psycopg2; system python3 does NOT.
- **Token budgeting**: Cap input 24k, output from JSON size, half-batch retry on >50% defaults
- **Pipeline version tracking**: `pipeline_version` (git SHA) in output JSON for triage/audit
- **House-Oversight**: Single-page OCR'd JPGs named `DOJ-OGR-*` (not EFTA*). SINGLE-only (no MAP). Frontend is dataset-agnostic.
- **Occupation extraction (v5)**: PROMPT_VERSION="v5". LONG prompts return `document_type`/`document_date`. `DOCUMENT_TYPES` in `scripts/shared/constants.py`.
- **Split pipeline non-determinism**: 84% exact match vs monolithic — LLM variance, not bugs.
- **vLLM backend abstraction**: `src/ner/llm_backend/` (ABC, factory, transformers + vLLM). `extract_names_v2.py --llm-backend vllm` for NER Tier 2. `generate_summaries.py` always uses vLLM. AWQ model: `Qwen/Qwen3-32B-AWQ` (~19GB). Tests: `tests/test_vllm_backend.py` (40 tests).
- **Qwen3 thinking mode**: `enable_thinking=False` in `_apply_chat_template()`. Both backends strip `<think>...</think>` from ALL output via `_THINK_RE` regex as defense-in-depth.
- **Per-worker clean_names**: `generate_summaries.py` auto-cleans only worker's chunk via `process_files()`. No separate `clean_names.py` step needed before summaries.
- **SLURM jobs don't refresh dirs**: Submit new jobs for new files. Scripts are idempotent.
- **ID ranges ≠ file counts**: DS10 has 942K range but 504K actual files. Always verify.
