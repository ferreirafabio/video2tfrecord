# Lessons Learned

## Algorithm Design

### Two-Phase Filtering for Merge Decisions
**Problem**: Mixing ambiguity detection with frequency gates caused subtle bugs. A candidate with low `share_vs_t` would be the only one passing both gates, so it auto-merged (1 candidate = no ambiguity check), even though multiple real candidates existed.

**Solution**: Separate into two distinct phases:
1. **Phase 1 (Ambiguity)**: Filter candidates by count >= 5. Among those, check if the top candidate dominates (`top_count / sum_of_all_candidate_counts`).
2. **Phase 2 (Frequency)**: Only AFTER a winner is selected, check `share_vs_t` (winner_count / single_token_count) and high-frequency guards.

**Key insight**: `top_share` for ambiguity uses `top_count / sum(candidate_counts)`, NOT `top_count / t_count`. These are fundamentally different denominators measuring different things.

### Don't Mix Test Concerns
When two tests have overlapping data structures (e.g., same candidate names) but different expected outcomes, they can create contradictions. Each test should be self-contained with unique names so its expected behavior is unambiguous.

Example: Test 7 (ambiguity blocks "Maxwell") and Test 20 (suffix handling allows "Smith") both had a second candidate with count=8. Resolved by lowering Test 20's competitor below the count>=5 gate.

### SLURM File-List Distribution
Instead of scanning directories to find unprocessed files (fragile, slow on 300k+ files), use `extract_from_list.sh` which reads a pre-built file list. More flexible: can target reprocessing, specific datasets, or triage lists. Pattern: `triage_reprocess.py` generates list → `extract_from_list.sh` processes it.

## Text Processing

### Punctuation in Name Matching
OCR output often has trailing punctuation on names: "Epstein," or "Epstein." or "Epstein's". The `effective_last_token()` function must:
1. Strip name suffixes (Jr., Sr., III, etc.) first
2. Strip trailing punctuation (`.,:;!?`)
3. Strip possessives (`'s`)

Order matters — suffixes are checked before punctuation stripping.

### Unicode Confusables Are a Policy Decision
Cyrillic 'е' (U+0435) looks identical to Latin 'e' (U+0065) but they're different codepoints. We deliberately do NOT normalize these because:
- OCR is unlikely to produce Cyrillic in English documents
- False positive matches are worse than false negatives
- Adding Unicode normalization adds complexity with minimal benefit

### Token Budgeting for LLM Batches
**Problem**: Fixed `max_new_tokens` causes JSON truncation on large batches. Small fixed values waste capacity on small batches.

**Solution**: Compute output budget dynamically from expected output size: `tokens_per_item * batch_size + overhead`. Cap at model max (2048 for Qwen 32B). If output budget would exceed cap, reduce batch size rather than raising ceiling. Use real tokenizer when available, fall back to `len(text) // 4 * 1.5` estimate.

**Half-batch retry**: If >50% of names in a classifier batch default to `mentioned`, re-run at half batch size. Context pressure was too high — smaller batch gives LLM more room per item.

## Database

### ON CONFLICT DO UPDATE...RETURNING for Batch Upserts
When batch-inserting rows that may already exist and you need the IDs back, use `ON CONFLICT (key) DO UPDATE SET key=EXCLUDED.key RETURNING key, id`. Using `DO NOTHING` won't return existing rows, so you'd miss IDs for conflicts. The dummy `SET key=EXCLUDED.key` is a no-op that forces RETURNING to work.

### Cross-File DB Batching Pattern
**Problem**: Per-file SQL commits cause ~98k round trips for a 500-file import.

**Solution**: Accumulate tuples across files in lists, flush to DB every N files with `execute_values(page_size=5000)`. Commit per flush, not per file. Write a progress log (one line per flushed file) for resume support via `--start-after`. Reduced ~98k SQL round trips to ~200 (490x improvement).

### Separate Pure INSERT vs Upsert Verification
For `--verify` checks, only assert `cursor.rowcount == len(rows)` on pure INSERT tables (no ON CONFLICT). For upsert tables, rowcount doesn't match prepared rows due to conflicts — only assert no exceptions.

### Shell Escaping with Special Characters
The DB password `w@7bGMivi7eEzHWu9!R8R` contains `@` and `!` which break bash expansion. Solutions:
- **Never**: `DB_PASSWORD='...' python script.py` (bash will mangle it)
- **Always**: Set via Python `os.environ` or use `scripts/db/db_connection.py`
- For SQL: Write password to a file, pipe to psql

### Always Use full_import.py for Database Imports
**Never** run `update_database.py` directly for production imports. It only handles name insertion — summaries, metadata, URLs, and grouping are all skipped. Use `full_import.py` which orchestrates all 8 steps in the correct order. For backfilling without re-importing names, use `--reprocess-missing --apply`.

### Bulk Update via COPY + Temp Table (not execute_values)
**Problem**: `execute_values()` with UPDATE-FROM-VALUES generates a single SQL string. For 33K rows with URLs, this string is megabytes — parsing and planning it takes longer than the actual UPDATE. Over a remote connection (75ms latency), even chunked execute_values at 5K rows per chunk timed out at 120s.

**Solution**: COPY into a temp table, then UPDATE JOIN:
```python
cursor.execute("CREATE TEMP TABLE _bulk (file_id TEXT, url TEXT) ON COMMIT DROP")
buf = io.StringIO()
for fid, url in rows:
    buf.write(f"{fid}\t{url}\n")
buf.seek(0)
cursor.copy_from(buf, "_bulk", columns=("file_id", "url"))
cursor.execute("""
    UPDATE source_documents AS sd SET hosted_url = _bulk.url
    FROM _bulk WHERE sd.file_id = _bulk.file_id
""")
```
COPY uses the binary COPY protocol (streaming, no SQL parsing). The UPDATE JOIN uses the index on `file_id`. Result: 33K rows in 1.6s (was timing out at 120s+).

**Shared helper**: `bulk_update()` in `db_connection.py` wraps this pattern. Handles None→\N escaping, tab/newline sanitization, and dynamic column definitions.

**Zombie prevention**: Added `statement_timeout=300000` (5 min) and `connect_timeout=10` to `get_db_connection()`. Timed-out Python scripts previously left active UPDATE queries holding row locks — new connections would block waiting for locks, creating cascading zombies. The statement_timeout ensures the DB kills orphaned queries automatically.

**Key rule**: NEVER use `execute_values()` for >1K rows over remote connections. Use COPY + temp table instead. `execute_values()` is fine for small batches (<500 rows) or localhost connections.

### Shared DB Connection Module
Before: 8 scripts each had their own `get_db_connection()` with different env var names (DB_*, PG*, POSTGRES_*).
After: Single `scripts/db/db_connection.py` with waterfall priority: `DATABASE_URL` > `DB_*` > `PG*` > `POSTGRES_*` > defaults.

### Schema Changes Need Migration Files
When removing columns (like `confidence`/`verified` from `name_entities`), always create a migration SQL file AND update all INSERT statements in Python code. Stale INSERT lists cause runtime errors.

## Code Organization

### Shared Constants Prevent Drift
`ROLE_PRIORITY` was duplicated in `clean_names.py` and `generate_summaries.py`. Now lives in `scripts/shared/constants.py`. Any constant used in 2+ files should be extracted.

### pyproject.toml for Test Configuration
Adding `[tool.pytest.ini_options]` with `pythonpath = ["."]` and `testpaths = ["tests"]` eliminates the need for `sys.path.insert(0, ...)` hacks at the top of every test file.

### Monitoring Consolidation
9 separate bash/python monitoring scripts → single `monitor.py` with subcommands. Python is better than bash for: counting files, JSON parsing, table formatting, email composition. Keep the SLURM orchestrator (`pipeline_monitor.sh`) separate — it actively submits jobs, not just monitors.

### Percentage Calculation: Avoid +1 Denominator Hack
**Problem**: `$((X * 100 / (Y + 1)))` to avoid division by zero causes 525/525 = 99% instead of 100%.
**Solution**: Use ternary: `$(( Y > 0 ? X * 100 / Y : 0 ))`.

### DOJ Summaries Are 100% Complete
The 89 DOJ docs without summaries have no extractable text (scanned images, redacted pages). They are not "remaining" — they are unsummarizable. Don't report them as incomplete.

## SSH / Remote Access

### KISLURM SSH Access
**NEVER** use `ssh kisbat1` — the `Host kisbat1` entry has a commented-out ProxyJump so it can't resolve.
**ALWAYS** use: `ssh kis3bat1.rz.ki.privat`
The wildcard rule `Host kis3bat*.rz.ki.privat` has `ProxyJump aadlogin.informatik.uni-freiburg.de` configured.
User alias: `kis3bat1ssh="ssh kis3bat1.rz.ki.privat"`

### KISLURM Data Paths
- Raw PDFs: `data/raw/doj/dataset-{N}/*.pdf` (in subdirectories, NOT flat)
- Names: `data/processed/names_v2/{dataset}/*_names.json`
- Summaries: `data/processed/summaries/{dataset}/*_summary.json`
- Working directory: `/work/dlclarge1/ferreira-involved/aretheyinvolved`

## Pipeline Tracking

### Pipeline Version Marker
Tag every `_names.json` with `pipeline_version` (git short SHA). Enables triage: find files produced by buggy code version AND matching affected patterns. Makes reprocessing auditable — you can diff output from two pipeline versions on the same input.

### Document Processing Status
A document is "processed" if its output file exists. No output = not done. Track explicitly:
- **Text extraction**: `data/processed/text/{dataset}/{file_id}.json` exists → done
- **NER names**: `data/processed/names_v2/{dataset}/{file_id}_names.json` exists → done
- **Summaries**: `data/processed/summaries/{dataset}/{file_id}_summary.json` exists → done
- **Skipped**: Raw PDF exists but no text JSON → not yet processed (or pipeline error)

### sys.path in scripts/extraction/ and scripts/db/
Scripts at `scripts/extraction/*.py` and `scripts/db/*.py` need `parent.parent.parent` (3 levels) to reach the project root. Using `parent.parent` (2 levels) resolves to `scripts/` which breaks `from src.xxx import ...`.

### SLURM Script Paths
SLURM scripts in `scripts/slurm/` must reference extraction scripts at `scripts/extraction/extract_text.py` (not `scripts/extract_text.py`). All scripts were moved to `scripts/extraction/` subdirectory.

### Bug Triage Pattern
Scan output JSON for patterns (e.g., `pipeline_version`, `extraction_method`) to generate targeted file lists for reprocessing. Example: find all files produced by buggy v2 pipeline → generate list → feed to `extract_from_list.sh`.

### Extraction Method Distribution
The `extraction_method` field in text JSON predicts compute needs: `pdf_text` = CPU-only (fast), `ocr_scanned`/`ocr` = GPU (slow). DOJ is 87% CPU, House is 100% GPU. Use this to estimate SLURM job duration and GPU budget.

## R2 Upload Policy

### Only Upload Text-Like Documents for DOJ
**Rationale**: The website is a name search engine. Photos without extracted text have no names, so they'll never appear in search results. Hosting them on R2 wastes storage.

**Rule**: Always use `--text-only` flag for DOJ uploads. A document is "text_like" if its `full_text` is >= 50 characters.

**R2 Directory Structure**: Mirrors local: `doj/dataset-4/EFTA00005705.pdf` (not flat `doj/EFTA00005705.pdf`).

**Usage**:
```bash
python scripts/db/upload_to_r2.py --dataset doj --sub-dataset dataset-4 --text-only --apply
```

## LLM Summary Generation

### Cross-Document Batching Design
**Problem**: Processing documents one-by-one (MAP→REDUCE→next) wastes GPU cycles. vLLM's continuous batching is most efficient with many concurrent prompts.

**Solution**: Collect prompts from 8 documents at a time:
1. Phase 1: Prepare all docs (load text, names, pages)
2. Phase 2: Collect MAP prompts from all multi-page docs with routing metadata `(doc_idx, page_idx)`
3. Phase 3: Batch MAP call via `_chunked_batch()` (chunks of 64 for vLLM)
4. Phase 4: Route MAP responses back to correct docs, build REDUCE prompts
5. Phase 5: Batch REDUCE call (short + long per doc = 2*N prompts)
6. Phase 6: Write outputs

**Key insight**: Single-page docs skip MAP entirely and go straight to REDUCE with direct prompts.

### Token-Budget-Aware Chunking
**Problem**: Fixed chunk size (count-based) can cause OOM when some prompts are huge (97-page doc summaries).

**Solution**: `_build_token_aware_chunks()` splits by BOTH prompt count (max 64) AND cumulative token estimate (max 96k). Uses `len(text) // 4` as token estimate. This prevents KV cache pressure on large documents.

### Hierarchical Reduce for Oversized Documents
**Problem**: 97-page police reports produce REDUCE prompts exceeding `MAX_MODEL_TOKENS` (32k).

**Solution**: Split page summaries into groups, reduce each group separately, then feed condensed group summaries into final REDUCE. Also catches vLLM `ValueError` for prompts exceeding `max_model_len` and falls back to truncation.

### Per-Item Retry for Failed Prompts
**Problem**: Whole-batch retry wastes GPU on the 95% that succeeded.

**Solution**: Track failed/empty responses by index. Retry only those individually (max 2 attempts). Fill remaining failures with empty strings.

### vLLM Guarantees Positional Correspondence
vLLM's synchronous `llm.generate()` returns responses in the same order as input prompts. No need for request_id keying. Assert `len(responses) == len(prompts)` as a safety check.

### House-Oversight Documents Are Always Single-Page
All 33,228 house-oversight files are individual OCR'd JPG images. They have `text` and `full_text` fields but no `pages` list. This means MAP phase is never used for house-oversight — only REDUCE with direct prompts.

### KISLURM Email Infrastructure
- `mail.informatik.uni-freiburg.de` does NOT resolve from KI cluster DNS
- `smtp.uni-freiburg.de` resolves but requires SMTP AUTH (rejected without credentials)
- **Nullmailer/sendmail silently fails**: `sendmail` returns exit 0 but queues to `/var/spool/nullmailer/failed/` — never delivered. Spent hours debugging because it looked like it worked.
- **Working solution**: `smtplib` via `smtp.informatik.uni-freiburg.de` (port 25, no auth needed)
- **Recipient constraint**: Only `@informatik.uni-freiburg.de` addresses accepted by this relay
- **Email format**: Send as HTML with `<pre>` monospace block — plain text tables get scrambled by email clients using proportional fonts

### Bash Octal Trap in Cron Scripts
**Problem**: `MINUTE=$(date +%M)` returns zero-padded values ("08", "09"). Bash `(( ))` interprets leading-zero numbers as octal, causing `((: 08: value too great for base` errors at minutes 08 and 09.
**Solution**: Force base-10: `MINUTE=$((10#$(date +%M)))` or use `date +%-M` (no padding).

### SLURM Memory for vLLM
Qwen3-32B-AWQ on H200 uses ~19GB model weights + KV cache scales to available VRAM. Need `--mem=80G` minimum. Old 48G was for transformers fallback (sequential inference, no KV cache).

### Per-Worker clean_names in Summary Pipeline
**Problem**: Running `clean_names.py` on the whole directory (850K files) before summaries means every SLURM worker redundantly cleans all files.
**Solution**: `generate_summaries.py` now auto-cleans only the files each worker will process. Uses `process_files(file_list)` instead of `process_directory(dir)`. Each worker determines its text file chunk first, then cleans only those names files missing `_names_clean.json`. No separate `clean_names.py` step needed before summaries.

### Qwen3 Think Block Defense-in-Depth
**Problem**: Qwen3's `<think>...</think>` blocks leak into output even with `enable_thinking=False` in chat template — especially via `generate_raw()` which bypasses the chat template entirely.
**Solution**: Both backends (transformers + vLLM) strip `<think>...</think>` from ALL output via `_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)`. Applied universally in `generate()`, `generate_batch()`, `generate_raw()`, and `generate_raw_batch()`.

## Multi-Tier Pipeline

### SLURM Jobs Don't Refresh Directories
**Problem**: Expected SLURM array jobs to dynamically discover new files during execution.
**Reality**: Each job discovers files once at startup and distributes them statically across workers (block distribution). New files added later are invisible.
**Solution**: Submit new jobs when new data lands. Scripts pre-filter files with existing output, so resubmission is safe (idempotent).

### ID Ranges ≠ Actual File Counts
**Problem**: Assumed DS10 had 942K files based on EFTA ID range (EFTA01262782-EFTA02205654).
**Reality**: Only 504,031 actual files in the zip (unzip -l reported 504,539 including directories). Many IDs in the range have no corresponding file.
**Lesson**: Always check `unzip -l file.zip | tail -5` for actual file count. Don't trust ID arithmetic.

### Split Pipeline Validation: Expect LLM Non-Determinism
Running the same pipeline twice (monolithic vs split) produces ~84% exact match on names. The 16% mismatches are:
- LLM role classification variance (sender vs mentioned vs recipient) — ~75% of mismatches
- NER span boundary differences (XLM-R aggregation non-determinism) — ~15%
- LLM correctly dropping OCR garbage in one run but not the other — ~10%

This is expected and acceptable. The split pipeline (`--ner-only` → `--from-ner`) is functionally correct.

### vLLM for Both NER Tier 2 and Summaries
Both `extract_names_v2.py` (via `--llm-backend vllm`) and `generate_summaries.py` use vLLM. The backend abstraction in `src/ner/llm_backend/` (ABC base, factory, transformers + vLLM implementations) makes this transparent. AWQ+Marlin quantization (~19GB) gives 4.4x token throughput vs BitsAndBytes 4-bit (~32GB). Default for NER stays `transformers` for backwards compat.

### Inter-Document Batching for Tier 2 (Stages 2 & 4)
**Problem**: Tier 2 processes files one at a time through the LLM. Each `generate()` call handles a single document's prompt. On H200 (141GB VRAM, Qwen2.5-32B at ~18GB), the GPU is severely underutilized.

**Solution**: `generate_batch()` in `shared_model.py` — left-padded batched generation with OOM fallback. `classify_spans_batch()` and `repair_names_batch()` collect prompts across K documents and batch the `generate()` call.

**Generation-budget bucketing**: HF `generate()` runs ALL sequences in a batch until the LONGEST hits EOS or max_new_tokens. Sorting prompts by `max_new_tokens` and chunking into groups of `inter_batch_size` ensures each chunk has a tight budget. A 10-span doc (400 tokens) never shares a batch with an 80-span doc (2500 tokens). True outliers naturally fall into their own small chunk — no hard threshold needed.

**Golden comparison split**: Comparing batch output vs baseline must separate Golden-A (semantic: normalized_name+role, tests batching contract) from Golden-B (offset/evidence, known to vary from LLM non-determinism). Only Golden-A gates the exit code.

**Validation results**: batch=1 vs batch=8 = 100% match. batch=8 vs golden = 99.7% (1000 files, 3 mismatches are improvements). Throughput: ~200 files/min with batch=8 on H200.

**Stderr buffering on NFS**: Python logging stderr appears frozen for minutes on KISLURM. Jobs ARE processing even when stderr doesn't update. Don't cancel jobs just because stderr looks stalled — check output file counts instead.

### Block Distribution vs Work-Stealing
Current: Static block distribution (worker N gets files[start:end]). If worker finishes early, its GPU sits idle.
Trade-off: For 500K+ files at 4-10s each, workers finish within ~10% of each other. The simplicity outweighs the 5-10% idle time vs. implementing a file-lock work-stealing queue.

### Lazy Imports for Optional GPU Dependencies
**Problem**: `src/extractors/__init__.py` imported `OCRExtractor` unconditionally, which imports `LightOnOcrForConditionalGeneration` from transformers. On RTX 2080 nodes where the transformers version doesn't have LightOnOCR, this crashes even when `--no-ocr` flag is used (import happens at module load time).
**Solution**: Use `try/except ImportError` in `__init__.py` for optional extractors (OCR, Audio). The `TextExtractor` class already lazily instantiates them, so `None` is fine when the dependency is missing.
**Lesson**: When a module has heavy GPU dependencies, always make imports conditional at the package level. Don't assume all nodes have the same Python environment.

### SLURM afterok vs afterany Dependencies
**Problem**: `afterok` requires ALL array tasks to exit 0. Worker 7 processed all files successfully but crashed during `torch.cuda.empty_cache()` cleanup, giving exit code 1. This blocked the entire downstream Tier 2 chain.
**Solution**: Use `afterany` for dependencies where the downstream job can handle partial upstream failures (Tier 2 skips missing NER files). Reserve `afterok` only when ALL upstream outputs are strictly required.

### unzip File Count vs Expected Count
**Problem**: `unzip -l` output includes directory entries and non-file entries. DS10 reported 504,539 entries but only 504,031 actual files were extracted, causing the watcher script to loop forever.
**Lesson**: When writing file-count watchers, use `>=` threshold with a margin, or poll for completion by checking if the unzip process itself has exited rather than counting files.

### House-Oversight Files Are Named DOJ-OGR-*
Not EFTA*. When checking SLURM logs to identify DOJ vs House jobs, look for `DOJ-OGR-` = House Oversight, `EFTA` = DOJ datasets.

## KISLURM Virtual Environment

### KISLURM .venv/bin/activate (FIXED)
**Problem**: `.venv/bin/activate` had a hardcoded macOS path (`/Users/fabioferreira/.../.venv`), prepending the wrong directory to `PATH`. Jobs silently used system Python instead of venv packages.

**Root cause fix**: Corrected the `VIRTUAL_ENV` path in `.venv/bin/activate` to `/work/dlclarge1/ferreira-involved/olmo3/.venv`. `source .venv/bin/activate` now works correctly on KISLURM.

**Manual workaround** (no longer required, remains in scripts as safe fallback):
```bash
ln -sf /usr/bin/python3 .venv/bin/python 2>/dev/null || true
ln -sf /usr/bin/python3 .venv/bin/python3 2>/dev/null || true
export VIRTUAL_ENV="$WORK_DIR/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="$WORK_DIR"
```

**Updated scripts** (as of 2026-02-09): `slurm_summaries.sh`, `slurm_ner_tier1.sh`, `slurm_ner_tier2.sh`, `cpu_postprocess.sh`, `cpu_extract_text.sh`, `pipeline_monitor.sh`, `extract_from_list.sh` -- these use the manual workaround but it is now redundant.

## Legacy Code Cleanup

### Verify Before Deleting
Before deleting any file, grep for all imports/references across the codebase. Files that seem unused may still be imported by test files, monitoring scripts, or SLURM jobs.

Checklist:
1. `grep -r "import <module>"` across all Python files
2. Check SLURM scripts that reference the file
3. Check monitoring scripts (observe_jobs.sh, auto_fix_jobs.py)
4. Check test files
5. Check CLAUDE.md and other docs

### Update All References After Deletion
After deleting legacy files, update:
- `__init__.py` files (remove stale imports)
- CLAUDE.md (directory structure, key scripts tables)
- Memory files (MEMORY.md, topic-specific .md files)
- Monitoring scripts (they reference extraction scripts by path)
- SLURM scripts (they reference pipeline scripts)
