# llama.cpp tuning knobs for 4× RTX 3080 20GB (no P2P), Qwen3 27B Q8_0, build 0.4.0-dev / 304665f

**Research date:** 2026-09-24 (system clock). **Staleness rule applied:** anything before ~2026-03-24 is flagged.

**Method note / access limits.** github.com HTML and git clone are blocked from this host, and the GitHub REST API hit its
unauthenticated 60/h limit mid-session. `raw.githubusercontent.com` works, so the **primary evidence below is the actual
source I downloaded and read**, at both `master` and the pinned commit `304665f`:
- `common/arg.cpp`, `common/common.h`, `common/common.cpp`, `common/fit.cpp` (master + 304665f)
- `src/llama-context.cpp` (master + 304665f), `src/llama-graph.cpp`, `src/llama-model.cpp`, `src/llama-kv-cache.cpp`
- `ggml/CMakeLists.txt`, `ggml/src/ggml-cuda/CMakeLists.txt` (master + 304665f), `ggml/src/ggml-cuda/fattn.cu`,
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-backend.cpp`, `docs/build.md`
- GitHub **search API** (worked): issue/PR titles + bodies, used for measured numbers and dates.

Discussion #18074 and #23262 bodies could **not** be fetched (GitHub Discussions are GraphQL-only and GraphQL returned
`limit: 0`). Anything about them below is title-level + mechanism inference, not quoted measurement.

Local source mirror kept at `C:\Users\<user>\lcpp-research\` if the parent wants to re-grep.

---

## Priority 1 — the three things that actually matter on this rig

### A. `--fit off` is not optional for you; it is mandatory with `--split-mode tensor` (correctness, not speed)

*(No effect on prefill or decode — it prevents an abort.)*

- `common/common.h:476` (both **master** and **304665f**): `bool fit_params = true;` — **fit is ON by default** in current
  llama.cpp. The help string is generated from that value (`common/arg.cpp:2863-2876`):
  `-fit/--fit [on|off]` — *"whether to adjust unset arguments to fit in device memory (default: 'on')"*.
- So `--fit off` is **not** "the neutral setting" — you are explicitly turning *off* a default-on allocator that would
  otherwise try to re-solve your `-ngl`/`-ts`/`-c`.
- Parent-verified from `common/fit.cpp`: it throws `llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`, and it
  aborts if `n_gpu_layers` or `tensor_split` were user-set. You set both (`-ngl -1 -ts 1,1,1,1`).
- Supporting flags exist in `common/arg.cpp:2877-2922`: `-fitp/--fit-print`, `-fitt/--fit-target` (default 1 GiB margin
  **per device** — `common.h:481`), `-fitc/--fit-ctx` (default 4096 — **fit may shrink `-c`**, `common.h:478`).
- **Keep `--fit off`.** Confidence: **high** (source, two revisions). URLs:
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/common.h and `.../common/arg.cpp` (fetched 2026-09-24).
- Side note: `-fit` is confirmed still buggy in adjacent tools — **issue #28970** (2026-09-15, `0.4.1-dev` build 10989)
  `llama-batched-bench` ignores `-fit`; **PR #27265** (2026-08-17) fixes a divide-by-zero in the multi-device fit path.

### B. KV cache quant: for a **fully VRAM-resident** model at 262144 ctx, q8_0 KV is the *only* knob that buys real headroom — and it is not free

- **Memory:** f16 → q8_0 K and V cuts KV bytes/token/layer by ~47%.
- **Requires FA?** Yes, for V. `src/llama-context.cpp:3781-3789` (master): *"quantized V cache requires flash_attn to be
  enabled"*; AUTO silently upgrades to ENABLED (*"enabling flash_attn since it is required for quantized V cache"*), explicit
  `-fa off` → `LLAMA_LOG_ERROR` + `return nullptr` (clean context-init failure, not a crash). Quantized **K** with FA also
  requires `n_embd_head_k % blck_size == 0` (`:3792-3801`). Confidence **high**.
- **Build requirement — you do NOT need `-DGGML_CUDA_FA_ALL_QUANTS` for `q8_0/q8_0`:**
  - Pinned `304665f`, `ggml/src/ggml-cuda/CMakeLists.txt:115-125`: without `GGML_CUDA_FA_ALL_QUANTS`, the compiled vec
    instances are exactly `f16-f16`, `q4_0-q4_0`, **`q8_0-q8_0`**, `bf16-bf16`.
  - `master` makes this explicit: `ggml/CMakeLists.txt:207-208` — `GGML_CUDA_FA_QUANTS` (new cache var) defaults to
    `"q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"`; `GGML_CUDA_FA_ALL_QUANTS` is now a **deprecated alias** for
    `GGML_CUDA_FA_QUANTS=all` (`docs/build.md:303-304`, fetched 2026-09-24; introduced by **PR #28079**, 2026-08-31).
  - Confidence **high**.
- **What happens if the pair is missing from the build (answering "silently fall back / error / refuse"):**
  neither silent nor fatal. `ggml/src/ggml-cuda/fattn.cu:501-513` — logs **once, at WARN**:
  *"no FlashAttention vector kernel compiled for K/V types X-Y, **converting K and V to f16 instead (slow)**. Add \"X-Y\" to
  GGML_CUDA_FA_QUANTS to compile it."* and then runs the f16-f16 vec case. So a build lacking the kernel **works but
  pays a per-call conversion**. Confidence **high** (source). This is the exact string to grep your server log for to prove
  your build has the q8_0-q8_0 path.
- **Is there a dequant cost in the FA path?** Yes, and its shape depends on batch:
  - *Decode* (`Q->ne[1] == 1`), Ampere GA102 (cc 8.6, `turing_mma_available` true), quantized K or V:
    `fattn.cu:649-658` routes to **`BEST_FATTN_KERNEL_VEC`**, i.e. the specialised `q8_0-q8_0` vec kernel that dequantizes
    tiles in-register while reading half the bytes. This is why q8_0 KV is often *near-free or faster* at short context.
    Confidence **high**.
  - *Prefill* (large `Q->ne[1]`): `fattn.cu:664` returns **`BEST_FATTN_KERNEL_MMA_F16`**, and the MMA-f16 kernel is
    f16-K/V. Quantized K/V on the wide path therefore pays a conversion. Confidence **medium-high** (dispatch is
    source-verified; the exact conversion site is inferred from the kernel contract + the vec-path warning).
  - `ggml_cuda_fattn_kv_type_supported()` (`fattn.cu:525-539`) whitelists F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0; anything else
    → `BEST_FATTN_KERNEL_NONE` → generic path.
- **Measured decode cost (from parent's own verified sources, not re-verified here):** SYCL Qwen3.8-27B sweep
  (2026-08-22): q8_0-KV vs f16 decode penalty grows with context — **-2.2% @0, -13.2% @4K, -51% @32K**, prefill within
  **1.5%**; plus a 4×5060Ti / 2×3090 llama.cpp bench measuring **~15% q8-KV decode tax at 255k**.
- **Category:** *memory (yes, large)* + *decode at long ctx (measured penalty)*; prefill ≈ unchanged.
  **Recommendation:** keep `-ctk q8_0 -ctv q8_0` **only if** you need the ctx; at 262144 on 29 GB across 4×20 GB you almost
  certainly do. Verify with `grep "converting K and V to f16"` — absence of that warning = your build is fine.

### C. Flash attention is hard-required by `--split-mode tensor` (already satisfied, but now provable)

- `src/llama-context.cpp:3762-3774` (master, 2026-09-24): if `split_mode == LLAMA_SPLIT_MODE_TENSOR`:
  AUTO → forced ENABLED, logging *"enabling flash_attn since it is required for SPLIT_MODE_TENSOR"*; if explicitly
  `DISABLED` → `LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled")` then `return nullptr`.
- **So in current master it fails cleanly with an error, not a segfault.** The older "segfault" folklore predates this
  guard; I could not date the guard without GitHub blame access. Your `-fa on` already satisfies it. Confidence **high**
  for current behavior; **medium** for the historical segfault claim.
- Also `src/llama-model.cpp:358-359`: `LLAMA_SPLIT_MODE_TENSOR` throws for unsupported architectures.

---

## Priority 2 — measured numbers for `-b` / `-ub`, and the ubatch↔VRAM↔tg coupling

### D. `-b` vs `-ub`: what each controls, and whether either touches decode

*(`-ub`: helps prefill. `-b`: no measurable effect on decode or prefill beyond chunking. Both are decode-neutral.)*

- Exact help strings, `common/arg.cpp` (master, 2026-09-24) — line ~1524 region for `-b`/`-ub`:
  - `-b, --batch-size N` → *"logical maximum batch size"* → `params.n_batch`
  - `-ub, --ubatch-size N` → *"physical maximum batch size"* → `params.n_ubatch`
  - (There is a duplicated `-ub` entry in master, one carrying env `LLAMA_ARG_BATCH` and one `LLAMA_ARG_UBATCH`; the second
    wins for the env var. Cosmetic bug worth noting.)
- Defaults, `common/common.h:451-452` (**identical at `304665f`**): `n_batch = 2048` *"logical batch size for prompt
  processing (must be >=32 to use BLAS)"*, `n_ubatch = 512` *"physical batch size for prompt processing"*.
- Clamping / subdivision:
  - `llama-context.cpp:246-248`: `n_batch = min(n_ctx, n_batch)`; `n_ubatch = min(n_batch, n_ubatch ?: n_batch)` — **ubatch can never exceed batch**.
  - `llama-context.cpp:1782`: `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` — `-b` is a hard cap on tokens per
    `llama_decode()` submission. Exceeding it **aborts**.
  - `llama-context.cpp:1810`: `mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all)` — the *compute* granularity.
  - `src/llama-kv-cache.cpp:702-713`: `llama_kv_cache::init_batch(..., uint32_t n_ubatch, ...)` →
    `balloc.split_simple(n_ubatch)` (single stream) / `split_equal(n_ubatch, ...)` (multi-stream) — **this is the actual
    ubatch slicer**.
  - `llama-context.cpp:1867-2040`: the decode loop iterates ubatches and accumulates `n_tokens_prev += ubatch.n_tokens`.
- **Does `-ub` affect decode? No, mechanically.** A single-token generation step submits a 1-token batch; it becomes
  exactly one ubatch of 1 token regardless of `n_ubatch`. With MTP `--spec-draft-n-max 3` a verify step submits ≤4 tokens;
  with `-np 4` continuous batching, ≤4 tokens. All ≪ 512. **`-ub` is a prefill-only knob.**
- **Does `-b` affect decode? No.** It only bounds how a *prompt* is chunked into decode calls. For decode steps the batch is
  1 (or np / 1+ndraft), far below 2048.
- Confidence **high** on the mechanics (source-read at two revisions). Caveat: `-b` **does** matter in one edge case —
  batched/parallel prompt ingestion where a single submission would exceed `n_batch` (assert), and for `llama-batched-bench`
  style workloads where "logical batch" = concurrency.
- **Measured numbers I could verify:**
  - llama-bench README *"Prompt processing with different batch sizes"* (parent-verified), 7B Q4_0 **CUDA**, pp1024:
    `n_batch 128 → 1436`, `256 → 1932`, `512 → 2254`, `1024 → 2498` t/s. Because `n_ubatch = min(n_batch, 512)` by
    default, that sweep is effectively an **ubatch** sweep up to 512 — and it says **prefill keeps improving with ubatch**
    (+12% from 512→1024). Source: https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench (master, 2026-09-24).
    Confidence **medium-high** (numbers relayed from parent's verification; page itself not re-fetched).
  - **Issue #28760** (created **2026-09-11**) — a production workload remarkably close to yours: *"Qwen3 27B IQ3_S,
    2× CMP 50HX (TU106, 20 GB), `--split-mode tensor --ubatch-size 2048`, FA on, KV q4_0, ctx 512K, MTP on"*. Evidence that
    **ub=2048 is a used, viable setting** on 20 GB tensor-split rigs with MTP. URL:
    https://github.com/ggml-org/llama.cpp/issues/28760
- **`-ub` and the tg regression you asked about (discussion #18074, "Should higher ubatch-size be lowering tg/s in vram
  constrained environments?").** I could **not** fetch the thread (GitHub Discussions unreachable; GraphQL limit 0), so I
  will not quote numbers from it. Mechanism, source-grounded:
  1. Ubatch drives **reserved** compute/scratch graph memory (`llama-context.cpp:1810` + graph reserve), so raising `-ub`
     enlarges the per-device compute buffer.
  2. On a VRAM-constrained rig the extra compute buffer comes out of the **KV cache / layer-offload budget**. With `-c`
     fixed at 262144, a bigger compute buffer can force fewer GPU layers or a smaller KV → decode gets slower even though
     prefill gets faster.
  3. In extreme cases the scheduler spills (`GGML_CUDA_ENABLE_UNIFIED_MEMORY`, or host-resident tensors) and *both* suffer.
  Confidence: **medium** (mechanism = source-verified; the causal claim for #18074 = inference, not a fetched measurement).
- **Recommendation:** sweep `-ub 512 / 1024 / 2048` (and `-b` at ≥ ub, e.g. 2048/4096) measuring **pp and tg together**, at
  your real `-c 262144`. Expect pp to rise then plateau in the 1024–2048 band; **treat any tg drop as a memory squeeze
  signal**, not as an inherent ubatch cost. Keep `-b` ≥ `-ub`.

---

## Priority 3 — CUDA graphs, MMQ/cuBLAS, peer knobs

### E. CUDA graphs: real decode gains, and the MTP + multi-GPU interaction is the risky part

*(Helps decode. Measured deltas exist. Risks are MTP- and multi-process-specific.)*

- Mechanism/source: `ggml/CMakeLists.txt:209` — `option(GGML_CUDA_GRAPHS "ggml: use CUDA graphs (llama.cpp only)" ${GGML_CUDA_GRAPHS_DEFAULT})`;
  lines 117-119 force `GGML_CUDA_GRAPHS_DEFAULT OFF` **only if unset**, so the llama.cpp top-level build is what turns it ON
  (parent-verified). Compile define `GGML_CUDA_USE_GRAPHS` at `ggml/src/ggml-cuda/CMakeLists.txt:123-125`.
- Runtime: `ggml/src/ggml-cuda/ggml-cuda.cu:2595-2604` — `ggml_cuda_graph_update_required()` logs at DEBUG
  `"CUDA Graph id %zu reused"`; the user-visible line is `src/llama-context.cpp:4364`
  `"%s: graphs reused = %10d"` from `data.n_reused` (incremented at `:1415`, surfaced as
  `llama_perf_context_data::n_reused` at `:3424`).
- **Measured deltas:**
  - **PR #27721** (created **2026-08-25**) — enabling graphs on Pascal sm_61: GTX 1080, MoE Ling-3.0-tiny Q6_K
    **41.2 → 57.6 tok/s (+39.8%)**, long-gen **47.7 → 72.2 (+51.4%)**, prefill 857.0 → 853.0 (**-0.5%**, noise);
    dense 9B Q5_K_M 26.3 → 28.0 (+6.4%); llama-bench pp512 **-1.4%**, **tg64 +78%** (1935/41.47 → 1908/73.85).
    URL: https://github.com/ggml-org/llama.cpp/pull/27721 → **decode-only gain, prefill untouched.** Confidence high
    (self-reported bench, dated, with variance).
  - **PR #25749** (2026-07-16, closed/merged) *"Enable CUDA graphs on Volta+Turing"* — author's V100×8 comparison reports
    *"mostly the same in `-sm layer` case, with some **major gains in `-sm tensor`**"*. **This is the closest published
    statement to your tensor-split question, and it is favourable.** URL:
    https://github.com/ggml-org/llama.cpp/pull/25749. Confidence medium (prose, not tabulated).
  - Parent-verified: **PR #28549** *"Enable CUDA graph for MTP draft"* — fixes MTP graph re-capture, **+4–5% decode**.
    URL: https://github.com/ggml-org/llama.cpp/pull/28549.
- **Does graph capture work with `-sm tensor` + NCCL all-reduce?** The only *architectural* gate in master is
  `ggml_cuda_graph_check_compability()` (`ggml-cuda.cu:2557-2589`): it disables graphs solely for `GGML_OP_MUL_MAT_ID`
  when `ggml_cuda_mul_mat_id_needs_sync()` is true (ref **PR #18958**), refined by **PR #26802** (2026-08-09,
  *"only disable CUDA graphs when mul_mat_id actually needs a stream sync"*). There is **no** split-mode gate. PR #25749's
  tensor-split gains corroborate that. Whether the NCCL all-reduce node itself is capture-safe is **not** proven by these
  sources → mark as **speculation, medium confidence**, and note `ncclAllReduce` is capturable on modern CUDA in general.
- **Known failure modes (dated, all relevant to your MTP + multi-GPU config):**
  - **Issue #25835** (2026-07-17, still open as of 2026-09-22): **VRAM leak with CUDA graphs on V100**, model
    *Qwen3.6-27B-UD-Q4_K_XL MTP*, `-ngl 999 --flash-attn on --spec-type ...`; grows until OOM over 5-10 turns;
    **workaround `GGML_CUDA_DISABLE_GRAPHS=1`**. ← nearly your exact shape (27B + MTP + graphs).
    URL: https://github.com/ggml-org/llama.cpp/issues/25835
  - **Issue #28404** (2026-09-04): two co-resident `llama-server` processes (one per GPU) → the second dies right after
    **CUDA-graph reuse** (`ggml-cuda.cu:107`), `GGML_CUDA_DISABLE_GRAPHS=1` fixes it. Relevant if you ever shard by process.
  - **Issue #27330** (2026-08-18): graphs hang the GPU channel (Xid 8) on RTX 5090 Laptop / sm_120; build config shows
    `GGML_CUDA_GRAPHS = ON`. Arch-specific (Blackwell), but shows graphs are not universally safe.
  - **Issue #27009** (2026-08-13, open): *"CUDA graphs for multi-slot decode via shape-stable padded ubatches"* — i.e.
    **graphs currently do not cover variable-shape multi-slot decode**; relevant if you try `-np > 1`.
- **Recommendation:** confirm `USE_GRAPHS = 1` in your startup log (`system_info: CUDA : ... USE_GRAPHS = 1`) and that
  `graphs reused` climbs during decode. GA102 is above the Volta gate, so you get graphs by default. Then, **because you
  run MTP**, keep `GGML_CUDA_DISABLE_GRAPHS=1` in your back pocket and A/B it if you see VRAM creep or an Xid.

### F. `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` for Q8_0 on GA102

*(Both are prefill-oriented; MMQ is already the default tuned path for your arch.)*

- `docs/build.md:301-302` (fetched **2026-09-24**), verbatim:
  - `GGML_CUDA_FORCE_MMQ` (default `false`): *"Force the use of custom matrix multiplication kernels for quantized models
    instead of FP16 cuBLAS **even if there is no int8 tensor core implementation available (affects V100, CDNA and RDNA3+)**.
    **MMQ kernels are enabled by default on GPUs with int8 tensor core support.** With MMQ force enabled, **speed for large
    batch sizes will be worse but VRAM consumption will be lower**."*
  - `GGML_CUDA_FORCE_CUBLAS` (default `false`): *"Force the use of FP16 cuBLAS instead of custom matrix multiplication
    kernels for quantized models. … memory use will be higher. **Prompt processing may become faster on recent datacenter
    GPUs (the custom kernels were tuned primarily for RTX 3000/4000)**."*
- **Interpretation for 4× RTX 3080 (GA102 has int8 tensor cores):** MMQ is **already on by default**, and the docs state
  explicitly that the custom kernels were tuned for **RTX 3000/4000** — your arch. So:
  - `FORCE_MMQ` → **no expected gain**, and it *hurts large-batch (prefill)*; only buys VRAM.
  - `FORCE_CUBLAS` → documented prefill gain is scoped to *datacenter* GPUs; on GA102 you would be leaving the tuned path.
    **Speculation, medium confidence: don't do either; keep build defaults.** No Q8_0-specific measured A/B was found.
- Confidence: **medium** (official doc text + arch reasoning; no direct 3080 measurement located).

### G. `GGML_CUDA_PEER_MAX_BATCH_SIZE` — gone from current source

*(No measurable effect: the knob no longer exists.)*

- **Not present** in `ggml/src/ggml-cuda/common.cuh` nor `ggml/src/ggml-cuda/ggml-cuda.cu` at **either** `master` or
  **`304665f`** (grepped directly). The old batched-peer-copy path is replaced by plain
  `cudaMemcpyPeerAsync` (`ggml-cuda.cu:834` and `:2523`). A GitHub issue/PR search for the identifier returned no
  llama.cpp issue or PR matching it.
- **Current peer knobs instead:**
  - `GGML_CUDA_NO_PEER_COPY` — CMake option `ggml/CMakeLists.txt:203` *"ggml: do not use peer to peer copies"*;
    feature string `NO_PEER_COPY` at `ggml-cuda.cu:5713-5714`.
  - `GGML_CUDA_P2P` — **runtime env var**, `docs/build.md:291-293`: *"can be set to enable peer-to-peer access between
    multiple GPUs … Requires driver support (usually restricted to workstation/datacenter GPUs). May cause crashes or
    corrupted outputs for some motherboards and BIOS settings (e.g. IOMMU)."* Note `ggml-cuda.cu:606` —
    `bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;` (opt-in), and `:608` sets it true on some path
    (worth reading in situ if this matters).
- **For your rig:** since kernel-direct peer writes are broken/2.67 GB/s on this box, **do not set `GGML_CUDA_P2P`**;
  there is nothing left to tune via `PEER_MAX_BATCH_SIZE`. Confidence **high** that it's absent; **medium** on the exact
  removal commit (could not get `git log` — clone blocked).

---

## Priority 4 — complete answers to the remaining questions

### H. `--no-kv-offload` / `--no-op-offload`

- Definitions (source, master 2026-09-24):
  - `common/arg.cpp:2410-2417`: `-kvo/--kv-offload` vs `-nkvo/--no-kv-offload` — *"whether to enable KV cache offloading
    (default: enabled)"*, env `LLAMA_ARG_KV_OFFLOAD`; maps to `cparams.offload_kqv = !params.no_kv_offload`
    (`common/common.cpp`, `common_context_params_to_llama`).
  - `common/arg.cpp:2942-2949`: `--op-offload` vs `--no-op-offload` — *"whether to offload host tensor operations to
    device (default: true)"* → `cparams.op_offload`.
- **`--no-kv-offload` would HURT you, for two independent reasons:**
  1. It **disables scheduler pipeline parallelism** for `-sm layer`. `src/llama-context.cpp:429-434` — `pipeline_parallel`
     requires `n_devices > 1 && n_gpu_layers() > n_layer_all && split_mode == LAYER && **cparams.offload_kqv** && !has_tensor_overrides()`.
     (`-sm tensor` never gets pipeline parallel in the first place.)
  2. With FA + host-resident KV, the FA node acquires copied inputs — exactly the case in **PR #22133** (2026-04-19):
     *"such as a **flash attention node when KV cache is fully offloaded to host memory**"* overflowing
     `GGML_SCHED_MAX_SPLIT_INPUTS` → assertion failure / crash (fixes **issue #21730**).
     URL: https://github.com/ggml-org/llama.cpp/pull/22133
  - **Verdict: keep KV offloaded (default).** Category: *hurts decode* (pipeline parallelism off) and *risks a crash*.
  - Confidence: **high** on mechanics; **medium** on the crash reproducing in your exact config.
- **`--no-op-offload`:** `cparams.op_offload` is threaded into `ggml_backend_sched_new(...)` (`llama-context.cpp:643`, `:682`;
  default `true` at `:3728`). The mechanism is `ggml-backend.cpp:970`:
  `if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer))` — i.e. it
  exists to **copy host-resident tensors up to the GPU so ops can still run on the GPU**. In a fully VRAM-resident config
  there are almost no host tensors, so the flag's effect ≈ 0 while generic docs say it can free a little VRAM.
  **Category: memory-only, expected no measurable effect here. Confidence: low-medium — no measurement found.**
- Related flag you may not know: `--no-host` / `LLAMA_ARG_NO_HOST` (`arg.cpp:2426-2432`) — *"bypass host buffer allowing
  extra buffers to be used"*; `mparams.no_host` (`common.cpp`). Also `GGML_CUDA_NO_VMM` (`ggml/CMakeLists.txt:204`) and
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (`docs/build.md:287`) if you ever need spill-to-RAM instead of OOM.

### I. `-ngl` partial offload vs full offload with no P2P

*(No measured evidence found. This answer is mechanism + speculation, clearly labelled.)*

- Split-mode semantics from the help text (`common/arg.cpp:2803-2808`, master 2026-09-24):
  `-sm layer` = *"split layers and KV across GPUs (pipelined)"*; `-sm row` = *"split weight across GPUs by rows
  (parallelized)"*; `-sm tensor` = *"split weights and KV across GPUs (parallelized, **EXPERIMENTAL**)"*.
- **Layer-split mechanism (the "writes activations to host" you asked about):** consecutive layers live on different GPUs;
  the activation tensor between layer *k* on GPU0 and layer *k+1* on GPU1 must move. With peer access that is a
  device→device copy; **without P2P it is staged through pinned host memory by the scheduler's copy path** (the scheduler
  inserts a copy node because the backends differ). Enabling `pipeline_parallel` (`llama-context.cpp:429-434`) lets the
  host-side copies overlap with compute, which is precisely why your `layer` config reaches prefill 1400 t/s.
- **Tensor-split mechanism:** every weight matrix is sharded column/row-wise across all 4 GPUs, so **every** matmul needs an
  all-reduce (`GGML_OP_ALLREDUCE` / NCCL) before the next op — per layer, per token. Without P2P that all-reduce is
  host-staged. That is the mechanism behind your split: `tensor` prefill **860.7** ≪ `layer` prefill **1400.1** (many small
  all-reduces), while `tensor` decode **78.3** > `layer` decode **52.7** (fewer, larger, better-overlapped collective +
  better SM occupancy per GPU).
- **Does partial `-ngl` ever help with no P2P?** **Speculation (confidence low):** no, not in `layer` mode for a
  *throughput* reason — the pipeline already pays one host staging per GPU boundary, so parking layers on CPU does not
  remove a boundary; it only adds slow CPU matmuls and reduces the GPU work per boundary. In `tensor` mode, partial offload
  is worse still: the CPUs cannot participate in the all-reduce the same way. The one scenario where partial offload helps
  is when full offload **does not fit** (KV + compute buffer OOM) — then a few CPU layers is better than spilling or
  shrinking ctx. I found **no** llama.cpp measurement of partial-vs-full `-ngl` on a PCIe-only multi-GPU rig.
- **Recommendation:** keep `-ngl -1` (all on GPU). Your own numbers already justify it. Spend effort on `-ub` and on the
  tensor-vs-layer choice instead.

### J. `--parallel` / `-np` + `--cont-batching`, and KV sizing

*(`-np > 1`: helps aggregate throughput; changes per-slot context. Marked partly speculation.)*

- Flags (source, master 2026-09-24):
  - `common/arg.cpp:2541-2549` (server) — `-np/--parallel N` = *"number of server slots (default: 1, **-1 = auto**)"*;
    non-server examples get a different help string (*"number of parallel sequences to decode"*). Maps to
    `cparams.n_seq_max = params.n_parallel` (`common/common.cpp`).
  - `common/arg.cpp:2566-2573` — `-cb/--cont-batching` vs `-nocb/--no-cont-batching`, default **enabled**.
- **KV sizing — source-verified pieces:** `common/common.h:573` `bool kv_unified = false;` (default **off**) and
  `:630` `int32_t kv_unified_per_slot = 0;`. `tools/server/server.cpp:156-173`:
  `-np -1` → *"n_parallel is set to auto, using n_parallel = 4 and kv_unified = true"*; and with
  `--kv-unified-per-slot` the pool is sized `n_ctx = n_parallel * kv_unified_per_slot`.
  ⇒ With the **default non-unified** KV, `-c` is the total pool and the per-slot capacity is `n_ctx / n_parallel`
  (parent-verified behaviour; I verified everything around it but not that division line itself). **Total KV VRAM does not
  change with `-np`** — only how it is carved up. Confidence **medium-high**; verify from your own startup log
  (`n_ctx_per_seq`).
  - **Practical warning:** if you set `-np 4` for aggregate throughput, each slot drops to ~65536 tokens at `-c 262144`.
    Use `--kv-unified` (or `-np -1`) if you want dynamic sharing, or `--kv-unified-per-slot`.
- **Effect on throughput:** `-np > 1` batches np tokens into one decode step. On a no-P2P tensor rig, an all-reduce per
  step has a large fixed latency; batching np sequences **amortises** it, which is the mechanism by which aggregate tok/s
  should rise even if per-sequence latency does not improve. **This is mechanism-based, not measured — mark as speculation
  (medium confidence).** No llama.cpp measurement of `-np` aggregate throughput on a no-P2P rig was found.
  - Related, dated: **Issue #27009** (2026-08-13) argues graphs don't cover variable-shape multi-slot decode and proposes
    padded buckets — so `-np > 1` may *reduce* CUDA-graph effectiveness today.
  - Related, dated: **PR #26802** (2026-08-09) benchmarks *parallel decode on a quantized MoE* with
    `llama-batched-bench -npp 64 -ntg 64 -npl 4,16,32` on RTX 5090 — a good template for your own `-np` sweep
    (`llama-batched-bench ... -npl 1,2,4`). URL: https://github.com/ggml-org/llama.cpp/pull/26802
- **Interaction with speculative decoding:** a third-party RCA
  (`docs/evaluations/bug3070-nextn-multislot-serialization.md`, ManniX-ITA/opencoti-llamafile) reports
  *"Qwen NextN self-spec **serialises the verify decode across slots**"* — i.e. MTP/NextN + `-np > 1` can serialise the
  verification step instead of batching it. Third-party, undated in what I read ⇒ **low confidence**, but it is a concrete
  reason to sweep `-np` *with MTP on* rather than assuming linear scaling. Your draft-acceptance numbers (0.587 tensor /
  0.722 layer) come from the spec path, so a `-np` change that serialises verify could cost you more than it gains.

### K. `--load-mode` (`mmap` vs `none` vs `dio`)

*(Load-time and memory-behaviour knob. Expected: no steady-state decode effect for a fully VRAM-resident model.)*

- Exact semantics, `common/arg.cpp:2686-2704` (master 2026-09-24), verbatim option list:
  - `auto` (default): *"mmap, unless a device does not support it"*
  - `none`: *"no special loading mode"*
  - `mmap`: *"memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)"*
  - `mlock`: *"force system to keep model in RAM rather than swapping or compressing"*
  - `mmap+mlock`
  - `dio`: *"use DirectIO if available"*
  - Default constant: `common/common.h:484` `enum llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;` (**same at `304665f`**).
- **Does mmap hurt decode for a 29 GB model?** In a fully-offloaded config the weights are copied into VRAM during load and
  the file is not re-read during decode, so **no steady-state effect** is expected. The `arg.cpp` caveat is about
  **pageouts** during/after load if the page cache cannot hold the mapping (RAM pressure) — which is a *load-time* and
  *RAM-pressure* hazard, not a decode-path cost. The two cases where mmap **does** reach steady state:
  1. **Partial offload** — any CPU-resident weights are read through the mapping on every forward pass; evicted pages then
     become major faults during decode. (Another reason to keep `-ngl -1`.)
  2. `--lazy-mode` (`arg.cpp:2705-2717`): `on/auto/off`, *"on-demand reading of certain tensors … (requires mmap)"* — this
     deliberately reads rows from disk per step for very large tensors; `auto` applies it only to tensors > 4 GiB.
- **Measured on your own box** (from the parent's `p2p-nccl-tensor-results.md`, 2026-09-24): load time
  `-sm layer` **29 s** vs `-sm tensor` + `NCCL_P2P_DISABLE=1` **7.4 s** — so load mode/split interacts strongly with load
  time here; steady-state inference was not isolated.
- **Verdict:** keep `--load-mode mmap`. **Category: load-time/memory only; no expected decode effect.** Confidence
  **medium** (source semantics + reasoning; no isolated measurement found).

### L. `-t 24` / `--threads-batch 48` on an EPYC host with 4 GPUs

*(Expected: no measurable effect on GPU-resident inference. Speculation, medium confidence.)*

- Wiring: `common/common.h:69-70` `cpuparams.n_threads = -1` (auto); mapped in `common/common.cpp`
  (`common_context_params_to_llama`): `cparams.n_threads = params.cpuparams.n_threads;
  cparams.n_threads_batch = params.cpuparams_batch.n_threads == -1 ? params.cpuparams.n_threads : params.cpuparams_batch.n_threads`.
  `common/arg.cpp:1524` is `-tb/--threads-batch`.
- With `-ngl -1` (all layers on GPU) the CPU backends only see: tokenization, sampling, graph construction/scheduling, any
  op the scheduler leaves on CPU, and (if `--no-op-offload`) host-op workloads. The **host staging for a no-P2P
  all-reduce/copy is done by the CUDA copy path and NCCL's own machinery, not by the ggml CPU threadpool**, so raising `-t`
  does not speed it up.
- **A real, source-visible cost of `-t != --threads-batch`:** `common/common.cpp` `common_threadpools::init()` creates a
  **second** threadpool when the two params differ, and *starts the non-batch pool paused*; the code comment cites
  **PR #27138** for the "pools must match n_threads exactly" rule. So `-t 24 -tb 48` allocates two pools — extra memory and
  two sets of threads, no proven gain.
- **Recommendation (speculation):** for a fully GPU-resident model these two numbers are near-irrelevant. If you want them,
  prefer `-t` = physical cores per NUMA node and let `-tb` default, and only tune them if `llama-bench` with `-ngl 0`
  shows you care about CPU-bound work. The one genuine reason to lower them: leave cores free for the host-staging copies /
  NCCL progress threads, so if you see the host CPU pegged during tensor-mode decode, drop `-t`/`-tb` and re-measure.
  Confidence **medium** (source mechanics verified; the "no effect" conclusion is inference, no measurement found).

---

## Summary table (effect categories)

| # | Flag / knob | Exact setting to use | Prefill | Decode | Category | Confidence |
|---|---|---|---|---|---|---|
| 1 | `--fit off` | keep `off` | — | — | correctness (tensor+fit aborts) | high |
| 2 | `-ctk q8_0 -ctv q8_0` | keep (need ctx) | ~unchanged (±1.5%) | penalty grows with ctx (-2%→-51%) | memory + long-ctx decode cost | med-high |
| 3 | `-DGGML_CUDA_FA_ALL_QUANTS` | **not needed** for q8_0/q8_0 | — | — | none | high |
| 4 | `-fa on` | required by `-sm tensor` | — | — | required | high |
| 5 | `-ub` | sweep 512 → 1024 → 2048 | **helps (plateaus)** | none mechanically; watch VRAM squeeze | prefill | high (mech) / med (mag) |
| 6 | `-b` | ≥ `-ub`, e.g. 2048 | none beyond chunking | **none** | none | high |
| 7 | `-t 24` / `-tb 48` | make them equal, or default | none expected | none expected | none | medium (speculation) |
| 8 | CUDA graphs | keep ON (`USE_GRAPHS=1`) | ~0 (-0.5…-1.4%) | **+6…+78%** measured (other archs); +4-5% MTP fix | decode | high/med |
| 9 | `GGML_CUDA_DISABLE_GRAPHS=1` | fallback only | — | — | workaround for leak/Xid with MTP | med |
| 10 | `--no-kv-offload` | **do not use** | hurts (kills pipeline parallel) | hurts + crash risk | anti-recommendation | high mech |
| 11 | `--no-op-offload` | unnecessary | ~0 | ~0 | memory-only, probably nothing | low-med |
| 12 | `FORCE_MMQ` | leave OFF | worse at large batch per docs | ~0 | — | medium |
| 13 | `FORCE_CUBLAS` | leave OFF | docs scope gain to datacenter GPUs | ~0 | — | medium |
| 14 | `PEER_MAX_BATCH_SIZE` | **gone from source** | — | — | none | high (absent) |
| 15 | `--load-mode mmap` | keep `mmap`/auto | — | none | load-time/memory | medium |
| 16 | `-ngl -1` | keep full offload | **best** | **best** | — | med (no direct measurement) |
| 17 | `-np > 1` | sweep with MTP ON | — | aggregate ↑ (speculation) | throughput, changes per-slot ctx | low-med |
| 18 | `--kv-unified` | use if `-np > 1` | — | — | KV carving | med-high |
| 19 | `--cont-batching` | keep on | — | — | required for slots | high |

## Suggested next experiments (cheap, in order)

1. `grep -i "converting K and V to f16" server.log` — proves your build has the q8_0-q8_0 FA kernel (Q3).
2. Confirm `USE_GRAPHS = 1` and rising `graphs reused` in the log (Q9).
3. `-ub` sweep {512,1024,2048} × `-b` {2048,4096} at real `-c 262144`, recording **pp and tg together** (Q2) — this is the
   single highest-expected-value experiment on this rig.
4. `llama-batched-bench -npl 1,2,4 -npp 1024 -ntg 128` with and without MTP, tensor split, to test whether `-np` gains
   survive speculative decoding (Q8) — template from PR #26802.
5. A/B `GGML_CUDA_DISABLE_GRAPHS=1` with MTP on, watching VRAM across turns (Q9, issue #25835 shape).
