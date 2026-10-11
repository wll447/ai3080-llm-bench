# llama.cpp decode tuning on 4× Ampere (GA102, non-P2P) — measured-evidence report

**Target rig:** 4× RTX 3080 20 GB (GA102, sm_86), 320-bit @ 19 Gbps = **760 GB/s each / 3 040 GB/s aggregate**, 80 GB VRAM, PCIe 4.0 x16 slots, **no P2P** (BAR1 256 MiB → host-staged inter-GPU traffic). Dense 27B at Q8_0 ≈ 29 GB → **78.3 tok/s decode** (baseline).

**Report compiled:** 2026-09-24.
**Staleness rule applied:** anything dated before ~2026-03-24 is tagged `⚠STALE(>6mo)` and should be re-derived on the actual rig.

**Labels used throughout:** `MEASURED` = someone published before/after numbers with a URL; `ESTIMATED` = my arithmetic/projection, no direct measurement; `UNVERIFIED` = single-source, weak methodology, or a mechanism claim without numbers.

**Environment limitation (affects sourcing):** `github.com` HTML pages (issues/discussions/PRs) were **unfetchable** from this session — every `web_fetch` to `github.com/...` failed with a transport error, including retries. I worked around it with `raw.githubusercontent.com`, the unauthenticated GitHub REST API (which then rate-limited to HTTP 403), `app.semanticdiff.com` PR mirrors, and search-index snippets. Consequently issue/PR **bodies** for #24485, #24489, #23567, #21788, #26409 could not be quoted verbatim — only their titles, dates and URLs. `raw.githubusercontent.com` was itself intermittently flaky and needed retries.

---

## 0. The one number that reframes everything

| Metric | Value | Label |
|---|---|---|
| Declared aggregate bandwidth | 4 × 760 = **3 040 GB/s** | spec |
| Bytes read per token (Q8_0, 29 GB) | 29 GB | given |
| Measured decode | **78.3 tok/s** | given |
| Implied aggregate bandwidth actually used | 29 GB × 78.3 = **2 271 GB/s** | ESTIMATED |
| **Fraction of peak bandwidth achieved** | **74.7 %** | ESTIMATED |

**Consequence:** you are already extracting ~75 % of theoretical DRAM bandwidth. A memory overclock that adds +5.3 % of *nominal* bandwidth can only pay off if the remaining 25 % is DRAM-limited rather than kernel-issue/dequant-limited. Everything in §1 depends on that.

Also note: if your current mode is `--split-mode layer` (the default), 78.3 tok/s already **exceeds the naive serialized pipeline ceiling** for 4 cards — 29 GB ÷ 760 GB/s = 38.2 ms → 26 tok/s. So your cards are demonstrably streaming weights **concurrently** (async pipelining across tokens), not taking turns. That is important: it means the "4 GPUs work in parallel" win that tensor-parallel is supposed to buy **is already partly banked on your rig**. Treat the 2-card `layer→tensor` gains in §4 as an **upper bound that will not transfer**, not as a forecast.

---

## 1. Memory overclocking: +500 / +1000 / +1500 MHz

### 1.1 Null result first

**No published before/after measurement of a GDDR6X memory offset on llama.cpp (or ik_llama.cpp) decode tok/s was found.** I ran these queries and got nothing usable:

- `RTX 3090 GDDR6X memory overclock 19.5 Gbps bandwidth measured tokens per second LLM reddit`
- `reddit LocalLLaMA overclock VRAM memory speed AI inference faster benchmark 3090`
- `"memory overclock" OR "mem offset" 3090 stable diffusion LLM tokens per second gain reddit`
- `llama.cpp benchmark GDDR6X memory clock speed scaling bandwidth test`
- `\u201c+1000\u201d memory overclock 3090 tokens per second llama.cpp gain percent`
- `overclock VRAM improve LLM tokens per second test results GPU`
- `llama.cpp issue memory overclock VRAM speed decode tokens`
- `"VRAM overclock" 3090 "tok/s" LLM improvement measured reddit LocalLLaMA`
- `"memory clock" overclock llama.cpp benchmark tok/s gain 3090 4090 measured percent`

GitHub search hits for `"memory overclock"` in `ggml-org/llama.cpp` returned **3 results, none about GPU memory OC** (the top hit is PR #16000 `--numa mirror`). Search for `mem clock overclock VRAM` returned **1 result**, also unrelated.

**ECC:** no consumer-GDDR6X ECC on/off data exists because GeForce cards have no ECC. Every "ECC on/off" measurement in the wild is about datacenter parts (A100/H100/GDDR6-HBM). **Null result.**

### 1.2 Bandwidth arithmetic (ESTIMATED, arithmetic only)

Your cards are 320-bit @ 19 Gbps = 760 GB/s. Afterburner/nvidia-smi memory offsets apply to the **9 501 MHz** memory clock (= 19 Gbps ÷ 2, DDR):

| Offset | Resulting mem clock | Effective rate | Bandwidth (320-bit) | Δ bandwidth |
|---|---|---|---|---|
| stock | 9 501 MHz | 19.0 Gbps | 760 GB/s | — |
| **+500** | 10 001 MHz | 20.0 Gbps | 800 GB/s | **+5.3 %** |
| **+1000** | 10 501 MHz | 21.0 Gbps | 840 GB/s | **+10.5 %** |
| **+1500** | 11 001 MHz | 22.0 Gbps | 880 GB/s | **+15.8 %** |

(For reference, the same offsets on a 384-bit 3080 Ti / 3090 at 19.5 Gbps / 936 GB/s give +5.1 %, +10.3 %, +15.4 %.)

Sanity check of the bus/rate model against an independent source: the RTX 4090 is listed at **1 008 GB/s** and the 3080 Ti at **912 GB/s** in the Rookery GPU guide's bandwidth table — https://raw.githubusercontent.com/lance0/rookery/main/docs/gpu-guide.md (undated, content references 2026) — which matches 21 Gbps × 384-bit and 19 Gbps × 384-bit exactly. Your 760 GB/s figure for the 20 GB 3080 mod is consistent.

### 1.3 The mechanism that caps this: EDR plateau (vendor-described, no LLM numbers)

`MEASURED (vendor mechanism, 2020)` — Hardwareluxx, 2020-09-13, describing NVIDIA's GA102 GDDR6X **Error Detection and Replay (EDR)**, sourced from the GA102 documentation. Verbatim (Russian, my translation in brackets):

> «С помощью Error Detection and Replay происходит исправление ошибок, но **пропускная способность памяти перестает увеличиваться. Она превращается в плато**, только после его прохождения увеличивается вероятность крахов и вылетов. Как только плато достигнуто, смысла в дальнейшем увеличении частоты уже нет.»
> [Errors get corrected, but **memory bandwidth stops increasing — it becomes a plateau**; only after passing through it does the probability of crashes and driver resets rise. Once the plateau is reached, there is no point raising the clock further.]

https://www.hardwareluxx.ru/index.php/news/hardware/grafikkarten/50330-podrobnosti-ampere-tekhnologiya-error-detection-and-replay-edr-uproshchaet-razgon-gddr6x.html — dated 13.09.2020. `⚠STALE(>6mo)`

Same article: the RTX 3080 FE's GDDR6X runs at **1.188 GHz** (= 19 Gbps effective), and EDR uses CRC to detect and *replay* corrupted transfers, so OC does not produce visual artifacts — it silently converts into replayed (bandwidth-consuming) traffic.

**This is the answer to your ECC question, and it is a mechanism, not a measurement:** GDDR6X OC does not "reduce performance via error correction"; it **stops producing bandwidth** at the plateau while still looking stable. So a +1 500 offset that appears stable can yield *less* than +15.8 %, or nothing at all.

Thermal context — `MEASURED`, TechPowerUp 2020-09-22, reporting Igor's Lab's infrared measurement on an RTX 3080 FE: the hottest GDDR6X chip hit **104 °C**, with TJ Max estimated at 105–110 °C, and NVIDIA chose 19 Gbps over 21 Gbps for thermal reasons:

> "The speedup from a slower 19 Gbps chip to faster 21 Gbps chips is plus 10%, meaning that the heat output would follow the percentage, resulting in even greater TGP of the card."

https://www.techpowerup.com/272420/the-reason-why-nvidias-geforce-rtx-3080-gpu-uses-19-gbps-gddr6x-memory-and-not-faster-variants (Sep 22nd, 2020); primary: https://www.igorslab.de/en/gddr6x-am-limit-ueber-100-grad-bei-der-geforce-rtx-3080-fe-im-chip-gemessen-2/ `⚠STALE(>6mo)`

**Implication for 4 cards in one chassis:** GDDR6X was already at 104 °C junction on a single FE at stock 19 Gbps in 2020, and 3080 GDDR6X on most AIB boards sits under the backplate. Four cards stacked in a server board with server airflow is *better* than an FE in a case, but a +1 000/+1 500 offset is very likely to hit either the EDR plateau or memory thermal throttling (which costs you bandwidth without limping the core clock). **Recommendation: prove the plateau on your own cards before assuming linearity** — measure achieved bandwidth with a memory-bound kernel, not with artifacts.

### 1.4 The only measured VRAM-OC → decode numbers found (not Ampere)

`MEASURED (single source, secondary summary, not Ampere)` — local-inference-lab `rtx6kpro` daily summary for **2026-07-30** (file `daily-summaries/2026-07/2026-07-31.md`), on RTX PRO 6000 Blackwell (GDDR7, 96 GB, SM120):

> "**VRAM overclocking (+6000 mem)** found to give **~10 tok/s decode boost on GLM 5.2 EXL3**, since decode is bandwidth-bound — safe since no voltage increase"

https://github.com/local-inference-lab/rtx6kpro/blob/master/daily-summaries/2026-07/2026-07-31.md — dated 2026-07-30. `MEASURED` (community/Discord-attributed; via raw: https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/daily-summaries/2026-07/2026-07-31.md)

Two caveats make this only weakly transferable:
1. **Unit ambiguity.** The same repo's `hardware/blackwell-power-limit-sweep.md` lists PRO 6000 WS memory clock at **13 365 MHz**; "+6000" against that is a ~+45 % offset, which is far outside what GDDR6X on a 3080 will take.
2. **Different memory tech:** GDDR7 vs GDDR6X, different controller, no EDR plateau documented for GDDR7 in the same way.

The same source gives the qualitative claim you'd expect (`decode is bandwidth-bound`), which is the useful part.

`CONTEXT ONLY (not an isolated A/B)` — Madreag's `turbo3-cuda` README benchmarks a **RTX 3090 Ti SM86 24 GB "OC +1000 core, +2200 mem (golden sample), 516W"**, and reports `q8_0` 32K = **77.44 tok/s** and `turbo2` 32K = **81.58 tok/s** on Qwen 3.5 9B Q8_0 (https://github.com/Madreag/turbo3-cuda/blob/release/cuda-optimized/README.md, 2026). Because the OC is applied to *every* run, this cannot be used to size the OC effect — only to note that the fastest published SM86 32K numbers in the wild were obtained on memory-overclocked cards.

### 1.5 Does decode scale linearly with memory bandwidth?

**Within one kernel/quant path: approximately yes. Across paths: no — format and kernel choice dominate.**

Supporting:

- `MEASURED` — Johannes Gaessler's llama.cpp performance page (llama.cpp developer), on memory-frequency scaling: *"the most important factor for performance is memory bandwidth"*, and for CPU decode *"the performance is almost proportional to just the memory frequency"* (chart `memory_scaling_1.png`), with the counterpoint that *"more threads can actually be detrimental"*. https://johannesgaessler.github.io/llamacpp_performance — **undated, clearly old** `⚠STALE(>6mo)`, CPU-only evidence.
- `MEASURED` — the Rookery guide states the rope model explicitly: `gen_tok/s ≈ memory_bandwidth_GB/s / model_size_GB`, and lists 3080 10 GB (760 GB/s) → 8B Q5_K_M ≈ **85 tok/s** vs 3090 (936 GB/s) → 27B Q4_K_M ≈ **35 tok/s**. https://raw.githubusercontent.com/lance0/rookery/main/docs/gpu-guide.md (2026).
- `MEASURED` — club-3090's cross-rig TP=4 data cleanly separates decode from interconnect: @MoppelMat's 4× 3090 on **mixed x4/x4/x8/x8** beat @ryanmpelletier's 4× 3090 on **all-x16** on decode (**79.23/101.61** vs **74.76/90.83**) while losing ~25–30 % on prefill. club-3090's own verdict: *"Clean decode(compute) / prefill(interconnect) split"* and *"decode is ~lane-insensitive at TP=4"*. https://raw.githubusercontent.com/noonghunna/club-3090/master/BENCHMARKS.md (rows 2026-07-05 and 2026-07-07).

Contradicting (i.e. decode is **not** purely bandwidth):

- `MEASURED` — habr.com, 2026-08-29, 2× RTX 3090: *"Файл bartowski больше, но быстрее, так что дело не в пропускной способности памяти."* [The bartowski file is **larger** but **faster**, so it is not a matter of memory bandwidth.] Their measured pair, same day, same flags: unsloth Q8_0 **57.6** vs bartowski Q6_K_L (24.1 GB) **60.2** tok/s. https://habr.com/ru/articles/1076080/
- `MEASURED` — steveseguin's B70 sweep (2026-08-22, Intel Arc Pro B70, SYCL — **not CUDA**): *"Decode is inverse to file size: the smallest file (UD-Q4_K_XL) is the slowest, the largest (Q4_K_M) the fastest, at every depth. So decode is not purely bandwidth-bound on this build."* https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/qwen38-27b-b70/notes/2026-08-22-qwen38-q4km-tp1-context-kv-sweep-finding.md

**Verdict:** a raw +5.3 % / +10.5 % / +15.8 % bandwidth increase should translate to roughly the same tok/s delta **only if** your decode is DRAM-limited. Your 74.7 % of peak says it is *mostly* DRAM-limited. Realistic expectation: **+3 to +5 % at +500, +6 to +10 % at +1000, and a high risk of the EDR plateau at +1500.** `ESTIMATED`

---

## 2. KV cache quantization: does `--cache-type-k/-v q8_0` (or q4_0 / turbo3) change DECODE speed?

### 2.1 Direct answer

| Regime | f16 → q8_0 KV decode cost | Evidence |
|---|---|---|
| **Short context, CUDA, modern build** | **−1.6 % to −2.9 %** | `MEASURED` |
| Short context, `turbo3`-class KV, CUDA | **−12 % to −17 %** | `MEASURED` |
| Short context, `turbo3`-class KV, badly-optimised CUDA kernels | up to **−83 %** | `MEASURED` (superseded fork) |
| Long context (32 K), CUDA, SM86/89 | q8_0-class KV ≈ **+3.5 % to +5.4 %** *faster* than the same bytes in a plain quantised path, and the KV-type delta is only a few % | `MEASURED` |
| Long context (16 K–32 K), **SYCL** | q8_0 KV is **−36 % to −51 %** vs f16 | `MEASURED`, **backend-specific** |
| 0 K → 131 K with q8_0/turbo4 KV, **SM86 vs SM89** | **−71 %** on 3090 vs **−54 %** on 4090 | `MEASURED` |

### 2.2 The strong short-context measurement

`MEASURED` — @Defilan, **2× RTX 5060 Ti 16 GB (Blackwell SM120), CUDA 12.8, tensor parallelism via llama.cpp**, build `AmesianX/TurboQuant v1.5.2` with `-DCMAKE_CUDA_ARCHITECTURES="86-real;120a-real"`, **4 concurrent requests, 5 min, 8K context**, posted in llama.cpp Discussion #20969 on **2026-04-08T20:23:23Z**:

| Model | f16 KV | q8_0 KV | tbq3/tbq3 | tbqp3/tbq3 | previous fork (tq3_0) |
|---|---|---|---|---|---|
| Llama 3.1 8B Q5_K_M | 42.5 tok/s | 41.8 (**−1.6 %**) | 37.3 (**−12.2 %**) | 36.2 (−14.8 %) | 8.4 (−83 %) |
| Qwen 2.5 14B Q5_K_M | 24.1 tok/s | 23.4 (**−2.9 %**) | 20.8 (**−13.7 %**) | 20.2 | — |

Quote: the column he labels `(-1.6%)` for q8_0 is against his own f16 baseline. Source: https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-... (comment id in the API dump; thread URL https://github.com/ggml-org/llama.cpp/discussions/20969)

**This is the single most on-point number for your question:** at 8K context on CUDA, `--cache-type-k q8_0 --cache-type-v q8_0` costs about **2 %** of decode. Not free, not fatal.

### 2.3 Additional CUDA measurements of the same kind

| Config | f16 baseline | quantised KV | Δ decode | Date / source |
|---|---|---|---|---|
| RTX 4090 (SM89), Llama-3.1-8B-Q4_K_M, `q8_0`-K + `turbo4`-V (asymmetric), spiritbuun v2 CUDA | — | — | **+8.4 % prefill / −6.2 % decode** | 2026-04-04, https://github.com/ggml-org/llama.cpp/discussions/20969 (WaveboSF; summarised by TheTom same thread) |
| RTX 4090, symmetric `turbo4`/`turbo4` | — | — | **−16.8 % decode** | same |
| RTX 5090, `q8_0`+`turbo4`, spiritbuun kernels | — | — | **−25.2 % decode** | same |
| RTX 5090, same config, `turboquant_plus` fused mmvq kernels | — | — | **−6.9 % decode** (−7.2 % at `q8_0`+`t3`; −7.3 % at `t3/t3`) | same |
| DGX Spark GB10 (CUDA), GLM-4.7-Flash | 67.5 TG t/s | `tbq3/tbq3` 49.7 | **−26 %** | 2026-04-04, same thread (AmesianX, v1.4.2 release notes) |
| RTX PRO 6000 (CUDA), Q3 KV (PolarQuant plugin) | 43.5 tok/s | 44.1 | **"zero speed overhead" (+1.4 %)** | 2026-04-06/08, caiovicentino, same thread |
| **CPU**, Qwen3.5-35B-A3B Q4_K_M, 4 threads | f16 gen 10.6 | `q8_0` 10.4 · `q4_0` **12.5** · `tq3_0` 11.4 | q8_0 **−1.9 %**, q4_0 **+17.9 %** | 2026-03-25, same thread (Aaryan-Kapoor) |

Note the CPU row: **`q4_0` was 17.9 % faster than f16 and 20 % faster than `q8_0`** at short context. That is the "cheaper KV can be *faster*" signal, and it recurs on CUDA (§2.5).

### 2.4 Long context — where the trade actually bites

**a) The harshest published SM86 number.** `MEASURED` — sztlink/turboquant-cuda-bench, IQ4_NL reproduction, **2026-04-27**, via club-3090's hardware notes:

> single-stream token-generation rate degrades **~−71 % from 0K to 131K context** on **SM86 (Ampere consumer, RTX 3090)** with `q8_0/turbo4` KV vs **~−54 %** on **SM89 (Ada, RTX 4090)** for the same model + KV config. Their writeup: "SM86 has a weaker warp dispatch path for the turbo4 dequant kernel. The degradation is architectural, not model-specific."

https://github.com/sztlink/turboquant-cuda-bench/blob/main/bench/iq4nl-repro/results.md, surfaced by @lkaupp at https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-16733925; quoted in https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/HARDWARE.md (`⚠` that specific repro was on **turbo4**, not plain q8_0/q8_0).

club-3090's operational conclusion, verbatim: *"even when you can fit the prompt + KV in VRAM ... single-stream TG above ~32K-65K on SM86 pays a structural per-token rate tax that you can't fix by changing KV format choice."*

**b) The backend-specific blowup (do not generalise to CUDA).** `MEASURED` — steveseguin, **2026-08-22**, single Intel Arc Pro B70 (32 GiB), llama.cpp **SYCL** AOT build, Q4_K_M, flash-attn on, `llama-bench` pp2048+tg128, 5 reps:

| depth | KV f16 | KV q8_0 | q8 vs f16 |
|---:|---:|---:|---:|
| 0 | 24.81 | 24.27 | −2.2 % |
| 2 048 | 24.46 | 22.45 | −8.2 % |
| 4 096 | 24.25 | 21.05 | −13.2 % |
| 8 192 | 23.83 | 18.68 | −21.6 % |
| 16 384 | 23.10 | 14.86 | −35.7 % |
| 24 576 | 22.42 | 12.40 | −44.7 % |
| 32 768 | 21.77 | 10.66 | **−51.0 %** |

Prefill was KV-dtype-independent (< 1.5 % at every depth). Verbatim finding: *"The q8_0-KV decode penalty grows with context — the headline result. Near parity at 0 ctx (−2 %), it widens monotonically to **−51 % at 32K** ... This is the opposite of the common 'KV8 is free speed + memory' assumption."*

https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/qwen38-27b-b70/notes/2026-08-22-qwen38-q4km-tp1-context-kv-sweep-finding.md — dated 2026-08-22. **Backend = SYCL, not CUDA.** Do not apply this to your 3080s; it does establish the *shape* (per-token dequant work scales with cached length).

**c) On CUDA/SM86, long-context q8_0 is not a catastrophe.** `MEASURED` — Madreag/turbo3-cuda, Qwen 3.5 27B Q6_K on RTX 5090 and Qwen 3.5 9B Q8_0 on RTX 3090, `llama-bench -d <depth>` tg128:

| GPU | KV type | short | 32 K | 64 K | 131 K | 256 K |
|---|---|---:|---:|---:|---:|---:|
| RTX 3090 (SM86, 9B Q8_0) | `q8_0` | — | 61.0 | — | — | — |
| RTX 3090 (SM86, 9B Q8_0) | `turbo2` | — | **63.12** | 56.91 | — | — |
| RTX 3090 Ti (SM86, 9B Q8_0, OC +2200 mem) | `q8_0` | 91.01 | 77.44 | OOM | — | — |
| RTX 3090 Ti, same | `turbo2` | 90.75 | **81.58** | 72.79 | — | — |
| RTX 5090 (27B Q6_K) | `q8_0` | 63.40 | 55.60 | — | — | — |
| RTX 5090, same | `turbo2` | 65.50 | **58.61** | 63.94 | 51.23 | 42.57 |

https://github.com/Madreag/turbo3-cuda/blob/release/cuda-optimized/README.md (2026). So on real SM86 CUDA, going to a *quarter-size* KV bought only **+3.5 %** (3090) to **+5.3 %** (3090 Ti) at 32 K. That is the honest magnitude of the long-context "read less KV" win **once the kernel path is good**.

### 2.5 The actual mechanism behind the "q8_0 KV is 3× slower" folklore

Two independent things can make quantised KV look catastrophically slow, and neither is "dequant ALU cost":

**(i) Flash-attention kernel availability at build time.** llama.cpp only compiles FA kernels for quantised KV when built with `GGML_CUDA_FA_ALL_QUANTS=ON`. @WaveboSF's finding in the #20969 thread, quoted by TheTom on 2026-04-04: *"The FA build flags finding is a great catch too. That's a **free 7.4 %** that people are definitely leaving on the table."* This is corroborated by an open feature request: **Issue #24485 — "[FR] Make `GGML_CUDA_FA_ALL_QUANTS` the default or add runtime warning for missing quantized FA kernels"** https://github.com/ggml-org/llama.cpp/issues/24485 (2026; title/URL only — body unretrievable) and by an open PR: **PR #28456 — "cuda : warn when flash attention falls back to the CPU backend due to unsupported KV cache types"** https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/28456/overview (Open).

**(ii) `-sm tensor` + quantised KV.** Discussed in §4.

The club-3090 `q8_0` "~3× slower" claim is exactly this class of artefact, and club-3090 itself flags it as unverified. Verbatim, from https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/engines/LLAMA_CPP.md:

> | `q8_0` | ~6 KB | ⚠️ fits at ~23 GB but slow | **~3× slower** | Community-reported on Qwen3.6 27B + 3090; flash-attention path doesn't optimize q8 the way it does q4_0 |
> | **`q4_0`** ⭐ | ~3 KB | ✅ fits at ~21 GB | full speed (40 tok/s flat) | The right pick for max context on consumer hardware |
>
> "**Test it on your own rig** before trusting the 3× claim — but if you're on `f16` or `q8` KV at 262K and seeing slow decode, swap to q4_0 first."

Against that: club-3090's own Luce/DFlash run on 1× 3090 measured KV-format changes as **irrelevant** at short scale — same-card default KV **73.97 tok/s** vs **K8V4** (`-ctk q8_0 -ctv q4_0`) **74.68 tok/s = +1 %**, verdict *"KV-format optimization doesn't help at HE-scale (<150-tok prompts × 128-tok gen) where KV pool isn't the bottleneck."* https://raw.githubusercontent.com/noonghunna/club-3090/master/BENCHMARKS.md

### 2.6 The 2026 KV-quant lineage (for context, not per-token decode)

- **llama.cpp Discussion #20969 — "TurboQuant — Extreme KV Cache Quantization"** https://github.com/ggml-org/llama.cpp/discussions/20969 (opened 2026-03-25; ~100+ comments through 2026-04-10). Contents include the measured tables above, the QJL-is-unnecessary finding, the `ggml` column-major transposition bug, and the Metal/CUDA speed journey (prefill **739 → 2 095 tok/s = 0.78× of q8_0's 2 694** after moving the Walsh-Hadamard rotation into the graph — TheTom, 2026-03-25 evening).
- **Discussion #21155 — "TurboQuant: 5.2× KV cache compression with near-lossless quality (ICLR 2026, Google DeepMind)"** https://github.com/ggml-org/llama.cpp/discussions/21155
- **Upstream adoptions:** Hadamard KV rotation **PR #21038**, fast-WHT CPU kernel **#22631**, **CUDA #23615**, **Vulkan #23687**; vLLM merged the TurboQuant attention backend as **PR #38479 (April 2026)** with `--kv-cache-dtype turboquant_k8v4`. All per https://raw.githubusercontent.com/TheTom/turboquant_plus/main/README.md (2026).
- **Dequant-skip is the real speed lever, not bit-width:** "sparse V dequant ... gave **+22.8 % decode at 32 K** on M5 Max, and **+5 % even on plain q8_0**" — TheTom, 2026-03-27, #20969. Upstream PR **#21119**. `MEASURED` (Metal) — the +22.8 % is a >10 % lever that is **not** available in stock CUDA llama.cpp today.
- **CUDA-specific turbo fork numbers (SM86!):** independent kernel work moved turbo KV decode a lot — on **RTX 3090, 9B Q8_0, 32 K**: `q8_0` 56.91 → **61.0 (+7.2 %)**, `turbo4` 35.63 → **60.28 (+69 %)**, `turbo3` 44.79 → 56.82, `turbo2` 53.21 → 63.12; at **64 K**: `turbo3` 33.43 → 49.27 (+47 %), `turbo2` 42.45 → 56.91 (+34 %). https://github.com/Madreag/turbo3-cuda/blob/release/cuda-optimized/README.md `MEASURED`

### 2.7 Bottom line for §2

- **Short context (≤ 8 K): `--cache-type-k q8_0 --cache-type-v q8_0` costs ≈ 2 % decode on CUDA.** Keep it. There is no meaningful short-context dequant penalty worth optimising.
- **Long context: the read-less-KV win is real but small on SM86 with good kernels (+3.5 % to +5.3 % at 32 K).** The published −36 %/−51 % q8_0 penalties are **SYCL**, and the −71 %-over-0→131K SM86 figure is a **turbo4** dequant-kernel dispatch problem, not a q8_0/q8_0 problem.
- **If you do see a 3× q8_0 penalty, it is almost certainly a missing FA kernel or a `-sm tensor` interaction, not ALU cost.** First action: verify the build was made with `-DGGML_CUDA_FA_ALL_QUANTS=ON` (worth ~7.4 %), and confirm FA is not falling back.

---

## 3. Other reliable >10 % levers on Ampere for llama.cpp decode

Ranked by measured size, with the honest confounds.

### 3.1 ✅ Speculative decoding (MTP / DFlash) — the biggest lever found, up to +81 %

`MEASURED` — habr.com, kotafey, **2026-08-29**, 2× RTX 3090 / Threadripper 3960X, llama.cpp `ghcr.io/ggml-org/llama.cpp:full-cuda` **build 10666**, Qwen3.8-27B dense (24.1 GB):

| Config | decode tok/s |
|---|---:|
| `-sm layer`, no MTP | 32.4 |
| `-sm layer`, MTP n=3 | **58.5** (**+80.6 %**) |
| `-sm tensor`, no MTP | 48.2 |
| `-sm tensor`, MTP n=3 | **74.8** |

Flags: `--spec-type draft-mtp --spec-draft-n-max 3 -ctkd q4_0 -ctvd q4_0`. The model's MTP head (`blk.64.nextn.*`) is unused by default.

Draft-depth sweep (measured, same source): `2 → 52.8`, `3 → 57.3`, `4 → 57.6`, `6 → 54.3` tok/s → **use 3–4**. The draft KV cache is separate; `-ctk/-ctv` do **not** apply to it — without `-ctkd/-ctvd` it silently sits in f16 and eats VRAM.

Mainline support: **PR #22673** merged on mainline per club-3090 (`docs/engines/LLAMA_CPP.md`, "MTP spec-decode merged on mainline (PR #22673, 2026-05-16)"), also cited in club-3090 BENCHMARKS `llamacpp/mtp` row (server-cuda-b9246, 2026-05-23). https://habr.com/ru/articles/1076080/ · https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/engines/LLAMA_CPP.md

**Caveat:** whether this transfers depends entirely on whether your 27B GGUF ships an MTP head. If it does, this outranks every memory/power/PCIe knob on this list. If it doesn't, use a matched external drafter (DFlash) instead — but note DFlash's context cost.

### 3.2 ⚠️ `--split-mode tensor` — up to +28 % short-context / +61 % at 240 K on a non-P2P 2× 3090, but with strings attached

Full detail in §4.

### 3.3 ✅ Power limit (`nvidia-smi -pl`): **lower power HURTS**, and by a lot at aggressive caps

`MEASURED` — club-3090 `docs/HARDWARE.md` power-cap anchor table, **llama.cpp + Qwen3.6-27B Q3_K_XL**, dense, narrative/code TPS (https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/HARDWARE.md):

| Rig | Cooling | Cap | Narr TPS | Code TPS | TPS/W |
|---|---|---:|---:|---:|---:|
| 3090 | water | 230 W | 25.15 | 24.86 | 0.109 |
| 3090 | water | **330 W** ⭐ | **36.35** | **36.26** | 0.110 |
| 3090 | water | 388 W (stock) | 38.23 | 37.97 | 0.098 |
| 3090 | air | **290 W** ⭐ | 32.26 | 32.17 | **0.111** |
| 3090 | air | 370 W (stock) | 34.66 | 34.67 | 0.104 |
| 3090 | air | 390 W (max) | 36.26 | 36.06 | 0.093 |
| 3090 | air | 290 W (`decode-concurrent` N=4) | 31.74 | 29.98 | **0.110** |
| 3090 | air | 370 W (`decode-concurrent` N=4) | 34.13 | 32.46 | 0.102 |
| 3090 | air (MoE) | 210 W ⭐ | 114.59 | 113.79 | **0.546** |
| 3090 | air (MoE) | 370 W | 136.84 | 136.65 | 0.386 |

Derived deltas (`ESTIMATED` from the above): **290 W vs 370 W stock = −6.9 % decode**; **290 W vs 390 W max = −11.0 %**; **230 W (water) vs 388 W stock = −34.2 % decode** — for **−22 % to −40 % power**. TPS/W actually *improves* at the knee.

club-3090's verdict, verbatim:

> "**The '230W is the sweet spot' lore is wrong** — it traces back to early thermal-constrained recommendations and 3-cap-resolution data. Dense 10W sweeps show 230W costs ~16% efficiency vs 290W (decode) and ~4% vs 250W (prefill)."
>
> "**Caveat — 230W on llama.cpp + GDN models is more aggressive than it looks**: cross-rig data from [@syangsao] (1× water-cooled 3090, llama.cpp + Qwen3.6 Q3_K_XL) shows **230W costs ~34% TPS** vs stock (25 vs 38 TPS) because the chunked_gated_delta_rule kernel is genuinely compute-bound on this model, not memory-bound. ... **Recommendation**: use 290W (air) / 330W (water) as the default cap on either engine."

Prefill behaves the same way: air 3090 llama.cpp `prefill-heavy` = 250 W **906.79** / 370 W **1051.07** / 390 W **1104.81** pp t/s (the 250 W "prefill sweet spot" is an *efficiency* optimum, not a speed one).

**Answer to "does lower power improve decode?": No.** For llama.cpp on Ampere, decode is not bandwidth-pure enough for a power cap to be free — 230 W costs 34 % on this exact engine/card. If you currently run reduced caps, raising them back to ~290–370 W is a **>10 % lever**; dropping them is a large **regression**.

**Important counter-evidence on newer silicon** — `MEASURED`, rtx6kpro `hardware/blackwell-power-limit-sweep.md` (driver 595.58.03, CUDA 13.2, 16× RTX PRO 6000 Blackwell, `gpu_burn -tc`, `nvidia-smi -pl` in 50 W steps, 1 Hz telemetry):

> "**Memory clock is invariant within each variant regardless of power limit. Power limit gates only GFX/SM clock** — memory always runs at its variant's fixed clock."

and for a purely memory-bound kernel (`4 GB memcopy`): **682 iter/s at 401 W (WS)** vs **680 iter/s at 365 W (Server)** — i.e. *identical throughput at very different power*. Their explicit caveat: *"`gpu_burn -tc` is a synthetic compute-bound stress. Memory-bound kernels (attention, KV cache) will respond very differently to power limit since memory clock is invariant."* https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/hardware/blackwell-power-limit-sweep.md (2026)

**Reconciliation for your rig:** on Ampere, `nvidia-smi -pl` does not lower the GDDR6X clock either — so a PLAIN weight-streaming GEMV should be nearly PL-insensitive. The club-3090 −34 % is measured with MTP/spec-decode and hybrid-attention kernels in the loop (extra compute). Expect your own sensitivity to land **between** those: measure it, don't assume it.

### 3.4 ⚠️ `nvidia-smi -lgc` / `-lmc` clock locking — the best *quality* of evidence, but on Blackwell

`MEASURED` — @apnar's 5090 sweep, reproduced in club-3090 `docs/HARDWARE.md`; source data at https://github.com/noonghunna/club-3090/discussions/86#discussioncomment-16845745:

- Setup: **5090 air-cooled + Gemma-4-31B-AutoRound + vLLM-MTP K=3 + `decode-concurrent` N=6, 35-point sweep across 5 mem-clock tiers × 7 GPU-clock points**.
- Verbatim: *"**TPS climbs nearly linearly with mem clock** — at 405 MHz mem (lowest), TPS caps at ~53 regardless of GPU clock; at 14001 MHz mem (max), TPS climbs through 800+ TPS at GPU 3090 MHz."*
- Peak efficiency: **7001 MHz mem / 1635 MHz GPU = 2.025 TPS/W (211 W, 428 narr TPS)** — *"1.42× more efficient than the 400W power-cap sweet spot (1.43 TPS/W)"*.
- Pareto point: **14001 MHz mem / 2122 MHz GPU = 1.92 TPS/W, 314 W, 602 narr TPS** — *"strictly better than the 400W cap on both axes: +5% more TPS at −22% less power."*
- Verbatim caveat: *"**Clock-locking is not portable to Ampere** — `-lmc` was removed from non-datacenter cards in some driver versions, and the 3090's voltage curve doesn't have the same headroom for sub-power-cap operation."*

**Answer to "does locking LOW core clocks help a bandwidth-bound workload?": the Blackwell data says yes at the margin** (locking GPU down to 1635 MHz while holding mem at 7001 MHz produced the single most efficient operating point), **but the gain was in TPS/W, not TPS**: 428 tok/s at 2.025 TPS/W vs 571 tok/s at the 400 W cap. Total throughput went *down*. And no Ampere LLM measurement of `-lgc`/`-lmc` was found. **Null result for Ampere.**

Corroborating Ampere detail from the same doc: on the air-cooled 3090, *"SM locks at 1560 MHz across 340-370W (boost-clock plateau)"* and SM is **1380 MHz at the 290 W sweet spot** — i.e. at the efficient cap the core clock is already well below max, which is consistent with decode not being core-clock-starved there.

### 3.5 ⚠️ Persistence mode (`nvidia-smi -pm 1`)

`UNVERIFIED, no tok/s number` — club-3090 `docs/HARDWARE.md`, "Note for older host platforms (PCIe Gen 3 + older CPUs)":

> "**Enable persistence mode** (`sudo nvidia-smi -pm 1`) — common to find this off on KVM/VM hosts; with it disabled the driver tears down between idle periods and adds per-request init latency."

Also noted as standard practice in the rtx6kpro power sweep ("Persistence mode: Enabled"). No decode tok/s delta published. Treat as a latency/hygiene fix, not a >10 % throughput lever.

### 3.6 ❌ `CUDA_DEVICE_MAX_CONNECTIONS` — nothing found

The **only** llama.cpp hit for `CUDA_DEVICE_MAX_CONNECTIONS` in the whole repo search was unrelated: **Issue #19219** (2026-01-30) — *"MoE model decode hangs on Jetson Orin AGX (SM87) since b7309"*. https://github.com/ggml-org/llama.cpp/issues/19219

**No measured effect on multi-GPU llama.cpp decode was found.** The general CUDA statement — serialising the hardware work queues slows execution but is useful for debugging — appears in NVIDIA's CUDA Programming Guide (https://docs.nvidia.com/cuda/cuda-programming-guide/, section on disabling asynchronous execution). Note llama.cpp's CUDA backend and NCCL set this internally in places; if you override it globally you may be disabling work-queue concurrency that the CUDA backend relies on. **Null result; do not touch this speculatively.**

### 3.7 ✅ `--no-mmap` / `--mlock` — measured FLAT

`MEASURED` — club-3090 BENCHMARKS, `beellama/carnice-v2-dual-q8-mtp` row, **2026-06-16**, 2× 3090 PCIe, no NVLink, 370 W:

> "**`-b`/`-ub`/`--no-mmap` A/B'd FLAT** — KV type was the only prefill lever."

https://raw.githubusercontent.com/noonghunna/club-3090/master/BENCHMARKS.md

Same row, the thing that *did* move prefill: **KV A/B = +17 % prefill (1003 vs 860 t/s)** from `q8_0` vs KVarN software compression — i.e. the *path*, not the bit-width.

Also, one deployment-specific reason to use `--no-mmap` anyway: Madreag's README tip — *"Use `--no-mmap` on WSL2 to disable mmap (avoids GPU stalls from page cache)."* https://github.com/Madreag/turbo3-cuda/blob/release/cuda-optimized/README.md

**Verdict: use `--no-mmap --mlock` for load-time determinism / to prevent page faults and swap, not for decode speed.** `-b`/`-ub` are also flat here; the only `-ub` effect found was small and positive — club-3090 `llamacpp/mtp` row: default `-ub 512`, and *"`CTX_SIZE=131072 UBATCH_SIZE=1024` is the faster-prefill override (~51 / 60 decode)"* vs *"50.27 / 58.92"* at `-ub 512` = **~+1.5–2 % decode** (2026-05-23).

### 3.8 ⚠️ CPU threads (`-t`), `--main-gpu`, `--tensor-split`

- **`-t`:** `MEASURED` but CPU-era. Johannes Gaessler's page: more threads can be *detrimental* (RX 3700X drops going 8→9 threads, only 8 physical cores; Xeon E5-2683 v4 was best at 29 threads with 16 physical cores) — because perf tracks memory bandwidth, and extra threads add overhead. https://johannesgaessler.github.io/llamacpp_performance `⚠STALE(>6mo)`. For a fully-offloaded 27B, `-t` only affects non-GPU ops (sampler, tokenisation, output) → small.
- **`--tensor-split` on asymmetric slots:** `MEASURED` — bric.pe.kr, **2026-05-23**, 2× GTX 1080 Ti (x16 + x8), `--tensor-split 12,10`: *"About **3–5 % gain** in our tests for 30B-class models. Negligible for 8B."* https://bric.pe.kr/blog/llama-cpp-split-mode-row-vs-layer-old-gpu
- **`--main-gpu`:** relevant only for `--split-mode none`; no perf data found.
- **`-ts` is IGNORED under `-sm tensor`:** `MEASURED` — habr 2026-08-29: *"`-ts` не работает: `llama_params_fit is not implemented for SPLIT_MODE_TENSOR`."* Upstream docs confirm: `--fit` "**Not supported with `tensor`**". https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md (retrieved 2026-09-24)

### 3.9 ✅ PCIe lane width / gen — **prefill-sensitive, decode-insensitive**

`MEASURED (vLLM, TP=4)` — club-3090 BENCHMARKS, same recipe (`multi-fast`/`multi-max`, fp8 KV, 262 K), two different rigs:

| Rig | Lanes | Power | Narr / Code TPS | Prefill @10K→90K |
|---|---|---:|---|---|
| @ryanmpelletier (4× 3090, Proxmox VM passthrough, no NVLink) | **all x16** | 350 W | 74.76 / 90.83 | **1288 → 1175** |
| @MoppelMat (4× 3090, bare metal, AMD 5950X) | **mixed x4/x4/x8/x8** | 300 W | **79.23 / 101.61** | **964 → 887 (−25 to −30 %)** |

club-3090's verdict: *"Decode beats ryan's x16 rig ... **prefill trails ~30 %** (964/887 vs 1275/1141) — the mixed x4/x8 lanes are the bandwidth-bound bottleneck, while **decode is ~lane-insensitive at TP=4**. Clean decode(compute) / prefill(interconnect) split."*

**Self-correction the repo published about its own comparison** (important, quote it): *"⚠️ **Attribution SOFTENED 2026-07-26** — this used to read 'All-x16 topology beats @Whamp's mixed-lane 59/74 → PCIe lane width matters for TP=4 all-reduce'. **That pair is confounded — two ways, not three ...** lanes (all-x16 vs x4/x16/x8/x16) **and** power cap (**350 W vs 230 W**). ... Lane width is still the most likely driver ... but it is a **hypothesis, not a measurement**."* So: **the lane-width effect on decode is not cleanly measured; the prefill effect is.**

`MEASURED` — PCIe gen, and the crucial confound. club-3090 `docs/HARDWARE.md`:

> "**PCIe Gen 3 x16 ≈ 15.75 GB/s** per direction vs **Gen 4 x16 ≈ 31.5 GB/s**. TP=2 all-reduce on the residual stream every layer is GB/s-class traffic — halving interconnect bandwidth roughly halves the all-reduce wall time, and decode-TPS is sensitive to that."

But then the decisive worked datapoint (same vLLM SHA, same compose):

| Rig | CPU | PCIe cross-card | Narr / Code TPS |
|---|---|---|---|
| @mgabor3141 — X399 + Threadripper 1950X (**Zen 1**) | slow | **Gen 3 x16 + x16 (full, symmetric)** | 63.48 / 80.36 |
| @OVDEN13 — Ryzen 5700X (**Zen 3**) | fast | **Gen 4 x4 + x8 (half the bandwidth)** | **75.80 / 99.04** |

> "which is **17–19 % below** ... **Better interconnect, slower CPU, lower TPS — so on this rig class, don't under-spec the CPU to splurge on PCIe lanes.**"

And the extreme case: **Issue #137** — Xeon Gold 6138 + PCIe Gen 3 x16 + 2× 3090 (KVM passthrough) → **32 / 41 TPS** vs Ryzen 5950X + Gen 4 + same KV config → **89 / 117 TPS** (https://github.com/noonghunna/club-3090/issues/137, cited in HARDWARE.md). ~2.8× — but confounded by CPU *and* virtualisation.

Symptom signature to recognise the failure mode: *"GPU utilization asymmetry during decode (e.g. `GPU 0: 28% util / 174W` vs `GPU 1: 85% util / 254W`) — communication-starved TP=2, where one card finishes its half-step and stalls waiting on all-reduce."*

**Answer:** for a host-staged non-P2P setup, **PCIe lane width matters for prefill/TTFT and for TTFT at depth; decode is comparatively insensitive** (and for llama.cpp single-stream, P2P bandwidth is outright irrelevant — see §4.3). The hidden variable with a much larger effect is **host CPU class**.

### 3.10 ⚠️ NUMA pinning on EPYC

What I found:

- `MEASURED-adjacent` — llama.cpp merged **PR #16000 `--numa mirror`** (2025-09-15): *"mirrors model weights to each Numa node in the system, and uses a thread-local var in the OMP threadpool to select the correct mirror copy local to the thread at runtime, to eliminate cross-socket..."* https://github.com/ggml-org/llama.cpp/pull/16000 `⚠STALE(>6mo)`
- `UNVERIFIED, no tok/s` — club-3090 HARDWARE, Proxmox VFIO row (2026-05-21): *"NUMA-aligned (vCPUs+mem+GPU same node), 8 vCPU sufficient (vLLM uses ~3)"*.
- `MEASURED, stability not speed` — club-3090 HARDWARE, "Note for AMD platforms": sustained TP=2 inference on AMD-Vi can drive **Xid 154** fatal MMU faults on both GPUs. *"Fix: add `iommu=pt` to the kernel command line ... Device DMA bypasses page-table translation while the IOMMU stays enabled."* Triage: `dmesg | grep -E "AMD-Vi.*IO_PAGE_FAULT|Xid.*154"`. Reported by @mgabor3141 on X399 + TR 1950X + 2× 3090, 2026.
- `NOT RETRIEVED` — **llama.cpp Discussion #11733, "Dual Epyc Genoa/Turin token generation performance bottlenecks"** — https://github.com/ggml-org/llama.cpp/discussions/11733. A search snippet shows the table header `P2 | S0 | NPS1 | likwid-bench load | 460.8 | 359.9 | 78.1%`, i.e. it does contain NPS1/NPS2 + measured memory-bandwidth numbers on EPYC. **I could not fetch it** (github.com blocked; the GitHub API returned HTTP 403 rate-limit). **This is the most likely place to find your EPYC NUMA answer — retrieve it manually.**

**Null result: no measured tok/s delta for `numactl --cpunodebind` / `--membind` / `--interleave` on a host-staged multi-GPU llama.cpp rig was found.** Given §3.9's finding that host CPU class dominates PCIe width, NUMA placement is a plausible large lever on a 4-card EPYC box — but it is **unmeasured in the public record I could reach.**

### 3.11 ⚙️ `GGML_CUDA_*` / `LLAMA_CUDA_*` env vars

| Var | Finding | Label |
|---|---|---|
| `GGML_CUDA_P2P` | **Opt-in, OFF by default** regardless of driver grant. Upstream: *"P2P requires driver support (usually restricted to workstation/datacenter GPUs) and **may cause crashes or corrupted outputs** on some motherboards or BIOS configurations (e.g. when IOMMU is enabled)."* Measured gain on llama.cpp single-stream: **~0** (§4.3). | MEASURED (≈0) |
| `GGML_CUDA_FA_ALL_QUANTS` | Needed for quantized-KV FA kernels. Without it you can silently fall back. @WaveboSF: *"a **free 7.4 %**"*. Open FR #24485 to make it default; open PR #28456 to warn on FA→CPU fallback. | MEASURED (7.4 %) |
| `GGML_CUDA_FORCE_MMQ` | **No measured decode number found.** Related: llama.cpp **PR #8075 "CUDA: use MMQ instead of cuBLAS by default"** https://github.com/ggml-org/llama.cpp/pull/8075. | NULL |
| `GGML_CUDA_FORCE_CUBLAS` | **No measured decode number found.** | NULL |
| `GGML_CUDA_DMMV_X` | **Obsolete.** DMMV was removed: **PR #10318 "CUDA: remove DMMV, consolidate F16 mult mat vec"** https://github.com/ggml-org/llama.cpp/pull/10318 | DEAD VARIABLE |
| `GGML_CUDA_MMV_Y` | Still present in the CUDA backend; **no measured decode number found.** | NULL |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Build-time option for small-VRAM cards. **No measured decode number found.** | NULL |
| `LLAMA_CUDA_*` | Legacy names; superseded by the `GGML_CUDA_*` namespace. | — |
| `GGML_CUDA_DISABLE_GRAPHS` | Complete workaround for a 2026-08-18 CUDA-graphs hang (RC watchdog + Xid 8) on RTX 5090 Laptop / sm_120: **Issue #27330** https://github.com/ggml-org/llama.cpp/issues/27330. Not an Ampere perf lever. | Issue only |
| `GGML_CUDA_NCCL` | **CMake default is `ON`** (upstream docs, §4.2), **but the official Docker images ship without NCCL** — see §4.2. Check your build. | MEASURED (build-dependent) |

### 3.12 ✅ `--flash-attn`

- Upstream docs: *"`-fa` … **Required when using `--split-mode tensor` and/or quantized V cache.**"* https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md (2026-09-24)
- club-3090: *"`-fa on` — flash attention on (**memory + speed both win on Ampere+**)"* (LLAMA_CPP.md).
- bric.pe.kr (Pascal, `MEASURED`, 2026-05-23): *"`--flash-attn` … still improves memory locality and reduces KV cache reads. **~10-15 % speedup on long contexts**."* (Pascal-specific.)
- Do not confuse `-fa` with the FA-kernel-coverage issue (§2.5): turning FA *on* while your quantised KV type has **no compiled kernel** is the worst case — PR #28456 exists precisely because that fallback was silent.

### 3.13 `--no-kv-offload`

No measurement found. It forces KV to CPU, which for a 4× 80 GB rig would add host-staged traffic per token — strictly a memory-relief flag. **NULL.**

---

## 4. `--split-mode tensor` vs `layer` vs `row` on NON-P2P systems

### 4.1 The single best data point — a 2× 3090 with **no NVLink**, and tensor wins big

`MEASURED` — habr.com, kotafey, **2026-08-29**. Stand: **2× RTX 3090 (24 GB, PCIe 4.0 x16, без NVLink / no NVLink)**, Threadripper 3960X, 121 GB RAM, Ubuntu 26.04, driver 595.71.05, `ghcr.io/ggml-org/llama.cpp:full-cuda` **build 10666**, Qwen3.8-27B dense (24.1 GB). Both columns with MTP n=3, 3 runs per point, spread ±1 tok/s:

| | `-sm layer` | `-sm tensor` | Δ |
|---|---:|---:|---:|
| Generation, short context | 58.5 | **74.7** | **+27.7 %** |
| Generation @ 130 K | 41.4 | **58.9** | **+42.3 %** |
| Generation @ 240 K | 34.4 | **55.4** | **+61.0 %** |
| Two parallel slots | 36.3 + 36.3 | **47.4 + 50.3** | +33 % aggregate |
| Prompt processing @ 40 K | **1410** | 1121 | **−20.5 %** |
| Prompt processing @ 240 K | **764** | 743 | −2.7 % |
| VRAM | 42.4 GB | 43.6 GB | +1.2 GB |

Without MTP, the same pair: `layer` **32.4** → `tensor` **48.2** = **+48.8 %**.

Their own (correct) framing of why, verbatim: *"Чем длиннее контекст, тем больше выигрыш. Моё объяснение (не измерение)..."* — the mechanism is **labelled by the author as explanation, not measurement**: with layer split each card reads its own KV shard while the other waits; with tensor split both read in parallel and only the all-reduce activations cross the bus, and that does not scale with context length.

Cost, in their words: *"обработка промпта −20 % на 40K, −11 % на 130K, −3 % на 240K. Для интерактивной работы выгодно; для batch-прогонов коротких промптов — не факт."* [prefill −20 % at 40 K, −11 % at 130 K, −3 % at 240 K; worthwhile for interactive work, not necessarily for short-prompt batch runs.]

Operational gotchas they hit (all `MEASURED`):
- **`--ipc=host` is mandatory** in Docker, or the first decode dies with `CUDA error: unhandled system error` in `ggml_backend_cuda_comm_allreduce_nccl`. The Docker error message does not hint at this.
- **`-ts` is ignored** (`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`) — no auto-fit, size `-c` by hand.
- **Sampling moves to CPU, once per slot**: `spec common_specu: backend offload failed for seq_id=N; using CPU sampler`. Under `-sm layer` there are **zero** such lines; under `-sm tensor`, exactly one per slot. Their rig (24 cores, both cards on x16) absorbed it; they explicitly warn that a weaker CPU, many slots, or x1/x4 risers could make the CPU sampler + logit round-trip the new bottleneck.
- **`-sm row` does not work at all on this config:** `error loading model: device CUDA0 does not support split buffers`.
- Their own caveat: *"У меня обе карты на PCIe 4.0 x16. На x8/x4 all-reduce дороже, результат может быть другим — не проверял."* [Both my cards are PCIe 4.0 x16. On x8/x4 the all-reduce is more expensive and the result may differ — I did not test that.]

Source: https://habr.com/ru/articles/1076080/ (29 Aug 2026).

### 4.2 What upstream llama.cpp itself says (retrieved 2026-09-24)

`MEASURED (documentation)` — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md, verbatim:

> | `layer` (**default**) | Pipeline parallelism. Each GPU holds a contiguous slice of layers... | Default and most compatible multi-GPU choice... **Can tolerate slow interconnect speeds between GPUs.** |
> | `row` | **Deprecated.** Older row-split tensor-parallel path with comparatively poor performance. Splits only dense weights across GPUs. Superseded by `tensor` which should be universally superior if it can be used. | **Avoid in new deployments.** |
> | `tensor` | **EXPERIMENTAL.** Tensor parallelism that splits both weights *and* KV across the participating GPUs via a "meta device" abstraction. | You want more memory than a single GPU provides and your priority is fast token generation... Treat as experimental as the code is less mature than pipeline parallelism. Performance should be good for multiple NVIDIA GPUs using the CUDA backend, no guarantees otherwise. |

> "**KV cache types must be non-quantized: `f32`, `f16`, or `bf16`. Support for quantized KV cache is not implemented and trying to use it will result in an error.**"

> "| Performance is worse with multi-GPU than single-GPU | The performance is bottlenecked by GPU interconnect speed. For `--split-mode tensor`, verify that NCCL is being used. **Try `--split-mode layer`** (less communication than `tensor`). Increase GPU interconnect speed via more PCIe lanes or e.g. NVLink (if available). |"

Also from the same doc:
- `--fit` "**Not supported with `tensor`**."
- `tensor` is **not implemented** for MoE/hybrid and SSM/RWKV architectures (Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H, Granite-Hybrid, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1, Mamba/Mamba2, PLAMO2, MiniCPM3, Gemma-3n, OLMo2, BitNet, T5) → *"`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '...'"*. **A dense 27B is fine.**
- NCCL: *"it's selected at build time (`-DGGML_CUDA_NCCL=ON`, **this is the default**)"*, and *"When NCCL is missing on a multi-GPU build, you'll see this one-time warning and **performance will be lower**: `NVIDIA Collective Communications Library (NCCL) is unavailable, multi GPU performance will be suboptimal`"*.
- `GGML_CUDA_P2P`: *"It is **opt-in** at runtime ... P2P requires driver support (usually restricted to workstation/datacenter GPUs) and **may cause crashes or corrupted outputs**."*

**Direct answer to "does `--split-mode tensor` require P2P or fall back to slow paths?":** upstream does **not** require P2P. It requires **NCCL at build time**, `-fa`, non-quantized KV, and a supported architecture. Without NCCL it falls back to an internal all-reduce and *"performance will be lower"*. P2P is a separate opt-in that is **off by default** and does not gate correctness.

**Critical conflict to verify on your build:** the docs say quantized KV + `tensor` **errors out**, but the habr 2×3090 run used `-ctk q4_0 -ctv q4_0` with `-sm tensor` on build 10666 and reported 74.8 tok/s. Open feature requests exist — **Issue #21788 "Feature Request: Allow `SPLIT_MODE_TENSOR` with KV cache quantization"** and **Issue #23567 "Support quantized KV cache with `--split-mode tensor`"** — and **PR #23225 "ggml: allow split-mode tensor to use different quantization types"** was **closed unmerged** (~2026-06-01, per the SemanticDiff mirror showing "closed this 115 days ago" relative to 2026-09-24): https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/23225/overview · https://github.com/ggml-org/llama.cpp/issues/21788 · https://github.com/ggml-org/llama.cpp/issues/23567

**So: test `-sm tensor -ctk q8_0 -ctv q8_0` on your exact binary first. If it errors, you must switch KV to f16 — which for a 27B dense at ≤32 K context is cheap (order of 1–3 GB), but at 240 K+ is not.**

Related, corroborating: **Issue #26409 — "Eval bug: SYCL `--split-mode tensor` is 3x slower than a single GPU, and hangs with a quantized KV cache"** https://github.com/ggml-org/llama.cpp/issues/26409 — measured 3× regression on a *different* backend; do not extrapolate to CUDA, but it shows the quantized-KV+tensor interaction is a real trap. Also **Issue #27198** (2026-08-16) SYCL `-sm tensor` crashes `dev2dev_memcpy` (DEVICE_LOST) on dual Arc Pro B70 *"despite working P2P"*; **Issue #28453** (2026-09-05) Windows V100 NVLink fatal errors with a custom tensor P2P all-reduce; **PR #26610** (2026-08-05) adds RPC `-sm tensor` on 2× Spark over RDMA.

### 4.3 How much is lost to host-staged allreduce, and what per-layer latency?

`MEASURED` — club-3090 `docs/PCIE_P2P.md` (§0a), which contains the most explicit published decomposition I found:

| mode | inter-GPU traffic | does P2P help? |
|---|---|---|
| **PP (`-sm layer`)** | activations at the boundary — **~7 MB/s measured** | **Barely.** Nothing to accelerate |
| **TP (`row`/`tensor`)** | all-reduce every layer — **~1400 MB/s**, many small serialized transfers | **Yes — and via LATENCY, not bandwidth.** |
| EP | all-to-all | not expressible in llama.cpp |

Verbatim, on the latency figure: *"**Where P2P does pay is latency, not bandwidth.** Tensor-parallel all-reduce is many small serialized transfers; the **measured 15.23 µs → 1.01 µs** drop matters far more than the GB/s figure. If you reason only about bandwidth you will wrongly conclude P2P cannot help — utilisation is often under 20% while the latency saving is 15×."*

**So the reported host-staged vs P2P per-layer sync latency is 15.23 µs → 1.01 µs** (a 15× reduction), on a rig with a patched driver; and the all-reduce payload is ~1400 MB/s versus ~7 MB/s for layer split. https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/PCIE_P2P.md (page carries updates dated 2026-07-30 and 2026-08-04)

**But for llama.cpp specifically, more peer bandwidth bought nothing:**

> "**llama.cpp's measured gain is ZERO.** A community 3× 3090 rig with transfer-verified P2P (**6.08 → 13.17 GB/s, 2.2×**) on its *best* pair saw **no TPS change at all** with `GGML_CUDA_P2P=1` vs unset, `--split-mode tensor` at ~150 TPS. **Single-stream llama.cpp moves ~2 GB/s against 13 GB/s available — it is not interconnect-bound, so P2P has no headroom to give back.**"

and, for the tensor path on a patched dual-3090 rig forced to host staging:

> "*`--split-mode tensor` hangs in warmup* → **`NCCL_P2P_DISABLE=1`** — verified: boots 16 s, **benches on par with `layer`** (**141.6 vs 140.7** same-session)."
> "*P2P 'makes no difference' on llama.cpp* → Peer access is **opt-in** via `GGML_CUDA_P2P` (off by default), and `layer` split barely uses the link. Set `GGML_CUDA_P2P=1` *and* use `row`/`tensor`. **Expect ~0 anyway** (§6)."
> "On the **shipped llama.cpp image, patched P2P buys essentially nothing at any split mode** — and there are *two* independent reasons ... `-sm layer`/PP: +0.2 % decode · ~+1 % prefill (disc #921) ... `-sm tensor` (TP=2): ~noise: **61.5→64.2** narrative, **73.5→72.2** code, prefill **1313→1346** (disc #903) — there *is* an all-reduce, but **the ggml-org image is built WITHOUT NCCL**."

The last point matters enormously for you: *"Verified independently: `ghcr.io/ggml-org/llama.cpp:server-cuda-b10236` contains **zero NCCL symbols and no NCCL linkage**."* The ggml-org image logs at boot:

```
NCCL not compiled in; falling back to internal AllReduce.
Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.
```

**If you run a stock ggml-org Docker image, `-sm tensor` may be silently running the internal all-reduce fallback, not NCCL.** habr used `full-cuda` build 10666 and got a large gain, so either that image has NCCL or the internal path was good enough at 2 GPUs. **Check your image for NCCL symbols before trusting any tensor-mode result.**

### 4.4 Is `layer` faster than `tensor` on 4-GPU non-P2P? — the honest answer

**There is no clean 4× 3080/3090-no-NVLink `tensor`-vs-`layer` llama.cpp measurement in the public record I could reach.** What exists:

| Evidence | n GPUs | P2P? | Verdict |
|---|---|---|---|
| habr 2026-08-29, 2× 3090 | 2 | **no NVLink** | **`tensor` much faster for decode (+27.7 % short ctx → +61 % @240 K), −20 % prefill @40 K** |
| club-3090 PCIE_P2P, patched dual-3090 with `NCCL_P2P_DISABLE=1` | 2 | host-staged | **`tensor` ≈ `layer`** (141.6 vs 140.7) |
| club-3090 PCIE_P2P, 3× 3090, transfer-verified P2P, `-sm tensor` ~150 TPS | 3 | P2P verified | **P2P on/off = 0 change** |
| bric.pe.kr 2026-05-23, `row` vs `layer` on Pascal/Turing PCIe-only | 2 | no | **`layer` beats `row` by 10–28 %**; their table says "2× RTX 3090 PCIe only: Layer wins (~5–10 %)" |
| upstream docs 2026-09-24 | any | any | `row` **deprecated**; on poor interconnect, **"Try `--split-mode layer`"** |
| vLLM TP=2 vs TP=4, 4× 3090, P2P verified, matched recipe (@alesha-pro, 2026-07-27) | 4 | P2P verified | decode **+4.0 % / −2.6 %** (flat); prefill **+54.6 % @10K, +62.0 % @90K**; TTFT@90K 73.1→45.6 s |

**The `row` vs `tensor` distinction is the crux and is widely conflated.** `row` (old, deprecated) loses on PCIe-only — that is the bric.pe.kr result and the community folklore. `tensor` (new, NCCL-based, "meta device") wins on the one clean non-NVLink dual-3090 measurement — that is the habr result. Both are true. **Do not apply the "layer is faster on non-P2P" folklore to `tensor`.**

Two reasons to expect the habr gain **not** to transfer to your 4-card rig:
1. Your 78.3 tok/s already exceeds the serialized 4-card ceiling (§0), so you are already getting cross-token pipelining that tensor-parallel is supposed to provide.
2. The all-reduce participant count goes 2 → 4, and each added rank adds a serialized host-staged exchange. club-3090's PCIE_P2P reasoning on why vLLM caps custom all-reduce at 2 PCIe GPUs is exactly this: *"at 4 on a shared PCIe fabric the N−1 peer writes per rank contend for the same host-bridge bandwidth."*

`ESTIMATED`: on your rig I would budget **tensor-split decode upside of roughly +0 % to +15 %**, not +28 %, with a real **prefill cost of −10 % to −25 %**. Measure it — `llama-bench -p 512 -n 128` in each mode, three reps, same session.

Also note: club-3090's @alesha-pro 4× 3090 result shows **more cards + more watts ≠ single-stream speed** — *"their TP=2 @220 W beats the published TP=4 @350 W row single-stream."* Your 4-card decode advantage over a 2-card rig is small; the 4 cards buy **capacity** (29 GB model + long KV), not linear speed.

### 4.5 Where tensor mode costs you something real

- **Prefill**: −20.5 % at 40 K, −11 % at 130 K, −2.7 % at 240 K (habr). For RAG/agent workloads with big prompts, this is the whole trade.
- **Sampling on CPU** (once per slot), habr.
- **VRAM**: +1.2 GB (habr).
- **`-ts` and `--fit` unavailable**; you hand-size `-c`.
- **Quantized KV may be forbidden** (upstream docs vs habr — verify on your binary).
- **The one documented correctness hazard** (vLLM-specific, not llama.cpp, but worth knowing): club-3090's `PCIE_P2P.md` reports a rig where a patched P2P collective **completed fast and returned wrong output** with every indicator green, caught only by reading generated text — *"That is strictly worse than a hang."* Their `verify-full` ladder exists for this. If you enable `GGML_CUDA_P2P`, generate real text and read it.

---

## 5. Practical recommendations for this rig, ordered by expected value

| # | Action | Expected effect | Confidence | Basis |
|---|---|---|---|---|
| 1 | **Check whether your 27B GGUF has an MTP head; if so enable `--spec-type draft-mtp --spec-draft-n-max 3 -ctkd q4_0 -ctvd q4_0`** | **+50 % to +81 %** decode at short context | HIGH, if the head exists | habr 2026-08-29: 32.4 → 58.5 (layer); measured draft-depth optimum 3–4 |
| 2 | **Raise power caps back to ~290–370 W** if you reduced them | up to **+50 %** vs 230 W; ~+7 % vs 290 W | HIGH for llama.cpp on 3090 | club-3090: 25.15 → 38.23 tok/s, 230 W → 388 W |
| 3 | **Rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON`** and confirm FA is not falling back | **~+7 %** (prefill; decode benefit type-dependent) | MEDIUM | @WaveboSF in #20969, 2026-04-04; FR #24485; PR #28456 |
| 4 | **Verify your build has NCCL**, then A/B `-sm tensor` vs `-sm layer` at short/32 K/128 K | **−5 % to +15 %** decode on 4 cards; **−10 % to −25 % prefill** | LOW (extrapolated from 2 cards) | habr (2 cards, +28 %); club-3090 (tensor≈layer when host-staged); upstream docs (EXPERIMENTAL) |
| 5 | **A/B memory offset +500 → +1000; stop at the EDR plateau** | **+3 % to +10 %** decode, if DRAM-limited | LOW-MEDIUM | arithmetic (§1.2); EDR mechanism (Hardwareluxx); only non-Ampere OC datapoint = +~10 tok/s |
| 6 | **Kill `--cache-type-k/-v q8_0`** toward `q4_0`, especially at long context | −2 % to +5 % at short ctx; +3.5 % to +5.3 % at 32 K on SM86; −51 % in the worst reported case | MEDIUM | Defilan (q8_0 = −1.6/−2.9 %); Madreag SM86 (+3.5/+5.3 % @32 K); steveseguin (SYCL, −51 %) |
| 7 | **NUMA: pin GPUs + host memory to one EPYC node; test `--numa` / `numactl`** | UNMEASURED, potentially large (host-staged traffic) | UNKNOWN | §3.10; retrieve Discussion #11733 |
| 8 | **Lock memory clock to max and try a low `-lgc`** | TPS/W up; total TPS likely *down* | LOW on Ampere | apnar's 5090 35-point sweep; Ampere `-lmc` support uncertain |
| 9 | `nvidia-smi -pm 1` | latency hygiene; no published tok/s | — | club-3090 HARDWARE |
| 10 | `--no-mmap --mlock`, `-ub 1024`, `--tensor-split` asymmetry | ≈0 decode; ~+2 % decode for `-ub 1024` | HIGH | club-3090: "A/B'd FLAT" |
| 11 | `CUDA_DEVICE_MAX_CONNECTIONS`, `GGML_CUDA_FORCE_MMQ/CUBLAS`, `GGML_CUDA_MMV_Y` | **no evidence** — leave alone | — | NULL results |
| 12 | PCIe lane width / gen | **decode ~insensitive; prefill & TTFT sensitive**. Host CPU class dominates | MEDIUM | MoppelMat vs ryanmpelletier; mgabor3141 vs OVDEN13 |

**Measurement protocol** (worth adopting verbatim from habr, which measured all of the above): `llama-server` returns `timings` with `predicted_per_second` and `prompt_per_second` in the `/completion` response — no separate bench needed. Three runs, `"cache_prompt": false`, `temperature: 0`, one short prompt and one long (32 K / 130 K / 240 K). *"Если разброс больше пары процентов — сравнивать рано."* [If the spread is more than a couple of percent, it is too early to compare.]

---

## 6. Queries / sources that found **nothing**

Reported as required:

1. **Any measured before/after llama.cpp (or ik_llama.cpp) decode tok/s for a GDDR6X memory offset** on 3080 / 3080 Ti / 3090. Nine distinct queries (§1.1); GitHub search for `"memory overclock"` in `ggml-org/llama.cpp` returned 3 unrelated results.
2. **ECC on/off for consumer GDDR6X** — no such thing exists on GeForce; no data.
3. **Ampere-specific `nvidia-smi -lgc` / `-lmc` LLM measurement** — only the Blackwell 5090 35-point sweep exists, and club-3090 explicitly says clock-locking "is not portable to Ampere".
4. **`CUDA_DEVICE_MAX_CONNECTIONS` measured effect on multi-GPU llama.cpp** — the only repo hit is an unrelated Jetson hang (#19219).
5. **`GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_MMV_Y`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY` measured decode deltas** — none found. `GGML_CUDA_DMMV_X` is **obsolete** (DMMV removed in PR #10318).
6. **`numactl --cpunodebind` / `--membind` / `--interleave` tok/s delta on a host-staged multi-GPU llama.cpp rig** — none found. The most promising source, llama.cpp **Discussion #11733** ("Dual Epyc Genoa/Turin token generation performance bottlenecks"), **could not be retrieved** (github.com blocked; API 403). A search snippet shows it contains `NPS1 | likwid-bench load | 460.8 | 359.9 | 78.1%`-style measured NUMA rows. **Manual retrieval recommended.**
7. **`--no-kv-offload` perf measurement** — none.
8. **4× RTX 3080 (20 GB mod) decode benchmarks** — none specific to 4 cards. Nearest 2-card Ampere-3080-20GB data points (all vLLM, club-3090):
   - **@troymroberts, 2026-05-02, 2× 3080 20 GB @ 200 W/card, TQ k8v4 KV + MTP K=3, `gpu-memory-utilization=0.82`, 262 K target, KV pool 5.2 GB/card:** *"verify-full 10/10 pass; bench **49 TPS wall single-stream, 210 TPS aggregate at n=8**. First published SM86 / 40 GB combined data point outside the 3090 family."* https://github.com/noonghunna/club-3090/discussions/25#discussioncomment-16787782
   - **@efschu, 2026-05-04, 2× 3080 20 GB, `--kv-cache-dtype fp8_e5m2`, mem-util 0.82:** *"bench **82.4 narr / 107.9 code TPS**, full 257K-token auto-discovery needle PASS at 90 % depth"* https://github.com/noonghunna/club-3090/issues/47
   - **@efschu, 2026-07-14, 2× 3080 20 GB + 1× 5090 (uneven TP=3, custom `shvllm` fork):** 68.1 / 87.6 decode; prefill the ceiling (TTFT 78 s @ 90 K) https://github.com/noonghunna/club-3090/issues/707
9. **A clean 4-GPU (≥3 cards) non-P2P `tensor` vs `layer` llama.cpp decode comparison** — does not exist publicly as far as I could reach. §4.4 has the fragments.
10. **llama.cpp issue/PR bodies for #24485, #24489, #23567, #21788, #26409, #27198** — titles/dates/URLs only. `github.com` HTML was unfetchable from this session and the unauthenticated GitHub REST API returned **HTTP 403** after a handful of calls.

---

## 7. Source index (every number above, with date)

| Source | Date | URL |
|---|---|---|
| Hardwareluxx — GA102 EDR / GDDR6X bandwidth plateau | 2020-09-13 ⚠STALE | https://www.hardwareluxx.ru/index.php/news/hardware/grafikkarten/50330-podrobnosti-ampere-tekhnologiya-error-detection-and-replay-edr-uproshchaet-razgon-gddr6x.html |
| TechPowerUp — why 3080 uses 19 Gbps (cites Igor's Lab IR measurement, 104 °C) | 2020-09-22 ⚠STALE | https://www.techpowerup.com/272420/the-reason-why-nvidias-geforce-rtx-3080-gpu-uses-19-gbps-gddr6x-memory-and-not-faster-variants |
| Igor's Lab — GDDR6X >100 °C on 3080 FE (primary) | 2020-09 ⚠STALE | https://www.igorslab.de/en/gddr6x-am-limit-ueber-100-grad-bei-der-geforce-rtx-3080-fe-im-chip-gemessen-2/ |
| Johannes Gaessler — llama.cpp performance testing (perf ∝ memory bandwidth/frequency) | undated ⚠STALE | https://johannesgaessler.github.io/llamacpp_performance |
| llama.cpp — `docs/multi-gpu.md` (split modes, tensor constraints, NCCL, P2P) | retrieved 2026-09-24 | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multi-gpu.md |
| llama.cpp — Discussion #20969 TurboQuant (all KV-quant measurement tables) | 2026-03-25 → 2026-04-10 | https://github.com/ggml-org/llama.cpp/discussions/20969 |
| llama.cpp — Discussion #21155 TurboQuant ICLR 2026 | 2026 | https://github.com/ggml-org/llama.cpp/discussions/21155 |
| llama.cpp — Issue #21788 FR: allow SPLIT_MODE_TENSOR with KV quant | 2026 | https://github.com/ggml-org/llama.cpp/issues/21788 |
| llama.cpp — Issue #23567 support quantized KV with `-sm tensor` | 2026 | https://github.com/ggml-org/llama.cpp/issues/23567 |
| llama.cpp — PR #23225 tensor split quants (CLOSED unmerged ~2026-06) | 2026 | https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/23225/overview |
| llama.cpp — Issue #24485 FR: make GGML_CUDA_FA_ALL_QUANTS default | 2026 | https://github.com/ggml-org/llama.cpp/issues/24485 |
| llama.cpp — PR #28456 warn on FA→CPU fallback for unsupported KV types (Open) | 2026 | https://app.semanticdiff.com/gh/ggerganov/llama.cpp/pull/28456/overview |
| llama.cpp — Issue #26409 SYCL `-sm tensor` 3× slower than single GPU | 2026 | https://github.com/ggml-org/llama.cpp/issues/26409 |
| llama.cpp — Issue #27198 SYCL `-sm tensor` DEVICE_LOST (dual B70) | 2026-08-16 | https://github.com/ggml-org/llama.cpp/issues/27198 |
| llama.cpp — Issue #28453 Windows V100 NVLink tensor P2P all-reduce fatal errors | 2026-09-05 | https://github.com/ggml-org/llama.cpp/issues/28453 |
| llama.cpp — PR #26610 RPC `-sm tensor` (2× Spark, RDMA) | 2026-08-05 | https://github.com/ggml-org/llama.cpp/pull/26610 |
| llama.cpp — Issue #19219 CUDA_DEVICE_MAX_CONNECTIONS (Jetson hang, unrelated) | 2026-01-30 | https://github.com/ggml-org/llama.cpp/issues/19219 |
| llama.cpp — Issue #27330 CUDA graphs hang on sm_120; `GGML_CUDA_DISABLE_GRAPHS=1` | 2026-08-18 | https://github.com/ggml-org/llama.cpp/issues/27330 |
| llama.cpp — Issue #11733 Dual EPYC Genoa/Turin token-gen bottlenecks (**NOT RETRIEVED**) | 2026 | https://github.com/ggml-org/llama.cpp/discussions/11733 |
| llama.cpp — PR #16000 `--numa mirror` | 2025-09-15 ⚠STALE | https://github.com/ggml-org/llama.cpp/pull/16000 |
| llama.cpp — PR #8075 MMQ instead of cuBLAS by default | — | https://github.com/ggml-org/llama.cpp/pull/8075 |
| llama.cpp — PR #10318 remove DMMV (retires `GGML_CUDA_DMMV_X`) | — | https://github.com/ggml-org/llama.cpp/pull/10318 |
| habr.com (kotafey) — 2× RTX 3090, no NVLink, layer vs tensor, MTP | 2026-08-29 | https://habr.com/ru/articles/1076080/ |
| club-3090 — `docs/PCIE_P2P.md` (7 MB/s vs 1400 MB/s; 15.23 µs → 1.01 µs; llama.cpp P2P = 0) | 2026-07-30 / 2026-08-04 | https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/PCIE_P2P.md |
| club-3090 — `docs/HARDWARE.md` (power-cap table, lgc/lmc, EDR-adjacent, PCIe Gen3/CPU, 3080 20 GB rows) | 2026-05 → 2026-08 | https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/HARDWARE.md |
| club-3090 — `BENCHMARKS.md` (TP=2/TP=4, lane width, `-b/-ub/--no-mmap` FLAT, Luce K8V4) | 2026-04 → 2026-08 | https://raw.githubusercontent.com/noonghunna/club-3090/master/BENCHMARKS.md |
| club-3090 — `docs/engines/LLAMA_CPP.md` (q8_0 "~3× slower" claim, q4_0 262 K recipe) | 2026 | https://raw.githubusercontent.com/noonghunna/club-3090/refs/heads/master/docs/engines/LLAMA_CPP.md |
| club-3090 — Discussion #86 (apnar 5090 35-point mem-clock × GPU-clock sweep) | 2026 | https://github.com/noonghunna/club-3090/discussions/86#discussioncomment-16845745 |
| club-3090 — Discussion #25 (2× 3080 20 GB, 49 TPS wall / 210 agg) | 2026-05-02 | https://github.com/noonghunna/club-3090/discussions/25#discussioncomment-16787782 |
| club-3090 — Issue #47 (2× 3080 20 GB fp8, 82.4/107.9) | 2026-05-04 | https://github.com/noonghunna/club-3090/issues/47 |
| club-3090 — Issue #137 (Gen3 Xeon KVM 32/41 vs Zen3 Gen4 89/117) | 2026 | https://github.com/noonghunna/club-3090/issues/137 |
| club-3090 — Issue #707 (2× 3080 + 5090 uneven TP=3) | 2026-07-14 | https://github.com/noonghunna/club-3090/issues/707 |
| Madreag/turbo3-cuda — CUDA turbo KV, SM86 3090/3090 Ti + 5090 tables | 2026 | https://github.com/Madreag/turbo3-cuda/blob/release/cuda-optimized/README.md |
| TheTom/turboquant_plus — upstream merge map, asymmetric K/V | 2026 | https://raw.githubusercontent.com/TheTom/turboquant_plus/main/README.md |
| steveseguin B70 — context+KV sweep (SYCL; q8_0 KV −51 % @32 K) | 2026-08-22 | https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/qwen38-27b-b70/notes/2026-08-22-qwen38-q4km-tp1-context-kv-sweep-finding.md |
| sztlink/turboquant-cuda-bench — SM86 −71 % vs SM89 −54 %, 0→131 K | 2026-04-27 | https://github.com/sztlink/turboquant-cuda-bench/blob/main/bench/iq4nl-repro/results.md |
| rtx6kpro — daily summary (VRAM +6000 mem → ~+10 tok/s decode) | 2026-07-30 | https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/daily-summaries/2026-07/2026-07-31.md |
| rtx6kpro — Blackwell power-limit sweep (mem clock invariant; memcpy PL-insensitive) | 2026 | https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/hardware/blackwell-power-limit-sweep.md |
| bric.pe.kr — `row` vs `layer` on 2× 1080 Ti, no NVLink, tensor-split asymmetry | 2026-05-23 | https://bric.pe.kr/blog/llama-cpp-split-mode-row-vs-layer-old-gpu |
| knightli.com — multi-GPU expectations (qualitative, no numbers) | 2026-05-09 | https://knightli.com:8443/en/2026/05/09/llama-cpp-multi-gpu-offload-performance/ |
| lance0/rookery — GPU bandwidth/rope table, KV quant guidance | 2026 | https://raw.githubusercontent.com/lance0/rookery/main/docs/gpu-guide.md |
