# Speculative Decoding Research Report — llama.cpp 2025/2026

**Compiled:** 2026 (research session date). **Method:** primary source fetch of `ggml-org/llama.cpp` master + practitioner benchmark repos + papers. Every number is labelled **MEASURED** (someone ran it and reported the number) or **ESTIMATED** (computed here from other numbers).

## Source index (all fetched this session)

| # | Source | Date / version |
|---|---|---|
| S1 | `docs/speculative.md` — https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md | master @ `f830688e91214cccfa25ed0a2b9a708ee5a855c3`, commit dated **2026-09-24T06:57:31Z** |
| S2 | `common/arg.cpp` — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp | same commit |
| S3 | `common/common.h` (struct defaults) | same commit |
| S4 | llama.cpp issue #23322 "Low MTP Draft Acceptance Rate with SWA/Hybrid Memory Models (Qwen3.6)" | created **2026-05-19T07:01:53Z** |
| S5 | llama.cpp issue #23658 "MTP draft acceptance rate collapses at specific context sizes" | created **2026-05-25T10:16:53Z**, closed 2026-05-26 |
| S6 | `tfriedel/qwen3.6-rtx3090-lab` README + FINDINGS | Qwen3.6/3.8 era; bench dates **2026-07-12 / 07-13** |
| S7 | `thc1006/qwen3.6-speculative-decoding-rtx3090` README + v4 audit README | archival 2026-04-21→05-07; controlled tier **2026-08-25→08-31**; audit 2026-08-25 |
| S8 | `steveseguin/b70-optimization-lab` MTP sweeps (Gemma-4-26B-A4B) | **2026-06-23** |
| S9 | `aliasocracy/enoch-ai-research-corpus` Qwen3-32B + Qwen3-0.6B suite | artifact dated 2026-05-01 |
| S10 | AngelSlim EAGLE-3 model cards (Qwen3-1.7B/4B/8B/14B/32B) | card updated 2026-01-13 |
| S11 | arXiv **2601.11580v2** "Speculative Decoding: Performance or Illusion?" | v1 2025-12-31, v2 **2026-03-18** |
| S12 | `steven0226/speculative-decoding-bench-rtx4090` dataset + EVAL_REPORT | llama.cpp commit `e3546c7`, **2026-07-11** |
| S13 | https://openclawdc.com/blog/speculative-decoding-slower-llama-cpp/ | "Why Speculative Decoding Made My Local LLM Slower" |
| S14 | Leviathan et al. 2023 (formula), via S11 Eq. (1) | — |
| S15 | llama.cpp PR #18039 (EAGLE-3 support), #22105 (DFlash), #25173 (DSpark) | EAGLE-3 merged; DFlash merged **2026-06-28** |

---

## Q1 — llama.cpp speculative-decoding flags: current names, defaults, migration

### `--spec-type` accepted values — **verbatim from S1 (master, 2026-09-24)** and S2 enum

```
--spec-type [none|draft-simple|draft-eagle3|draft-dflash|draft-dspark|draft-mtp|ngram-cache|
             ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
                                        comma-separated list of types of speculative decoding to use
                                        (default: none)
                                        (env: LLAMA_ARG_SPEC_TYPE)
--spec-default                          use default speculative decoding config
                                        (enables ngram-mod)
```

Enum in `common/common.h` (S3, lines 171–183) confirms exactly 11 members:
`NONE, DRAFT_SIMPLE, DRAFT_EAGLE3, DRAFT_MTP, DRAFT_DFLASH, DRAFT_DSPARK, NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD, NGRAM_CACHE`.

**Flags that do NOT exist in current master** (verified by exact-string grep of S2 — these are *reporting-nothing* results):

| Queried name | Result |
|---|---|
| `--spec-replace` | **NOT FOUND** in `common/arg.cpp` |
| `lookup` (as a `--spec-type`) | **NOT FOUND**. Only `-lcs/--lookup-cache-static`, `-lcd/--lookup-cache-dynamic` exist (S2 L1622, L1629) |
| `--draft-p-min` | **EXISTS as a legacy alias** of `--spec-draft-p-min` (S2 L4193) |
| `--draft-p-split` | **EXISTS as a legacy alias** of `--spec-draft-p-split` (S2 L4186) |

### Core flag list — verbatim from S1 §"Draft Model Parameters"

```
--spec-draft-model, -md, --model-draft  FNAME
                                        draft model for speculative decoding (default: unused)
                                        (env: LLAMA_ARG_SPEC_DRAFT_MODEL)
--spec-draft-hf, -hfd, -hfrd, --hf-repo-draft  <user>/<model>[:quant]
                                        HuggingFace repository for the draft model
--spec-draft-n-max                      N
                                        number of tokens to draft for speculative decoding (default: 3)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MAX)
--spec-draft-n-min                      N
                                        minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MIN)
--spec-draft-p-split, --draft-p-split   P
                                        speculative decoding split probability (default: 0.10)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_SPLIT)
--spec-draft-p-min, --draft-p-min       P
                                        minimum speculative decoding probability (greedy) (default: 0.00)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_MIN)
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft  N
                                        max. number of draft model layers to store in VRAM, either an exact
                                        number, 'auto', or 'all' (default: auto)
--spec-draft-device, -devd, --device-draft  <dev1,dev2,..>
                                        comma-separated list of devices to use for offloading the draft model
                                        (none = don't offload, default: follows --device)
```

Defaults confirmed in `common/common.h` (S3) `struct common_params_speculative_draft`:

```cpp
int32_t n_max = 3;    // maximum number of tokens to draft during speculative decoding
int32_t n_min = 0;    // minimum number of draft tokens to use for speculative decoding
float p_split = 0.1f; // speculative decoding split probability
float p_min   = 0.0f; // minimum speculative decoding probability (greedy)
```

Also present and relevant (from S2): `--spec-synth-rates P0,P1,...` and `--spec-synth-len L` (benchmarking-only synthetic acceptance, S1 §"Synthetic Acceptance"), `--spec-draft-backend-sampling` / `--no-spec-draft-backend-sampling`, `--spec-draft-conf-min P` (documented in S1 for `draft-dspark` only), plus per-ngram-family knobs `--spec-ngram-mod-n-match` (24), `--spec-ngram-mod-n-min` (48), `--spec-ngram-mod-n-max` (64), `--spec-ngram-{simple,map-k,map-k4v}-size-n` (12), `-size-m` (48), `-min-hits` (1).

### What changed vs the old `--draft-max/--draft-min` API

This is now a **hard removal, not a deprecation** — the old spellings throw. Verbatim from S2 (L4383–4391):

```cpp
    add_opt(common_arg(
        {"--draft", "--draft-n", "--draft-max"}, "N",
        "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use --spec-draft-n-max or --spec-ngram-mod-n-max");
        }
    ).set_spec()...set_env("LLAMA_ARG_DRAFT_MAX"));
    add_opt(common_arg(
        {"--draft-min", "--draft-n-min"}, "N",
        "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use --spec-draft-n-min or --spec-ngram-mod-n-min");
        }
    ).set_spec()...set_env("LLAMA_ARG_DRAFT_MIN"));
```

with:

```cpp
[[noreturn]] static void arg_removed(const std::string & msg) {
    throw std::invalid_argument("the argument has been removed. " + msg);
}
```

Same treatment for `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` (now per-family).

**The single most important behavioural change for this user** — independent corroboration from S7 (README, §Reproduction):

> "Note that llama.cpp renamed the speculative arguments after `bcb5eeb64` (`--draft-max` → `--spec-draft-n-max`) and that **`--spec-type` now defaults to `none`, so `-md` alone loads a draft model and never speculates**; the runner's `BENCH_FLAVOR` switch handles both spellings."

An independent 2026-07 notebook (S12 `docs/PLAN.md`) states the same and adds a per-request detail:

> "**2026-06 起 `-md` 不再自動啟用，必須 `--spec-type draft-simple`**；`--spec-draft-n-max` 預設 16→3、`--draft-p-min` 預設 0.00；**PR #22838 起可在 HTTP request body 覆寫 n_max/n_min/p_min**（調整不用重啟 server）"

i.e. as of 2026-06, `-md` no longer auto-enables speculation; `--spec-draft-n-max` default went 16→3; and from PR #22838 the HTTP request body can override `n_max/n_min/p_min`. Acceptance is read from response `timings.draft_n` / `draft_n_accepted` and the server log line `draft acceptance = ...`; `llama-bench` does **not** support draft. (Statement from a third-party notebook, not from the llama.cpp repo itself — treat as **secondary**.)

---

## Q2 — Tuning `--spec-draft-n-max`, and the MTP picture

### 2a. Sweeps across n_max — the core evidence table

**MEASURED.** All rows are real published runs. Note the split between *self-speculative* (MTP/DFlash, target's own layers) and *external draft model* — the two behave completely differently.

| n_max | Target / drafter | aggregate tok/s | vs no-spec | acceptance | Source, date |
|---:|---|---:|---:|---:|---|
| 1 | Qwen3.5-0.8B draft → Qwen3.6-35B-A3B | 31.1 | **−74.8 %** | 68.7 % | S7 Run C, 2026-08-25 |
| 2 | same | 34.2 | −72.2 % | 60.3 % | S7 Run C |
| 4 | same | **35.6** | −71.1 % | 45.4 % | S7 Run C |
| 8 | same | 32.1 | −74.0 % | 29.7 % | S7 Run C |
| 16 | same | 23.7 | −80.8 % | 15.0 % | S7 Run C |
| 32 | same | 17.3 | −86.0 % | 8.0 % | S7 Run C |
| — | *baseline* | *123.4* | — | — | S7 Run C |
| 1 | DFlash (target's own layers) | 120.2 | +8.7 % | 82.0 % | S7 Run K, 2026-08-26 |
| 2 | DFlash | 129.5 | +17.1 % | 72.8 % | S7 Run K |
| **3** | **DFlash** | **130.0** | **+17.6 %** | **63.6 %** | S7 Run K |
| 4 | DFlash | 129.8 | +17.3 % | 55.6 % | S7 Run K |
| 6 | DFlash | 100.8 | −8.9 % | 43.0 % | S7 Run K |
| 8 | DFlash | 93.2 | −15.8 % | 37.2 % | S7 Run K |
| — | *baseline* | *110.6* | — | — | S7 Run K |

Verbatim conclusion from S7 Run K:

> **"There is no peak, there is a plateau and then a cliff.** `n_max` 2, 3 and 4 land at 129.5, 130.0 and 129.8; separated by less than the run-to-run SD of the baseline, so calling any one of them "the optimum" would be reading noise. What the sweep does establish is the shape: one token of draft is not enough (+8.7 %), two to four are worth about +17 %, and by six the arm is already losing. The sign change sits between 4 and 6."

**MTP-specific n-sweep on the same rig (S7 Run M/O2, headline Latin-square table):**

| arm | pooled tok/s | change | acceptance |
|---|---:|---:|---:|
| `spec-dflash-n2` | **146.2** | **+26.3 %** | 72.3 % |
| `spec-mtp-n2` | 141.9 | +22.7 % | 78.4 % |
| `spec-dflash-n4` | 137.9 | +19.2 % | 55.2 % |
| *no speculation* | *115.7* | — | — |
| `ngram-map-k4v-m8` | 115.4 | −0.3 % | 50.0 % |
| `ngram-mod-n24` | 103.1 | −10.9 % | 5.0 % |
| `ngram-cache` | 93.7 | −19.0 % | 5.2 % |
| `spec-draft-n8` (0.8B external) | 30.9 | −73.3 % | 29.5 % |
| `spec-draft-n1` (0.8B external) | 29.2 | **−74.8 %** | **69.7 %** |

Note `spec-mtp-n2` = 78.4 % acceptance — **the highest acceptance in the entire matrix, and still only +22.7 %** because MTP's per-round cost is real.

**vLLM MTP k-sweep, Qwen3.6-27B dense on 2× RTX 3090 (S6, 2026-05-06; re-benched 2026-07-13):**

| Config | narrative-story | code-mergesort | narrative-explain |
|---|---:|---:|---:|
| baseline (no MTP, prefix cache ON) | 149 | 149 | 149 |
| MTP k=1 | 162 (+8 %) | 178 (+19 %) | 165 (+11 %) |
| MTP k=2 | 182 (+22 %) | 229 (+54 %) | 191 (+28 %) |
| **MTP k=3** ← chosen | **179 (+20 %)** | **264 (+77 %)** | **200 (+34 %)** |
| MTP k=4 | 171 (+15 %) | 276 (+85 %) | 190 (+27 %) |

Verbatim (S6):

> "`k=3` is the cross-prompt sweet spot. `k=4` keeps gaining on code (276) but starts regressing on narrative — verify-cost dominates beyond k=3 for natural-language acceptance patterns."

**Crucial caveat — and it is the finding that most changes tuning advice.** S6 FINDINGS, 2026-07-12, after fixing CUDA-graph capture:

> **"Finding 3 — the k=3 MTP optimum was an artifact of the slow verify pass.** With full graphs + custom AR, re-sweeping k: k=3 → 118/154, **k=4 → 117/166**, k=5 → 114/165 (narr/code). k=4 is the new sweet spot; May's "k=4 regresses narrative" disappears once the verify pass is cheap."

So the optimal n_max on MTP is **not a property of MTP** — it moves when the verify pass gets cheaper. Practical read: **sweep n_max again after any change to cudagraph/attention-backend settings.**

**Gemma-4-26B-A4B MTP sweep, Intel Arc Pro B70, llama.cpp `dec5ca557` (S8, 2026-06-23):**

| n_max | tok/s after TTFT | mean accepted length | per-position acceptance |
|---:|---:|---:|---|
| 2 | 46.15 | 2.61 | (0.858, 0.756) |
| **3** | **48.35** | 3.26 | (0.858, 0.752, 0.650) |
| 4 | 43.56 | 3.61 | (0.837, 0.701, 0.598, 0.469) |

Same file, later run on a longer "filled-long" prompt shape (588 in / 512 out): **n=3 → 68.19 tok/s, n=4 → 74.39 tok/s (new best)** — verbatim decision:

> "The filled-long shape changed the optimum. Short-prompt `long` remained best at AOT `n=3`, but filled-long strongly favored `n=4` … `n=2` is too conservative for filled-long."

And the current best fresh-response MTP lane in that repo uses `--spec-draft-n-max 7 --spec-draft-n-min 2 --spec-draft-p-min 0.12` at 92.40 tok/s first-request (S8, `20260623T2301`).

**RTX 4090, Qwen3-8B + Qwen3-0.6B draft, llama.cpp `e3546c7` (2026-07-11) (S12):**

| n_max | 程式碼 (code) tok/s (acc) | 創意對作 (creative) | JSON tok/s (acc) | 翻譯 tok/s (acc) |
|---:|---|---|---|---|
| 2 | 138.2 (89.1 %) | 87.6 (38.2 %) | 139.3 (98.0 %) | 101.1 (53.9 %) |
| 4 | 177.5 (81.9 %) | 80.4 (24.2 %) | 198.8 (95.9 %) | 102.6 (39.0 %) |
| 8 | 196.9 (73.0 %) | 58.1 (12.9 %) | 216.7 (89.4 %) | 73.4 (22.3 %) |
| 16 | 189.5 (65.1 %) | 39.5 (6.5 %) | 269.6 (82.5 %) | 51.7 (11.5 %) |
| *baseline* | *141.3* | *138.3* | *141.6* | *140.6* |

Verbatim recommendation: **"n_max=4 附近是相對安全的折衷值"** (n_max≈4 is the relatively safe compromise) — JSON wants larger, prose/translation gets worse at every n_max.

### 2b. Is 3 optimal for MTP?

**Answer: it is a reasonable default, not a natural constant.** Synthesis of the measured evidence:

- **Dense 27B-class + vLLM MTP:** k=3 was the cross-prompt optimum *before* the cudagraph fix; k=4 became the optimum after (S6).
- **Self-speculative on llama.cpp (DFlash/MTP, single 3090):** n_max 2–4 is a **plateau** (129.5 / 130.0 / 129.8 tok/s) statistically indistinguishable from each other; the cliff is at 6 (S7).
- **MTP on Gemma-4-26B (Arc B70):** n=3 best on short prompts, n=4 best on long/filled prompts (S8).
- **External small draft model:** optimum is much lower (1–4) and the whole family loses on Ampere regardless (S7, S12).

**Practical recommendation for this rig:** keep 3 as the incumbent, but A/B **2, 3, 4** on your own workload — and re-run that A/B after any cudagraph/backend change. Going to 5–6 is where measured curves turn down (S7: −8.9 % at 6; S12: 0.52× at 8 on 翻譯).

### 2c. `--spec-draft-p-min` — measured effect

Defaults: **0.00** (S1/S3). `--draft-p-min` still works as an alias (S2 L4193).

**The one clean published p_min sweep** is S7 Run H (`n_max` 8, external drafter, A3B target, on the binary path where speculation is net-negative):

| configuration | real acceptance | pooled tok/s | vs baseline |
|---|---:|---:|---:|
| no speculation | — | 123.8 | — |
| `n_max` 8, `p_min` 0 | 29.7 % | 32.7 | −73.6 % |
| `n_max` 8, `p_min` 0.75 | **80.2 %** | **42.8** | **−65.5 %** |
| `n_max` 8, `p_min` 0.90 | **88.2 %** | 42.5 | −65.6 % |

Verbatim from S7:

> "**Nine drafted tokens in ten accepted, correctly counted, and still nearly three times slower than not speculating.** The intuition was right; the evidence for it was not … `p_min` also turns out to be the knob that matters rather than `n_max`: at `p_min` 0.75, `n_max` 32 and `n_max` 128 are byte-identical, at 6159 drafted tokens, 70.9 % acceptance and 42.0 tok/s each."

**Read this carefully:** p_min raises acceptance *substantially* (29.7 % → 88.2 %) but on that configuration it does **not** rescue throughput, because it does nothing about the fixed per-round cost. p_min's real job is **capping wasted draft compute**; it is not a speed lever on its own.

**p_min is workload/regime dependent — negative results:**

- S8 (Gemma-4 MTP): `p-min=0.05` and `p-min=0.12` appear in winning configs, but a paired test of `n=3, n-min=2, p-min=0.05` vs plain `n=3` was a **loss** (46.89 vs 47.43 tok/s).
- S9 (Qwen3-32B + 0.6B, temp 0): lowering `p_min` 0.8 → 0.5 produced **no change at all** in accepted counts (23 vs 23, 22 vs 22, 28 vs 28) and ~identical tok/s. Verbatim explanation:

> "This is expected under deterministic (temperature 0) generation: when the draft model's top token is accepted, the probability threshold is irrelevant; when it is rejected, lowering the threshold does not help."

**So: if you run greedy/temp 0, expect p_min to do nothing.** Typical values in the wild: 0.00 (default), 0.05, 0.12, 0.5, 0.75, 0.8, 0.9. Values ≥0.75 are what practitioners reach for to *raise reported acceptance*, not necessarily throughput.

### 2d. Which models support MTP in llama.cpp

From S1 + S2 + third-party reports:

- `--spec-type draft-mtp` "Use Multi Token Prediction (MTP) heads from the main model" — the head lives **in the main model's GGUF**, no separate `-md` needed (S6 FINDINGS confirms for Qwen: "Qwen's MTP head ships in-checkpoint").
- `--mtp` is a **download-time** flag: *"also download the multi-token prediction (MTP) head, if available"* (S2 L3080, under `LLAMA_EXAMPLE_DOWNLOAD`) — i.e. for HF sidecar heads.
- S1 lists EAGLE-3 support for **Qwen3** (`AngelSlim/Qwen3-{1.7B,4B,8B,14B,32B,a3B}_eagle3`, `Tengyunw/qwen3_8b_eagle3`, `Tengyunw/qwen3_30b_moe_eagle3`), **Llama-3.1/3.3**, **Gemma-4 31B/26B-A4B**, **gpt-oss-20b/120b**.
- S1 DFlash/DSpark support: `z-lab/Qwen3-4B-DFlash`, `deepseek-ai/dspark_qwen3_4b_block7`; **"Currently only drafts with a Qwen3 backbone are supported"** for DSpark.
- S15 PR #15225 = *"server: implement GLM-style MTP"* → **GLM-4.5/4.6 family MTP is supported**.
- S11 confirms MTP in vLLM for **DeepSeek/GLM** families and quantifies GLM's decay: *"position-wise acceptance rate (0.92 → 0.68 → 0.38 on GPQA-Main)"*.

**Known MTP failure modes worth knowing (both are real, both bite at long context):**

- S4 (issue #23322, 2026-05-19): Qwen3.6-27B-MTP on SWA/hybrid — acceptance **0.367 / 0.833 / 0.354** across tasks, with `forcing full prompt re-processing due to lack of cache data`. Reporter expected "70–90 % for models with compatible attention mechanisms".
- S5 (issue #23658, 2026-05-25): MTP acceptance collapses at ~2048-aligned context sizes — ctx 12032 → 1.12× @ **16 % AR**; ctx 12288 → **1.91× @ 71 % AR**. Later table: 17024 → 0.88× @ 0 %, 18432 → 1.68× @ 61 %, 24064 → 0.89×, 24576 → 1.75×. Closed by a maintainer as AI-generated noise, but a commenter (2026-07-30) reported reproducing the same collapse pattern from an **unstable undervolt**, fixed by restoring default voltage.
  - **Actionable:** if you see MTP acceptance collapse at particular ctx sizes, check for 2048-aligned ctx values and verify GPU stability/undervolt before filing a bug.

---

## Q3 — MTP head vs separate draft model for a 27B/32B Qwen-class target

### 3a. The headline answer: on this class of hardware, self-speculation wins and external drafts lose

**MEASURED, single RTX 3090, Qwen3.6-35B-A3B, llama.cpp `3737e4137`, Latin-square controlled, 2026-08-26 (S7):**

| drafter type | best arm | tok/s | vs baseline | acceptance |
|---|---|---:|---:|---:|
| **target's own layers (DFlash)** | `spec-dflash-n2` | 146.2 | **+26.3 %** | 72.3 % |
| **target's own MTP head** | `spec-mtp-n2` | 141.9 | **+22.7 %** | 78.4 % |
| **target's own layers (DFlash, longer)** | `spec-dflash-n4` | 137.9 | +19.2 % | 55.2 % |
| *none* | *baseline* | *115.7* | — | — |
| **external Qwen3.5-0.8B draft** | `spec-draft-n1` | 29.2 | **−74.8 %** | 69.7 % |
| **external Qwen3.5-0.8B draft** | `spec-draft-n8` | 30.9 | −73.3 % | 29.5 % |

Verbatim from S7 root README:

> "**The negative observation survives for the methods v1 tested, and only for those.** With an external draft model, speculation still loses badly here, and batching widens the gap rather than closing it. With the target's *own* layers as the drafter, DFlash and the model's built-in multi-token-prediction head, it wins at short draft windows and one request at a time."

**The mechanism, and why it will not transfer cleanly to a dense 27B** — S7 (external drafter pays a fixed per-round cost that no acceptance rate can amortise): a full checkpoint the server reports at **82.079 MiB**, saved and restored, plus a dense forward pass. Measured: **772 full-checkpoint creates and 709 restores in one ten-prompt arm-run**, at 82.079 MiB each, nominal **118.7 GiB** by event count × logged size (S7: "an estimate and not measured memory traffic"). Timed with instrumentation: **39.07 s of a 71.4 s excess = 54.7 %**, replicated at 54.6 % in a second balanced run.

**Critical caveat to carry over:** that target is a **35B-A3B MoE** activating ~3 B params, so a 0.8 B *dense* drafter costs a quarter of a target step before any state management: **17.24 s in `generate()` for the external drafter vs 1.89 s for MTP at `n_max` 1 and 3.43 s for DFlash at 2** (S7). On a **dense 27B**, the target step is far more expensive and the dedicated infrastructure (`--spec-draft-device`) can dodge the checkpoint cost — so the external-draft penalty should be smaller. **Do not read the −74 % as a dense-27B prediction.**

### 3b. Published acceptance rates & speedups for 27B–32B Qwen targets

| Target | Drafter | Hardware / engine | Speedup | Acceptance | Source, date | Label |
|---|---|---|---|---|---|
| Qwen3-32B Q4_K_M | Qwen3-0.6B Q8_0 | GB10, llama.cpp, temp 0 | mean **1.37×** (1.08–2.55×) | mean **19.87 %** | S9, 2026-05-01 | MEASURED |
| Qwen3-32B, code/text prompt | Qwen3-0.6B, d=4, p_min 0.8 | same | **2.55×** | **66 %** | S9 | MEASURED |
| Qwen3-32B, prose prompts | Qwen3-0.6B, d=4 | same | 1.29–1.30× | **19.6–20.4 %** | S9 | MEASURED |
| Qwen3-32B, prose, d=8 | Qwen3-0.6B, d=8 | same | 1.08–1.13× | 9.8–10.7 % | S9 | MEASURED |
| Qwen3.6-27B **dense** + MTP-3 | in-checkpoint MTP | 2× RTX 3090, vLLM | **+5–8 %** (94–97 → 117–127 narr / 122–125 → 164–169 code) | 53–80 % | S6, 2026-05-06 / 07-13 | MEASURED |
| Qwen3.8-27B + MTP-3 | in-checkpoint MTP | 2× RTX 3090, vLLM | **no gain** ("its tp2-mtp variant adds nothing") | — | S6, 2026-08-13+ | MEASURED |
| Qwen3.6-35B-A3B MoE + MTP | in-checkpoint MTP | 2× RTX 3090, vLLM | **+20/+77/+34 %** (k=3) | 1.7–1.9 mean len, pos-0 71–94 % | S6 | MEASURED |
| Qwen3-32B Q4_K_M | Qwen3-0.6B Q8_0 | GB10 | ambiguity in this user's baseline — see note | — | S9 | MEASURED |

**Important negative result for the exact target size:** S6, FINDINGS §QWEN38:

> "the 3.6 tp2-mtp throughput trick doesn't transfer" — for Qwen3.8-27B the `tp2-mtp` variant adds nothing over `tp2`.

And S6's blanket recommendation for llama.cpp on this class of consumer Ampere:

> "Don't enable llama.cpp's `--spec-type ngram-cache`, `--spec-type ngram-mod`, or `--model-draft` on `gguf` mode; every variant is net-negative on consumer Ampere."
>
> "llama.cpp for the dense 27B — 3× slower than vLLM" (as an engine choice, not a spec-decode claim)

**EAGLE-3 head results on Qwen3 (S10 AngelSlim, vLLM v0.11.2, single H20, tp=1, num_speculative_tokens=2, batch_size=1, output_len=1024):**

| Target | Baseline tok/s | EAGLE-3 tok/s | Speedup | accept length |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 381.05 (mean) | 642.93 (mean) | **1.69×** | 2.17 |
| Qwen3-4B | 233.26 | 386.94 | **1.66×** | 2.08 |
| Qwen3-8B | 151.81 | 257.52 | **1.70×** | 1.99 |
| Qwen3-14B | 93.26 | 145.33 | **1.56×** | 1.79 |
| **Qwen3-32B** | **43.32** | **~75–80** (GSM8K 80.43 / HumanEval 72.49) | **~1.67–1.86×** | **1.90–2.01** |

AngelSlim card summary, verbatim: *"with an accept length of 1.8–3.5 and a maximum speedup of 1.4–1.9×."* Note the **accept-length trend falls with model size** (2.17 → 1.79 from 1.7B to 14B), so a 32B target should be at the bottom of that band. There is a **Qwen3-32B EAGLE-3 sidecar** (`AngelSlim/Qwen3-32B_eagle3`, card updated 2026-01-13), and llama.cpp supports `--spec-type draft-eagle3` (S1). **This is the most promising untried option for a 27B dense Qwen3-class target.**

### 3c. VRAM cost of a draft model

| Component | Size | Source | Label |
|---|---|---|---|
| Qwen3-0.6B Q8_0 GGUF (draft model file) | **~0.75 GiB** | S9 §2.2 | MEASURED |
| Qwen3-32B Q4_K_M GGUF (target, for scale) | ~18.4 GiB | S9 §2.2 | MEASURED |
| Draft model + its KV cache above baseline | **+1.3 to +1.4 GB** (8.0–8.1 GB baseline → 9.4–9.5 GB peak) | S12 §6 | MEASURED |
| Qwen3.6 MTP head at BF16 | **3.7 GB** — *"cannot sit beside a 21 GB target on a 24 GB card, so it is quantised"* | S7 | MEASURED |
| EAGLE-3 head static overhead (Llama3.1-8B, FP16) | **+0.79 GiB** static (14.96 → 15.75 GiB) and **+4 KiB per token** KV (128 → 132 KiB) | S11 §5.2 | MEASURED |

**Your 4×3080 20 GB = 80 GB case:** 0.6–0.8 B Q8 drafts are ~0.75 GiB + KV — negligible against 80 GB. A 3.7 GB BF16 MTP head is also affordable *if* you have headroom. Your stated setup already uses the in-checkpoint MTP head, so **no additional VRAM is being spent on a draft model at all** — and that is a genuine advantage over the external-draft path.

### 3d. Does the draft need its own GPU? — **No, but it should get one**

**Source-of-truth on the flag (S2, verbatim):**

```
{"--spec-draft-device", "-devd", "--device-draft"}, "<dev1,dev2,..>",
"comma-separated list of devices to use for offloading the draft model (none = don't offload, default: follows --device)\n"
"use --list-devices to see a list of available devices",
```

**Default is `follows --device`** — i.e. by default the draft lands on the same devices as the target. It does **not** need its own GPU; you opt into separate placement with `-devd`.

**Real reports on contention vs separation (S13 blog, "Why Speculative Decoding Made My Local LLM Slower"):**

> "**Same-device contention is a real failure mode**, not a tuning problem. A Vulkan bug report has draft per-token time exploding when both models sit on the same iGPU, wiping out any gain."

> "On a two-GPU box, putting the draft on the second card removes the contention entirely. On a single-device machine there is nowhere to move it to, and the answer is to stop using a draft model."

Troubleshooting table from the same source:

| Symptom | Likely cause | Fix |
|---|---|---|
| Catastrophic slowdown, not a mild one | Same-device contention (Vulkan/iGPU) | **Split devices with `-devd`**, or disable |
| Refuses to load, vocab/tokenizer error | Tokenizer mismatch | Use a same-family draft |
| Uniformly slower on a Mac | Metal net loss, shared bandwidth | Drop the draft; try n-gram |

**Quantified contention evidence, "spare capacity" framing (S13):** Metal-backend draft-model speculation is a net loss of roughly **−11 % to −24 %** on typical workloads, because unified memory means the draft is not running on spare capacity.

**"Draft occupies its own GPU" on a fully-occupied rig — what the evidence actually says:**

- **MEASURED (S7, 2-GPU rig with target on one card):** the audit explicitly ran Draft+Target coexisting on one 24 GB card at the VRAM edge — *"Run J's telemetry peaks at 23946 MiB of 24576 with the drafter resident (630 MiB of headroom, 2.6 % of the card) and a 120 MiB allocation still failed."* That is the **draft-shares-GPU** case, and it is the case that produced the −71 % to −75 % external-draft results.
- **MEASURED (S6):** for the **dense 27B**, TP=2 across 2 cards is the throughput optimum and **TP=4 is slower, not faster, on PCIe-only 3090s** — "Doubling cards 1→2 gives ~30 % decode speedup; doubling 2→4 gives essentially nothing … TP=2 is one round-trip; TP=4 is a tree-reduce with two hops and four-card PCIe contention." All 4 cards on the same NUMA node.
- **I found no published head-to-head measurement of "draft on its own GPU vs shared GPU" on a 4×consumer-GPU rig with a 27B target.** **This is a genuine gap.** Reported as a null result below.

**Special case — separate-GPU drafts over PCIe with no P2P:** also **no direct measurement found.** What I can bound it with: (a) the contention mechanism is *bandwidth/compute on the shared device*, not interconnect, so moving the draft off-device removes contention by construction (S13); (b) the cost you take on instead is cross-device traffic per verify step — and on this exact topology S6 measured that *all* cross-card traffic is expensive: S6's rig overcame it by using a **BAR1 P2P driver patch** (`aikitoria BAR1-P2P patched`) and reported **+10 % code TPS** from enabling vLLM's CUSTOM allreduce over P2P. Without P2P, llama.cpp's draft device split will pay host-memory round-trips per verify step, which is structurally the same penalty that makes TP=4 pointless on this rig.

**Concrete suggestion for your rig:** your target already occupies all 4 GPUs at 80 GB total. Since `-devd` defaults to following `--device`, your MTP head is currently **sharing** the target's devices. Options: (1) leave it — MTP's 1.89 s `generate()` cost is small (S7); (2) reserve a slice of one card via `-ngl`/`--spec-draft-ngl` and pin the draft with `-devd`; (3) measure the A/B yourself, since no published number exists.

---

## Q4 — Techniques beyond classic spec decoding, with measured results

| Technique | Engine / availability | Measured speedup | Hardware / model | Source, date |
|---|---|---|---|---|
| **EAGLE-3** | **llama.cpp** (`--spec-type draft-eagle3`, S1/S15 #18039), vLLM, SGLang, TensorRT-LLM | **1.4–1.9×**, accept length 1.8–3.5 | H20, Qwen3-1.7B→32B (S10, 2026-01-13) | MEASURED |
| EAGLE-3 (per-size) | same | Qwen3-32B **~1.67–1.86×**, accept len **1.90–2.01** | H20, vLLM v0.11.2 | MEASURED |
| EAGLE (v2-era, for contrast) | vLLM | 1.73× → 1.21× as batch 1→128 | Llama3.1-8B, GSM8K (S11) | MEASURED |
| EAGLE, 70B, batch 1→32 | vLLM | 1.96× → 1.72× (−14.0 %) | Llama3-70B (S11) | MEASURED |
| **EAGLE-3 in llama.cpp — SGLang cross-check** | llama.cpp PR #18039 comment | **0.46–0.71× baseline at batch 1** (a *loss*) | DGX Spark, gpt-oss-120b + EAGLE3 (S7 citing #18039 comment 3755925892) | MEASURED |
| **MTP (multi-token prediction)** | **llama.cpp** (`--spec-type draft-mtp`), vLLM (`method=mtp`) | **+22.7 %** (self-spec, S7); **+20/+77/+34 %** (vLLM k=3, S6); 1.3–1.8× (GLM-4.5-Air, S11) | 3090 / 2×3090 / H100 | MEASURED |
| **DFlash** (block-diffusion draft, target's hidden states) | **llama.cpp** (`--spec-type draft-dflash`, merged **2026-06-28** per #22105); vLLM (PR #45163 for DiffusionGemma) | **+26.3 %** at n2; **+18.7 %** at n4 | RTX 3090, Qwen3.6-35B-A3B (S7) | MEASURED |
| DFlash on 3090, Gemma-4-26B-A4B | vLLM | 179 TPS code (**+40 %** vs gguf) but 95 TPS narrative (**−27 %**); collapses to ~50 TPS past 16 K ctx | 1× RTX 3090 (S6) | MEASURED |
| **DSpark** (DFlash + Markov head) | **llama.cpp** (`--spec-type draft-dspark`, PR #25173) | no independent measurement found | — | — |
| **n-gram / prompt-lookup** (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`) | **llama.cpp** (all five + `--spec-default` = ngram-mod) | **−0.3 % to −40 %** on prose; **1.09×–2.81×** on structured/JSON/code | see below | MEASURED |
| Medusa heads | vLLM / research forks; **not in llama.cpp** | no 2025/26 consumer-GPU number found in this sweep | — | — |
| Lookahead decoding | research / forks; **not in llama.cpp** | no 2025/26 consumer-GPU number found | — | — |
| REST / SV / Hydra / ReDrafter / SAM / Falcon | not in llama.cpp | no 2025/26 consumer-GPU number found | — | — |
| **Adaptive combination of multiple SD methods** | research (S11 simulator) | **up to 4.9×** vs no-SD | simulation over real benchmark data (S11) | ESTIMATED (simulated) |

**n-gram family — the concrete numbers (both directions):**

*S7, 3090, Qwen3.6-35B-A3B (a MoE case where n-gram mostly loses):* `ngram-map-k4v-m8` −0.3 % @ 50.0 % acceptance but only **216 drafted tokens across 27 000 generated** (one per 125 — it barely fires); `ngram-mod-n24` −10.9 %; `ngram-cache` −19.0 %; `ngram-simple` −4.2 %.

*S12, RTX 4090, Qwen3-8B (no MoE problem) — n-gram is the big winner:*

| Task | draft-simple (Qwen3-0.6B) | **ngram-mod (no draft)** | ngram acceptance | draft acceptance |
|---|---:|---:|---:|---:|
| code | 1.23× | **2.76×** | 76.3 % | 94.7 % |
| 英翻繁中 (translation) | 0.75× | **1.99×** | 72.9 % | 44.4 % |
| 創意對作 (creative) | 0.65× | **1.09×** (only sub-baseline task to go positive) | 61.9 % | 31.9 % |
| JSON | 1.28× | **2.81×** | 100.0 % | 98.7 % |

Verbatim conclusion (S12 §6): **"`ngram-mod` 在全部四個任務都大幅勝過 `draft-simple`，且零額外 VRAM 成本"** — ngram-mod beats draft-simple on all four tasks at zero extra VRAM. Its acceptance is high because *structured output repeats itself*; on low-repetition tasks it will not fire.

*S6 GEMMA_FINDINGS, 1× 3090:* "llama.cpp + ngram-mod" for Gemma-4-26B-A4B gives **127–131 TPS flat at full 262 K ctx**, 95 % SM util — a *single-GPU* win where the draft-model path lost.

### What is realistically usable on a 4×3080 llama.cpp rig

| Rank | Option | Verdict for this rig |
|---|---|---|
| 1 | **Keep `draft-mtp`** (incumbent) | Already measured at 78.3 tok/s / 2.76 mean accepted. Cost is near-zero per S7 (1.89 s `generate()` at n_max 1). |
| 2 | **`--spec-type draft-eagle3`** with `AngelSlim/Qwen3-32B_eagle3` | **Highest-upside untried option.** 1.67–1.86× on a 32B Qwen3 in vLLM (S10); supported in llama.cpp (S1); tiny VRAM (+0.79 GiB class). Caveat: the one llama.cpp EAGLE-3 datapoint at batch 1 on gpt-oss-120b was **0.46–0.71×** (S7), so measure before believing. |
| 3 | **`--spec-type ngram-mod` / `--spec-default`** | Free (no model, no VRAM). Try only on structured/repetitive output; on prose it measured −10.9 % (S7) to −40 % (ngram-cache). |
| 4 | **Combine** `--spec-type ngram-mod,draft-mtp` | Explicitly supported (S1 gives a multi-type example). Independently tested on Gemma-4 MTP: **a loss** — best combo 91.69 vs 92.40 tok/s for MTP alone (S8, 2026-06-23). |
| 5 | `draft-dflash` | +18.7 % to +26.3 % measured (S7) but **needs a Qwen3-backbone DFlash sidecar** and its win was on a MoE target. Worth a pilot if a Qwen3 27B DFlash sidecar exists. |
| 6 | Separate small draft model (Qwen3-0.6B/1.7B/4B) | **Last resort.** 1.08–2.55× on GB10 (S9) but −71 % to −75 % on the controlled 3090 MoE run (S7). On a **dense** 27B with `-devd` splitting the penalty should be far smaller — but nobody has published that measurement. |
| N/A | Medusa, lookahead, REST/SV, Hydra, ReDrafter | **Not available in llama.cpp.** No 2025/26 consumer-GPU numbers surfaced. |

---

## Q5 — Theoretical ceiling and the acceptance model

### 5a. The standard speedup formula (S11 Eq. 1, attributed to Leviathan et al. 2023)

Verbatim:

> "Let `c` denote the execution-time ratio of a single forward pass between the drafting method and the target model, `α` is the token acceptance rate and `k` is the proposed length. The expected overall wall-time speedup achieved by speculative decoding, can be expressed as follows:"

```
E(speedup) = (1 − α^(k+1)) / [ (1 − α) · (k·c + 1) ]
```

> "This formulation assumes that the `k + 1` simultaneous evaluations of the target model take approximately the same amount of time as generating a single token in parallel. This assumption typically holds when the system is memory-bound, which is common at small batch sizes, but may break down once the system becomes compute-bound."

> "Importantly, the absolute execution time of the drafting method is irrelevant; only its runtime relative to the target model affects the achievable speedup."

**Consequence:** the numerator `(1 − α^(k+1))/(1 − α)` is exactly `Σᵢ₌₁..ₖ αⁱ + 1` = the **expected accepted length including the target's own token (the bonus token)**. So:

```
E[accepted length] = Σ_{i=1..k} P(first i drafts all accepted) + 1
```

**The correct per-position formula.** If each draft position has its own *conditional* acceptance probability `pᵢ`, then with `Pᵢ = Π_{j≤i} pⱼ`:

```
E[accepted length] = 1 + Σ_{i=1..k} Pᵢ = 1 + Σ_{i=1..k} Π_{j=1..i} pⱼ
E[drafts accepted] = Σ_{i=1..k} Pᵢ
```

Under a **uniform** conditional rate `α`, this collapses to `E[L] = (1 − α^(k+1))/(1 − α)`.

**Important ambiguity in llama.cpp's counter.** S7's audit found llama.cpp's server counter reports "tokens re-verified after truncation to the accepted prefix" and can read 100 % when true token-level acceptance is 53.7 %:

| counter | value | what it counts |
|---|---:|---|
| server `draft_n_accepted / draft_n` | **115 / 115 = 100.0 %** | tokens re-verified after truncation |
| drafter `#acc tokens / #gen tokens` | **115 / 214 = 53.7 %** | true token-level acceptance |
| drafter `#acc drafts / #gen drafts` | **33 / 81 = 40.7 %** | true draft-sequence acceptance |

So: **"draft acceptance rate" in llama.cpp logs may be the conditional/unconditional per-position rate, not `P(all k accepted)`.** Your 0.587 should be interpreted as one of these; the mean-accepted-length of 2.76 is the more trustworthy number.

### 5b. Numerical ceiling for **this** configuration

Inputs: measured decode **78.3 tok/s**, `--spec-draft-n-max 3`, mean accepted length **2.76**.

**ESTIMATED** baseline (no speculation) decode rate if MTP contributes a pure accepted-length multiple:
`78.3 / 2.76 = 28.4 tok/s`.

**Scenario A — acceptance saturates (extra positions add nothing), only draft cost grows.**
n_max 4/5/6/7/8 all give L = 2.76 → **78.3 tok/s and then strictly worse**, because the extra verified positions are free of benefit but not of cost. This is the **pessimistic-but-observed** pattern (S7: 130.0 → 100.8 tok/s from n=3 → n=6).

**Scenario B — acceptance length grows as measured for self-spec families.** Using S8's measured mean accepted lengths for a self-speculative drafter at the same n_max values, and holding per-token cost constant (optimistic):

| n_max | mean accepted length (S8, MEASURED) | ESTIMATED tok/s | ESTIMATED vs current |
|---:|---:|---:|---:|
| 2 | 2.61 | 74.0 | −5 % |
| **3 (current)** | **2.76** (your measurement) | **78.3** | — |
| 4 | 3.61 | **102.4** | **+31 %** |

**Scenario C — the hard ceiling you asked for: every drafted token accepted, `L = n_max + 1`.** (ESTIMATED; ignores the extra draft compute)

| n_max | L = n_max+1 | ESTIMATED tok/s | ESTIMATED speedup vs 28.4 tok/s baseline |
|---:|---:|---:|---:|
| 1 | 2 | 56.7 | 2.0× |
| 2 | 3 | 85.1 | 3.0× |
| **3 (current)** | **4** | **113.5** | **4.0×** |
| 4 | 5 | 141.8 | 5.0× |
| 5 | 6 | 170.2 | 6.0× |
| 6 | 7 | 198.6 | 7.0× |
| 8 | 9 | 255.3 | 9.0× |

**Answer to "is 3 optimal for MTP, or is higher better?":** At n_max=3 you are capturing **2.76 of a theoretical 4.0 accepted tokens = 69 %** of the ceiling at that depth. Raising n_max to 4 buys at most `(5−4)/4 = 25 %` more *if* acceptance holds and draft cost stays flat — and every measured curve in this report shows cost rising faster than the extra accepted tokens beyond n = 4–6. Your realistic headroom is therefore **~+15 % to +30 %**, not +100 %, and the measured shape (S7: cliff between 4 and 6) says the ceiling at n_max ≈ 5 is roughly **where you already are**.

**The α needed for speculation to be worth it (from the formula):** setting `E(speedup) = 1` gives the break-even. Solve `(1−α^(k+1))/((1−α)(kc+1)) = 1`. For small `c` this is satisfied at very low α, which is why speculative decoding *can't* be usefully described by a single α threshold — the per-round cost `c` dominates. Empirically, S7 measured the threshold directly:

> "Across all 60 points in run L (ten prompts, three draft lengths, two workloads) acceptance and speed-up correlate at **r = +0.946**, and the least-squares line crosses zero at **48.2 % acceptance**. Below it 24 of 25 points are slower; above it 35 of 35 are faster."

> "So the usable statement is narrower than it first looked: **within a self-speculative family on this target, a configuration is worth running when it clears roughly 48 % draft acceptance.** It is not a statement about acceptance in general. An external 0.8 B drafter is 75 % slower at 68.7 % acceptance, because it pays a fixed per-round cost that no acceptance rate can amortise."

**Your 0.587 acceptance is above that 48.2 % empirical threshold** — consistent with your MTP configuration being a net win.

### 5c. What practitioners consider "good"

| Claim | Value | Source | Label |
|---|---|---|---|
| Empirical break-even for self-speculative families | **~48 %** draft acceptance | S7, 2026-08-26 | MEASURED (86 arm-runs scored) |
| "A configuration is worth running when it clears roughly 48 %" | 48 % | S7 verbatim | MEASURED |
| Expected acceptance for compatible-attention MTP models | **70–90 %** | S4 (reporter's expectation) | ESTIMATED (opinion) |
| OpenAI-grade target | ">0.7, >0.8" | *no source found* | — |
| Metal practical guidance | "no universal threshold … read the acceptance stats and compare tok/s with and without the draft on your actual prompts" | S13 | opinion |
| "The threshold transfers; the slope does not" | sign prediction 78/86 arm-runs, mean magnitude error +8.1 pp, worst +52.2 pp | S7 | MEASURED |

**Reporting note:** I looked specifically for a citation-backed ">0.7 is good / >0.8 is good" convention in the 2025–2026 literature and **found none.** The only empirically derived threshold located is S7's ~48 %, and S7 itself stresses it does not generalize across drafter families. Treat any "0.8 = good" rule as folklore.

### 5d. When spec decoding HURTS

All MEASURED:

| Condition | Evidence |
|---|---|
| **High n_max + low acceptance** | S7: n_max 6 → −8.9 %; 8 → −15.8 %; 16 → −80.8 %; 32 → −86.0 %. S12: n_max 16 on 翻譯 = 0.37×, on 創意 = 0.29×. |
| **External draft model on Ampere** | S7: −71 % to −75 % *even at 68.7–69.7 % acceptance*. |
| **Draft sharing a device with the target at the VRAM edge** | S7: 23 946 of 24 576 MiB used, 630 MiB headroom, a 120 MiB allocation failed. S13: "the VRAM you spend on the draft comes straight out of your context window or your quant — which is exactly how people end up slower than they started." |
| **High batch / concurrency** | S7: DFlash at 1 request = +17.3 %; at 4 = +0.4 %; at **8 = −74.1 %** (154.7 → 39.6 tok/s). S11: EAGLE 1.73× → 1.21× from batch 1→128 on Llama3.1-8B. |
| **MoE targets with few active experts** | S6: every llama.cpp variant net-negative on A3B; mechanism "8/256 expert routing means the verify pass loads the *union* of K positions' expert sets; at K ≪ 94 … the union load wipes out the savings"; worst case −60 % on structured prompts. **Caveat: S7's audit later showed no MoE-specific knee exists** (residual step at the 95.3 % coverage point = −0.39 pp) and that the true model is `ms/token = 27.00 + 4.040 × (draft tokens per generated token)`, R² = 0.99303. So the A3B loss is ordinary SD economics, not an expert-threshold effect. |
| **Low-draftability prose** | S9: 10–20 % acceptance on prose → 1.08–1.30×. S12: 翻譯 and 創意 tasks 0.29–0.79×. S13: same setup 4×+ on refactoring, ~1.3× on "explain the rules of chess". |
| **MTP with mismatched attention (SWA/hybrid)** | S4: 35–37 % acceptance on Qwen3.6-27B-MTP; full prompt re-processing adds 300–3000 ms per request. |
| **MTP at ~2048-aligned context sizes** | S5: 0.88×–1.12× at ctx 12032/17024/22016/24064 vs 1.68×–1.91× at ctx 12288/18432/22336/24576. |
| **Spec decoding + fp8 KV (vLLM)** | S6: MTP silently forced the *whole model* from full to piecewise cudagraphs, which is why the dense 27B saw only +8 % while the MoE saw +77 %. Fixed by `--attention-backend FLASH_ATTN` + fp16 KV. |

---

## Queries / sources that found nothing

Reported as required:

1. **`--spec-replace`** — exact-string grep of `common/arg.cpp` (master `f830688e`): **NOT FOUND**. Reported in the task as a flag to check; it does not exist in current llama.cpp.
2. **`lookup` as a `--spec-type` value** — **NOT FOUND**. Only `-lcs/--lookup-cache-static` and `-lcd/--lookup-cache-dynamic` exist. The old `llama-lookup` example has been folded into the `ngram-*` families.
3. **Published Medusa / lookahead / REST / SV / Hydra / ReDrafter / SAM / Falcon speedups on consumer GPUs, 2025–2026** — **no numbers found** in this sweep, and none of these are in llama.cpp. If the parent needs these, a separate paper-focused pass (not repo/forum-focused) is required.
4. **Head-to-head "draft on its own GPU vs shared GPU" on a 4× consumer-GPU rig with a ~27B target** — **no direct measurement found.** The closest proxies are S13's general statement ("On a two-GPU box, putting the draft on the second card removes the contention entirely") and S6/S7's shared-device VRAM-edge results.
5. **Separate-GPU drafts over PCIe with no P2P, measured** — **no direct measurement found.** Bounded indirectly by S6's TP-scaling findings and the BAR1-P2P patch result.
6. **A citation-backed ">0.7 / >0.8 acceptance is good" convention** — **not found.** Only S7's empirical ~48 % threshold exists in the located literature.
7. **Any published EAGLE-3 + Qwen3-27B/32B result *in llama.cpp*** — **not found.** The only llama.cpp EAGLE-3 datapoint located was a *negative* one on gpt-oss-120b (0.46–0.71× at batch 1, S7). This is the single highest-value experiment this user could run.
8. **`thc1006/qwen3.6-vllm-2x3090`** README — HTTP 404 on both `main` and `master` raw paths via the available proxy; the repo's findings are only available second-hand through S6's citations (the +27.5 % decode-rate claim S6 partially confirmed at +20/+28/+77 %).
9. **`shaunliew/qwen3vl-speculative-decoding-dgx-spark`** README — HTTP 404 on the fetched path; cannot report contents.

## Staleness flags (per instruction, anything older than ~6 months)

| Source | Date | Flag |
|---|---|---|
| S4 issue #23322 | 2026-05-19 | **~4 months — borderline fresh** |
| S5 issue #23658 | 2026-05-25 | **~4 months — borderline fresh** (closed) |
| S6 FINDINGS May section (2026-05-06) | MTP k-sweep | **>4 months — the k=3 conclusion was explicitly superseded by S6's own 2026-07-12 revision.** Do not cite the May k=3 result as current. |
| S9 Qwen3-32B suite | 2026-05-01 | **~5 months — borderline**; single-run-per-config, no CIs, llama.cpp commit unpinned (author's own limitations §4) |
| S7 archival tier (v1/v2/v3) | 2026-04-21 → 05-07 | **~5 months, and the repo retracted the headline mechanism.** Use only the controlled tier (2026-08-25→08-31). |
| S10 AngelSlim EAGLE-3 cards | card updated 2026-01-13; vLLM v0.11.2 | **~8 months — STALE for software version**, though the Qwen3 accept-length ratios are still the best available |
| S11 Leviathan formula | 2023 | **Foundational, not stale** — but its *constant-c* assumption is explicitly flagged by S11 as breaking down when compute-bound |
| S12 4090 bench (llama.cpp `e3546c7`) | 2026-07-11 | **~2.5 months — fresh** |
| S13 blog | undated on fetch | **Date unknown** — cite with caution |
| S14 MTP PR #15225 (GLM-style) | — | merge date not captured |
| Everything from S1/S2/S3 | 2026-09-24 | **CURRENT — primary source of truth** |

## Highest-value next actions for this user

1. **Sweep `--spec-draft-n-max` 2/3/4/5** on the real workload and record mean accepted length — the measured shape (plateau 2–4, cliff at 6) predicts +15–30 % available at 4 *at best*.
2. **Pilot `--spec-type draft-eagle3` with `AngelSlim/Qwen3-32B_eagle3`** (converted via `--target-model-dir`). Published 1.67–1.86× / accept-length 1.90–2.01 on a 32B Qwen3 in vLLM; the only llama.cpp EAGLE-3 datapoint is negative, so this is genuinely unknown and high-upside.
3. **Try `--spec-type ngram-mod,draft-mtp`** only if the workload is repetitive/structured — it measured as a loss on the one paired MTP test found (S8: 91.69 vs 92.40).
4. **Check whether ctx size sits on a 2048-aligned MTP collapse boundary** (S5) and whether GPU undervolt is stable (S5 comment, 2026-07-30).
5. **If you ever add an external draft, use `-devd` to place it off the target devices** — the default is `follows --device`, which is the contention case.
6. **Re-run the n_max sweep after any cudagraph/attention-backend change** — S6 showed the k=3 "optimum" was an artifact of a slow verify pass.
