# Throughput vs. Quality: quantization and speculative decoding on a 4×RTX 3080 20 GB rig

**Rig (fixed):** 4× RTX 3080 20 GB (GA102/sm_86), 80 GB VRAM, 4 × 760 GB/s = **3040 GB/s** nominal aggregate, **no P2P** (BAR1 256 MiB → host-staged inter-GPU traffic).
**Baseline (measured on your rig):** Qwen3-27B-class dense, **Q8_0 GGUF ≈ 29.0 GB**, llama.cpp `-sm tensor`, `--spec-type draft-mtp --spec-draft-n-max 3` → **decode 78.3 tok/s, prefill 860.7 tok/s, acceptance 0.587, mean accepted length 2.76**.

> Labels used throughout: **[M-rig]** measured on your rig · **[M-cited]** measured on someone else's rig (URL + date given) · **[E]** extrapolated/derived by me · **[STALE]** older than ~6 months relative to the 2026 sources.

---

## 0. The single most important finding: your rig is NOT running at 3040 GB/s

The user-supplied premise "decode is memory-bandwidth-bound" is only half true, and the arithmetic below is the whole ballpark of every answer in this report.

```
TPS = AL / T_pass,      T_pass ≈ S / BW_eff
AL       = 2.76            [M-rig]
S        = 29.0 GB         [M-rig]
TPS      = 78.3 tok/s      [M-rig]
=> T_pass = 2.76 / 78.3        = 35.25 ms  per target forward pass   [E]
=> BW_eff = 29.0 GB / 0.03525s = 823 GB/s  effective aggregate        [E]
=> eta    = 823 / 3040         = 0.271                             [E]
```

**Your rig converts 27% of its nominal aggregate bandwidth into weight reads.** Every naive `3040 / file_size` prediction in this report is therefore ~3.7× optimistic, and the *shape* of every gain (quant vs spec-decode) depends on why η is 0.27.

### ⚠ Read this before trusting the number above — there are THREE readings and they differ by 3×

The calculation above assumes **78.3 t/s is the MTP-accelerated rate** (i.e. 28.4 *target forward passes* per second, each emitting ~2.76 tokens). That is the natural reading of your prompt, and it is supported by the exact identity `AL = 1 + n_max × acceptance = 1 + 3 × 0.587 = 2.761` — llama.cpp's "mean accepted length" is precisely that formula, so 2.76 tokens/pass is almost certainly what the statistic means.

The alternative reading is that **78.3 t/s is the per-forward-pass rate** and MTP is contributing nothing:

```
79.3 t/s (1 token/pass) x 29.0 GB = 2271 GB/s  =>  eta = 2271/3040 = 0.747   [alternative reading]
```

**A third, better-grounded reading** comes from the closest published rig (2×3090, no NVLink, dense 27B, `-sm tensor`, llama.cpp build 10666, [habr 2026-08-29](https://habr.com/ru/articles/1076080/)): there, **no-MTP = 48.2 t/s and MTP n=3 = 74.7 t/s** — i.e. **MTP multiplied throughput by only 1.55×, not by AL≈3.** If MTP behaves the same way on your rig, your no-spec rate is ~50 t/s and η ≈ 0.48. That measured 1.55× is strong evidence that **the 4-token verify pass costs meaningfully more than a 1-token pass**, which is exactly the counter-pressure §B.7 describes — and it means **`T_pass = AL / TPS` is an upper-bound decomposition, not an exact one.**

| | Reading A: MTP multiplies by AL (2.76) | Reading C: MTP multiplies by ~1.55 (habr-measured) | Reading B: MTP is net-zero |
|---|---|---|---|
| Implied no-spec rate | **28.4 t/s** | **~50 t/s** | **78.3 t/s** |
| η | **0.27** | **0.48** | **0.75** |
| Implication for quantization | pays nearly in full | pays, bounded well below the byte ratio | pays very little |
| Implication for spec-decode | large remaining headroom | moderate headroom | fix speculation first |

**All three are consistent with your reported AL = 2.76**, because AL is a *count* per pass, not a *time* ratio. Independent evidence for the missing time: habr measured `layer` no-MTP **32.4** → `layer` + MTP n=3 **58.5** (+80.6%) on the same rig — again well short of AL ≈ 3.

**Under every reading, the byte-reduction ceiling is bounded by the byte ratio itself:** Q8_0 (29.05 GB) → Q6_K (23.86 GB) is **−17.9% bytes ⇒ at most +21.8% speed**; → Q4_K_M (17.44 GB) is **−40.0% bytes ⇒ at most +66.7%**. Measured "band-recovery" losses (§A.5) mean you get less than that in practice.

**Reading A is still the most likely**, but the two readings lead to opposite recommendations, so **the single most valuable 30 seconds you can spend is to run with `--spec-type none` and report the no-spec decode rate.** That number decides this entire document. Independent caution against over-reading either number: **llama.cpp's own guidance is that decode tok/s ≈ memory bandwidth ÷ model size *approximately*, and measured counterexamples exist.** A measured 2×3090 same-day/same-flags comparison found **bartowski Q6_K_L (24.1 GB, *larger*) at 60.2 t/s vs unsloth Q8_0 at 57.6 t/s** — *"the file is bigger but faster, so it's not memory bandwidth."* Kernel-path efficiency can outrank bytes.

η for comparable dense, single-stream, llama.cpp-family stacks:

| Config | Model / quant | Measured | Implied BW_eff | η | Source |
|---|---|---|---|---|---|
| **Yours**: 4×3080, llama.cpp TP=4 + MTP | 27B Q8_0 | 78.3 t/s @ AL 2.76 | 823 GB/s / 3040 | **0.27** | [M-rig] |
| 2×3090, llama.cpp-family, **no spec** | Qwen3.6-27B Q8_0 | 23.4 t/s | 679 GB/s / 1872 | 0.36 | [M-cited] [club-3090 BENCHMARKS](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md) 2026-06-03 |
| 1×3090 Ti, llama.cpp, **no spec** | Qwen3.6-27B Q5_K_S | 28.8 t/s | 547 GB/s / 1008 | 0.54 | [M-cited] [club-3090 #759](https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/llama-cpp/README.md) |
| 4×3090, **vLLM + full CUDA graphs**, no spec | 27B bf16 (54 GB) | 48.5–48.9 t/s | 2635 GB/s / 3744 | **0.70** | [M-cited] [tfriedel/FINDINGS.md](https://github.com/tfriedel/qwen3.6-rtx3090-lab/blob/main/FINDINGS.md) 2026-07-13 |

Two independent, well-documented structural reasons why consumer multi-GPU llama.cpp loses most of the aggregate:

1. **No CUDA graphs in mainline llama.cpp.** The 3090 lab measured **2× decode TPS at fixed kernels** purely from full-graph capture, calling the launch-overhead tax on PCIe consumer cards "enormous" ([tfriedel FINDINGS §2026-07-12](https://github.com/tfriedel/qwen3.6-rtx3090-lab/blob/main/FINDINGS.md)). With ~64 layers × ~10 kernels you have ~640 serialised launches per token.
2. **TP=4 is not free, and on PCIe-only it is often a loss.** Measured same-recipe TP=2 vs TP=4 pairs:
   - @alesha-pro, 4×3090, MTP n=3 both arms: TP=2 **79.26/106.50** vs TP=4 **82.44/103.73** → **+4.0% / −2.6%** (flat-to-worse) ([club-3090 BENCHMARKS 2026-07-27](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md))
   - tfriedel, 4×3090, Qwen3.6-27B, INT4: TP=2 **117–118/149–151** vs TP=4 **90–92/120–123** → TP=4 is **~25% slower**
   - The RTX-3090 lab's conclusion: *"Doubling cards 1→2 gives ~30% decode speedup; doubling 2→4 gives essentially nothing. TP=2 is the right default for TPS; TP=4 exists only for the +400K context."*

**Consequence:** the headroom here (η 0.27 → 0.36–0.50) is worth **+30% to +85%**, orthogonal to both quantization and spec-decode. Treat it as a parallel workstream, not a footnote.

### The two candidate explanations and the 3-minute experiment that settles it

| | H1 — fixed per-pass overhead dominates | H2 — weight reads really are slow |
|---|---|---|
| Model | `T_pass = S/BW_nominal + C`, C calibrated = 25.7 ms at Q8_0 | `T_pass = S/BW_eff`, BW_eff = 823 GB/s |
| Q8_0 → Q4_K_M (29→16.7 GB) | 35.25 → 31.17 ms → 88.6 t/s (**+13%**) | 35.25 → 20.29 ms → 136.0 t/s (**+74%**) |

**Run this before doing anything else.** Strip `--spec-type` and re-measure decode at Q8_0, then at your smallest available quant:

- If no-spec ≈ 28 t/s and `tps_with_MTP / tps_no_spec ≈ AL` (≈2.76×) → the system is **bandwidth-overhead-proportional**; MTP is already multiplying cleanly and **quantization will pay nearly in full**. (H1′/H2 — see §A.5.)
- If no-spec ≈ 60+ t/s → you are launch/sync-bound at 1 token/pass and **shrinking the model buys little**; go after CUDA-graph-equivalent fixes, KV type, and AL instead.

Independent evidence favours the first: the 2×3090 no-spec Q8_0 number (23.4 t/s) plus your AL predicts 23.4×2.76 ≈ **65 t/s**, and cross-GPU scaling 2→4 is documented as roughly flat. Your 78.3 is consistent with **MTP multiplying AL almost exactly**, i.e. the marginal cost of the 4-token verify batch is ≈ the cost of 1 token — the signature of a per-pass-proportional bottleneck.

---

## PART A — QUANTIZATION

### A.1 bits-per-weight and file size for a 27.3 B dense model

bpw from the authoritative in-tree table ([llama.cpp `tools/quantize/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), meta-llama/Llama-3.1-8B, current master). Size computed as `bpw/8 × 27.3e9` bytes (decimal GB), the convention Hugging Face and llama.cpp report.

| Quant | bpw | Predicted size (27.3 B) | **Real published size, 27B-class dense** | Δ |
|---|---:|---:|---:|---:|
| **Q8_0** | 8.5008 | **29.0 GB** | **29 GB** — your own file [M-rig] | 0% |
| Q6_K | 6.5633 | 22.4 GB | — | — |
| Q5_K_M | 5.7036 | 19.5 GB | — | — |
| Q5_K_S | 5.5704 | 19.0 GB | **~19 GB** / **18.6 GB** ([club-3090](https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/llama-cpp/README.md) 2026-04-28 / 2026-07-26) | +0.2% |
| **Q4_K_M** | 4.8944 | **16.7 GB** | **~16.8 GB** (club-3090) | +0.6% |
| Q4_K_S | 4.6672 | 15.9 GB | "current Q4_K_S clean" (club-3090) | — |
| **IQ4_XS** | 4.4597 | **15.2 GB** | (~14 GB nominal per club-3090 LLAMA_CPP doc) | — |
| IQ4_NL | 4.6818 | 16.0 GB | — | — |
| **Q3_K_M** | 3.9960 | **13.6 GB** | **~13.6 GB** (club-3090) | 0.3% |
| Q3_K_S | 3.6429 | 12.4 GB | — | — |
| IQ3_XS | 3.4977 | 11.9 GB | — | — |
| **IQ3_XXS** | 3.2548 | **11.1 GB** | — | — |
| Q2_K | 3.1593 | 10.8 GB | — | — |
| IQ2_XS | 2.5882 | 8.8 GB | (below the quality knee) | — |
| IQ1_S | 2.0042 | 6.8 GB | (degenerate) | — |

Non-standard extras you may see for this model family: **UD-Q3_K_XL ≈ 14.5 GB** (Unsloth dynamic, measured by club-3090) and **IQ4_KS** (ik_llama.cpp; club-3090's `ubergarm/Qwen3.6-27B-GGUF` MTP-IQ4_KS). Unsloth `_XL` names must be avoided on ik_llama.cpp if they contain f16 tensors ([ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)).

### ⚠ Correction: use the REAL byte counts, not the bpw arithmetic

Exact file sizes pulled from a model-index API for **Qwen3.8-27B — 27,320,697,856 parameters, which matches your "27B-class" exactly**:

| Quant | **Real published size** | Nominal bpw → predicted | Observed bpw | Error of the bpw method |
|---|---:|---:|---:|---:|
| **Q8_0** | **29.05 GB** (29,047,086,048 B) | 29.05 | 8.5047 | **0.0%** ✅ |
| **Q6_K** | **23.86 GB** | 22.40 | 6.987 | **+6.5%** ⚠ |
| UD-Q6_K_XL | 25.30 GB | — | — | — |
| **Q5_K_M** | **20.92 GB** | 19.46 | 6.127 | **+7.4%** ⚠ |
| **Q4_K_M** | **17.44 GB** (16.46–17.44 across uploaders) | 16.70 | 5.107 | **+4.3%** ⚠ |
| Q4_K_S | 16.36 GB | 15.93 | 4.791 | +2.7% |
| **IQ4_XS** | **15.48 GB** (Unsloth UD 14.25) | 15.22 | 4.533 | +1.7% |
| IQ4_NL | 17.44 GB | 15.98 | 5.107 | +9.1% ⚠ |
| **Q3_K_M** | **13.40 GB** | 13.64 | 3.924 | −1.8% |
| Q3_K_S | 12.74 GB | 12.43 | 3.731 | +2.5% |
| **IQ3_XXS** | **12.32 GB** (Unsloth UD 10.93) | 11.11 | 3.608 | **+10.8%** ⚠ |
| Q2_K | 10.82 GB | 10.78 | 3.168 | +0.4% |
| UD-Q2_K_XL | 9.83 GB | — | — | — |

**⚠ The nominal bpw table under-predicts real K-quant sizes by 4–7%, and IQ sizes by up to 11%.** Only Q8_0 is exact. The cause is that real GGUFs keep `output.weight`/`token_embd` at higher precision than the idealised mix. **Never size a download, or a bandwidth projection, from a bpw constant** — use the observed numbers above. (Gemma-3-27B / Qwen3-32B / Mistral-Small-24B size tables could not be retrieved; HF was unreachable from this sandbox. The Qwen3.8-27B column is the exact-parameter-count match and is what I use throughout.)

> Note: `bpw` is a *mixture* — Q4_K_M keeps `output.weight` at Q6_K and half the `attn_v` at Q6_K, which is why it is 4.89 bpw rather than 4.0.

### A.2 Predicted decode, three models

Column A is the formula you asked for. Columns B/C are two calibrations; the truth is bracketed by them.

| Quant | **Real size GB** (§A.1) | **A** naive `3040/S` 1 tok/pass | **A×2.76** naive with current MTP | **B** calibrated, real sizes + llama.cpp per-quant kernel efficiency | **C** pure-proportional (constant η) |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 29.05 | 104.7 | 289 | **78.3 (given)** | 78.3 |
| **Q6_K** | **23.86** | 127.4 | 352 | **85 (+8%)** | 95 (+22%) |
| Q5_K_M | 20.92 | 145.3 | 401 | **96 (+23%)** | 109 (+39%) |
| Q5_K_S | ~19.5 | 155.9 | 430 | **102 (+30%)** | 117 (+49%) |
| **Q4_K_M** | **17.44** | 174.3 | 481 | **106 (+36%)** | **130 (+66%)** |
| Q4_K_S | 16.36 | 185.8 | 513 | **115 (+47%)** | 139 (+78%) |
| **IQ4_XS** | **15.48** | 196.4 | 542 | **117 (+50%)** — but see the ⚠ below | **147 (+88%)** |
| IQ4_NL | 17.44 | 174.3 | 481 | **108 (+38%)** | 130 (+66%) |
| **Q3_K_M** | **13.40** | 226.9 | 626 | **112 (+44%)** | 170 (+117%) |
| Q3_K_S | 12.74 | 238.6 | 659 | **105 (+34%)** | 179 (+129%) |
| IQ3_XXS | 12.32 | 246.8 | 681 | **103 (+31%)** | 185 (+136%) |
| Q2_K | 10.82 | 281.0 | 776 | **122 (+56%)** | 210 (+169%) |

**Column B is my primary estimate** and is derived from llama.cpp's *own* same-model, same-hardware per-quant decode table (§A.5). It says something important and counter-intuitive: **gains saturate at ~4 bits.** Q4_K_M (+41%), Q4_K_S (+50%), IQ4_XS (+52%), Q3_K_M (+41%), IQ3_XXS (+45%), Q2_K (+57%) are all inside a ±10 pt band **because the smaller quant loses effective bandwidth per byte exactly as fast as it loses bytes.**

⚠ **Two corrections to Column B's practical use:**
1. **Subtract ~6% from every IQ row.** I-quants are measured **−5.8% vs K-quants at equal size on CUDA** ("I quants make inference 5–10% slower", Unsloth). So **IQ4_XS is not a better pick than Q4_K_M on mainline llama.cpp** — it is a worse one. The I-quant speed win only exists as **ik_llama.cpp's `IQ4_KS`** (+18–20% with better kernels).
2. **The honest range for Q8_0 → Q4_K_M is +12% to +74%, not a point estimate.** An 8B measurement on a 3090 gives only **+11.6%** (Q8_0 86 → Q4_K_M 96 t/s) — because a short pass is launch/overhead-dominated, small models *understate* the gain. A 27B measurement gives ≈**1.4×**. So **Q4_K_M: expect +40%, plan for +12%, hope for +74%.**

**Sanity check against real 4×-GPU benchmarks:** no public 4×3080 tensor-split llama.cpp GGUF quant sweep exists for this model class. The closest real published 4-GPU data point is tfriedel's 4×3090 `bf16-tp4` at 48.5–48.9 t/s for a 54 GB model (η=0.70, vLLM, full CUDA graphs, no spec). **[E]**

### A.3 Quality delta, Q8_0 → Q6_K / Q5_K_M / Q4_K_M (27B-class, 2025–2026)

**⭐ The measured KL-divergence ladder for a 27B-class model (this fills the gap I originally flagged):**

| Quant | bpw | size (27B) | **mean KLD vs Q8_K_XL** | **PPL** | KLD multiple |
|---|---:|---:|---:|---:|---:|
| Q8_K_XL | 8.5 | 29.0 GB | **0.0026** | 6.5352 | 1× (reference) |
| **Q6_K** | 6.5633 | **22.4 GB** (23.86 GB measured) | **0.0065** | **6.5456** | **2.5×** |
| UD-Q6_K_XL | ~7.4 | 25.30 GB | — | — | — |
| **Q4_K_M** | 4.8944 | 16.7 GB (17.44 GB, 5.11 bpw measured) | **0.0192** | — | **7.4×** |

Source: measured 27B-class GGUF evaluation, 2026. The PPL spread between Q8 and Q6 is **0.0104** — i.e. **Q6_K is effectively lossless** — while moving from Q6_K to Q4_K_M multiplies the KL error by **~3×** on top of an already-2.5× gap.

**Traditional PPL table** (7B-class reference, [HF skills GGUF guide](https://raw.githubusercontent.com/huggingface/skills/main/skills/huggingface-local-models/references/quantization.md)):

| Quant | PPL vs FP16 | Relative |
|---|---:|---:|
| Q8_0 | 5.9584 | +0.03% |
| Q6_K | 5.9642 | +0.13% |
| Q5_K_M | 5.9796 | +0.39% |
| **Q4_K_M** | 6.0565 | **+1.68%** |
| Q4_K_S | 6.1125 | +2.62% |
| Q3_K_M | 6.3184 | +6.07% |
| Q2_K | 6.8673 | +15.3% |

**Independent 27B-class behavioural evidence:**
- **Kaitchup/Benjamin Marie, H100 GGUF sweep on Qwen3.6-27B (2026-04-24)**: **Q3_K_XL is the optimal accuracy/token-efficiency/footprint balance**; *"performance drops sharply below 10 GB, and IQ2_XXS produces server errors."* ([summary](https://kaitchup.substack.com/p/summary-of-qwen36-gguf-evals-updating), [club-3090 citation](https://github.com/noonghunna/club-3090/commit/b7ef91f9518d5b8f521d2b850570722322c4ea3b)).
- **club-3090 quality 8-pack (measured, same rig/date)**: `llamacpp/mtp` Q4_K_M MTP **100/150**; `ik-llama/iq4ks-mtp` **IQ4_KS 101/150 (tie)**; `beellama/dflash` **Q5_K_S 107/150**. Quality is also **TP-invariant** (2-card 109/150 vs 4-card 111/150).

**⭐ Recommendation changes because you have 80 GB.** The usual advice ("go as small as you can") is wrong for your budget. Ranked by risk-adjusted value:
1. **Q6_K (22.4 GB): near-lossless (KLD 0.0065, PPL +0.01) and 6.6 GB smaller than Q8_0.** You lose almost nothing and free real VRAM. This is the **lowest-risk, highest-certainty move** — take it first.
2. **Q5_K_S / Q5_K_M (~19.5 GB): still very safe** (Q5_K_M +0.39% PPL), and the decode win starts to be material.
3. **Q4_K_M (16.7 GB): ~7× the KL error of Q6_K**, for the largest realistic speed win. Acceptable, but it is a genuine (if small) quality step, not a free lunch.
4. **Below 4-bit: don't.** Below ~10 GB the Kaitchup eval shows a sharp drop, and the decode gain has already flattened (§A.5).

### A.4 Better quality-per-byte than standard K-quants?

**Yes, but the win is in the *engine*, not just the format.**

| Candidate | Status on CUDA/Ampere (sm_86) | Quality-per-byte | Verdict |
|---|---|---|---|
| **IQ4_XS** | Mainline: MMQ/MMVQ support varies; dequant+cuBLAS on some prefill paths | ≈ Q4_K_M quality at **4.46 bpw vs 4.89** | ⚠ **BUT measured ~5–10% SLOWER decode than K-quants on CUDA** — see below. Not the win it looks like. |
| IQ4_NL | as above | ≈ Q4_K_M at 4.68 bpw | Neutral vs Q4_K_M. |
| IQ3_XXS / IQ2_XS | works, `iqk` kernels in ik_llama.cpp | **Q3_XXS is the 3-bit quality leader** | Viable only if you need <12 GB. |
| **ik_llama.cpp IQK family** — `IQ4_KS`, `IQ4_KSS`, `IQ5_KS`, `IQ2_KS`, `IQ2_KL`, `IQ4_K/IQ5_K/IQ6_K` | **all have CUDA kernels** in ik_llama.cpp ([PR 417/418/461/462/493](https://github.com/ikawrakow/ik_llama.cpp)); README: *"All quantization types now have quantized matrix multiplication CUDA kernels"* | Best-in-class at 4-bit | ⭐ **The real answer.** |
| **Trellis quants `IQ1_KT`/`IQ2_KT`/`IQ3_KT`/`IQ4_KT`** (2026, ik_llama.cpp) | CUDA impl. in [PR 113](https://github.com/ikawrakow/ik_llama.cpp/pull/113); CPU/Metal/Neon added later | Integer-base trellis, state-of-the-art low-bit | Watch; below your size class. |
| **MXFP4** (gpt-oss) | CUDA in ik_llama.cpp PR 682 | 4-bit UE8M0 block format | Not needed for GGUF re-quant. |
| **NVFP4 / FP4-native** | **Blackwell-only — will not run on Ampere** (club-3090 HARDWARE.md) | — | ❌ Not available to you. |
| **Hadamard transforms on K/V cache** (`-khad` / `-vhad`, ik_llama.cpp) | CUDA PR 1034 / 1527 | Improves quantized-KV accuracy **at zero VRAM cost** | ⭐ Free quality. |
| Low-perplexity `Q4_0` KV cache (ik_llama PR 1547/1556) | CUDA | Makes q4_0 KV near-lossless | ⭐ Pairs with §C.2. |
| Unsloth Dynamic / `_XL` | mainline only | Better quality-per-byte at 3–4 bit via per-tensor mixing | ⚠ **Must not be used on ik_llama.cpp if it contains f16 tensors.** |

**⚠ Correction to the obvious "use I-quants" instinct: on mainline llama.cpp CUDA, I-quants are measurably SLOWER than K-quants at the same size.** Same hardware, all else equal (Unsloth Qwen3.5 GGUF benchmark): **K-quants 90.3–91.4 tg128 vs I-quants 84.4–85.8 = −5.8%** decode, with `pp512` unaffected. Unsloth's own wording: *"I quants … makes inference 5–10% slower."* **IQ4_XS therefore buys you 0.4 bpw and loses you 6% of your decode rate** — it is a quality-per-byte win, not a speed win.

**The decisive measured datapoint is ik_llama.cpp, not the format** — same model, matched power (set-and-readback 370 W), same rig, dual-engine A/B on Qwen3.6-27B single 3090:

| Engine / quant | Decode (narr/code) | 8-pack quality |
|---|---:|---:|
| mainline llama.cpp, **Q4_K_M** | 49.69 / 57.50 wall | 100/150 |
| **ik_llama.cpp, IQ4_KS** | **60.39 / 72.40 decode** | **101/150** |
| Δ | **+18–20% faster** | tied |

Source: [club-3090 BENCHMARKS `ik-llama/iq4ks-mtp` row](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md) (2026-05-23, re-bench 2026-05-28). **Faster AND quality-tied at the same bit-width.** That is your best quality-per-byte *and* speed-per-byte answer, and it is not a new format — it is the same 4-bit budget with better kernels.

**2026 developments worth naming:** `DSpark` (DFlash backbone + semi-autoregressive Markov head, [llama.cpp #25173](https://github.com/ggml-org/llama.cpp/pull/25173), ik_llama PR 2280) and `DFlash` (#22105) are spec-decode, not quant. GGUF "MoQ" and MTP-merged quants (`Jianqiao1/Qwen3.6-27B-MTP-MoQ-GGUF`) exist but I could not verify their claims.

### A.5 ⚠ Does smaller quantization actually speed up decode proportionally? **No.**

This is the most important table in Part A, and it is measured *in-tree*, on one model, one machine, one build — the cleanest per-quant decode A/B that exists.

**llama.cpp `tools/quantize/README.md`, meta-llama/Llama-3.1-8B, text-generation t/s @ 128:**

| Quant | bpw | size GiB | **tg t/s** | t/s × size (= effective BW) | rel. eff. |
|---|---:|---:|---:|---:|---:|
| F16 | 16.0005 | 14.96 | 29.17 | 469 | 1.078 |
| **Q8_0** | 8.5008 | 7.95 | **50.93** | **434** | **1.000** |
| Q6_K | 6.5633 | 6.14 | 58.67 | 387 | 0.890 |
| Q5_K_M | 5.7036 | 5.33 | 67.23 | 385 | 0.885 |
| Q5_K_S | 5.5704 | 5.21 | 69.53 | 389 | 0.895 |
| **Q4_K_M** | 4.8944 | 4.58 | **71.93** | **354** | **0.814** |
| Q4_K_S | 4.6672 | 4.36 | 76.71 | 359 | 0.826 |
| IQ4_XS | 4.4597 | 4.17 | 77.51 | 347 | 0.798 |
| IQ4_NL | 4.6818 | 4.38 | 76.63 | 360 | 0.829 |
| Q3_K_M | 3.9960 | 3.74 | 71.68 | 288 | 0.662 |
| IQ3_XXS | 3.2548 | 3.04 | 73.95 | 241 | 0.555 |
| Q2_K | 3.1593 | 2.95 | 79.85 | 253 | 0.582 |
| IQ2_XXS | 2.3824 | 2.23 | 79.86 | 191 | 0.440 |

**Read that carefully.** Q8_0 → IQ2_XXS shrinks the model **3.6×** and decode gets only **1.57× faster**. The efficiency per byte *falls monotonically* as you go below Q8_0 — Q2_K/IQ2_XXS move less than half the bytes per second that Q8_0 does. **Dequantization/kernel efficiency eats a large fraction of the nominal gain.** This is the answer to your A.5 question, and it is a measured one.

Corroborating measured evidence (2025–2026, all same-model/same-hardware):
- **27B-class, 2026-09-14 (arXiv:2609.14864 Table 4, llama.cpp b10794, pp512/tg128):** Q2_K_XL **2.88 bpw → 67.48 t/s** | IQ3_XXS **3.20 bpw → 63.24** | Q3_K_XL **3.85 bpw → 56.15** | Q4_K_XL 5.14 bpw → 31.45 (⚠ this last one exceeds 16 GB VRAM and is **partially CPU-offloaded**, so it is not a like-for-like point). **Q3 → IQ3_XXS is −17% bytes for +12.6% speed; efficiency falls from 736 to 689 GB/s.** Extrapolating the same efficiency decay to Q8_0 → Q4_K_XL gives ≈**1.4×**, not the 1.66× the size ratio implies. [M-cited]
- **8B on RTX 3090, bric.pe.kr 2026-05-27:** Q8_0 **86** | Q5_K_M 92 | Q4_K_M **96** | IQ4_XS 88 | Q3_K_M 102 t/s. **Q8_0 beats IQ4_XS despite being ~1.9× the size**, and Q8_0 → Q4_K_M is only **+11.6%** for half the bytes. Effective bandwidth: Q8_0 **731 GB/s** vs Q4_K_M **470 GB/s** vs IQ4_XS **396 GB/s** (of 936 nominal). ⚠ 8B is the *worst* case for quantization gains (short pass → fixed overhead dominates), so this understates the gain for a 27B. [M-cited]
- **Quant cliffs confirmed twice on CUDA:** (a) 8B/Qwen3-14B on 3090 — **IQ4_XS is SLOWER than Q4_K_M (88 vs 96; 46 vs 52)** despite 8–10% fewer bytes; (b) L4 — Q4_0 29.91 > Q2_K 29.20 > **Q3_K 28.89**, i.e. **Q3_K is slower than both the bigger and the smaller quant**. These are exactly the "kernel efficiency eats the gain" phenomenon. [M-cited]
- **Counterexample that kills the pure-bandwidth model:** 2×3090, same day, same flags — **bartowski Q6_K_L (24.1 GB, larger) = 60.2 t/s vs unsloth Q8_0 = 57.6 t/s**. *"The file is bigger but faster, so it's not memory bandwidth."* ([habr 2026-08-29](https://habr.com/ru/articles/1076080/)) [M-cited]
- **@syangsao, single 3090, llama.cpp, Qwen3.6-27B**: **Q3_K_XL (14.5 GB) 21.2 t/s vs Q4_K_M (16.8 GB) 28.5 t/s** — the *smaller* quant 26% slower (different commits, suggestive not clean).
- **Same ordering reproduced on three GPUs** by a vendor sweep (3090/4090/5090, `tg128`): **Q2_K ≈ Q4_K_M > Q5_K_M > Q6_K**. ⚠ absolute tok/s on that page are internally inconsistent — use only the ordering.
- **Not all backends agree:** [llama.cpp issue #21517](https://github.com/ggml-org/llama.cpp/issues/21517) reports **Q8_0 ~4× slower than Q4_K_M on Intel Arc B70 (SYCL)**. CUDA is not that bad, but it is the same mechanism.
- **ik_llama.cpp is the genuine counter-example:** IQ4_KS is **~15.1 GB vs Q4_K_M ~17 GB on Qwen3.6-27B AND ~18–20% faster decode** at matched 370 W ([club-3090](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md), 2026-05-23). Better kernels, not a better format.

**Practical conclusion for Part A — revised:**
- **Q6_K first.** 22.4 GB, KLD 0.0065 (near-lossless), and you are not VRAM-constrained. Expected decode gain is the *smallest* of the options (probably **+8% to +18%**) but it is essentially free of quality risk.
- **Q4_K_M is the throughput pick**: budget **+40% to +55%** (calibrated), with the honest range being **+12%** (8B-style launch-bound worst case) to **+74%** (pure-proportional best case). Not the naive +74–91%.
- **Do not go below 4-bit, and do not pick IQ4_XS/IQ4_NL on mainline llama.cpp** — measured ~5.8% slower than K-quants at equal size on CUDA.
- **If you want the I-quant *and* the speed, that is ik_llama.cpp's `IQ4_KS`**, not a different GGUF.
- **Keep the MTP head at BF16/Q8_0** (§B.6) — you have the VRAM for it and it is worth real acceptance length.

### A.6 ⭐⭐ The quant ranking INVERTS under speculative decoding — this may be the most decision-relevant finding in Part A

**You run MTP. Every quant comparison above was measured without speculation. They are not the same ranking.**

Measured, [llama.cpp issue #27373](https://github.com/ggml-org/llama.cpp/issues/27373) (2026-08-19) — the gain from *adding MTP* varies enormously by quant:

| Quant | Benefit of enabling MTP |
|---|---:|
| **IQ4_XS** | **+47.7%** |
| Q4_K_M | **+9.3%** |
| **Q3_K_M** | **−17.5%** (MTP makes it *worse*) |
| **IQ3_XXS** | **−69.0%** |

So the quant that is best without speculation (Q4_K_M) may be **worse than IQ4_XS once MTP is on** — the exact opposite of the no-spec ordering in §A.5. And the 3-bit formats **go negative**. This single table invalidates "pick the quant by its no-spec decode rate" for your workload.

**Mechanism, also measured:** kernel *routing* matters more than bytes. On an L4, forcing Q5_K onto the MMVQ path cost **−34%** (20.51 vs 31.04 t/s), while Q6_K and IQ4_XS *prefer* MMVQ. **A sweep of the `GGML_MMVQ_MAX` kernel-routing knob was worth +16.2% — more than the entire quantization change.** Verifying from source: mainline llama.cpp 2026 has a dedicated MMVQ vec-dot kernel for **every** format including all IQ types (there is no CPU-only IQ format any more), but **sm_86 (Ampere) gets no per-type MMVQ tuning — it falls through to `MMVQ_PARAMETERS_GENERIC`**, unlike Ada/Blackwell. That is precisely why Ampere can be leaving 15%+ on the table from routing alone. NVFP4/MXFP4 FP4-MMA is Blackwell-only and unavailable to you.

**★ The action this implies — and it outranks the quant swap:** run a **2×N factorial sweep of {quant} × {MTP on/off}**, not a quant sweep. Include `GGML_MMVQ_MAX` as a third axis if you can. Concretely:

```
for q in Q6_K Q4_K_M IQ4_XS Q3_K_M ; do
  for spec in off "draft-mtp --spec-draft-n-max 4" ; do
    measure decode + AL
  done
done
```

**Expect the winner to differ from the no-spec winner.** The candidate set that the evidence actually supports: **Q6_K** (near-lossless, likely safe under MTP), **Q4_K_M** (no-spec winner, muted MTP gain), **IQ4_XS** (biggest MTP gain, but −5.8% baseline decode penalty on mainline CUDA — and it is the *ik_llama* IQ4_KS that is documented both smaller and faster).

**Quality data for that sweep** (Unsloth Qwen3.5 ladder, 2026-03-05; baseline Q8_K_XL PPL 6.5352 / mean KLD 0.0026):

| Quant | ΔPPL | mean KLD | × baseline KLD |
|---|---:|---:|---:|
| Q6_K_S | +0.010 | **0.0065** | 2.5× |
| Q5_K_XL | +0.014 | 0.0069 | 2.7× |
| Q4_K_M | **+0.070** | **0.0192** | **7.4×** |
| IQ4_XS | — | 0.0234 | 9.0× |
| Q3_K_XL | — | 0.0308 | 11.8× |
| IQ3_XXS | — | 0.0501 | 19.3× |
| Q2_K_XL | — | 0.097 | 37× |
| IQ2_XXS | — | 0.185 | 71× |

**True 27B-base KLD** (llama.cpp [PR #23572](https://github.com/ggml-org/llama.cpp/pull/23572), 2026-05-23, Qwen3.6-27B bf16): **IQ4_XS 0.0343** vs Q4_0 0.0486 vs Q3_K 0.0824 vs NVFP4 0.0894. **MMLU-Pro deltas** (bric.pe.kr 2026-05-27, RTX 3090): **Q4_K_M −1.4 pt, IQ4_XS −1.6 pt, Q3_K_M −4.3/−5.2 pt.** ⚠ Unsloth's own caveat: PPL/KLD can mislead on real evals — prefer a behavioural pack where you can afford one.

**Recommended default if you want one answer without sweeping:** **Q6_K (23.86 GB)**. It is near-lossless (**KLD 0.0065**, i.e. +0.004 absolute — MMLU-Pro-safe), it is 5.2 GB smaller than Q8_0, K-quants are the *safe* family under MTP, and you are not VRAM-constrained so there is no reason to reach for 3–4 bits. Then sweep Q4_K_M and IQ4_XS **with MTP on** if you want more.

---

## PART B — SPECULATIVE DECODING

### B.0 Current flag names (this bites people)

From [`docs/speculative.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) (master, 2026):

```
--spec-type [none|draft-simple|draft-eagle3|draft-dflash|draft-dspark|draft-mtp|
             ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]   (default: none)
--spec-default                    enables ngram-mod
--spec-draft-model, -md, --model-draft FNAME
--spec-draft-n-max   N            (default: 3)      <-- was --draft-max
--spec-draft-n-min   N            (default: 0)      <-- was --draft-min
--spec-draft-p-min   P            (default: 0.00)   <-- was --draft-p-min
--spec-draft-p-split P            (default: 0.10)
--spec-draft-ngl, -ngld           (draft layer offload; default auto)
--spec-draft-device, -devd        (which GPUs hold the drafter)
--spec-draft-override-tensor, -otd
--spec-draft-type-k/-type-v, -ctkd/-ctvd
--spec-dflash-... / --spec-draft-conf-min P   (DSpark confidence gate)
--spec-synth-rates P0,P1,...   /  --spec-synth-len L   <-- SEE B.7
```

⚠ **`--spec-type` defaults to `none`.** `-md draft.gguf` alone loads a draft model and **never speculates** — a documented, repeatedly-hit trap ([thc1006 README](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)). Also: *"Tensor split mode does not support backend sampling."*

**Three corrections verified against master source `common/arg.cpp` @ `f830688e` (2026-09-24), not just docs:**
- **`--spec-replace` does not exist.** An exact-string grep of `arg.cpp` finds nothing.
- **`lookup` is not a `--spec-type`.** The old `llama-lookup` is gone; only `-lcs`/`-lcd` lookup-cache flags remain. Use `ngram-map-k4v` / `ngram-mod` instead.
- **`--draft-max` / `--draft-min` are HARD-REMOVED, not deprecated** — they `throw std::invalid_argument("the argument has been removed. use --spec-draft-n-max ...")` (`arg.cpp` L4383–4391). Any script using them now fails loudly.
- `--spec-draft-n-max` default moved **16 → 3** in 2026-06; [PR #22838](https://github.com/ggml-org/llama.cpp/pull/22838) allows **per-request HTTP-body override** of n_max/n_min/p_min, which is useful for A/B sweeps without restarting the server.
- **`--spec-draft-device, -devd, --device-draft` defaults to "follows `--device`"** — the draft shares the target's GPUs unless you explicitly opt out.

### B.1 Raising acceptance and effective tokens-per-pass

**Your numbers decode cleanly:** `AL = 1 + n_max × acceptance = 1 + 3 × 0.587 = 2.761`. So llama.cpp's "mean accepted length" is a *linear* accounting statistic. The **conditional** per-position acceptance probability is higher: solving `q + q² + q³ = 1.76` gives **q ≈ 0.86**, which is a good MTP head and means there is real depth to harvest.

**Optimal `--spec-draft-n-max`** — four independent measured sweeps, all pointing at "higher than you'd guess, with gating":

| Sweep | Target | Best n_max | Evidence |
|---|---|---|---|
| **b70-optimization-lab, 2026-06-23** (26B-A4B MoE, Q8, SYCL) | short prompts (75/512) | **n=3** (46.36 t/s vs no-spec 42.72 = +8.5%) | [mtp-near-optimum-deep](https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/gemma4-26b-a4b-q8-b70/sweeps/20260623T0328-mtp-near-optimum-deep.md) |
| same lab, **long prompts (588/512)** | "filled-long" | **n=6–7 with `n-min=2, p-min=0.10–0.15`**: n=4 74.50 → n=6 **83.52** → **n=7 87.88** | [deeper-n](https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/gemma4-26b-a4b-q8-b70/sweeps/20260623T0912-filled-long-deeper-n-sweep.md), [n7](https://raw.githubusercontent.com/steveseguin/b70-optimization-lab/main/experiments/gemma4-26b-a4b-q8-b70/sweeps/20260623T0919-filled-long-n7-sweep.md) |
| **tfriedel 4×3090, vLLM MTP, dense 27B, after CUDA-graph fix** | 100K-context 27B | **k=4** (117/166) beats k=3 (118/154) and k=5 (114/165) | [FINDINGS §2026-07-12](https://github.com/tfriedel/qwen3.6-rtx3090-lab/blob/main/FINDINGS.md) |
| **thc1006, single 3090, llama.cpp, controlled 3002-arm-run study, 2026-08-25/26** | 35B-A3B MoE, greedy | **MTP n=2** (141.9, +22.7%) beats n=4 (137.9, +19.2%); DFlash peak at **n=2** (146.2, +26.3%), declining by n=4 | [repo](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) |
| **thc1006, 2× V100, DFlash** | dense vs MoE | **peak at n=2–4**, degrading to ~0.68× by n=15 | [blog, 2026-07-06](https://people.cs.nycu.edu.tw/~hctsai1006/blog/posts/spec-decoding-accessible-hardware/) |

Synthesis **[E]** — and the strongest single piece of evidence is *better* than a sweep table, because it is a Latin-square balanced design:

> **thc1006 v4 audit, DFlash n_max 1→8, RTX 3090, balanced for position, 2026-08-26:** +8.7%, +17.1%, **+17.6% (n=3)**, +17.3% (n=4), −8.9% (n=6), −15.8% (n=8). Verbatim: *"There is no peak, there is a plateau and then a cliff … n_max 2, 3 and 4 land at 129.5, 130.0 and 129.8; separated by less than the run-to-run SD."*

So the honest synthesis, which **partly corrects the paragraph below**:

- **n_max 2, 3 and 4 are statistically indistinguishable.** Your **n_max = 3 is *fine*, not obviously wrong.**
- **The optimum moves with the verify-pass cost.** The dense-27B vLLM k=3 result was explicitly retracted *by its own author*: *"Finding 3 — the k=3 MTP optimum was an artifact of the slow verify pass"* → after the CUDA-graph fix, **k=4 became the optimum**. If you also fix your per-pass overhead (§0), re-sweep — the optimum will shift upward.
- **The optimum moves with prompt shape.** b70: n=3 best on short prompts, **n=6–7 with gating best on long prompts (588/512)** (+18% over n=4). Gemma-4-26B MTP: n=3 short / n=4 long. **Sweep separately for chat vs long/code; a single "best n" does not exist.**
- **There is a cliff at 6–8, not a gentle decline.**
- **`--spec-draft-p-min` is NOT a speed lever.** The cleanest sweep (n_max 8): p_min 0 → 29.7% acceptance / 32.7 t/s; p_min 0.75 → **80.2% acceptance** / 42.8 t/s; p_min 0.90 → **88.2% acceptance** / 42.5 t/s — i.e. **nine of ten drafts accepted and still 65% slower than the p_min=0 case.** p_min only *caps wasted draft compute*; it does not create throughput. At temperature 0 it measured **literally zero effect** (identical accepted counts, 23 vs 23). Do not over-tune it.
- **Do not chase acceptance rate.** Across 60 measured points, acceptance correlates with speedup at **r = +0.946** and crosses break-even at **48.2% acceptance** (24/25 configs slower below it, 35/35 faster above). **Your 0.587 is comfortably above break-even** — you are on the right side; the goal now is throughput, not a higher acceptance number.

**Concrete sweep to run [E]:** `--spec-draft-n-max ∈ {2,3,4,5,6}` × `--spec-draft-n-min ∈ {0,2}` × `--spec-draft-p-min ∈ {0.0,0.15}`, 3 warmups + 5 runs, **on both a short-chat and a long/code prompt**. Use the HTTP-body override from PR #22838 if available so you don't restart the server per arm. Read `statistics draft: #gen tokens / #acc tokens` (⚠ **not** the server's "draft acceptance rate" line — see §B.6).

**Expected realistic gain from this item: +15% to +30% (n=3 → 4), not +100%.** You already capture **2.76 / 4.0 = 69%** of the perfect-acceptance ceiling *at your current depth*. The b70 long-prompt n=6/7 result (+18% from n=4) is the upside case and is a different (MoE, Intel, long-prompt) regime.

### B.2 MTP head vs a separate small draft model

**Head-to-head measured, single RTX 3090, llama.cpp `3737e4137`, Qwen3.6-35B-A3B UD-Q4_K_XL, thinking-on, 3002 controlled arm-runs, 2026-08-26** ([thc1006, archived-and-re-measured](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)):

| Arm | pooled tok/s | change vs no-spec | acceptance |
|---|---:|---:|---:|
| **`spec-dflash-n2`** (target's own hidden states, block drafter) | **146.2** | **+26.3%** | 72.3% |
| **`spec-mtp-n2`** (model's built-in MTP head) | 141.9 | +22.7% | 78.4% |
| `spec-dflash-n4` | 137.9 | +19.2% | 55.2% |
| **no speculation** | 115.7 | — | — |
| `ngram-map-k4v-m8` | 115.4 | −0.3% | 50.0% (fires on 1/125 tokens) |
| `ngram-mod-n24` | 103.1 | −10.9% | 5.0% |
| `ngram-cache` | 93.7 | −19.0% | 5.2% |
| **`spec-draft-n8`** — classic **external draft model** | **30.9** | **−73.3%** | 29.5% |
| **`spec-draft-n1`** — external draft model | **29.2** | **−74.8%** | 69.7% |

**Conclusion, with an important scope correction.** On consumer llama.cpp, an **external small draft model (Qwen3-0.6B/1.7B/4B) is catastrophic *for that target*** — ~−75%. Even at 69.7% acceptance the classic draft-then-verify path loses, and the measured cause is concrete: **each speculative round pays an 82.079 MiB checkpoint** (772 creates / 709 restores across a ten-prompt run; **54.7% of the 71.4 s of excess time**).

⚠ **Do NOT generalise that −75% to your dense 27B.** That target is a **35B-A3B MoE activating ~3B params**, so a 0.8B dense drafter costs ~¼ of a target step; on a dense 27B the target step is far more expensive and the same drafter is proportionally much cheaper. The measured dense-target numbers are **positive**:

| Target | Drafter | Measured | Acceptance |
|---|---|---|---|
| Qwen3-**32B** Q4_K_M, llama.cpp (GB10) | Qwen3-**0.6B** Q8 | **1.37× mean** (range 1.08–2.55×) | 19.87% mean; **code 66% acc → 2.55×**, prose 10–20% → ~1.3× |
| Qwen3.6-27B dense, vLLM 2×3090 | built-in MTP-3 | +5–8% only | — |
| Qwen3.8-27B | MTP | "adds nothing" | — |
| Qwen3-32B, vLLM/H20 | **EAGLE-3** (`AngelSlim/Qwen3-32B_eagle3`) | **1.67–1.86×**, accept length 1.90–2.01 | — |

So: an external Qwen3-0.6B draft on a dense 27B is **worth one cheap experiment on code-shaped workloads** (0.75 GiB VRAM, +1.3–1.4 GB with its KV), but it is *workload-fragile* (excellent on code/JSON, mediocre on prose) and the trained heads below are better-supported. **The thing that is definitely not worth it is a separate 4B draft model, or a draft model on a different GPU (see §B.3).**

**What beats MTP is a *trained head*, not a small model:**
- **DFlash** (block-diffusion drafter, `--spec-type draft-dflash`): +26.3% vs MTP's +22.7% (same rig, same study); and on club-3090's single-3090-Ti Qwen3.6-27B run, DFlash gave **2.37× mean over no-spec (28.79 → 68.16 t/s) at only 0.238 acceptance** ([club-3090 llama-cpp README](https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/llama-cpp/README.md), #759). ⚠ **DFlash costs context** — the drafter inherits `-c` and mainline has no `-cd` cap; **q4_0 drafter-KV does not recover it** (the binding constraint is the drafter's compute buffers).
- **vLLM's MTP k-sweep on the *dense* 27B, 2×3090**: k=3 → **+20% narr / +77% code** (179/264) vs no-MTP baseline 149; k=4 → 171/276 ([tfriedel FINDINGS](https://github.com/tfriedel/qwen3.6-rtx3090-lab/blob/main/FINDINGS.md)). Acceptance by position at k=3: **0.77–0.84 / 0.57–0.68 / 0.40–0.53**; mean accept length **2.7–3.0**. Your 0.587 "acceptance" and AL 2.76 sit right in this band → your MTP head is performing normally.
- **EAGLE-3** is now first-class in llama.cpp (`--spec-type draft-eagle3`, converted with `--target-model-dir`). Supported heads include `AngelSlim/Qwen3-32B_eagle3`, `AngelSlim/Qwen3-14B_eagle3`, `Tengyunw/qwen3_30b_moe_eagle3`. **I found no published EAGLE-3 head or measurement for Qwen3.6-27B specifically** — the closest is Qwen3-32B. Independent caution: the llama.cpp maintainer's SGLang cross-check of gpt-oss-120b + EAGLE-3 on DGX Spark measured **0.46–0.71× baseline at batch 1** (i.e. a *loss*) — cited in the thc1006 README as the strongest independent corroboration of the llama.cpp spec-decode negative result. ⚠ **EAGLE-3 on batch-1 consumer hardware is not a demonstrated win.**
- **DSpark** (2026, DFlash + Markov head, `--spec-type draft-dspark`) is the newest option; `--spec-draft-n-max 7`, clamped to the draft's trained block size, plus `--spec-draft-conf-min P`. **No consumer-GPU measurements found — [E] only.**

### B.3 Does the draft need its own GPU? **No — and you actively should not do that.**

- MTP and DFlash/EAGLE-3 heads are small (hundreds of MB to ~1.5 GB) and are placed with `--spec-draft-ngl` / `--spec-draft-device` / `--spec-draft-override-tensor` / `-otd`. Co-resident is the normal, measured-good configuration everywhere in the club-3090 and tfriedel data (single- and dual-card DFlash runs are all co-resident).
- **Putting the drafter on a separate GPU over non-P2P PCIe is measured to be ~worthless.** club-3090 measured the lucebox dual-GPU split (target GPU 0, draft GPU 1) on a **PHB / `CNS`** rig — exactly your topology class — at **+1.7% over same-card (75.24 vs 73.97 t/s)**, and concluded: *"dual-GPU split needs a P2P-capable interconnect (NVLink or peer-supported chipset) to deliver its value. PHB+CNS rigs see no benefit."* ([BENCHMARKS.md](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md))
- **But shared-GPU contention is a real, documented failure mode.** One report describes it as *"catastrophic slowdown, not a mild one"*, with the fix being *"split devices with `-devd`"*. Since **`-devd` defaults to following `--device`**, the drafter lands on the target's GPUs by default. For you (80 GB, 29 GB model) that is the right default; if you ever run near a VRAM ceiling, use `--spec-draft-device` deliberately rather than by accident.
- **Known null:** there is **no published head-to-head "draft on its own GPU vs shared GPU" on a 4-card consumer rig with a ~27B target**, and **no measured separate-GPU draft case with P2P disabled**. Treat any claim here as unverified.
- Your target already fills all 4 GPUs; the drafter adds ~0.5–1.5 GB somewhere. **The Qwen3.6 MTP head is 3.7 GB at BF16** — *"cannot sit beside a 21 GB target on a 24 GB card, so it is quantised."* **You have 80 GB: keep the MTP head at BF16/Q8_0** (see §B.6 — this is directly worth acceptance length). If VRAM is tight, drop the *main* model's quant, not the head's.

### B.4 Beyond classic spec decoding (2025–2026, measured on consumer GPUs)

| Technique | Available on your rig? | Measured speedup | Quality |
|---|---|---|---|
| **MTP head** | ✅ native, already running | +22.7% (MoE, 1×3090) / +20–77% (dense 27B, 2×3090 vLLM) | lossless |
| **DFlash** (`draft-dflash`) | ✅ mainline since #22105, 2026-06-28; drafter `Anbeeld/Qwen3.6-27B-DFlash-GGUF` | **+26.3%** (MoE, 1×3090, matched no-spec); **2.37×** (27B, 1×3090 Ti, AR 0.238) | lossless; **costs context** |
| **DSpark** (`draft-dspark`) | ✅ 2026, #25173 | no consumer measurements found | lossless in principle |
| **EAGLE-3** (`draft-eagle3`) | ✅ flag exists; **no Qwen3.6-27B head published** | maintainer's DGX-Spark SGLang check: **0.46–0.71×** (a loss) at batch 1 | lossless |
| **Medusa** | ❌ not in llama.cpp | — | — |
| **Lookahead decoding** | ❌ not in llama.cpp | — | — |
| **Self-speculative / layer-skip** | partially via `ngram-*` | layer-skip not implemented | — |
| **n-gram / prompt-lookup** (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`) | ✅ | **Extremely workload-dependent; a loss on MoE/prose, a large win on structured/repetitive text.** thc1006: ngram-cache **−19.0%**, ngram-mod −10.9%, ngram-map-k4v −0.3%. **BUT** a 4090 Qwen3-8B sweep found `ngram-mod` **beats draft-simple on all four tasks at ZERO VRAM: code 2.76×, JSON 2.81×, translation 1.99×, creative 1.09×**; club-3090 measured `--spec-type ngram-mod` on Qwen3.6-27B Q4_K_M at **+25% on code** (22/26 vs 21/21). ⚠ **The ngram-mod + draft-mtp combo was tested as a LOSS in mainline llama.cpp (91.69 vs 92.40 t/s)** — the +35% two-stage result comes from **ik_llama.cpp's purpose-built two-stage implementation ([PR #1789](https://github.com/ikawrakow/ik_llama.cpp/pull/1789))**, not mainline | lossless, free (no extra VRAM) |
| **Luce/DFlash fork, PFlash long-context compression** | fork only | ~24× faster prefill compression at 131K source on 1×3090 | ❌ **failed verify-stress 0/7 in the integrated server path** — not shippable |

**Two high-value, low-risk items here:** (1) **ik_llama two-stage `ngram-mod n_max=4` + `MTP n_max=3`** — measured **+35% code decode, lossless, zero extra VRAM**, with an inverted-U sweep (n=2→78.8, n=4→**97.8**, n=8→90.4, n=16→81.5); (2) **`--spec-type ngram-mod` alone** as a draftless add-on, measured +25% code on Qwen3.6-27B at zero cost.

### B.5 `--draft-max` / `--draft-min` / media equivalents

Renamed. Current spellings and defaults are in §B.0. `media` is not a llama.cpp concept. The equivalent knobs are `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-min`, `--spec-draft-p-split`. ⚠ Any tutorial or script you copy that still uses `--draft-max` is **pre-rename** (llama.cpp renamed after `bcb5eeb64`) — check before trusting its numbers.

### B.6 Acceptance-rate engineering (the small stuff that actually moves the needle)

- **Keep the MTP head at high precision — and you have the VRAM to do it.** club-3090's Carnice-V2-27B diagnostic found *"Un-quantizing 7 `mtp.layers.0.*` projections (BF16 overlay) recovered AL from 2.0 → 3.0"* — MTP-head quantization mismatch accounted for **~70% of the acceptance-length gap** ([qwen3.6-27b CHANGELOG, 2026-05-04](https://github.com/noonghunna/club-3090/blob/master/models/qwen3.6-27b/CHANGELOG.md)). The head is **3.7 GB at BF16**, which is why 24 GB single-card users are forced to quantize it — **you are not.** If you drop the main model to Q4_K_M, keep the head at BF16/Q8_0. ik_llama.cpp exposes `--mtp-requantize-output-tensor new_type` for exactly this.
- ⚠ **Two known llama.cpp MTP bugs may be depressing your 0.587 / 2.76 right now.**
  - **[issue #23658](https://github.com/ggml-org/llama.cpp/issues/23658) — acceptance collapses at 2048-aligned context sizes.** Measured: **1.12× at 16% acceptance @ ctx 12032 vs 1.91× at 71% @ ctx 12288**; also 17024 → 0.88× @ 0%; 24064 → 0.89×. One commenter traced the collapse to an unstable undervolt. **Action: sweep your `-c` across a 2048 boundary (e.g. 12032 / 12288 / 14336) and re-read AL.** This is a free, potentially large win that has nothing to do with quantization.
  - **[issue #23322](https://github.com/ggml-org/llama.cpp/issues/23322)** — SWA/hybrid models show low acceptance (35–37% reported on **Qwen3.6-27B-MTP**). Relevant if your 27B is a hybrid (DeltaNet/GDN) architecture.
  - Related: **[issue #24055](https://github.com/ggml-org/llama.cpp/issues/24055)** — context checkpoints always invalidated on hybrid/recurrent models, which is exactly the 82 MiB-per-round cost mechanism from §B.2.
- ⚠ **Trust mean accepted length, not the acceptance rate.** llama.cpp's server *"draft acceptance rate"* counter **can read 100% while true token-level acceptance is 53.7%** — a documented counter artefact in the thc1006 audit (the pre-audit "100% acceptance yet slower" headline was retracted for this reason). Use `statistics draft: #acc tokens / #gen tokens`, or your own `AL = 1 + n_max × accept`. **Your AL = 2.76 is a sound measurement.**
- **Reasoning/thinking text drafts far better than terse output.** club-3090: *"reasoning text drafts well, so a deeper draft pays off"* — thinking-on profile uses `n=5, p-min=0.5`, thinking-off uses `n=2, p-min=0.0`. **Your optimal n_max depends on whether you're streaming reasoning or answers.** [M-cited, single-card]
- **Temperature**: spec decoding is distribution-preserving under stochastic sampling, so it does not cost quality; but acceptance falls as temperature rises (higher entropy → less predictable). Note that **at temp 0, `--spec-draft-p-min` had literally no effect** on accepted counts.

### B.7 Theoretical ceiling — the arithmetic, and how to *measure* it

**Define:** `T_pass` = wall time of one target forward pass = **35.25 ms** [E, from `AL/TPS = 2.76/78.3`].
Then `TPS_ceiling(n_max) = (n_max + 1) / T_pass`, assuming perfect acceptance and unchanged per-pass cost.

| n_max | perfect AL | ceiling t/s | vs 78.3 |
|---:|---:|---:|---:|
| 1 | 2 | 56.7 | −28% |
| 3 | 4 | **113.5** | **+45%** |
| 4 | 5 | 141.8 | +81% |
| 5 | 6 | 170.2 | +117% |
| 6 | 7 | 198.6 | +154% |
| 7 | 8 | **226.9** | **+190%** |

**More realistic, using the conditional acceptance probability q ≈ 0.86** implied by your own numbers (E[accepted] = Σ q^i):

| n_max | E[accepted] | AL | t/s at constant `T_pass` |
|---:|---:|---:|---:|
| 3 (today) | 1.76 | 2.76 | 78.3 ✓ |
| 4 | 2.27 | 3.27 | 92.7 (+18%) |
| 5 | 2.81 | 3.81 | 108.1 (+38%) |
| 6 | 3.28 | 4.28 | 121.4 (+55%) |
| 7 | 3.73 | 4.73 | 134.2 (+71%) |
| 15 (DFlash, clamped) | ~5.0 | ~6.0 | 170 (+117%) |

**The standard closed-form** (Leviathan et al. 2023; restated as Eq. 1 of arXiv 2601.11580v2) is `E[speedup] = (1 − α^(k+1)) / [(1 − α)(k·c + 1)]`, where the numerator is exactly `Σ_{i≤k} α^i + 1` = expected accepted length including the bonus token, and `c` is the draft/verify cost ratio. Per-position form: `E[L] = 1 + Σ_{i=1..k} Π_{j≤i} p_j`. **This is why you must not read the AL table as free money: the `k·c` term in the denominator is the draft+verify cost, which is what turns every measured curve over between n=4 and n=8.**

**Two counter-pressures keep this from being free [E]:** (a) the verify batch grows, so per-pass *bytes moved* (activations, all-reduce payload) grows roughly linearly with AL — on a latency-bound all-reduce this is nearly free, on a bandwidth-bound one it costs; (b) the drafter itself runs n_max extra small forward passes, plus llama.cpp's ~82 MiB checkpoint per round. The b70, thc1006 and tfriedel data all show these turning the curve over: b70 found n=8+ loss, tfriedel found k=5 < k=4 for narrative, thc1006 found a **cliff at n=6–8**.

**Realistic target, stated honestly:** you already capture **2.76 / 4.0 = 69%** of the perfect-acceptance ceiling *at n_max = 3*. Going to a well-tuned n_max = 4 (and re-tuning after you fix per-pass overhead, which is what moved the dense-27B vLLM optimum from k=3 to k=4) is worth **~+15% to +30%**. Holding n = 6–7 with gating on long prompts (the b70 result) is worth up to **~+40%**, but is not guaranteed on your hardware/architecture. **Plan for +15–30%, treat +40% as the upside, and do not budget for the +117% the depth-6 row implies.**

**⭐ Measure the ceiling directly instead of guessing.** llama.cpp ships synthetic-acceptance benchmarking:

```bash
# pretend every draft is accepted, mean accept length = 4 (n_max=3, perfect)
llama-server ... --spec-type draft-mtp --spec-draft-n-max 3 --spec-synth-len 4
```

`--spec-synth-len L` sets the target mean acceptance length for `K` draft tokens, `L ∈ [1, K+1]`; `--spec-synth-rates P0,P1,...` sets per-position rates explicitly. **This isolates the verify-pass cost from acceptance quality in one run**, and it is the cheapest decision-relevant experiment available to you — it tells you your true ceiling without needing a drafter at all. (Generated output is invalid under synthetic acceptance — benchmarking only.)

---

## PART C — OTHER THROUGHPUT LEVERS

### C.1 ⚠ KV-cache quantization — **corrected**. It is NOT a 3× lever; the real lever is a missing kernel

My first pass reported club-3090's community claim that *"q8_0 KV is ~3× slower than q4_0"* and told you to check it first. **That claim is a misdiagnosis and I am retracting the recommendation.** The real measured picture:

| Claim | Status |
|---|---|
| "q8_0 KV ≈ 3× slower than q4_0" | **Artefact.** club-3090 itself warns *"Test it on your own rig before trusting the 3× claim."* |
| **q8_0/q8_0 KV cost at short context on CUDA** | **−1.6% to −2.9% decode** — measured, 2× RTX 5060 Ti, CUDA 12.8, llama.cpp, 8K ctx, 4 concurrent ([llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969), @Defilan 2026-04-08): Llama-3.1-8B Q5_K_M 42.5 → 41.8; Qwen2.5-14B 24.1 → 23.4 |
| **q8_0 KV win at long context, real SM86 CUDA** | **only +3.5% to +5.3% at 32K** (RTX 3090: 61.0 → 63.12; 3090 Ti: 77.44 → 81.58) — *not* a large lever |
| The scary **−51%** figure | **Single Intel Arc Pro B70, SYCL backend** — *do not apply to CUDA* |
| The **−71% over 131K on SM86** figure | A **turbo4** dequant-kernel warp-dispatch problem on SM86, **not** q8_0/q8_0. But note the general finding stands: *"single-stream TG above ~32K–65K on SM86 pays a structural per-token rate tax you can't fix by changing KV format choice."* |
| Other CUDA points (#20969) | 4090 asymmetric q8_0-K/turbo4-V = +8.4% prefill / **−6.2% decode**; symmetric turbo4/turbo4 = **−16.8% decode**; 5090 q8_0+t4 = −25.2% with one kernel set vs **−6.9%** with the fused one. Turbo-style KV formats are the risk, not plain q8_0. |

**⭐ The actual lever hiding behind that folklore: `-DGGML_CUDA_FA_ALL_QUANTS=ON`.** llama.cpp only compiles flash-attention kernels for **quantized** KV types when that CMake flag is set. Without it, a quantized-KV FA path silently falls back — and @WaveboSF measured it as *"a free **7.4%** that people are definitely leaving on the table."* Two open upstream items confirm the trap is real: FR [#24485](https://github.com/ggml-org/llama.cpp/issues/24485) ("make `GGML_CUDA_FA_ALL_QUANTS` the default or warn") and PR [#28456](https://github.com/ggml-org/llama.cpp/pull/28456) ("warn when flash attention falls back to the CPU backend due to unsupported KV cache types"). **Check your build, and check the server log for a fallback warning.**

**Quality cost of cheap KV is real but tail-shaped**: Anbeeld's long-context benchmarks put `turbo3_tcq` at **~82% 99.9th-percentile KLD tail precision** — visible loss on the worst 0.1% of positions (JSON keys, closing braces, tool calls). ik_llama.cpp's `-khad`/`-vhad` Hadamard transforms recover much of that **at zero VRAM cost**.

**Net answer to your C.9 question:** KV-cache quantization changes decode speed only marginally on CUDA (**≈−2% at short context, up to +5% at 32K from reading less**), so it is *mostly* a memory/context knob — but there is a **~7% free win** available from building FA properly. Also note: **`--split-mode tensor` upstream requires non-quantized KV (`f32`/`f16`/`bf16`)** — see §C.5 — so quantized KV may not even be an option for you.

### C.2 Memory overclocking (GDDR6X) — **NULL result on published evidence**

- **⚠ There is no published before/after llama.cpp or ik_llama.cpp decode measurement for ANY GDDR6X memory offset** — nine targeted queries returned nothing, and a GitHub repo search for "memory overclock" in `ggml-org/llama.cpp` returns three unrelated hits. Everything below is arithmetic and mechanism, not a measurement. (Consumer GDDR6X has **no ECC**, so the ECC on/off data in the wild does not apply to you.)
- **Physics [E]:** your 320-bit bus at 19 Gbps = 760 GB/s. Afterburner offsets apply to the 9501 MHz memory clock. **+500 → 20.0 Gbps → 800 GB/s (+5.3%)**; **+1000 → 21.0 Gbps → 840 GB/s (+10.5%)**; +1500 → 22.0 Gbps → 880 GB/s (+15.8%).
- **Does decode scale with bandwidth? Within one kernel path, approximately yes; across paths, no.** Supporting: llama.cpp's own perf notes — *"the most important factor for performance is memory bandwidth"*; and a vendor clock sweep on a 5090 measured *"TPS climbs nearly linearly with mem clock — at 405 MHz mem, TPS caps at ~53 regardless of GPU clock; at 14001 MHz mem, TPS climbs through 800+"*. Contradicting: the 2×3090 Q6_K_L-vs-Q8_0 result in §0 (larger file, faster) and the non-monotonic quant ranking in §A.5.
- **⭐ The mechanism that caps this, and it is not "error correction costs you perf":** NVIDIA GA102 GDDR6X uses **Error Detection and Replay (EDR)**. Per the GA102 documentation: errors *are* corrected, but **"bandwidth stops increasing — it becomes a plateau… once the plateau is reached there is no point in raising the clock further."** So a memory offset that looks perfectly stable, with no artifacts, can yield **nothing at all**. This is the dominant risk, and it is invisible without a before/after measurement.
- **Thermal context [M-cited, 2020, ⚠STALE]:** an IR-camera measurement of a 3080 FE at stock found the hottest GDDR6X chip at **104 °C**, with Tjmax estimated 105–110 °C ([igor'sLAB](https://www.igorslab.de/en/gddr6x-am-limit-ueber-100-grad-bei-der-geforce-rtx-3080-fe-im-chip-gemessen-2/), [TechPowerUp summary](https://www.techpowerup.com/272420/)). In a 4-card rig with tight spacing, memory-junction throttling is a live risk. **Monitor `nvidia-smi -q -d TEMPERATURE` memory junction under load before and after any OC.**
- The only VRAM-OC→decode datapoint found anywhere is **not Ampere**: an RTX PRO 6000 Blackwell GDDR7 note (2026-07-30) reports *"VRAM overclocking (+6000 mem) ~10 tok/s decode boost on GLM 5.2, since decode is bandwidth-bound."*
- **Verdict: worth 20 minutes as a free experiment, but budget +3–5% at +500 and +6–10% at +1000, watch for the EDR plateau, and do not treat it as a strategy.** Note also that **`-lmc` (memory clock lock) is not reliably available on Ampere consumer cards**, so you set offsets via Afterburner/LACT and cannot pin them.

### C.3 Power limits, clock locking, persistence — **direction corrected**

⚠ **If you have reduced your power caps, undo it. On llama.cpp, a lower cap costs you TPS, often badly.** Measured, llama.cpp + Qwen3.6-27B Q3_K_XL, single 3090:

| Rig | Cap | Decode (narr/code) | vs stock |
|---|---|---:|---:|
| air 3090 | 290 W | 32.26 / 32.17 | −6.9% vs 370 W |
| air 3090 | 370 W (stock) | 34.66 / 34.67 | — |
| air 3090 | 390 W (max) | 36.26 / 36.06 | +4.6% |
| water 3090 | **230 W** | **25.15** | **−34.2% vs 388 W stock** |
| water 3090 | 330 W | 36.35 | −4.9% |
| water 3090 | 388 W (stock) | 38.23 | — |

Source: [club-3090 HARDWARE.md §Power](https://github.com/noonghunna/club-3090/blob/master/docs/HARDWARE.md). *"The '230W is the sweet spot' lore is wrong."* The mechanism is specific and important: **`chunked_gated_delta_rule` is genuinely compute-bound, not memory-bound** on this model family — so a power cap bites hard where a pure weight-streaming workload would not feel it.

**The 290 W figure is a TPS/W optimum, not a TPS optimum.** For raw throughput, higher is better. Use 290 W only if you are power/thermal-limited; if you already capped to be "efficient," raising the cap back is a **>10% lever** (up to +50% vs 230 W).

**Counter-evidence on newer silicon [M-cited]:** a Blackwell 4 GB memcpy ran 682 iter/s @401 W vs 680 @365 W — *"memory clock is invariant within each variant regardless of power limit; power limit gates only GFX/SM clock."* On Ampere `-pl` likewise does not lower GDDR6X clock, so a pure weight-streaming GEMV should be PL-insensitive; the club-3090 penalty includes MTP + hybrid-attention compute. **Expect your sensitivity to land between the two — and measure it rather than assuming.**

| Other lever | Measured effect |
|---|---|
| **`nvidia-smi -pm 1`** (persistence) | *"common to find this off on KVM/VM hosts; with it disabled the driver tears down between idle periods and adds per-request init latency."* Latency hygiene, not a >10% TPS lever. Just do it. |
| **`nvidia-smi -lgc` / `-lmc`** (clock lock) | Large win on **5090** (2.025 TPS/W at 7001/1635 MHz = **1.42× the efficiency** of the 400 W cap; and a Pareto point at 14001/2122 giving **+5% TPS at −22% power**). ⚠ **NULL for Ampere** — *"Clock-locking is not portable to Ampere — `-lmc` was removed from non-datacenter cards in some driver versions, and the 3090's voltage curve doesn't have the same headroom."* |
| `-lgc` on 3090, indirect | At the 290 W efficient cap the core clock is already only ~1380 MHz (SM locks at 1560 MHz across 340–370 W) — i.e. the efficient cap is *already* effectively a low core-clock lock. |

### C.4 NUMA, PCIe, CPU, IOMMU

- **Keep all 4 GPUs in one NUMA domain.** Set BIOS **NPS1**; `nvidia-smi topo -m` should read `PHB` (or `PIX`), never `NODE`/`SYS`. *"Under NPS4, two GPUs in different CPU quadrants can report NODE (or worse) instead of PHB, adding cross-die latency to every all-reduce."*
- **PCIe generation and CPU IPC matter enormously for TP.** Measured: *"PCIe Gen 3 + older CPU → TP=2 paths take a **30–40% throughput hit**"*, with a worked example: **Xeon Gold 6138 + Gen3 x16 + 2×3090 → 32/41 TPS** vs **Ryzen 5950X + Gen4 → 89/117 TPS** on the same config ([club-3090 HARDWARE.md](https://github.com/noonghunna/club-3090/blob/master/docs/HARDWARE.md)). *"In this class CPU IPC can outrank cross-card PCIe bandwidth."*
- **Check trained link width under load**: `nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv` vs `...max`. Riser/BIOS bifurcation silently halves it. A chipset-attached card drags the whole collective down to its own link.
- **`iommu=pt`** on AMD hosts — without it, sustained TP DMA can drive **AMD-Vi `Xid 154` fatal MMU faults** ("Node Reboot Required") with misleading collateral symptoms. Triage: `dmesg | grep -E "AMD-Vi.*IO_PAGE_FAULT|Xid.*154"`.
- **`CUDA_DEVICE_MAX_CONNECTIONS`**: **NULL, and actively risky.** No measured multi-GPU llama.cpp decode effect exists anywhere. llama.cpp's CUDA backend and NCCL set it internally; a global override can **disable work-queue concurrency the backend relies on**. **Do not touch it.**
- **NUMA on EPYC is largely a NULL for tok/s.** What exists: llama.cpp PR #16000 adds a `--numa mirror` strategy (⚠ 2025-09, stale); one Proxmox-VFIO row is described as *"NUMA-aligned (vCPUs+mem+GPU same node)"* qualitatively only; and the concrete win is a **stability** one — `iommu=pt` (above). **No measured `numactl --cpunodebind/--membind/--interleave` tok/s delta exists for any host-staged multi-GPU llama.cpp rig.**
- **⭐ Best remaining unretrieved lead: [llama.cpp Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733) — "Dual Epyc Genoa/Turin token generation performance bottlenecks."** A search snippet shows it contains measured NUMA rows in the shape `NPS1 | likwid-bench load | 460.8 | 359.9 | 78.1%`. **This is the most likely place your answer lives — pull it manually.**

### C.5 Batching / misc llama.cpp flags

| Flag | Measured effect |
|---|---|
| `-fa on` | **Required for the q4_0-KV fast path** and for `-sm tensor` (without FA, `-sm tensor` segfaults in warmup — [llama.cpp #24489](https://github.com/ggml-org/llama.cpp/issues/24489) / PR #24491). |
| `-b` / `-ub` | b70 sweep: *"`BATCH_SIZE=1024` did not move the n=4 lane; do not prioritize batch size until MTP depth/gating is exhausted."* `-ub 512` costs ~2–4% TPS but buys context. |
| `--no-mmap` + `--mlock` | Changes load time / page-fault behaviour. Not shown to change steady-state decode in any source I found. **Not a >10% lever.** |
| `-t` threads | Only affects CPU-side ops (`"set to physical-cores / 2 typically"`). Irrelevant when everything is on GPU. |
| `--tensor-split` | On a 4-card unbalanced rig, worth rebalancing so no single card is the straggler (all-reduce runs at the slowest participant). |
| **`ik_llama.cpp` instead of mainline** | **+18–20% decode at matched power, quality tied** (§A.4). Plus `-khad`/`-vhad` free KV quality, plus the low-perplexity `Q4_0` KV work, plus two-stage ngram-mod+MTP (+35% code). |
| **`beellama.cpp` fork** | **Fastest single-card 3090 code path measured: 50.2 / 99.7 t/s (Q5_K_S + DFlash-IQ4_XS)** vs ik 60/72 and mainline 50/59 — narrative ~tie, **code ~+40%**. |
| **`--split-mode tensor` vs `layer` vs `row`** | See §C.6 — this is the most over-claimed lever on your list. Headline: **`row` is DEPRECATED**; `tensor` is upstream **EXPERIMENTAL**; and **whether you have NCCL compiled in changes the answer completely.** |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | **~+7.4% free** if you use quantized KV. See §C.1. |
| `GGML_CUDA_P2P` | Opt-in, **OFF by default**; upstream warns it *"may cause crashes or corrupted outputs."* Measured llama.cpp single-stream gain with verified P2P: **≈0** (single-stream llama.cpp moves ~2 GB/s against 13 GB/s available — *"it is not interconnect-bound"*). |
| `GGML_CUDA_FORCE_MMQ` / `FORCE_CUBLAS` / `MMV_Y` / `ENABLE_UNIFIED_MEMORY` | **NULL** — no measured decode deltas found. `GGML_CUDA_DMMV_X` is **obsolete** (DMMV was removed in llama.cpp PR #10318). |
| `CUDA_DEVICE_MAX_CONNECTIONS` | **NULL, and actively risky.** No measured multi-GPU llama.cpp decode effect exists. llama.cpp's CUDA backend and NCCL set it internally; a global override can disable work-queue concurrency the backend relies on. **Do not touch it speculatively.** |

### C.6 Split mode on non-P2P: `tensor` vs `layer` vs `row` — **the real picture (corrects my Action #6)**

This is where the folklore is worst. Three things are true simultaneously and are constantly conflated:

| Mode | Upstream status (docs, fetched 2026-09-24) | Inter-GPU traffic |
|---|---|---|
| **`row`** | **"Deprecated… comparatively poor performance… Superseded by `tensor`… Avoid in new deployments."** | all-reduce every layer |
| **`layer`** (default) | *"Can tolerate slow interconnect speeds between GPUs."* | activations at the boundary — **~7 MB/s measured** |
| **`tensor`** | **"EXPERIMENTAL… Performance should be good for multiple NVIDIA GPUs using the CUDA backend, no guarantees otherwise."** | all-reduce every layer — **~1400 MB/s measured, latency-bound** |

**The near-exact-match measurement for your topology** ([habr, 2026-08-29](https://habr.com/ru/articles/1076080/)) — 2× RTX 3090, PCIe 4.0 x16, **no NVLink**, Threadripper 3960X, llama.cpp build 10666, Qwen3.8-27B dense 24.1 GB, MTP n=3, 3 runs/pt:

| Metric | `-sm layer` | `-sm tensor` | Δ |
|---|---:|---:|---:|
| Generation, short ctx | 58.5 | **74.7** | **+27.7%** |
| Generation @130K | 41.4 | 58.9 | +42.3% |
| Generation @240K | 34.4 | 55.4 | +61.0% |
| **No-MTP** generation | 32.4 | 48.2 | **+48.8%** |
| **Prefill @40K** | **1410** | 1121 | **−20.5%** |
| Prefill @240K | 764 | 743 | −2.7% |

⚠ **But read the caveats before acting:**
1. **NCCL presence decides everything.** That error path names `ggml_backend_cuda_comm_allreduce_nccl` — so *that* build had NCCL. club-3090 verified that `ghcr.io/ggml-org/llama.cpp:server-cuda-b10236` contains **zero NCCL symbols** and logs `NCCL not compiled in; falling back to internal AllReduce`. On such a build, `tensor` measured **~noise vs `layer`** (61.5→64.2 narr, 73.5→72.2 code, prefill 1313→1346; club-3090 PCIE_P2P disc #903). Upstream CMake defaults `-DGGML_CUDA_NCCL=ON`, but the shipped images are not all built that way. **Check your binary — grep the boot log for the NCCL warning.** If you are on the internal fallback, that is a real, cheap, unexplored lever for you.
2. **Do not extrapolate 2 cards to 4.** All-reduce ranks go 2→4, each adding a serialized host-staged exchange. The one clean matched-recipe 4-card pair (vLLM, @alesha-pro, 4×3090 Gen3 all-x16) shows **decode flat: TP=2 79.26/106.50 vs TP=4 82.44/103.73 (+4.0% / −2.6%)**, with prefill **+54.6% @10K / +62.0% @90K**. My estimate for your 4-card `tensor`-vs-`layer` decode delta: **+0% to +15%**, with a real **prefill cost of −10% to −25%**. [E]
3. **`--ipc=host` is MANDATORY** in Docker. Without it the first decode dies with `CUDA error: unhandled system error` in `ggml_backend_cuda_comm_allreduce_nccl` — and Docker's message gives no hint.
4. **Upstream says `tensor` requires non-quantized KV** (`f32`/`f16`/`bf16`): *"Support for quantized KV cache is not implemented and trying to use it will result in an error."* Open FRs [#21788](https://github.com/ggml-org/llama.cpp/issues/21788) and [#23567](https://github.com/ggml-org/llama.cpp/issues/23567); PR #23225 closed unmerged. **But the habr run used `-ctk q4_0 -ctv q4_0` with `-sm tensor` successfully.** Verify on your exact binary before planning around it.
5. **`-sm row` fails outright on this class**: `error loading model: device CUDA0 does not support split buffers`.
6. **`--tensor-split` / `-ts` is IGNORED under `-sm tensor`** (and `--fit` is *"not supported with `tensor`"*). It is a `layer`-mode knob; there, measured **+3–5% for 30B-class models** on an asymmetric 2-card rig.
7. **Under `tensor`, sampling is forced to CPU once per slot** — the log shows `spec common_specu: backend offload failed for seq_id=N; using CPU sampler`, exactly one line per slot (zero under `layer`). On a weak CPU this can become the new bottleneck.
8. **Latency, not bandwidth, is what you are paying for.** club-3090's measured decomposition: tensor-parallel all-reduce is *"many small serialized transfers"* with per-layer sync **15.23 µs host-staged vs 1.01 µs with P2P — a 15× latency difference** — and *"if you reason only about bandwidth you will wrongly conclude P2P cannot help; utilisation is often under 20% while the latency saving is 15×."*
9. **Correctness hazard.** [PCIE_P2P §7a](https://github.com/noonghunna/club-3090/blob/master/docs/PCIE_P2P.md) documents a rig where a patched P2P collective **completed fast and returned wrong output** with every indicator green — *"caught only by reading the generated text. That is strictly worse than a hang."* Whatever you change here, **read the generated text**, don't just watch tok/s.

**Revised expectation for Action #6: [+0% to +28%] decode, at a possible [−10% to −25%] prefill cost — not the +30–85% I originally wrote.** The +30–85% number came from assuming your η=0.27 could be lifted toward the 0.70 seen on a *vLLM-with-CUDA-graphs* rig; that comparison crosses an engine boundary and is not a supported inference. The genuine, separate opportunity in that area remains the **NCCL-enabled build** if your binary lacks it.

---

## PRIORITIZED ACTION LIST

Ordered by (expected gain) / (effort × risk). Percentages are **[E]** unless marked.

| # | Action | Expected decode gain | Quality cost | Effort | Confidence |
|---|---|---|---|---|---|
| **1** | **Build with `-DGGML_CUDA_FA_ALL_QUANTS=ON`** and check the server log for an FA-fallback warning. *(Replaces my retracted "check your KV type" advice — plain q8_0 KV costs only ~2% on CUDA, and the "3× slower" claim is an artefact of this missing kernel.)* | **up to +7.4%**, "free 7.4% that people are definitely leaving on the table" | none | 30 min (rebuild) | high — measured, with two open upstream FRs/PRs confirming the trap |
| **2** | **Measure the no-spec baseline** (`--spec-type none`) and `--spec-synth-len 4,8` | no gain — **this diagnostic decides between η=0.27 and η=0.75 and therefore sizes every other item** | none | 10 min | — |
| **3** | **Re-tune MTP depth**: sweep `--spec-draft-n-max ∈ {2,3,4,5,6}` × `--spec-draft-n-min ∈ {0,2}` × `--spec-draft-p-min ∈ {0.0,0.15}`, separately for short chat vs long/code prompts — and **re-sweep after fixing per-pass overhead**, because that is what moved the dense-27B optimum k=3 → k=4 | **+15% to +30%** (n=3→4 realistically); up to +40% if n=6–7 holds on long prompts; you already capture 69% of the n=3 ceiling | **none** (lossless) | 1 h | high — but **n=2/3/4 are statistically indistinguishable** in the best-controlled design; expect a plateau then a cliff at 6–8 |
| **3b** | **Sweep `-c` across a 2048 boundary** while watching acceptance (llama.cpp [#23658](https://github.com/ggml-org/llama.cpp/issues/23658): 1.12× @16% acc at ctx 12032 vs 1.91× @71% at 12288) | free win if you are sitting on a bad alignment; 0% otherwise | none | 20 min | high — known upstream bug, exact mechanism you can test |
| **3c** | **Confirm `-ctkd`/`-ctvd` are set for the draft KV.** Draft KV is *separate* from `-ctk`/`-ctv`; without these the draft KV silently sits in **f16**, costing VRAM and draft time | small but free | none | 5 min | measured gotcha [M-cited] |
| **4** | **⭐ Sweep quant × MTP TOGETHER — not quant alone.** The quant ranking measured without speculation **inverts** under MTP (llama.cpp [#27373](https://github.com/ggml-org/llama.cpp/issues/27373), 2026-08-19: MTP gain is IQ4_XS **+47.7%**, Q4_K_M **+9.3%**, Q3_K_M **−17.5%**, IQ3_XXS **−69%**). Candidate set: **Q6_K (23.86 GB)**, Q4_K_M (17.44 GB), IQ4_XS (15.48 GB) | **Q6_K ≈ +10–18%** (but near-lossless); Q4_K_M up to **+40%** (real-size-corrected); the *winner under MTP may be IQ4_XS* despite being slower without it | Q6_K: **KLD 0.0065 = near-lossless**; Q4_K_M: **KLD 0.0192 = 7.4×**; IQ4_XS: **KLD 0.0234**, MMLU-Pro −1.6 pt; Q3_K_M: **−4.3/−5.2 pt** | 2 h (downloads) | **high value, moderate confidence** — the inversion is measured but on other rigs |
| **4b** | **Sweep `GGML_MMVQ_MAX` (kernel routing).** Ampere/sm_86 gets **no per-type MMVQ tuning** — it falls to `MMVQ_PARAMETERS_GENERIC` — and forcing Q5_K onto MMVQ cost **−34%** on an L4. A routing sweep was worth **+16.2%, more than the whole quant change** | **up to +16%**, quant-independent | none | 30 min | medium — measured on another arch, but the Ampere generic-fallback is verified in source |
| **4c** | ⚠ **Do NOT pick by no-spec decode rate, and do NOT use IQ4_XS/IQ4_NL on mainline llama.cpp expecting speed.** I-quants are **−5.8%** vs K-quants at equal size on CUDA, and IQ4_XS is *slower than the larger Q4_K_M* in two independent 3090 tests. Also: real K-quant files are **4–7% larger** than the bpw formula predicts (IQ up to +11%) — size your projections from the measured table in §A.1 | — | — | — | measured, 3 sources |
| **5** | **Keep the MTP head at BF16/Q8_0** when you drop the main model — it is 3.7 GB at BF16, which is why 24 GB users can't, and **you can** | recovers **AL 2.0 → 3.0** in the published case | none | 20 min | measured on a 27B fine-tune [M-cited] |
| **5b** | **Raise power caps back if you lowered them.** llama.cpp + this model family is partly **compute-bound**, so caps bite hard: 230 W = **−34%**, 290 W = **−6.9%** vs 370 W stock | **up to +50%** vs a 230 W cap; **+7%** vs 290 W | none | 5 min | high (3090/llama.cpp measured) |
| **6** | **Check whether your binary has NCCL, then A/B `-sm tensor` vs `-sm layer`** at short / 32K / 128K. (`row` is deprecated — ignore it.) Use `--ipc=host` in Docker. Re-check numpy/PCIe width, `nvidia-smi -pm 1` | **+0% to +28%** decode (2-card measured +27.7% with NCCL); **≈0%** if you are on the internal-allreduce fallback. **Prefill −10% to −25%** | none | 0.5 day | **medium-low on 4 cards** — flat in the one matched 4-card pair. *Revised down from +30–85%; see §C.6* |
| **6b** | **Build llama.cpp with `-DGGML_CUDA_NCCL=ON` + `NCCL_P2P_DISABLE=1`** if your current binary lacks NCCL (grep the boot log for `NCCL not compiled in; falling back to internal AllReduce`) | unmeasured on your topology; the 2-card NCCL-enabled build shows tensor +27.7% over layer, and the internal-fallback build shows ~0% | none | half day (build) | **unknown — genuinely unexplored, and cheap to test** |
| **6c** | **Try TP=2** (`CUDA_VISIBLE_DEVICES=0,1`) with Q5_K_S/Q4_K_M and compare against TP=4 | 4-card TP=4 vs TP=2 measured **flat** (vLLM, +4.0%/−2.6%); on 2 cards `tensor` beat `layer` by +28%. Frees 2 GPUs for a second model or a drafter | none | 30 min | medium |
| **7** | **Add `ngram-mod`** as a draftless second stage (ik_llama: ngram-mod n_max=4 **+** MTP n_max=3). Note the combo is a **loss in mainline llama.cpp** — use ik_llama's two-stage | **+25% code** (mainline, single flag) / **+35% code** (ik two-stage) | **lossless, 0 VRAM** | 15 min | measured [M-cited] |
| **8** | **Try DFlash** (`--spec-type draft-dflash --spec-draft-model <Anbeeld Qwen3.6-27B-DFlash-Q4_K_M> --spec-draft-n-max 15`) | **+26% over MTP** (matched); up to **2.37×** vs no-spec on 1×3090 Ti | lossless; **costs context** (drafter inherits `-c`) | 1 h | measured on 3 rigs [M-cited] |
| **9** | **Move to `ik_llama.cpp`** with **IQ4_KS** (~15.1 GB) — the one place an I-quant *and* a speed win coexist | **+18–20% decode at matched power**, quality tied (8-pack 101 vs 100) | none | half day (build) | measured A/B on the same model/rig [M-cited] |
| **10** | **Thermal hygiene**: verify no GDDR6X memory-junction throttling (a 3080 FE measures **104 °C** inside the chip at stock), correct fan curves and spacing | **0 to +10%** if you are currently thermally throttled | none | 30 min | measured (3090/3080 thermal) |
| **11** | **GDDR6X memory OC** +500…+1000 MHz | **+3–5%** at +500, **+6–10%** at +1000. ⚠ **NULL result: no published before/after llama.cpp decode measurement for ANY GDDR6X offset exists** (9 targeted queries). Also: GDDR6X **Error Detection & Replay creates a bandwidth PLATEAU** — the clock keeps rising, the bandwidth does not, and there are no artifacts to warn you | none | 20 min | **low — arithmetic + a non-Ampere (GDDR7) datapoint only** |
| **12** | **Try EAGLE-3** (`--spec-type draft-eagle3`) — heads exist for Qwen3-14B/32B, **none for Qwen3.6-27B** | vLLM/H20 on Qwen3-32B: **1.67–1.86×** (accept length 1.90–2.01). The single llama.cpp datapoint is **negative (0.46–0.71× at batch 1)** | lossless | 2 h | **low / high-variance — highest-upside untried option, but no evidence it works in llama.cpp at batch 1** |
| **12b** | **One cheap experiment: external Qwen3-0.6B Q8 draft** (`--spec-type draft-simple -md qwen3-0.6b-q8.gguf`, 0.75 GiB) | Qwen3-**32B** dense on GB10: **1.37× mean**, but code 2.55× / prose ~1.3× | lossless | 30 min | **workload-fragile.** Not for a 4B draft, and never on a separate GPU |
| **13** | ~~External Qwen3-0.6B/1.7B/4B draft model~~ | ⚠ **Corrected:** −75% was measured on an **MoE (35B-A3B, ~3B active)** target and does **not** generalise to your dense 27B — see #12b. **But a 4B draft, or any draft on its own GPU without P2P, is still not worth it.** Measured cause of the loss: an **82 MiB checkpoint per speculative round** | — | — | scope caveat [M-cited] |
| **14** | **Do NOT touch**: `CUDA_DEVICE_MAX_CONNECTIONS` (null + risk of disabling backend work-queue concurrency), `GGML_CUDA_FORCE_MMQ`/`FORCE_CUBLAS`/`MMV_Y` (null), `GGML_CUDA_DMMV_X` (obsolete), `-lgc`/`-lmc` on Ampere (not portable), Medusa/lookahead (not in llama.cpp) | 0% | — | — | explicit nulls |

**Combined realistic target [E] — revised downward for honesty.** The two independent multipliers are action 3 (AL 2.76 → ~4, ×1.25–1.45) and action 4 (Q8_0 → Q4_K_M, ×1.4 at the calibrated efficiency, up to ×1.74 proportional). They compound *weakly*: a smaller model also shrinks the bytes the deeper verify batch must move, and both share the same per-pass overhead.

- **Conservative (Q6_K + n_max 4):** 78.3 × 1.12 × 1.30 ≈ **114 t/s**
- **Central (Q4_K_M + n_max 4):** 78.3 × 1.45 × 1.30 ≈ **148 t/s**
- **Upside (Q4_K_M + n_max 6–7 with gating):** ≈ **185 t/s**
- **Plus a fixed multi-GPU/NCCL/KV-kernel stack (+0–25%):** **up to ~200–230 t/s**

**Plan for 110–150 t/s, treat 185+ as upside, and re-derive after measuring the no-spec baseline — if that comes back ≥60 t/s, discard the whole quantization column and go fix speculation first.**

The absolute ceiling if η were 1.0 and AL were perfect at n_max=7 is `3040/29 × 8 = 838 t/s`, so the headroom is real — but per §C.6 the most attractive-looking route to it (recovering η) is the one with the weakest evidence.

---

## WHAT IS MEASURED vs EXTRAPOLATED — quick key

**Measured on your rig:** Q8_0 = 29 GB; decode 78.3 t/s; prefill 860.7 t/s; acceptance 0.587; AL 2.76.

**Measured elsewhere, cited with date/URL:** all per-quant file sizes (club-3090 Qwen3.6-27B, 2026); the llama.cpp per-quant `tg128`/bpw table (in-tree master); the KLD ladder Q8_K_XL 0.0026 / Q6_K 0.0065 / Q4_K_M 0.0192; the Unsloth I-quant −5.8% penalty; bric.pe.kr 3090 8B quant sweep (2026-05-27); arXiv:2609.14864 27B quant sweep (2026-09-14); habr 2×3090 `layer` vs `tensor` + MTP (2026-08-29); all Part B acceptance/speedup figures (thc1006 2026-08-25/26, tfriedel 2026-07-12/13, b70-optimization-lab 2026-06-23, club-3090 2026-04→08); the TP=2-vs-TP=4 pairs; ik_llama +18–20%; ngram-mod +25/+35%; MTP-head-precision AL recovery; llama.cpp MTP ctx-alignment bug #23658; power-cap measurements; the KV-q8_0 CUDA cost (−2% short ctx, +3.5–5.3% at 32K); `GGML_CUDA_FA_ALL_QUANTS` +7.4%.

**Extrapolated by me [E]:** the η calibration (both readings) and all three projection columns in §A.2; the `T_pass` = 35.25 ms decomposition; the implied conditional acceptance q ≈ 0.86; the ceiling table in §B.7; the memory-OC percentages; the 2-card→4-card tensor extrapolation; the combined 110–150 t/s target.

**Explicit null results (searched, found nothing):**
- **Memory OC:** no published before/after llama.cpp or ik_llama.cpp decode measurement for **any** GDDR6X offset (9 targeted queries). Consumer GDDR6X has no ECC, so ECC on/off data does not apply.
- **NUMA:** no measured `numactl --cpunodebind/--membind/--interleave` tok/s delta on any host-staged multi-GPU llama.cpp rig. The most promising unretrieved lead is **llama.cpp Discussion #11733** ("Dual Epyc Genoa/Turin token generation performance bottlenecks"), which a search snippet shows contains `NPS1 | likwid-bench load | 460.8 | 359.9 | 78.1%`-style measured rows — **worth pulling manually.**
- **`CUDA_DEVICE_MAX_CONNECTIONS`, `GGML_CUDA_FORCE_MMQ/FORCE_CUBLAS/MMV_Y/ENABLE_UNIFIED_MEMORY`, `--no-kv-offload`:** no measured decode deltas anywhere.
- **`-lgc`/`-lmc` on Ampere:** no Ampere-specific LLM measurement exists.
- **Draft placement:** no published 4-GPU "draft on its own GPU vs shared" head-to-head, and **no measured separate-GPU draft case with P2P disabled**.
- **EAGLE-3 / DSpark / Medusa / lookahead / REST / SV / Hydra / ReDrafter / SAM on consumer GPUs:** nothing usable. Medusa and lookahead are not in llama.cpp at all.
- **No clean ≥3-card non-P2P `tensor` vs `layer` llama.cpp comparison exists publicly.**
- **No 4×RTX 3080-20GB decode data exists.** Nearest 2-card Ampere-3080-20GB points (all vLLM, club-3090): @troymroberts 2026-05-02, 2× 3080 20GB @200 W/card, TQ k8v4 KV + MTP K=3 → **49 TPS wall single-stream, 210 TPS aggregate at n=8**; @efschu 2026-05-04, 2× 3080 20GB, fp8_e5m2 KV, mem-util 0.82 → **82.4 narr / 107.9 code TPS**, 257K needle PASS at 90% depth.
- The "acceptance > 0.7 / > 0.8 is good" convention has **no citation** — it appears to be folklore. The only empirically derived threshold is the **48.2% break-even**.
- **`--spec-replace` does not exist**; **`lookup` is not a `--spec-type`**; `--draft-max`/`--draft-min` are hard-removed and throw.

**Sourcing limitation:** `github.com` HTML pages were unfetchable for part of this session and the GitHub REST API rate-limited, so some issue/PR **bodies** (llama.cpp #24485, #24489, #23567, #21788, #26409, #27198) are titles + dates + URLs only. Findings from those items are flagged as such.
