# GGUF quantization for a 27B dense model on Ampere (4× RTX 3080 20 GB, 80 GB, 3040 GB/s)

Research date: 2026 (assumed). Every table row is labelled **MEASURED** (published number from a real run) or **ESTIMATED/DERIVED** (arithmetic I did on measured inputs). Anything older than ~6 months is flagged.

Reference point for the user's rig: **Qwen3.8-27B dense**, 27.320697856 B params (65 layers, n_embd 5120, vocab 248320) — this is the exact model the two most useful 2026 sources measured.

---

## 0. Executive summary (the answers)

| # | Question | Answer |
|---|---|---|
| 1 | bpw + size | Authoritative bpw table from llama.cpp §1.1. Real 27B files: Q8_0 29.05–29.12 GB, Q6_K 23.86 GB, Q5_K_M 20.92 GB, Q4_K_M 17.44 GB, Q4_K_S 16.36 GB, IQ4_XS 15.48 GB, IQ4_NL 17.44 GB, Q3_K_M 13.40 GB, IQ3_XXS 12.32 GB, Q3_K_S 12.74 GB, Q2_K 10.82 GB (§1.3). Observed bpw runs **0.3 %–22 % above** the nominal figure; see §1.4. |
| 2 | Quality deltas | Q4_K_M ≈ +0.07 PPL / **7× the mean-KLD** of Q6_K (0.0192 vs 0.0065) on the Unsloth Qwen3.5 ladder. IQ4_XS mean KLD 0.0234 ≈ Q4_K_M, but measured *worse PPL* (6.6234 vs 6.6097) at a smaller size. Below Q4 the KLD explodes: Q3_K_M 0.0585, Q3_K_XL 0.0308, Q2_K_XL 0.097–0.156, IQ2_XXS 0.185–0.346. Detail in §2. |
| 3 | Better quality/byte? | **Yes: IQ4_XS/IQ4_NL beat Q4_K_S on quality-per-bit**, IQ3_XXS beats Q3_K_M on the Pareto frontier — but they *lose decode speed on CUDA* (§3.2). **ik_llama.cpp `IQ4_KS` is the only format that is simultaneously smaller AND faster than mainline Q4_K_M** (§3.4). MXFP4/NVFP4 are worse-or-equal at 4 bpw and NVFP4 is Blackwell-only. Trellis (IQ*_KT), IQ4_K/IQ5_K/IQ2_K/IQ3_K are ik_llama.cpp-only; all now have CUDA kernels but **not** as fast as mainline. |
| 4 | Does smaller = faster? | **No — and it is not even monotonic.** Three independent measured counterexamples: (a) RTX 3090, Q4_K_M **96 t/s** vs IQ4_XS **88 t/s** despite IQ4_XS being 8 % smaller; (b) L4, Q4_0 block **29.91** > Q2_K **29.20** > Q3_K **28.89 t/s**; (c) RTX 5080 Qwen3.8-27B, Q4_K_XL→Q3_K_XL is **−45 % bytes for +1.79× speed** (size ratio predicts 1.55×, so only ~15 % of the expected gain survived), and Q4→Q3 is far worse than Q4→Q8 in ratio terms. Full numbers §4. |
| 4b | Is Q8_0 unusually fast/slow? | **Q8_0 is unusually *fast* per byte at batch-1 decode and unusually *cheap* in quality per byte.** It has the simplest dequant path, and on Ada/Turing MMVQ it gets its own `nwarps` tuning. But it is *not* immune: on Intel SYCL it is ~4× slower than Q4_K_M (issue #21517), and it costs 8.5 bpw vs 4.9. |

**Actionable read for this rig:** with 80 GB and 29 GB currently spent on Q8_0, moving to **Q6_K (23.86 GB)** or **UD-Q6_K_XL (25.30 GB)** costs ~0.004 mean-KLD and buys 3.7–5.2 GB of VRAM. Moving to Q4_K_M buys 11.6 GB and +37 % decode speed at the cost of ~7× the KLD error — measurable but small in absolute terms (mean KLD 0.019 for a model whose baseline KLD headroom is ~0.003). The worst trade on this hardware is IQ3/Q2: bytes fall 55–65 %, decode rises ≤1.8×, and KLD rises 10–45×.

---

## 1. Bits-per-weight and real file sizes

### 1.1 Authoritative bpw table (llama.cpp upstream) — MEASURED

Source: [llama.cpp `tools/quantize/README.md`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md), fetched 2026. Measured on **meta-llama/Llama-3.1-8B** (8.03 B params).

| Quant | bits/weight | size (GiB) | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| IQ1_S | 2.0042 | 1.87 | 858.88 ±1.22 | 79.73 ±0.79 |
| IQ1_M | 2.1460 | 2.01 | 847.99 ±0.47 | 72.92 ±0.14 |
| IQ2_XXS | 2.3824 | 2.23 | 852.39 ±0.85 | 79.86 ±0.22 |
| IQ2_XS | 2.5882 | 2.42 | 826.99 ±12.51 | 78.04 ±0.46 |
| IQ2_S | 2.7403 | 2.56 | 783.55 ±13.73 | 77.30 ±2.47 |
| IQ2_M | 2.9294 | 2.74 | 787.68 ±7.00 | 74.44 ±0.15 |
| Q2_K_S | 2.9697 | 2.78 | 798.91 ±6.40 | **90.01 ±0.12** |
| Q2_K | 3.1593 | 2.95 | 784.45 ±7.85 | 79.85 ±0.20 |
| IQ3_XXS | 3.2548 | 3.04 | 813.88 ±6.53 | 73.95 ±0.20 |
| IQ3_XS | 3.4977 | 3.27 | 708.71 ±1.26 | 71.67 ±0.54 |
| IQ3_S | 3.6606 | 3.42 | 798.78 ±8.81 | 69.31 ±0.63 |
| Q3_K_S | 3.6429 | 3.41 | 752.17 ±7.94 | 69.84 ±0.18 |
| IQ3_M | 3.7628 | 3.52 | 768.70 ±13.73 | 70.15 ±0.33 |
| Q3_K_M | 3.9960 | 3.74 | 783.44 ±9.92 | 71.68 ±0.22 |
| Q3_K_L | 4.2979 | 4.02 | 761.17 ±7.55 | 69.38 ±0.49 |
| IQ4_XS | 4.4597 | 4.17 | 771.80 ±11.38 | 77.51 ±0.20 |
| Q4_K_S | 4.6672 | 4.36 | 818.55 ±9.58 | 76.71 ±0.20 |
| IQ4_NL | 4.6818 | 4.38 | 806.03 ±7.07 | 76.63 ±0.28 |
| **Q4_K_M** | **4.8944** | **4.58** | 821.81 ±21.44 | 71.93 ±1.52 |
| Q5_K_S | 5.5704 | 5.21 | 752.52 ±0.99 | 69.53 ±0.18 |
| Q5_K_M | 5.7036 | 5.33 | 758.69 ±7.43 | 67.23 ±1.08 |
| Q6_K | 6.5633 | 6.14 | 812.01 ±10.82 | 58.67 ±3.13 |
| Q8_0 | 8.5008 | 7.95 | 865.09 ±8.30 | **50.93 ±0.08** |
| F16 | 16.0005 | 14.96 | 923.49 ±0.53 | 29.17 ±0.04 |

Note the shape of the upstream table: on **this** 8B/M-series run tg128 *falls monotonically* with bpw (90.0 t/s at 2.97 bpw → 50.9 t/s at 8.5 bpw). §4 shows that this monotonicity **does not hold** on Ampere CUDA once IQ formats and mixed-precision K-quants are in play (IQ4_XS < Q4_K_M; Q3_K < Q2_K).

Cross-check I found but could **not** verify (llama.cpp docs mirror, unversioned, 7B numbers): Q8_0 5.9584 PPL, Q6_K 5.9642, Q5_K_M 5.9796, **Q4_K_M 6.0565 (+1.68 % vs FP16)**, Q4_K_S 6.1125 (+2.62 %), Q3_K_M 6.3184 (+6.07 %), Q2_K 6.8673 (+15.3 %). Source: [mintlify llama.cpp mirror — Model Quantization](https://mintlify.wiki/ggml-org/llama.cpp/concepts/quantization), fetched 2026; model not stated. Treat as **ESTIMATED/uncertain**.

### 1.2 The bpw → size formula and the overhead caveat

```
bytes ≈ bpw/8 × params + overhead      -- the naive form
```

llama.cpp's own numbers already include the overhead, which is why `Q8_0` is 8.5008 bpw and not 8.0, and `Q6_K` is 6.5633 and not 6.0. Two overhead sources: (a) per-block scale/zero-point metadata, (b) tensors that the mixed recipes deliberately keep high (token embeddings, `output.weight`, `ssm_alpha`/`ssm_beta` in hybrid models). The **token embedding alone is 248320 × 5120 = 1.27 B params** on Qwen3.8-27B; at Q8_0 that is 1.35 GB of the file regardless of the nominal quant.

### 1.3 Real published GGUF sizes for a 27B-class dense model — MEASURED

Primary model: **Qwen3.8-27B** (dense, 27.320697856 B params; `UD-IQ2_XXS` has 26.895998464 B / 64 layers). Exact byte sizes read from the HuggingFace tree API via `https://hf-mirror.com/api/models/<repo>/tree/main?expand=true`, retrieved 2026 (direct `huggingface.co` is unreachable from this environment).

| Quant | bedrock / ggml-org bytes | bartowski bytes | unsloth (UD) bytes | bed. GB | bart. GB | unsl. GB |
|---|---|---|---|---|---|---|
| `UD-IQ1_S` | — | — | 6 192 222 208 | — | — | 6.19 |
| `UD-IQ1_M` | — | — | 6 729 166 848 | — | — | 6.73 |
| `UD-IQ2_XXS` | 7 266 070 528 | 8 881 268 448 | 7 266 070 528 | 7.27 | 8.88 | 7.27 |
| `UD-IQ2_S` | — | 9 684 043 488 | 8 371 970 048 | — | 9.68 | 8.37 |
| `UD-Q2_K_XL` | 9 828 981 664 | 10 821 543 648 (`Q2_K`) | 9 828 981 664 | 9.83 | 10.82 | 9.83 |
| `UD-IQ3_XXS` | 10 934 860 704 | 12 320 167 648 | 10 934 860 704 | 10.93 | 12.32 | 10.93 |
| `UD-Q3_K_S` | — | 12 739 884 768 | — | — | 12.74 | — |
| `UD-Q3_K_XL` | 13 146 393 504 | 16 391 919 200 | 13 146 393 504 | 13.15 | 16.39 | 13.15 |
| `Q3_K_M` / `UD-Q3_K_M` | 13 404 051 168 | 13 404 051 168 | — | 13.40 | 13.40 | — |
| `UD-IQ3_S` | — | 12 799 604 448 | 12 040 883 104 | — | 12.80 | 12.04 |
| `UD-IQ4_XS` | — | 15 475 951 328 | 14 252 845 984 | — | 15.48 | 14.25 |
| `Q4_K_S` / `UD-Q4_K_S` | 16 363 513 568 | — | 15 358 213 024 | 16.36 | — | 15.36 |
| `Q4_0` | 16 348 767 968 | — | 16 056 478 688 | 16.35 | — | 16.06 |
| **`Q4_K_M`** | **17 442 399 968** | **17 442 399 968** | 16 464 440 224 (`UD-Q4_K_M`) | **17.44** | **17.44** | 16.46 |
| `UD-Q4_K_XL` | 17 559 178 144 | — | 17 559 178 144 | 17.56 | — | 17.56 |
| `UD-IQ4_NL` | — | 17 442 399 968 | — | — | 17.44 | — |
| `Q5_K_S` / `UD-Q5_K_S` | 19 566 913 248 | — | 18 665 753 504 | 19.57 | — | 18.67 |
| **`Q5_K_M`** | **20 923 877 088** | — | 19 771 509 664 (`UD-Q5_K_M`) | **20.92** | — | 19.77 |
| `UD-Q5_K_XL` | 20 876 938 144 | — | 20 876 938 144 | 20.88 | — | 20.88 |
| `UD-Q6_K` | — | — | 21 983 677 344 | — | — | 21.98 |
| `Q6_K` | 23 860 565 728 | — | — | 23.86 | — | — |
| `UD-Q6_K_XL` | 25 299 061 664 | — | 25 299 061 664 | 25.30 | — | 25.30 |
| `UD-Q8_K_XL` | — | — | 31 457 991 680 | — | — | 31.46 |
| **`Q8_0`** | **29 047 086 048** | 29 116 388 960 | 29 047 086 048 | **29.05** | 29.12 | 29.05 |

Sorting the *same* quant name across uploaders: `Q4_K_M` spans **16.46 → 17.44 GB** (6 % spread) and `Q4_K_XL`/`Q3_K_XL` add another 5–25 % on top. `UD-Q8_K_XL` is 31.46 GB — **8 % larger than plain Q8_0** because Unsloth raises the output/embedding tensors.

Cross-source sanity checks:
- The authoritative blog that published the same `Q4_K_M` file (`hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4`, README, measured 2026-08-17/18) quotes **17 559 178 144 bytes = 16.35 GiB** for `UD-Q4_K_XL`, matching the API byte count exactly. Note the author labels 16.35 **GiB** as "GB" in one table — the API bytes are ground truth.
- The same repo's size table for [`unsloth/Qwen3.8-27B-GGUF`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) matches the API within rounding: `UD-IQ2_XXS` 9.01 GB, `UD-Q2_K_XL` 10.68 GB, `UD-Q3_K_XL` 13.44 GB, `Q4_K_M` 17.11 GB, `UD-Q4_K_XL` 17.92 GB, `UD-Q5_K_XL` 20.22 GB, `Q6_K` 22.88 GB, `UD-Q6_K_XL` 25.92 GB, `Q8_0` 29.05 GB.
- Unsloth's own Dynamic-3.0 update table (Mar 5 2026, Qwen3.5) reports the same convention in GB: `UD-Q2_K_XL` 11.3, `UD-Q3_K_XL` 15.5, `UD-Q4_K_XL` 20.7, `UD-Q5_K_XL` 24.6 — that is the **35B-A3B** model, not the 27B, so do not mix columns.

**Not found.** I could not retrieve Gemma-3-27B-it, Qwen3-32B or Mistral-Small-24B file-size tables: `huggingface.co` and `cdn-lfs.huggingface.co` both time out from this environment, and `hf-mirror.com` served only the first few repositories before rate-limiting (`bartowski/google_gemma-3-27b-it-GGUF`, `bartowski/Qwen_Qwen3-32B-GGUF`, `unsloth/Qwen3.6-27B-GGUF` all returned empty on 4 attempts each). For a 27B-class dense model the Qwen3.8-27B table above is representative; a Gemma-3-27B (27.4 B, wider vocab 262144) file will run ~3–6 % larger at the same quant because of the embedding.

### 1.4 "bpw" vs observed size — the overhead, quantified

**DERIVED** from the bedrock/ggml-org exact bytes ÷ 27.320697856 B params:

| Quant | nominal-ish bpw | observed bpw | overhead vs naive |
|---|---|---|---|
| Q8_0 | 8.50 (upstream) | **8.506** | +0.06 % vs upstream value |
| `UD-Q8_K_XL` | — | 9.211 | +8.3 % over Q8_0 |
| Q6_K | 6.563 (upstream) | **6.987** | +6.5 % |
| `UD-Q6_K_XL` | — | 7.408 | +12.9 % over Q6_K |
| Q5_K_M | 5.704 (upstream) | **6.127** | +7.4 % |
| Q4_K_M | 4.894 (upstream) | **5.107** | +4.3 % |
| `UD-Q4_K_XL` | — | 5.142 | +0.7 % over Q4_K_M (but +0.3 GB) |
| IQ4_XS (bartowski) | 4.460 (upstream) | 4.532 | +1.6 % |
| `UD-IQ4_XS` (unsloth) | 4.460 | 4.173 | **−6.4 %** (fewer high-precision tensors than upstream's reference recipe) |
| IQ4_NL (bartowski) | 4.682 (upstream) | 5.107 | **+9.1 %** |
| Q3_K_M | 3.996 (upstream) | 3.925 | −1.8 % |
| Q3_K_S | 3.643 (upstream) | 3.730 | +2.4 % |
| IQ3_XXS (bartowski) | 3.255 (upstream) | 3.608 | **+10.8 %** |
| `UD-IQ3_XXS` (unsloth) | 3.255 | 3.202 | −1.6 % |
| Q2_K (bartowski) | 3.159 (upstream) | 3.169 | +0.3 % |
| `UD-Q2_K_XL` | — | 2.878 | −8.9 % vs Q2_K |
| `UD-IQ2_XXS` | 2.382 (upstream) | 2.161 | −9.3 % |

**Take-aways.** (1) The upstream bpw table is an 8B measurement — on a 27B hybrid model the *same quant name* can land 6–11 % off it, in either direction, because the recipe mixes per-tensor types differently. (2) **Never size a download from a bpw constant.** Use `bpw/8 × params` only as a ±10 % estimate, and prefer the repo's actual byte count. (3) `Q8_0` is the one exception: it is within 0.1 % of 8.5 bpw on every 27B file measured here because nothing can be promoted above it.

---

## 2. Published quality metrics for 27B-class dense models

### 2.1 The Unsloth Qwen3.5 ladder — PPL and full KLD distribution — MEASURED

Source: [Unsloth — Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks), page updated Mar 5 2026 (the "Full Benchmarks" table; the page also carries Qwen3.5-35B-A3B / 122B / 397B sections — the table below is the combined multi-quantizer table, which I believe is the **35B-A3B** model because the page's per-model figures are 35B-A3B, but Unsloth does not label the table itself, so treat the *absolute* PPL/KLD as 27B/35B-class and the *ratios* as the transferable result). Baseline: `Q8_K_XL` PPL **6.5352**, KLD99.9 **0.1033**, mean KLD **0.0026**.

| Quantizer | Quant | GB | PPL | ΔPPL vs Q8_K_XL | KLD 99.9 % | mean KLD | mean-KLD ×Q8 |
|---|---|---|---|---|---|---|---|
| Unsloth | IQ2_XXS | 9.09 | 7.7160 | +1.181 | 4.2221 | 0.1846 | 71× |
| Unsloth | Q2_K_XL | 12.04 | 7.0438 | +0.509 | 2.9092 | 0.0970 | 37× |
| Unsloth | IQ3_XXS | 13.12 | 6.7829 | +0.248 | 1.5296 | 0.0501 | 19× |
| Unsloth | IQ3_S | 14.13 | 6.7715 | +0.236 | 1.4193 | 0.0457 | 18× |
| Unsloth | Q3_K_M | 15.54 | 6.7320 | +0.197 | 0.9726 | 0.0324 | 12× |
| Unsloth | Q3_K_XL | 16.06 | 6.7245 | +0.189 | 0.9539 | 0.0308 | 12× |
| Unsloth | MXFP4_MOE | 18.17 | 6.6000 | +0.065 | 0.7789 | 0.0272 | 10× |
| Unsloth | **Q4_K_M** | 18.49 | 6.6053 | **+0.070** | 0.5478 | **0.0192** | 7.4× |
| Unsloth | Q4_K_L | 18.82 | 6.5905 | +0.055 | 0.4828 | 0.0150 | 5.8× |
| Unsloth | **Q4_K_XL** | 19.17 | 6.5918 | +0.057 | 0.4097 | **0.0137** | 5.3× |
| Unsloth | **Q5_K_XL** | 23.22 | 6.5489 | **+0.014** | 0.2360 | **0.0069** | 2.7× |
| Unsloth | Q6_K_S | 26.56 | 6.5456 | +0.010 | 0.2226 | 0.0065 | 2.5× |
| Unsloth | Q6_K_XL | 28.22 | 6.5392 | +0.004 | 0.1437 | 0.0041 | 1.6× |
| Unsloth | Q8_K_XL | 36.04 | 6.5352 | 0 | 0.1033 | 0.0026 | 1.0× |
| bartowski | IQ2_XXS | 8.15 | 9.3427 | +2.808 | 6.0607 | 0.3457 | 133× |
| bartowski | Q2_K_L | 11.98 | 7.5504 | +1.015 | 3.8095 | 0.1559 | 60× |
| bartowski | IQ3_XXS | 12.94 | 7.0938 | +0.559 | 2.1563 | 0.0851 | 33× |
| bartowski | Q3_K_M | 14.95 | 6.7720 | +0.237 | 1.7779 | 0.0585 | 23× |
| bartowski | Q3_K_XL | 15.97 | 6.8245 | +0.289 | 1.7516 | 0.0627 | 24× |
| bartowski | **IQ4_XS** | 17.42 | 6.6234 | +0.088 | 0.7265 | **0.0234** | 9.0× |
| bartowski | **Q4_K_M** | 19.77 | 6.6097 | **+0.075** | 0.5771 | **0.0182** | 7.0× |
| bartowski | Q5_K_M | 23.11 | 6.5828 | +0.048 | 0.3549 | 0.0106 | 4.1× |
| AesSedai | IQ3_S | 12.65 | 6.9152 | +0.380 | 1.8669 | 0.0613 | 24× |
| AesSedai | IQ4_XS | 16.40 | 6.6447 | +0.110 | 0.8067 | 0.0235 | 9.0× |
| AesSedai | Q4_K_M | 20.62 | 6.5665 | +0.031 | 0.3171 | 0.0096 | 3.7× |
| AesSedai | Q5_K_M | 24.45 | 6.5356 | +0.000 | 0.2100 | 0.0058 | 2.2× |
| Ubergarm | Q4_0 | 19.79 | 6.5784 | +0.043 | 0.4829 | 0.0142 | 5.5× |

Concrete numeric lines from that page:
> "Imatrix definitely helps reduce KLD & PPL, **at the cost of 5-10% slower inference**."
> "**I quants (iq3_xxs, iq2_s etc) makes inference 5-10% slower**, they're definitely better in terms of efficiency, but there is a tradeoff."

### 2.2 Direct 27B KLD / PPL from upstream llama.cpp PR — MEASURED

Source: [llama.cpp PR #23572, "CUDA: permit quantizing into NVFP4"](https://api.github.com/repos/ggml-org/llama.cpp/issues/23572), opened **2026-05-23**, updated 2026-09-08. Model: **`qwen3.6-27b-bf16` → 27B dense**, wikitext-2, 200 chunks, `-c 512 -b 4096 -ub 2048 -fa on`, RTX 5090.

| Quant | mean KLD | top-P agreement | Size | Speed (pp2048 / tg128) |
|---|---|---|---|---|
| **IQ4_XS** (baseline) | **0.0343** | 93.05 % | — | — |
| Q4_0 (baseline) | 0.0486 | 91.71 % | 14.63 GiB | 4097.30 / **84.63 ±0.40** |
| NVFP4 | 0.0894 | 88.18 % | 14.63 GiB | 5938.26 / 84.32 |
| Q3_K (baseline) | 0.0824 | 88.86 % | 12.56 GiB | 3455.00 / **87.39 ±0.71** |
| calibrated `unsloth/Qwen3.6-27B-NVFP4` | 0.0712 | 89.92 % | — | — |
| NVFP4 + quantize-time MSE | 0.0794 | 88.99 % | — | — |

Direct quotes:
> "| Mean KLD | Same Top-P | Optimization / Baseline |"
> "**0.0343 | 93.05% | IQ4_XS baseline**"
> "**0.0824 | 88.86% | Q3_K baseline**"
> "| qwen35 27B Q3_K - Medium | 12.56 GiB | tg128 | **87.39 ± 0.71** |"
> "| qwen35 27B Q4_0 | 14.63 GiB | tg128 | **84.63 ± 0.40** |"
> "This early iteration is slightly worse (**0.0895 KLD**) than `Q3_K` (**0.0826 KLD**)"

PPL lines from the same PR (Qwen3.6-27B, bf16 base PPL **6.793059 ±0.075277**):
- Q4_0: PPL(Q) **7.017811**, ratio **1.033085**, `Mean PPL(Q)-PPL(base)` = **0.224752**
- Q3_K: PPL(Q) **6.899628**, ratio **1.015688**, Δ = **0.106568**
- NVFP4 (calibrated import): PPL(Q) **7.219390**, ratio **1.062760**, Δ = **0.426331**

**Key derived comparison (DERIVED):** mean KLD IQ4_XS 0.0343 vs Q4_0 0.0486 = **IQ4_XS is 29 % closer to the bf16 distribution than Q4_0 while being 11 % smaller** (nominal). That is the strongest single piece of evidence that IQ4-class formats beat legacy 4-bit per byte on this model class.

### 2.3 The Ampere/4090 Q4/Q8/Q3 ladder and MMLU-Pro deltas — MEASURED

Source: [bric.pe.kr — "GGUF Quantization Showdown — Q4_K_M vs Q4_K_S vs IQ4_XS vs Q5_K_M (2026 Real Quality + Speed)"](https://bric.pe.kr/blog/gguf-quantization-showdown-q4-q5-iq4-2026), published **2026-05-27**. llama.cpp b3500+ via Ollama 0.6.x.

Llama 3.1 8B (wikitext / Korean PPL, MMLU-Pro = 140-question sample):

| Quant | Size | wikitext PPL | Δ | KR PPL | Δ | MMLU-Pro | Δ |
|---|---|---|---|---|---|---|---|
| Q8_0 | 8.5 GB | 6.42 | — | 8.71 | — | **47.1 %** | — |
| Q5_K_M | 5.7 | 6.45 | +0.5 % | 8.78 | +0.8 % | 46.4 % | **−0.7 pt** |
| Q5_K_S | 5.5 | 6.48 | +0.9 % | 8.82 | +1.3 % | 46.0 % | −1.1 pt |
| **Q4_K_M** | **4.9** | **6.51** | **+1.4 %** | **8.91** | **+2.3 %** | **45.7 %** | **−1.4 pt** |
| **IQ4_XS** | **4.5** | **6.54** | **+1.9 %** | **8.95** | **+2.8 %** | **45.5 %** | **−1.6 pt** |
| Q4_K_S | 4.7 | 6.58 | +2.5 % | 9.02 | +3.6 % | 45.1 % | −2.0 pt |
| Q3_K_M | 4.0 | 6.83 | **+6.4 %** | 9.45 | **+8.5 %** | 42.8 % | **−4.3 pt** |

Qwen 3 14B (denser analogue of a 27B dense model's behaviour):

| Quant | Size | wikitext PPL | Δ | KR PPL | Δ | MMLU-Pro | Δ |
|---|---|---|---|---|---|---|---|
| Q8_0 | 14.5 | 5.18 | — | 6.92 | — | **58.4 %** | — |
| Q5_K_M | 9.7 | 5.22 | +0.8 % | 6.99 | +1.0 % | 57.6 % | −0.8 pt |
| **Q4_K_M** | **8.4** | **5.29** | **+2.1 %** | **7.11** | **+2.7 %** | **57.0 %** | **−1.4 pt** |
| **IQ4_XS** | **7.6** | **5.32** | **+2.7 %** | **7.18** | **+3.8 %** | **56.6 %** | **−1.8 pt** |
| Q4_K_S | 8.0 | 5.37 | +3.7 % | 7.25 | +4.8 % | 56.1 % | −2.3 pt |
| Q3_K_M | 7.0 | 5.61 | **+8.3 %** | 7.62 | **+10.1 %** | 53.2 % | **−5.2 pt** |

So on this measurement Q4_K_M = **−1.4 MMLU-Pro points** and **+1.4/+2.1 % PPL**, not "indistinguishable", and Q3_K_M = **−5.2 points** — that is the cliff. Same source:
> "**Q4_K_M** — perplexity within 1.4-2.1% of Q8_0 ... fastest on both Ampere and Pascal"
> "**Q5_K_M** — 0.5-0.8% perplexity edge over Q4_K_M, needs ~15% more VRAM"
> "**Q3_K_M's 6.4% perplexity jump is noticeable in generation.**"

**Caveat — single-sample benchmark.** MMLU-Pro on a 140-question sample has roughly ±4 points of 95 % CI, so the 1.4-point Q4_K_M delta is not individually significant and the 5.2-point Q3 delta is only ~1.3σ. The PPL deltas are the trustworthy part. Treat this table as directionally correct and the MMLU column as indicative. It is also 8B/14B, not 27B — larger models are generally *more* quantization-robust, so the true 27B deltas will be smaller (Unsloth's 27B/35B-class ladder in §2.1 shows Q4_K_M at +0.07 PPL, i.e. ~1 %, consistent with that direction).

### 2.4 The 27B/2026 "Q4_K_M is indistinguishable" claim, tested

The closest 2026 source to a direct claim, [Unsloth docs](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks):
> "**Perplexity and KLD can be misleading** ... **Unsloth Dynamic IQ2_XXS performs better than AesSedai's IQ3_S on real world evals (LiveCodeBench v6, MMLU Pro) despite being 11GB smaller.** Yet, AesSedai's perplexity and KLD benchmarks suggest the opposite. (PPL: 0.3552 vs 0.2441; KLD: 9.0338 vs 8.2849 - lower is better)."

That is a **counterexample to the "Q4_K_M/KLD decides it" framing**, not a confirmation of "Q4_K_M indistinguishable". The honest 2026 position from all sources read: PPL/KLD are a *rough signal*; Q4_K_M is within ~1 % PPL and ~7× the mean KLD of Q8 for this model class; below Q4 the KLD degrades 10–100× and real-task scores do drop.

### 2.5 Where Q4_K_M measurably *does* degrade — MEASURED

- **Agentic / tool-call behaviour.** [club-3090 BENCHMARKS.md](https://raw.githubusercontent.com/noonghunna/club-3090/v0.10.2/BENCHMARKS.md) (rows dated 2026-05 to 2026-07): the `cli-40` pack separates 4-bit from 8-bit consistently. 2×3090 Tess-4-27B NVFP4 (4-bit): **106/150** think-off; FP8 W8A16 (8-bit) on the identical rig: **111/150** think-off and **117/150** think-on — "the ON ties the GGUF (117), **+4/+5 over nvfp4 A0, all in cli-40** (OFF 22, ON 24 ...)". Quote: "So the nvfp4 undershoot was **4-bit weight-quant damage to agentic behavior, reversible by precision — NOT a serving-path artifact**."
- **ssm/recurrence tensors are the sensitive ones in hybrid Qwen3.5/3.6/3.8.** Unsloth: "Quantizing `ssm_out` (Mamba layers) is not a good idea ... `ssm_out` at q2_k does dramatically worse. **Quantizing any `attn_*` is especially sensitive** for hybrid architectures." And the L4 report's 2026-08-19 re-cut moved 48 `ssm_alpha` + 48 `ssm_beta` tensors from `Q4_K`→`Q8_0`, with 306 of 866 tensors changing type: "+3.4 % throughput and **+18 432 tokens of context**" and a repair of the json workload (32.09 → **40.37 tok/s**).

---

## 3. Is there a better quality-per-byte format for this class?

### 3.1 Mainline llama.cpp CUDA support matrix (the real answer to "what runs fast on Ampere sm_86")

**MEASURED from source** (fetched 2026, `master`):

`ggml/src/ggml-cuda/mmvq.cu` — `get_vec_dot_q_cuda()` has a **dedicated vector-dot kernel for every one of**: `Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ4_NL, IQ4_XS, IQ3_S`. **So there is no "CPU-only IQ format" in mainline llama.cpp in 2026** — the historically slow/absent CUDA IQ kernels have been filled in. What remains different is *speed*, not availability.

`ggml/src/ggml-cuda/mmq.cuh` — `DECL_MMQ_CASE` for the MMA/DP4A batch path covers: `Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4`. Ampere and everything below Blackwell routes through `mmq-config-ampere.cuh` (`ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA` → `ggml_cuda_mmq_get_config_ampere`).

**NVFP4/MXFP4 are the exception and it matters.** From the same source, in `ggml_cuda_mmq_get_config`, Blackwell takes a different table; and the `block_fp4_mmq` comment says plainly: `// this struct is used for fp4 data types (currently only used for Blackwell)`. NVFP4 *loading* works on Ampere but with no FP4 MMA path it is a dequant-to-16-bit fallback — i.e. strictly worse than Q4_0/K-quants on sm_86. Confirmed by the NVFP4 PR itself (an `GGML_NVFP4_FORCE_DEQUANT` escape hatch was added) and by `PR #28823 cuda: fall back to cuBLAS for IQ quants on Blackwell` (title only — I could not read it; GitHub API rate-limited).

Also source-verified: `ggml_cuda_should_use_mmvq()` has per-arch per-type batch cutoffs `// tuned on RTX 4090` (Ada) and `// tuned on RTX 5090` (Blackwell) for `Q2_K/Q3_K/Q4_K/Q5_K/Q6_K`, but **Ampere sm_86 falls through to the generic `ne11 <= MMVQ_MAX_BATCH_SIZE`** path. The L4 report makes the consequence explicit: `// The first guess is the MMVQ launch configuration, since **sm_89 matches no tuned table and falls through to MMVQ_PARAMETERS_GENERIC**`. That is true for sm_86 too.

### 3.2 Quality-per-byte ranking that the data supports

**On CUDA, at equal bytes**, the ordering from §2 is: **IQ4_XS ≈ IQ4_NL > Q4_K_M > Q4_K_S > Q4_0**, and **IQ3_XXS > Q3_K_M > Q3_K_S ≈ Q2_K**. Concretely, measured on the same bf16 base (PR #23572, Qwen3.6-27B): IQ4_XS mean KLD **0.0343** vs Q4_0 **0.0486** — a 29 % KLD reduction at 11 % fewer nominal bytes. And in Unsloth's ladder: bartowski IQ4_XS (17.42 GB) mean KLD **0.0234** vs Unsloth Q4_K_M (18.49 GB) **0.0192** — IQ4_XS is *not* better than a well-made Q4_K_M, but it is 1 GB smaller, so it wins on the frontier.

### 3.3 The catch: those formats lose decode speed on CUDA

Three independent measurements, §4 has the full tables. In one line each:
- IQ4_XS is **−8.3 %** (96→88 t/s, RTX 3090 8B) and **−11.5 %** (52→46, Qwen3-14B) versus Q4_K_M while being only 8 % smaller.
- I-quants as a class are **−5.8 %** tg128 vs K-quants on the same host with pp512 unaffected (Unsloth Qwen3.5 bench: K 90.33–91.36 vs IQ 84.37–85.79).
- Unsloth states it twice in prose: "I quants ... makes inference **5-10% slower**" and "Imatrix ... at the cost of **5-10% slower inference**".

So for this user's bandwidth-bound 4-GPU rig the I-quant trade is: **−8 % bytes for −6…−12 % decode speed** → the two roughly cancel, and you pay in quantization complexity for a small PPL win. That is the honest verdict for *mainline* llama.cpp.

### 3.4 The one format that breaks the trade: ik_llama.cpp IQ4_KS

**MEASURED.** Source: [club-3090 `docs/engines/IK_LLAMA.md`](https://raw.githubusercontent.com/noonghunna/club-3090/v0.10.2/docs/engines/IK_LLAMA.md), rows dated **2026-05-23/24** (corrected from an earlier "tie" in [#184](https://github.com/noonghunna/club-3090/discussions/184)).

| Metric | IQ4_KS + MTP (ik_llama.cpp) | Q4_K_M + MTP (mainline) |
|---|---|---|
| Decode TPS narr / code | **~60 / ~72** | ~50 / ~59 |
| Ratio | **+18–20 %**, confirmed by set-and-readback power-cap A/B at **both 230 W and 370 W** | — |
| Weight size (Qwen3.6-27B) | **~15.1 GB** | **~17 GB** |
| Max ctx, 1× 3090 @ q4_0 KV | 262 K | 262 K (tie) |
| VRAM @ 262 K | ~22.5 GB | +0.5–0.8 GB more |
| Quality 8-pack | 103/150 | 102/150 (tie) |
| IQ5_KS (~18.5 GB) | author's published PPL **~6.93** vs IQ4_KS **~6.97** | — |

Quote: "**`IQ4_KS` is ~15.1 GB vs `Q4_K_M`'s ~17 GB** on Qwen3.6-27B — smaller weights leave room for **262K context** on a single 24 GB card."
Quote: "**The honest finding: ik is ~18–20% faster on decode TPS** (quality + context tied, ~0.5–0.8 GB leaner) — **faster *and* leaner**."

The mechanism is stated by the quanitzer himself, [ik_llama.cpp discussion #8, "New quantization types IQ2_K, IQ3_K, IQ4_K, IQ5_K"](https://api.github.com/repos/ikawrakow/ik_llama.cpp/discussions/8) (created 2024-08-01, last updated **2025-07-22** — *older than 6 months, partially stale, but the design rationale has not changed*):
> "The lookup is handled quite OK by modern GPU's, but leads to a **massive performance penalty on CPU's** (and, from what I gather from `llama.cpp` user comments, **also on older GPU's**). The new `IQK` quants use a non-linear mapping ... these lookup tables are much smaller (4, 8, 16, 32 INT8 values ...), so they fit into 1 or 2 SIMD registers... resulting in a performance that is **(nearly) the same as corresponding linear mapping**."
> "`IQ4_K` and `IQ5_K` here are **miles ahead of any 4- or 5-bpw quantization type in mainline llama.cpp**." ([discussion #288](https://api.github.com/repos/ikawrakow/ik_llama.cpp/discussions/288), 2025-03-25 — *stale, >6 months*)

### 3.5 Newer / exotic formats — status as of 2026

| Format | Where | Ampere CUDA status |
|---|---|---|
| `IQ4_K`, `IQ5_K`, `IQ2_K`, `IQ3_K`, `IQ2_KS`, `IQ4_KS`, `IQ5_KS`, `IQ4_KSS`, `IQ2_KL`, `IQ6_K` | **ik_llama.cpp only** (never upstreamed) | CUDA kernels exist. Per the ik README: "`IQ4_K, IQ5_K, IQ6_K` [PR 417]", "`IQ2_KS, IQ2_K, IQ3_K` [PR 418]", "`IQ4_K_R4/IQ5_K_R4/IQ2_K_R4/IQ3_K_R4` [PR 461]", "`IQ4_KS_R4`/`IQ5_KS_R4` [PR 493]", and **"All quantization types now have quantized matrix multiplication CUDA kernels"** ([ik_llama.cpp README](https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/README.md), fetched 2026). Caveat for multi-GPU: **"k-quants (`Q2_K, Q3_K, Q4_K, Q5_K, Q6_K`) do not have CUDA row-interleaved implementation"** → with `-rtr` those tensors get computed **on the CPU even when offloaded**. |
| Trellis `IQ1_KT`/`IQ2_KT`/`IQ3_KT`/`IQ4_KT` | **ik_llama.cpp only** (PR 113 + 441/471/475/529/616) | Yes, CUDA + Metal + Neon + CPU. Novel integer-base trellis; still niche. |
| `MXFP4` | mainline + ik | In mainline: MMA config on Blackwell only, `MMQ_DP4A_TXS_Q8_1` on Ampere; MMVQ vec-dot exists everywhere. **Unsloth's own finding: "MXFP4 is much worse on many tensors ... MXFP4 uses 4.25 bits per weight, whilst Q4_K uses 4.5 bits per weight. It's better to use Q4_K than MXFP4."** And they **retired MXFP4** from all Q2_K_XL/Q3_K_XL/Q4_K_XL quants. |
| `NVFP4` | mainline (PR #21095 merged; #23572 open) | **Blackwell-only for the MMA fast path.** No FP4 MMA on sm_86. Counted as strictly dominated on Ampere. |
| `TQ1_0`, `TQ2_0`, `Q1_0`, `UD-IQ1_XS/XXS/XXXS` | mainline + unsloth fork | Ternary/1-bit; **irrelevant at 80 GB** — 1.19–1.5625 bpw, and the measured quality is brutal: Unsloth's own Qwen3.8-2.4T numbers are IQ1_S PPL 2.579 / KLD 0.565 vs UD-IQ1_XXXS PPL **4.489** / KLD **1.110**. |
| `QTIP`, `AQLM`, `BitNet` | research / non-GGUF | No llama.cpp path. ik's discussion #8 explicitly credits QuIP#/AQLM for the codebook idea the i-quants use; BitNet-b1.58-2B-4T got model support (PR 337) but is a *trained* 1.58-bit model, not a post-training quant you can apply to a Qwen3.8-27B. |
| Hadamard KV (`-khad`/`-vhad`), `Q8_KV`, `int8_per_token_head`, `TQ3` KV | ik only / vLLM | KV-cache axis, separate from weights. ik's Hadamard transforms are the only mechanism found that "recovers accuracy lost to KV quantization". |

**Query that found nothing useful:** multiple attempts to locate a 2025–2026 published "Trellis/QTIP/BitNet beats IQ4_K at 4 bpw on CUDA" result returned only the ik trellis PRs and the original QuIP#/AQLM papers, i.e. **no independent CUDA decode measurement of trellis quants exists** in the sources I could reach. Treat trellis as unproven for this rig.

---

## 4. Does a smaller quant actually decode faster? — MEASURED evidence

### 4.1 The single best dataset: Qwen3.8-27B dense, same model, same builds

Source: **arXiv:2609.14864, "GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS THREE SYSTEMS"**, submitted **2026-09-14**, Section 4.2 + Tables 2/4/5. All configs: llama.cpp `b10794`, flash attention, **F16 K/V**, 5 repetitions, fresh process, `-ngl 99`, depths 0/4096/16384. Files are `unsloth/Qwen3.8-27B-GGUF`. Decode = `llama-bench` `tg128`; prefill = `pp512`.

**RTX 5080 16 GB, CUDA** (device-copy bandwidth measured at **801.06 GB/s**; FP16 matmul 118.83 TFLOP/s):

| Quant | bytes | GB | DERIVED bpw | tg128 depth 0 | tg128 @4K | tg128 @16K | DERIVED eff. GB/s | pp512 depth 0 |
|---|---|---|---|---|---|---|---|---|
| `UD-IQ2_XXS` (66→64 layers) | 7 266 070 528 | 7.27 | 2.161 | **76.76** | 75.58 | 70.36 | 519 | 1 725 |
| `UD-Q2_K_XL` | 9 828 981 664 | 9.83 | 2.878 | **67.48** | 66.30 | 62.22 | 618 | 2 043 |
| `UD-IQ3_XXS` | 10 934 860 704 | 10.93 | 3.202 | **63.24** | 62.46 | 58.63 | 644 | 2 024 |
| `UD-Q3_K_XL` | 13 146 393 504 | 13.15 | 3.850 | **56.15** | 55.49 | 52.65 | 688 | 2 042 |

**MacBook Pro M4 Max 64 GB (Metal, 12 P-cores)** — note the *opposite ordering* at the small end:

| Quant | tg128 depth 0 | tg128 @4K | tg128 @16K |
|---|---|---|---|
| `UD-IQ2_XXS` | 23.56 | 20.41 | 19.71 |
| `UD-IQ3_XXS` | 23.49 | 22.20 | 19.63 |
| `UD-Q2_K_XL` | **24.08** | 23.14 | 21.30 |
| `UD-Q3_K_XL` | 22.80 | 21.89 | 17.49 |
| `Q4_K_XL` | 20.50 | 19.70 | 16.77 |
| `Q5_K_XL` | 18.88 | 18.33 | 15.82 |
| `Q6_K_XL` | 16.86 | 15.92 | 14.97 |
| `Q8_0` (historical probe) | **7.10** | — | — |

Direct quotes:
> "The near-family Qwen3.8-27B ladder varies across the three host/runtime stacks. IQ2 has 1.55% fewer parameters and one fewer layer. **On both Apple hosts, Q2 is only 2.2% faster than IQ2 at depth zero, whereas on RTX, IQ2 is 13.7% faster.**"
> "**Across the three conventional Q4/Q8 pairs, Q4 is 1.32–1.43× faster on MacBook, 1.29–1.41× on Studio, and 1.37–1.49× on RTX**, while its prefill rate is usually slightly lower. **File compression alone therefore does not specify kernel efficiency.**"
> "Three effects complicate a raw memory roofline: **quantized kernels attain different fractions of peak bandwidth**."

**DERIVED — the Q4→Q3 comparison on RTX 5080:** bytes fall from 17 559 178 144 to 13 146 393 504 = **−25.1 %**, so a pure-bandwidth model predicts decode to rise by **1.335×**, i.e. 56.15 → **75.0 t/s**. Measured is **56.15 → 67.48 t/s = 1.202×**. So **60 % of the predicted speedup materialised and 40 % was eaten by kernel/dequantisation overhead.** Compare with the paper's own Q4/Q8 pairs, which achieved **1.37–1.49× against a size ratio of ~1.66× (83–90 % recovered)** — i.e. **the loss grows sharply as bpw falls below 4.5**. That is the cliff, expressed as a number.

Also note the RTX 5080 §4.2 quote applies here: "**File compression alone therefore does not specify kernel efficiency.**"

**Effective bandwidth reveals the real story (DERIVED):** only the Q3_K_XL file gets close to the 801 GB/s device-copy figure (688/801 = 86 %); Q2_K_XL reaches 618 (77 %), IQ3_XXS 644 (80 %), and IQ2_XXS collapses to 519 (65 %). **The 2-bit I-quant moves 3× fewer bytes but streams them at 65 % of the achievable rate, so it is only 1.37× faster than a 7.4× larger Q3 file.** Compare MacBook Q8_0: 192 GB/s against a 380 GB/s copy ceiling (51 %) — Q8_0 is *badly* kernel-limited on Metal, which is exactly why the Apple ladder looks monotonic and the CUDA one does not.

**A caveat the paper states itself:** "The largest RTX analytical working set exceeds reported VRAM before graph buffers, so **requested full layer offload is not proof of physical residency** ... Windows shared-memory spill cannot be excluded." Do not over-read the RTX effective-bandwidth numbers as pure kernel efficiency.

### 4.2 The cleanest "quant cliff" dataset: Q4 vs Q5 vs Q3 vs Q2, same model, same GPU — MEASURED

Source: [bric.pe.kr "GGUF Quantization Showdown"](https://bric.pe.kr/blog/gguf-quantization-showdown-q4-q5-iq4-2026), **2026-05-27**, llama.cpp b3500+, batch 1, 256 prompt / 256 gen.

**RTX 3090 24 GB (Ampere sm_86 — closest published analogue to the user's 3080s):**

| Quant | Llama 3.1 8B t/s | VRAM (4K ctx) | Qwen 3 14B t/s | VRAM |
|---|---|---|---|---|
| Q8_0 | 86 | 9.8 GB | 41 | 15.8 GB |
| Q5_K_M | 92 | 7.0 | 48 | 11.2 |
| Q5_K_S | 94 | 6.8 | — | — |
| **Q4_K_M** | **96** | **6.2** | **52** | **9.8** |
| **IQ4_XS** | **88 ← slower** | **5.8** | **46 ← slower** | **9.2** |
| Q4_K_S | 99 | 6.0 | 53 | 9.5 |
| Q3_K_M | 102 | 5.3 | 56 | 8.6 |

Quotes:
> "**IQ4_XS is slower than Q4_K_M on Pascal (-25%)** despite using less VRAM. ... **On Ampere it's only -8%, much more competitive.**"
> "**On a 1080 Ti or P40, prefer Q4_K_M over IQ-quants even when IQ is smaller.**"

**GTX 1080 Ti (Pascal sm_61):** Q8_0 19, Q5_K_M 21, Q5_K_S 22, **Q4_K_M 25**, **IQ4_XS 19 (−25 %)**, Q4_K_S 26, Q3_K_M 28 t/s. Qwen3-14B: Q4_K_M 13, IQ4_XS 11, Q4_K_S 14, Q3_K_M 16.

**So on one Ampere card at fixed model:** Q4_K_M → Q4_K_S (+3 %), → Q3_K_M (+6 %) while IQ4_XS goes **backwards 8 %**. The monotonic size→speed assumption fails at exactly the 4-bit boundary where I-quants start.

### 4.3 Same-model, same-rig Q4 vs I-quant at equal size — MEASURED

Source: [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks), Mar 5 2026. `pp512` / `tg128` (hardware not disclosed; likely 5090-class given the pp numbers):

| Type | pp512 | tg128 | vs K-quant mean |
|---|---|---|---|
| mxfp4 | 1978.69 | 90.67 | −0.2 % |
| **q4_k** | 1976.44 | **90.38** | baseline |
| q3_k | 1972.61 | **91.36** | +1.1 % |
| q6_k | 1964.55 | 90.50 | +0.1 % |
| q2_k | 1964.20 | 90.77 | +0.4 % |
| q8_0 | 1964.17 | 90.33 | −0.1 % |
| q5_k | 1947.74 | 90.72 | +0.4 % |
| **iq3_xxs** | **2030.94** | **85.68** | **−5.2 %** |
| **iq2_xxs** | 1997.64 | **85.79** | **−5.1 %** |
| **iq3_s** | 1990.12 | **84.37** | **−6.7 %** |
| **iq2_xs** | 1967.85 | **85.19** | **−5.8 %** |
| **iq2_s** | 1952.50 | **85.04** | **−5.9 %** |

**This is the cleanest statement of the cliff: pp512 is flat within ±2 % across a 2.0× bpw range (i.e. prefill is compute-bound and does not care), while tg128 is flat across the K-quants (Q2_K 90.77 ≈ Q8_0 90.33, within noise) and *drops by a consistent 5–7 % for every I-quant*.** Smaller K-quants gain nothing at decode; I-quants lose 5–7 % and never gain. Prefill slightly *favours* the I-quants.

### 4.4 The non-monotonic quant ordering, spelled out as tokens/s — MEASURED

Source: [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4 README](https://raw.githubusercontent.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4/main/README.md), measured **2026-08-17/18** on a single **NVIDIA L4 24 GB (sm_89, 72 W cap)**, `-ub 256 -b 2048`, q4_0 KV, DFlash-2 drafting n-max 7, kernel routing `GGML_MMVQ_MAX=2`.

| Config | Decode (tok/s) | Note |
|---|---|---|
| Stock deployment (native MTP head, depth 2) | 24.14 | baseline |
| + route speculative verification to MMQ | 28.05 | +16.2 % |
| + separate cheaper head with Q2_K output tensor | 30.70 | +9.4 % |
| + GPU sampling (`-bs`) | 31.00 | +0.9 % |
| + block-diffusion drafter (DFlash 2) | 35.99 | +15.0 % |
| + 2026-08-19 weight rebuild | **37.86** | +5.2 % |
| **shipped seven-workload average** | **32.96** | 1.36× stock |

**Draft-head size vs speed — the "quant cliff" at 5 GB scale:**
| Draft head configuration | Read/step | Predicted @252.8 GB/s | Measured | Mean accepted length |
|---|---|---|---|---|
| Embedded, 994.6 MiB Q6_K output | 1.237 GiB | 5.25 ms | 4.9–5.3 ms | 2.88 |
| `mtp-…-Q4_0` sidecar, 682.0 MiB | 0.888 GiB | 3.77 ms | 3.94 ms | 2.99 |
| **same, output retyped Q2_K, 397.9 MiB** | **0.611 GiB** | **2.60 ms** | **2.73 ms** | **3.01** |

Quotes:
> "The block is the other half of the draft read, 227.6 MiB against 157 MiB for the sub-head, and a cheaper quantization does not help it. **Rebuilding it at Q3_K with the imatrix gives 30.67 tok/s and accepted length 2.89, against 31.77 and 2.95 at Q4_0.** Cumulative draft time also rises, 5143 ms against 4831, **despite the smaller file, because the Q3_K dequantization kernel is slower per byte than the Q4_0 one.**"
> "**Q3_K sits between Q4_0 and Q2_K in size but measured worse than both, 28.89 tok/s against 29.20 and 29.91** on the same two workloads: its first-position acceptance fell to 0.746, against 0.779 for Q4_0 and 0.763 for Q2_K, which cost more than the bandwidth it saved."

**This is the direct answer to "is there a quant cliff where Q4_K_M is not faster than Q5_K_M":** a stronger version is true — **a *smaller* quant (Q3_K) is slower than *both* its larger (Q4_0) and smaller (Q2_K) neighbours on the same bytes-per-step model, because the dequant kernel efficiency dominates.** The mechanism is stated quantitatively in the same README — dequantization burns enough power to hit the cap and throttle the SM clock:
> "The chain is: **dequantization burns power, the cap throttles the SM clock to compensate, and SMs at a throttled clock with dequant work interleaved cannot issue memory requests fast enough to keep the bus full.** Decode sits at the 72 W cap. ... the 40 GB/s difference from a pure stream **is the power cost of dequantization**."
> "It is recoverable only by **doing less ALU work per byte read, which means a different kernel family, not a different setting.**"

### 4.5 MMQ vs MMVQ kernel efficiency per quant type on Ampere — MEASURED

Same source (L4, sm_89, `UD-Q4_K_XL`, 15.75 GiB read per forward pass, effective bandwidth **252.8 GB/s** vs a 300.05 GB/s datasheet peak and a **293.4 GB/s** measured streaming probe = 97.8 % of datasheet):

**Cost of one target forward pass vs verification width:**
| Verify width | `mul_mat_vec_q` (stock MMVQ) | `mul_mat_q` (MMQ) |
|---|---|---|
| 1 | **66.9 ms** | n/a (MMVQ correct at batch 1) |
| 2 | — | 74.1 ms |
| 3 | 85.5 ms | 80.6 ms |
| 4 | 111.7 ms | 83.2 ms |
| 8 | — | 89.7 ms |
| 12 | — | 100.1 ms |
| 16 | — | 106.2 ms |

Fitted: `T_MMVQ(B) = 66.9 + ~9.3·(B−1) ms`, `T_MMQ(B) = 66.9 + 9.7 + 1.98·(B−1) ms`. "**The 9.7 ms is a fixed admission cost: MMQ stages weights through shared memory in tiles rather than streaming them, so it does not reach the MMVQ bandwidth on the same bytes.**"

**Per-quant-type routing cost at draft depth 5** (routing one type back to MMVQ, everything else on MMQ):
| Routed to MMVQ | Share of weights | Decode (tok/s) |
|---|---|---|
| nothing (all MMQ) | — | 31.04 |
| Q4_K | 4 % | 31.05 |
| **IQ4_XS** | **17 %** | **31.76 (+2.3 %)** |
| **Q6_K** | **6 %** | **31.83 (+2.5 %)** |
| **Q5_K** | **72 %** | **20.51 (−34 %)** |

Quote: "**Q5_K must stay on MMQ: it is 72 % of the weights and the MMVQ per-column cost lands on all of it, costing a third of throughput.** Q6_K and IQ4_XS are small enough that the MMVQ cheaper entry beats its worse slope."
Quote: "**sm_89 matches no tuned table and falls through to `MMVQ_PARAMETERS_GENERIC`**" — and, from source §3.1, **so does sm_86**. The user's 3080s get the generic MMVQ launch config, not the Turing or Ada tuned one.

Also, decisive for MTP/speculative decoding: **`ggml_cuda_should_use_mmvq` returns true for `ne11 <= MMVQ_MAX_BATCH_SIZE` (= 8) and is consulted before `ggml_cuda_should_use_mmq`** — so every speculative verification batch (3–9 wide) lands on MMVQ unless patched. That is why speculation and quant choice interact (§4.6).

### 4.6 IQ quants and speculative decoding — MEASURED, and it flips the §4.3 conclusion

Source: [llama.cpp issue #27373, "Misc. bug: [SYCL] Degraded performance with MTP and Q3 quants"](https://github.com/ggml-org/llama.cpp/issues/27373), opened **2026-08-19**. `Qwen3.5-9B`, INTEL SYCL, `-ngl 99`, `--spec-type draft-mtp`, simple llama-cli runs (author labels them "one-shot results ... not a proper bench").

| Model | no MTP | MTP | Δ |
|---|---|---|---|
| `Qwen3.5-9B-Q4_K_M` | 56.0 | 61.2 | **+9.3 %** |
| `Qwen3.5-9B-IQ4_XS` | 39.0 | 57.6 | **+47.7 %** |
| `Qwen3.5-9B-Q3_K_M` | 56.0 | 46.2 | **−17.5 %** |
| `Qwen3.5-9B-UD-IQ3_XXS` | 22.6 | 7.0 | **−69.0 %** |

Quote: "**IQ4 also receives much more speedup than Q4, which I find strange.**"

This inverts §4.3 under speculation: IQ4_XS is the *worst* non-speculating quant (39.0 vs 56.0) and the *best* speculating one (+47.7 %). Because the user runs MTP speculative decoding, **this is the single most transferable finding in the report** — caveat: it is SYCL, one-shot, and the author says so. I found **no equivalent CUDA measurement**; searching for one returned nothing usable.

### 4.7 The 27B number that matters most for this rig — MEASURED

Source: same L4 README; model `UD-Q4_K_XL`, 16.35 GiB on disk, **15.75 GiB actually read per forward pass** (minus a 682 MiB `token_embd` row-lookup and a 272 MiB MTP block).

| Quantity | Value |
|---|---|
| **Stock (no speculation) decode** | **14.95 tok/s** |
| Target forward pass, batch 1 | **66.9 ms** |
| Effective bandwidth | **252.8 GB/s** (84 % of 300.05 GB/s datasheet; a pure `float4` stream reaches **293.4 GB/s = 97.8 %**) |
| SM clock during decode | oscillates **1230–1935 MHz**, `SW Power Cap` active, pinned at 72 W |
| Memory clock during decode | **6251 MHz, never moves** — no memory-clock headroom |
| Streaming probe | holds 2040 MHz at **45.6 W** |

**DERIVED for the user's rig.** Q8_0 at 29.05 GB, 78.3 tok/s: weight bytes streamed per token = 29.05 GB, so aggregate effective bandwidth = 29.05e9 × 78.3 = **2.275 TB/s of the 3.04 TB/s nominal = 74.8 %**. The L4 comparison point is 252.8 GB/s of a 300.05 GB/s datasheet = **84 %**, against a *measured* streaming ceiling of 293.4 GB/s = **97.8 %**. So this rig is running a few points below the L4's dequant-limited fraction, with the difference plausibly being 4-way tensor-split sync rather than anything quant-specific. **Implication: the headroom for a smaller quant is bounded by that ~75 % figure. A 25 % byte reduction cannot yield more than ~25 % speed, and §4.1 shows that below 4.5 bpw you actually recover ~60 % of even that.**

If the L4's "bytes actually read per forward pass" logic transfers (only touched weights count; the L4 reads 15.75 GiB of a 16.35 GiB file), a Q4_K_M 27B on this rig would stream ~15.6 GiB instead of ~27 GiB, i.e. **−42 % bytes**, and the paper's RTX 5080 Q4/Q8 ratio of **1.37–1.49×** implies **~107–117 tok/s** versus 78.3 today — but only if four cards' worth of aggregate dequant ALU does not become the binding constraint. **ESTIMATED, not measured on 4× 3080.**

---

## 5. Practical conclusions for this specific rig

1. **Reported "bpw" is a guideline, not a size.** On the exact Qwen3.8-27B model, observed bpw deviates from the upstream 8B table by **−6 % to +11 %** depending only on *which uploader's recipe* you download (§1.4). `Q4_K_M` alone ranges 16.46–17.44 GB across bedrock/unsloth/bartowski. Use API byte counts.
2. **Q8_0 → Q6_K is the free move.** Within 0.004 mean KLD and 0.010 PPL of the 8-bit/unsloth-8bit reference, and 3.7–5.2 GB smaller (23.86 GB bedrock, 25.30 GB `UD-Q6_K_XL`) (§2.1, §1.3). On Ada/Blackwell, Q6_K even gets *favourable* MMVQ routing (§4.5). On Ampere sm_86 it gets the generic table, so expect less.
3. **Q8_0 is genuinely a fast format, not a slow one.** It has the simplest vec-dot, and on Metal it is the *only* reason the Apple ladder looks monotonic. Its cost is 8.5 bpw, not kernel inefficiency. Expectations of a big speed win from shrinking it are the thing to distrust.
4. **The cliff is real and it is at ~4.5 bpw.** Above it, bytes→speed converts at 83–90 %. Below it, at 60 % or worse; and at the 4-bit I-quant / 3-bit K-quant boundary the ordering stops being monotonic at all (IQ4_XS < Q4_K_M on 3090, Q3_K < Q2_K and < Q4_0 on L4).
5. **I-quants are a quality-per-byte play, not a speed play, on mainline CUDA.** Expect **−5 to −12 % decode** for **−8 to −11 % bytes**; the two cancel. Worth it only if you need the bytes (you do not — you have 80 GB) or if you are speculating (§4.6, IQ4_XS gained +47.7 % from MTP while Q4_K_M gained +9.3 %).
6. **The only format documented to be both smaller and faster is ik_llama.cpp `IQ4_KS`** — 15.1 vs 17 GB on a 27B, **+18–20 % decode** at matched power, quality tied (§3.4). Costs: a fork, and k-quants lose their CUDA row-interleaved path under `-rtr` on multi-GPU setups. For a 4-GPU rig, verify `-rtr` behaviour before committing.
7. **Do not chase MXFP4 or NVFP4 on Ampere.** NVFP4's FP4 MMA is Blackwell-only ("currently only used for Blackwell" in `mmq.cuh`); MXFP4 is 4.25 bpw with measured-worse quality than Q4_K's 4.5 bpw per Unsloth's own study.
8. **Speculative decoding changes the quant ranking.** Quant ranking measured *without* speculation does not transfer to the speculative path — the L4 report shows Q6_K and IQ4_XS both *prefer* MMVQ while Q5_K must stay on MMQ (−34 % if routed wrong), and the SYCL MTP report shows IQ4_XS gaining 5× more from MTP than Q4_K_M. Since the user runs MTP, **the quant should be chosen with MTP on**, and the kernel routing (`GGML_MMVQ_MAX`) sweep is worth more than the quant change itself: routing alone was **+16.2 %** on the L4.

---

## 6. Sources, dates, and staleness

| Source | Date | Staleness | Used for |
|---|---|---|---|
| [llama.cpp `tools/quantize/README.md`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md) | master, fetched 2026 | current | §1.1 bpw table |
| [arXiv:2609.14864](https://arxiv.org/abs/2609.14864) + [HTML](https://arxiv.org/html/2609.14864v1) | 2026-09-14 | current | §4.1, §1.3 byte sizes |
| [llama.cpp PR #23572](https://api.github.com/repos/ggml-org/llama.cpp/issues/23572) | opened 2026-05-23, upd 2026-09-08 | current | §2.2 |
| [bric.pe.kr GGUF Showdown](https://bric.pe.kr/blog/gguf-quantization-showdown-q4-q5-iq4-2026) | 2026-05-27 | ~3 mo | §2.3, §4.2 (Ampere 3090) |
| [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) | Mar 5 2026 update | ~6 mo ⚠️ | §2.1, §4.3 |
| [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) | measured 2026-08-17/18 | current | §4.4, §4.5, §4.7, sizes |
| [club-3090 `QUANTIZATION.md`](https://raw.githubusercontent.com/noonghunna/club-3090/v0.10.2/docs/QUANTIZATION.md) | tag v0.10.2, rows to 2026-07 | ~3 mo | §3, §2.5 |
| [club-3090 `BENCHMARKS.md`](https://raw.githubusercontent.com/noonghunna/club-3090/v0.10.2/BENCHMARKS.md) | rows 2026-04→2026-07 | ~3 mo | §2.5, §3.4 |
| [club-3090 `engines/IK_LLAMA.md`](https://raw.githubusercontent.com/noonghunna/club-3090/v0.10.2/docs/engines/IK_LLAMA.md) | 2026-05-23/24 | ~4 mo | §3.4 |
| [ik_llama.cpp README + discussion #8 + #288](https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/README.md) | #8 upd 2025-07-22, #288 2025-03-25 | **⚠️ >6 mo, stale** | §3.4, §3.5 ik quant design/CUDA kernels |
| [llama.cpp issue #27373](https://github.com/ggml-org/llama.cpp/issues/27373) | 2026-08-19 | current | §4.6 MTP × quant |
| [llama.cpp `mmq.cuh`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmq.cuh) / [`mmvq.cu`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmvq.cu) | master, fetched 2026 | current | §3.1, §4.5 |
| [hf-mirror tree API](https://hf-mirror.com/api/models/bartowski/Qwen3.8-27B-GGUF/tree/main?expand=true) | fetched 2026 | current | §1.3 exact bytes |
| [willitrunai.com quant comparator](https://willitrunai.com/quant-compare) | © 2026 | current | cross-check only — **ESTIMATED, "calibrated against real GGUF file sizes"**, Qwen3.5-27B: Q2_K 10.5, Q3_K_S 13.2, Q4_K_M 16.5, Q5_K_M 19.4, Q6_K 22.1, Q8_0 28.9 GB — agrees with the measured bytes to ±3 %, except its Q2_K (10.5 GB) which matches bartowski, not bedrock (9.83 GB) |
| [mintlify llama.cpp mirror](https://mintlify.wiki/ggml-org/llama.cpp/concepts/quantization) | unversioned | **⚠️ model unstated, ESTIMATED** | §1.1 cross-check |
| [llama.cpp `PR #8075`](https://github.com/ggml-org/llama.cpp/pull/8075) (MMQ-default MMVQ) | 2024 | ⚠️ stale | mechanism context only |
| [HF skills GGUF guide](https://raw.githubusercontent.com/huggingface/skills/main/skills/huggingface-local-models/references/quantization.md) | 2026 | current | §1.1 cross-check; PPL table model unstated — **ESTIMATED** |

### Queries that found nothing useful
- No published **CUDA** measurement of the MTP/speculative-decoding × quant interaction (§4.6) — only the SYCL case.
- No independent CUDA decode measurement of **trellis** quants (`IQ*_KT`, §3.5).
- No 2025–2026 result showing **QTIP / AQLM / BitNet** landed in llama.cpp or ik_llama.cpp as a *post-training* format for a 27B dense model.
- Gemma-3-27B / Qwen3-32B / Mistral-Small-24B **GGUF size tables**: unreachable (HF blocked, mirror rate-limited) — 4 attempts each.
- "Q4_K_M is indistinguishable from Q8_0" as a *measured claim* for 27B: not found; the closest measured statements are +0.070 PPL / 7.4× mean-KLD (§2.1), −1.4 MMLU-Pro points (§2.3), and −4 to −5 agentic-pack points vs 8-bit (§2.5) — none of which supports "indistinguishable".
- r/LocalLLaMA direct: **blocked** (`www.reddit.com` resolves to a non-public IP in this environment; the JSON search endpoint is unreachable). No r/LocalLLaMA threads are cited above as a result.
