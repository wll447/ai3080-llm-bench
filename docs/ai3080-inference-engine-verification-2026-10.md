# ai3080：推理引擎查证报告（gufo / strata / halogen / vLLM / SGLang / …）

> 查证日期：2026-10（本次会话）
> 目标机：`ai3080` — 4× RTX 3080 20GB 改装卡（GA102 / **sm_86**）、80 GB 显存、
> 123 GiB 内存（8×16 GB DDR4-2400 ECC）、AMD EPYC 7402、P2P 已打通（驱动 610.57.04 + 补丁）
> 现役：llama.cpp 主线 `81bc6b8`（0.5.0-dev build 333）+ Qwen3.8-27B **Q8_0**、
> `--split-mode tensor -ts 1,1,1,1`、MTP `--spec-draft-n-max 2`
> → **prefill ~815–877 t/s / decode ~82–84 t/s**
> 状态：本次查证时 ai3080 **不在线**（Tailscale 无该节点、<TAILSCALE-IP> / <LAN-IP> 均不可达），
> 全部结论基于 workspace 内既有实测记录 + 外部公开来源，**未做线上复测**。

---

## 0. 一句话结论

**gufo / halogen / strata 都不适用于 ai3080。** 前两个是 AMD Strix Halo（`gfx1151`）专用引擎
——那是你 **mad395** 那台机器的方案（`WorkSpace/mad395-gufo-halogen-指南.md` 已记录）；
strata 是「125B MoE 靠 RAM/SSD offload 塞进 12 GB 卡」的另一个赛道。

真正可能比 llama.cpp 好的只有 **ik_llama.cpp / vLLM / SGLang**（+ ExLlamaV3 作参照），
而且**没有一个能全面碾压**：单流 decode 基本同档，赢面集中在
①减少每步同步次数 ②并发聚合吞吐 这两个维度。

---

## 1. 点名引擎逐个查证

| 引擎 | 真伪 | 支持 sm_86？ | 判定 |
|---|---|---|---|
| **gufo** | ✅ 真项目，MIT，活跃 | ❌ **仅 gfx1151** | 自述 *"specifically built and optimized for the AMD Strix Halo hardware: Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`)"*，依赖 ROCm/HIP、hipBLAS。**与 NVIDIA 无关。** |
| **halogen** | ✅ 真项目（全手写 HIP kernel） | ❌ **硬拒绝其它架构** | README 原文：*"The build **hard-rejects every other architecture**; this will not run on your discrete GPU, and that is deliberate."* 只支持 Qwen3.8-27B + 自定义 `.hgn` 权重 |
| **strata** | ✅ 真项目，MIT | ⚠️ 支持 RTX 30 系，但**赛道不同** | 目标是把 **Qwen3.8-Flash-Next 125B MoE** 跑在 12 GB+ 卡上：GPU 放热专家 + RAM 放全部 + CPU 算 + SSD 放查找表；需 32–96 GB RAM、~70–80 GB 下载、首启动冻结 1–3 分钟 |
| **vLLM** | ✅ 真，Apache-2.0 | ✅ | 见 §3 |
| **SGLang** | ✅ 真，Apache-2.0 | ✅ | 见 §3 |

### 1.1 两个必须先说的坑

**（a）gufo / halogen 的性能数字很漂亮，但一个都不能搬过来。**
gufo 官方在 395 + 128 GB 上：Qwen3.8-27B Q4 `656.33 pp / 70.56 tg`、8 并发 123.00 聚合；
Flash-Next `1,628.52 pp / 59.41 tg`。halogen 报 `pp32768 = 566 t/s`、decode 均值 31.71。
这些是 **LPDDR5X 统一内存 + gfx1151** 的成绩，与 4×3080 的 GDDR6X/PCIe 拓扑没有可比性。

**（b）strata 的仓库卫生需要审。**
同一份 README 出现在多个账号下（`Niko1221/Strata`、`architectds/Strata`），
且首屏就引导你把一段「帮我装好」的 prompt 粘进 Claude Code / Cursor / Codex
（`docs/AI_SETUP.md`），并提供一个 MCP server 让 AI 代理执行安装/启停。
**在生产服务器上跑之前应当先读脚本**，不要把「让 AI 自动安装」当成默认路径。

---

## 2. 判断「好不好」的正确口径

你 2026-09-26 的实测已经把根因钉死了，这决定了选引擎的标准：

| 证据（本机实测） | 含义 |
|---|---|
| tensor 无 MTP 单 pass = **18.90 ms**，每卡读 7.28 GB → **385 GB/s = 53% 峰值** | decode **不是**带宽瓶颈 |
| Q8_0 → Q4_K_M：字节 **−42.2%**，decode 只 **+12.0%**，单 pass 32.1 → 34.0 ms | 同样证伪「少读字节」路线 |
| 每个草稿 pass 恒定 **~5–6.6 ms**（tensor 6.6 / layer 4.9），与 split 模式无关 | 这是**内核启动 + 每步同步**的固定开销 |
| MTP 增益 tensor **+56%**、layer **+148%** | 真正有效的杠杆是「每 pass 多出 token」 |

> **所以：能减少「每步集合通信同步次数」的引擎 = 对你有用；其余都是噪声。**

同时记住 sm_86 的硬墙：**FP8 权重、NVFP4 都不可用**（需 sm_89+ / sm_100+），
可用范围是 W4A16（AutoRound / AWQ / GPTQ / Marlin）与 GGUF Q*/K*/IQ*。
网上大量「4 卡 fp8 配方」对你直接不适用。

---

## 3. 真正值得考虑的候选（按对症程度排序）

### 🥇 #1 — ik_llama.cpp，`-sm graph`（graph parallel）
**最对症。这是你既有研究里排第 1、但在 9/26 清理时被删掉的候选。**

- 机制（**源码级验证**，非营销）：`LLAMA_SPLIT_MODE_GRAPH = 3`，张量按行列同时切分、
  KV 按 GPU 切分、self-attention 并行，**每层只有 2 个同步点**（post-attention / post-FFN），
  而不是每个 matmul 一次 all-reduce。
- **不依赖 NCCL**：`ggml/src/ggml-cuda/reduce.cu` 是手写 ring all-reduce 走 `cudaMemcpyPeerAsync`；
  ikawrakow 自己的注释说 ring 比 NCCL 更快。NCCL 只在 `nhave == 2 || ne[1] < 32` 时用 —— TP=4 prefill **不走**。
- 实测：同功耗同质量 **+18 ~ +20%**；无 P2P 时退化温和（2×3090 / Llama-3-70B Q4_0：prefill −24%、decode −11%）。
  **你现在 P2P 是通的，实际只会更好。**
- 附赠：IQK imatrix 量化（`IQ4_KS` 等，GGUF 世界最佳质量/位）、Hadamard KV（`-khad`/`-vhad`，零 VRAM 提质）、
  `--spec-stage` **两段式 `ngram-mod` + MTP**（代码场景 **+35%**、无损、0 额外 VRAM —— 注意
  这个组合在 **主线 llama.cpp 里是净亏损**，只有 ik 的实现有效）。
- 成本：**半天**（另编译一棵树）。风险低（drop-in `llama-server` 替代）。

### 🥈 #2 — vLLM，TP=4 + AutoRound/AWQ INT4 + MTP
- **单流赢不了多少**：club-3090 上一台 P2P 已通的 4×3090 跑 vLLM + AutoRound-INT4 + MTP n=3 + fp8 KV
  → 单流 **82.44 / 103.73 t/s**。你现在 Q8_0 是 82–84 —— **基本同档**。
- **真正的赢面在并发聚合：3–10×**（continuous batching + paged attention + 前缀缓存）。
  你有 262144 上下文的服务，**如果实际有多个客户端，这是唯一量级上的提升**。
- 注意：TP=4 + 无 NVLink 时 vLLM 会禁用 custom all-reduce，走 NCCL。
  另有 4 卡 no-NVLink 的实测显示 vLLM **TP=2 在多数并发点优于 TP=4**，值得两个都试。
- 代价：部署重（Python/CUDA 环境、模型要转格式）、磁盘要留量。

### 🥉 #3 — SGLang
- 与 vLLM 同一条 NCCL/no-P2P 路径，机制上不比 vLLM 更会处理你这套拓扑。
- 但有数据点显示同 rig 单流 decode **+43% over vLLM**（W4A8 + DFlash2 n=8）。
- 代价：生产上比 vLLM 更脆、跨机器锚点更少。

### 参照项
- **ExLlamaV3**（`turboderp-org/exllamav3`）：Ampere 可用，EXL3 4-bit 量化。作为第三条对照线。
- **ktransformers**：只在**大 MoE 超出显存**时才有价值（router-aware 热专家缓存）。
  你 29 GB 的 27B dense 在 80 GB 里绰绰有余，**不适用**。
- **NInfer / `Don-Chad/ninfer-3090`**（2026-09 新出现）：宣称单 3090 跑 Qwen3.8-27B **71 t/s**、C8 163 t/s。
  **按证据给「存疑」**：3090 约 936 GB/s、Q4 权重 15–16 GB → 纯自回归上限 ~55–60 t/s，
  论坛帖里当场就有人据此质疑；且**只支持单卡**、只吃 NInfer 自有格式权重（不能换 GGUF）、
  FP8 KV 在 SM86 不可用。**对 4 卡 80 GB 的你没有意义。**

---

## 4. 「比 llama.cpp 好」要分维度谈

| 维度 | 现状（ai3080） | 谁更好 | 上限 |
|---|---|---|---|
| **单流 decode** | 82–84 t/s | 谁都赢不了太多（vLLM/ik 同档） | Q8_0 29 GB / 3040 GB/s ≈ **105 t/s** 硬顶，你在 75–80% |
| **每 pass token 数** | MTP n=2（AL 2.65） | ik 的两段式 ngram+MTP（代码 +35%） | 引擎半相关，最实在的杠杆 |
| **并发聚合吞吐** | 未测 | **vLLM / SGLang，数量级差异** | 这是唯一「真的更好」的轴 |
| **长 prompt 总时间** | tensor n=2 = 8012 ms | 你已有答案：`layer + n_max=6` = **7627 ms** 反超 | 提示 > 输出×12 时 layer 反超 |
| **每步同步开销** | 单 pass 18.9 ms / 53% 峰值 | **ik_llama `-sm graph`（每层 2 次同步）** | 148 t/s 理论值需改源码，属 PR 范畴 |

---

## 5. 建议动作顺序

1. **先做 ik_llama.cpp `-sm graph` + IQ4_KS 的 A/B**（半天，预期 +18–20%，最对症）。
   保持红线不变：**绝不设 `GGML_CUDA_P2P=1`**。
2. **再上 vLLM TP=4 INT4**，但**只测并发轴**（C1 vs C8 聚合），别和单流数字混谈。
   顺带测 TP=2 vs TP=4。
3. **gufo / halogen 留在 mad395**；**strata 只在你想跑 125B MoE 时**再评估（且要先审安装脚本）。
4. **统一口径**再做对比：4001 token prompt + 256 token 输出（`ignore_eos`）+ 3/3 正确性校验
   （Paris / 51 / Jupiter）+ 确认 `nvidia-smi topo -p2p` 全 OK，避免在被污染状态下测出一堆假结论。
5. 磁盘预算：现 ~51 G 已用 / 171 G 可用。vLLM 环境 + 模型转换要额外预留，
   ik_llama.cpp 另建一棵源码树约几 GB。

---

## 6. 来源

- gufo — <https://github.com/gufo-org/gufo>（README 原文：仅 `gfx1151`）
- halogen — <https://github.com/peonist-ai/halogen-server>（README 原文：硬拒绝其它架构）
- Strata — <https://github.com/Niko1221/Strata> / <https://github.com/architectds/Strata>
- club-3090 引擎对比（vLLM / llama.cpp / SGLang / ktransformers / ik_llama.cpp）—
  <https://github.com/noonghunna/club-3090/blob/master/docs/INFERENCE_ENGINES.md>
- NInfer SM86 移植帖（含质疑回复）— <https://lcz.me/topic/1644/3>
- ExLlamaV3 — <https://github.com/turboderp-org/exllamav3>
- 本机实测依据：`ai3080-final-status-2026-09-26.md`、`ai3080-production-deployment.md`、
  `ai3080-speedup-plan.md`、`no-p2p-engine-research.md`、`spec-decode-research-2026.md`、
  `quant-specdecode-4x3080-report.md`、`WorkSpace/mad395-gufo-halogen-指南.md`
