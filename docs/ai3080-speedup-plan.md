# ai3080：llama.cpp 提速研判（本机实测 + 全网调研）

**日期**: 2026-09-24
**硬件**: 4× RTX 3080 20GB，各自独占 AMD EPYC root complex，Gen4 x16，直插无交换机
**模型**: Qwen3.8-27B（`qwen35` 架构，65 层，**16 层全注意力 + 49 层 GDN 线性注意力**，KV 头 4，key_length 256）
**引擎**: llama.cpp 0.4.0-dev (commit 304665f, 2026-09-15)，`GGML_CUDA_NCCL=ON`，CUDA graphs ON

---

## 0. 一句话结论

> **decode 不是显存带宽瓶颈 —— 是「每个 forward pass 的同步开销」瓶颈。**
>
> **证据（本机实测）**：模型体积缩小 **42.2%**（27.1GB → 15.7GB），decode 只涨 **12.0%**。
> 每个 pass 的耗时几乎没变（Q8_0 32.1 ms → Q4 34.0 ms）。
>
> **所以方向不是「少读字节」，而是「减少同步次数 / 降低单次同步成本」。**

---

## 1. 本机实测矩阵（全部为 4001 token prompt + 256 token 输出 + ignore_eos）

| 模型 | split | MTP | **prefill** | **decode** | 接受长度 AL |
|---|---|---|---|---|---|
| Q8_0 (27.1GB) | tensor×4 | 无 | 842.5 | 72.53 | – |
| Q8_0 | tensor×4 | n=1 | 838.6 | 69.87 | 1.89 |
| **Q8_0** | **tensor×4** | **n=2** | 843.8 | **82.50** ⭐ | 2.65 |
| Q8_0 | tensor×4 | n=3 | 839.6 | 75.83 | 2.76 |
| Q8_0 | tensor×4 | n=5 | 846.0 | 72.47 | 3.10 |
| Q8_0 | tensor×4 | n=8 | 822.2 | 62.82 | 3.49 |
| Q8_0 | tensor×4 + nccl | n=2 | 849.8 | 81.32 | 2.65 |
| **Q8_0** | **layer×4** | n=2 | **1400.1** ⭐ | 52.7 | – |
| Q4_K_M (15.7GB) | tensor×4 | 无 | 780.1 | 90.76 | – |
| Q4_K_M | tensor×4 | n=2 | 784.8 | 86.97 | 2.51 |
| **Q4_K_M** | **tensor×4** | **n=3** | 805.2 | **92.40** ⭐ | 3.14 |
| Q4_K_M | layer×4 | n=2 | 1234.1 | 62.19 | – |
| Q4_K_M | tensor×**2** | n=3 | 834.0 | 71.99 | – |
| Q4_K_M | tensor×**2** | 无 | 835.3 | 71.41 | – |

### 读出来的结论

**① 量化收益极小 —— 因为 decode 不是带宽瓶颈**
```
Q8_0 → Q4_K_M:  字节 -42.2%   decode 只 +12.0%
单 pass 耗时:   32.1 ms → 34.0 ms   ← 几乎没变
```

**② MTP 在 Q8_0 上最优 n=2（+13.7%）；Q4_K_M 上 n=3 更好**
```
Q8_0: 无MTP 72.5 → n=2 82.5  (+13.7%)  ← 最优
Q4:   无MTP 90.8 → n=3 92.4  ( +1.8%)  ← 已接近饱和
```
> AL 到了 2.65~3.14，但速度只涨 14% ⇒ **验证 pass 的开销随 token 数线性增长，完全没有摊薄** —— 又一次指向「同步开销按 batch 大小增长」。

**③ 2 卡比 4 卡慢**（Q4: 72.0 vs 92.4）
> 网上 4×5060Ti 基准说「2 卡走 internal 更快」，**但那是因为他们的 4 卡走了 butterfly。我们已经是 NCCL，而且四张卡各占一条 CPU 直连 lane（正是那份基准说的最优拓扑），所以 4 卡优势正常。这条不适用我们。**

**④ `layer` 模式 prefill 巨大领先（Q8_0: 1400 vs 844，+66%），但 decode 大幅落后（52.7 vs 82.5）**
> 因为 PP 的单序列 decode 要在 4 卡上**串行**跑。**按负载二选一。**

**⑤ `GGML_CUDA_ALLREDUCE=nccl` 无效**（81.3 vs 82.5，噪声内）—— 因为本来就是 NCCL 路径。

### 自检确认（决定网上建议是否适用）

| 检查项 | 结果 | 含义 |
|---|---|---|
| `ldd` 有 `libnccl.so.2`，`GGML_CUDA_NCCL=ON` | ✅ | 走 NCCL，不是 internal/butterfly |
| `converting K and V to f16` 警告 | ✅ 无 | f16 KV 在 FA 快路径上 |
| `graphs reused = 91 / ~93 passes` | ✅ | CUDA graphs 生效 |
| `GGML_CUDA_FA_ALL_QUANTS=OFF` | ⚠️ 但无影响 | 用 f16 KV，不需要重编 |
| **`n_slots = 4`** | ⚠️ **-np 默认 4，非 1** | `-c 262144` 是 4 个 slot 共享 |

---

## 2. 已实测排除的（别再花时间）

| 项 | 实测 | 结论 |
|---|---|---|
| 2 卡（`CUDA_VISIBLE_DEVICES=0,1`）| 72.0 vs 92.4 | ❌ 更慢 |
| `GGML_CUDA_ALLREDUCE=nccl` | 81.3 vs 82.5 | ❌ 噪声内 |
| `-ts 1,1,0,0` 当 2 卡用 | 日志报 `n_devices != 2` | ❌ 无效，必须 `CUDA_VISIBLE_DEVICES` |
| 大幅降量化（-42% 字节）| decode 只 +12% | ❌ 收益远小于体积比 |
| 不等比 `-ts` | 开源 bug #28506 未修 | ❌ tensor 模式下不可靠 |

---

## 3. 优先级行动清单（按「收益/成本」排序）

### 🟢 立刻可做，零风险

| # | 动作 | 预期 | 依据 |
|---|---|---|---|
| **1** | **`-np 1`** 显式设定 | 稳定 graph 形状；当前是 4 slot 共享 | 本机自检 `n_slots=4` |
| **2** | **`-t 24` 且让 `--threads-batch` 等于它**（现在 48）| 少一个线程池 | 全 GPU 驻留时 CPU 线程无关 |
| **3** | **Q8_0 用 MTP `n_max=2`**（不是 3）| **72.5 → 82.5，+13.7%** | ✅ 本机实测 |
| **4** | **`NCCL_NVLS_ENABLE=0`** | 免费保险（NCCL 2.27/2.28 有 NVLS 分配失败导致 hang 的 bug）| NCCL 文档 |
| **5** | **`-c` 跨 2048 边界扫一下** | MTP 接受率在 2048 对齐的 ctx 上会崩（issue #23658：ctx 12032 → 16% 接受，12288 → 71%）| 20 分钟，免费 |

### 🟡 半小时实验，预期可观

| # | 动作 | 预期 | 依据 |
|---|---|---|---|
| **6** | **扫 `GGML_MMVQ_MAX` 内核路由** | **最高 +16.2%**（比整个量化变化还大）。sm_86 **没有** per-type MMVQ 调优，走 `MMVQ_PARAMETERS_GENERIC` | 同源实测 |
| **7** | **`-b`/`-ub` 成对扫描**：`-b 2048 -ub 1024`、`-b 4096 -ub 2048` | prefill 上升，decode 不变 | ⚠️ **只调 `-ub` 无效**：`n_ubatch = min(n_batch, n_ubatch)`，你 `-b 1024` 把它卡住了 |
| **8** | **layer 模式加 `CUDA_SCALE_LAUNCH_QUEUES=4x`** | prefill **+10~25%**（decode 不变）| PR #19042；**不会自动设**，2 月被回退成手动 |
| **9** | **量化 × MTP 二维扫描**（Q6_K/Q4_K_M/Q3_K_M × MTP 开/关）| ⭐ **量化排名在 MTP 下会反转**：IQ4_XS +47.7%、Q4_K_M +9.3%、Q3_K_M **-17.5%** | issue #27373 |
| **10** | **power limit / 温度检查** | 这台模型**部分是 compute-bound**（GDN 的 `chunked_gated_delta_rule`），230W 会 -34%，290W -6.9%。**别降压** | 实测 |

### 🔵 需要重编/换引擎，收益更大

| # | 动作 | 预期 | 依据 |
|---|---|---|---|
| **11** | **升级到含 PR #28549 的构建**（2026-09-16，MTP 的 CUDA graph）| **+4~5% decode**。你的 build 是 09-15，**不含** | 上游实测 |
| **12** | **换 ik_llama.cpp + `-sm graph`（图并行）+ IQ4_KS** | **+18~20%**（同功耗、质量持平）；graph 并行**每层只 2 次同步**（源码验证：`reduce.cu` 用 `cudaMemcpyPeerAsync` 手写 ring all-reduce，**不依赖 NCCL**，无 P2P 只损 -11% TG / -24% PP）| 源码级验证 |
| **13** | **`--spec-type draft-dflash` + DFlash 头** | 比 MTP **+26%**（matched），对无投机最高 2.37× | club-3090 #759 |
| **14** | **ngram-mod**（mainline 一个 flag）| 代码场景 **+25%**，无损、0 VRAM；ik_llama 的 ngram+MTP 两段式 **+35%** | 实测 |
| **15** | 显存超频 +500~1000 MHz | +3~10%（**Ampere 上无公开实测**，且 GDDR6X 有 EDR 平台期）| 外推 |

---

## 4. ⚠️ 必须知道的风险

| 风险 | 内容 |
|---|---|
| **tensor 模式会死锁** | issue #27110（未修）：无 P2P 的 2×5060Ti 上 `-sm tensor` **约 1/1000 请求死锁**（NCCL 与 internal 都会）；**`layer` 从不发生**。你的生产模式正是 tensor —— 这是稳定性权衡，不只是性能 |
| **MTP + CUDA graphs 显存泄漏** | issue #25835（未修）：Qwen3.6-27B + MTP + graphs，5~10 轮后涨到 OOM；逃生舱 `GGML_CUDA_DISABLE_GRAPHS=1` |
| **Ampere (sm_86) 硬限制** | **FP8 权重、NVFP4 都不可用**（需 sm_89+ / sm_100+）。可用的只有 W4A16（AWQ/GPTQ/Marlin）。网上不少 4 卡 fp8 配方对你不适用 |

---

## 5. 现实目标

| 组合 | 预期 decode |
|---|---|
| 现在（Q8_0 + MTP n=2） | **82.5** |
| ＋ `-np 1`、MMVQ 路由、MTP 深度微调、power 检查 | **95 ~ 110** |
| ＋ 升级 #28549、量化×MTP 二维扫描 | **110 ~ 130** |
| ＋ ik_llama `-sm graph` + IQ4_KS + DFlash | **150 ~ 190**（上界，需实测） |

**诚实说明**：字节减少有硬上限 —— Q8_0→Q4_K_M 是 -40% 字节，纯比例最多 +66.7%，而实测召回损失会吃掉一部分；**我们实测只拿到 +12%**，说明瓶颈在别处。

---

## 5.5 【实测更新】升级 #28549 + `-b/-ub` 隔离实验

### A. #28549 升级：**收益为零**

构建：`/home/<user>/llama.cpp/build-28549/`（`0.4.1-dev build 140, commit 2f3fd02`）
旧构建已备份：`/home/<user>/llama-backup-304665f/`（含 .so，可立即回滚）

同一套参数，只换二进制（各跑 2 轮）：

| 构建 | prefill | decode | graphs reused | AL |
|---|---|---|---|---|
| OLD (304665f) | 941 / 979 | 72.48 / **74.27** | 109 / 216 | 2.30 / 2.32 |
| NEW (2f3fd02, #28549) | 934 / 963 | 72.43 / **74.29** | 108 / 215 | 2.30 / 2.32 |

**74.27 vs 74.29 —— 完全相同。graph 复用次数也相同。→ #28549 在本机无可测收益。**

### B. ⚠️ 更重要的发现：`-b`/`-ub` 是**权衡**，不是白赚

调研说「`-b/-ub` 只影响 prefill、decode 中性」。**本机实测证伪：**

| 配置 | prefill | **decode** | 接受率 | AL |
|---|---|---|---|---|
| 原基线 `b1024 ub512` | 843.8 | **82.50** | 0.823 | 2.65 |
| `np1` + `b1024 ub512` | 838.9–855.2 | **81.9** | 0.82292 | 2.65 |
| `b2048 ub1024` + `tb24` | 904.4–931.7 | 73.3–74.5 | 0.66210 | 2.32 |
| `np1` + `b2048 ub1024` + `tb24` | 963.3–979.0 | 74.3 | 0.66210 | 2.32 |
| `np1` + `b2048 ub1024` + `tb48` | 904.9–937.8 | 72.3–73.7 | 0.652–0.662 | 2.30–2.32 |

**结论：**
- ✅ **`-np 1` 无害**（81.9 vs 82.5，噪声内）
- ✅ **`--threads-batch` 无关**（tb24 与 tb48 结果相同）
- ⚠️ **真正的变量是 `-b/-ub`**：`b2048 ub1024` 让 prefill **+9~11%**，但 decode **-10~12%**
- **机制**：大 ubatch 让验证走不同 kernel → 浮点结果微变 → **draft 接受率从 0.823 掉到 0.66** → 每 pass 产出的 token 变少

### C. 最终推荐配置（按负载二选一）

**decode 优先（聊天 / 单流）— 82 t/s**
```bash
NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 ./build-28549/bin/llama-server \
  -m /home/<user>/models/Qwen3.8-27B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 262144 -ngl -1 \
  --split-mode tensor -ts 1,1,1,1 \
  -b 1024 -ub 512 -t 24 --threads-batch 48 -np 1 \
  -fa on --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --cont-batching --load-mode mmap --jinja --fit off
```

**prefill 优先（长上下文 / 低 TTFT）— 938 t/s**
```bash
  ... -b 2048 -ub 1024 -t 24 --threads-batch 24 -np 1 ...   （其余同上）
```

**长上下文批量（prefill 最高）— 1400+**
```bash
  ... --split-mode layer ... CUDA_SCALE_LAUNCH_QUEUES=4x ...
```

### D. 回滚

```bash
# 立即回到旧构建
cp -a /home/<user>/llama-backup-304665f/bin/. /home/<user>/llama.cpp/build/bin/
# 或切回旧提交重新构建
cd /home/<user>/llama.cpp && git checkout 304665f
```

---

## 5.6 【实测更新】三个零编译实验，全部 0 效果 + 一个配置纠正

### A. `MMVQ_PARAMETERS_TURING` / `GGML_MMVQ_MAX`：**源码级证明是空操作**

`mmvq.cu` 的 `calc_nwarps()` 展开对比（我们的 Q8_0 权重）：

| 情形 | GENERIC（我们实际走的）| TURING（拿不到的那张）| 结果 |
|---|---|---|---|
| `ncols_dst = 1`（无 MTP）| `case 1: return 4` | `default: return 4` | **都是 4** |
| `ncols_dst = 3`（MTP n=2）| `case 2/3/4: return 4` | `case 2/3/4: return 4` | **都是 4** |

> TURING 与 GENERIC 的**唯一**差异：K 量化在 `batch=1` 时用 2 warp 而非 4 warp。
> **Q8_0 走 `default`，两条路径返回同一个值。→ 对我们 0 效果。**

`MMVQ_MAX_BATCH_SIZE = 8` 这个阈值：我们 decode 批次 = **3**、prefill 批次 = **512**，两头都远离 8
→ **任何合理取值都不改变路由。整条线是死路。**

### B. `CUDA_SCALE_LAUNCH_QUEUES=4x`（layer 模式）：**实测 0%**

| layer 模式（`-np 1`）| prefill | decode |
|---|---|---|
| 不加 | 1272 / 1418 | 43.00 / 43.00 |
| 加 `=4x` | 1262 / 1407 | 42.96 / 42.86 |

调研称 prefill +10~25%（测于 2×RTX Pro 6000）。**本机无效果。**

### C. ⚠️ 配置纠正：`-np 1` 对 layer 模式有害

| layer 模式 | decode |
|---|---|
| `-np 4`（默认）| **52.7** |
| `-np 1` | **43.0（-18%）** |

**原因**：PP 靠多序列才能流水；单序列时 4 张卡串行等待。
（club-3090：*"PP is not parallelism at `-np 1`"*）

| 模式 | `-np` 正确取值 |
|---|---|
| **tensor** | **`1`**（中性，且稳定 CUDA graph 形状）|
| **layer** | **保持默认 `4`**（多序列才让 PP 有意义）|

### D. 结论：**便宜的手段已经全部穷尽**

| 试过的 | 结果 |
|---|---|
| 升级 #28549 | **0%** |
| `GGML_MMVQ_MAX` / MMVQ 表 | **源码证明 0 效果** |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` | **0%** |
| `-b 2048 -ub 1024` | prefill +9~11% / decode **-10~12%**（权衡）|
| `-np 1`（tensor）| 0（中性）|
| `-np 1`（layer）| **-18%（有害）** |
| `GGML_CUDA_ALLREDUCE=nccl` | 0% |
| 2 卡 | -22% |
| Q4_K_M（-42% 字节）| decode 仅 +12% |

> **所有廉价旋钮都是 0 或权衡。** 这反过来强力确认了诊断：
> **瓶颈是结构性的（每 pass 的集合通信同步），只有改变执行策略才能撼动它。**

### E. 只剩三个真正可能撼动的方向

| 方向 | 机制 | 预期 | 成本 |
|---|---|---|---|
| **ik_llama.cpp `-sm graph`** | 每层只 2 次同步（vs 每个 matmul 一次）；手写 ring all-reduce 走 `cudaMemcpyPeerAsync`，**不依赖 NCCL** | **+18~20%** | 半天（另编译）|
| **DFlash 投机头** | 比 MTP 接受率高（matched +26%）| +26% | 中 |
| **vLLM TP=4 + INT4（aggregate）** | 完全不同的引擎 | 并发场景 3~10× | 高 |

---

## 6. 对「P2P 到底值多少」的最终交叉验证

三个独立来源，一致结论：**在我们这类拓扑上，P2P 对 decode 几乎无价值**：

| 来源 | 结论 |
|---|---|
| 4×5060Ti 基准（同模型类、同 Q8_0、同 MTP、无 P2P）| *"The limiter is collective latency, not the interconnect. PCIe runs only 15–20% utilized. A 2-card P2P A/B on a sister rig **changed nothing**."* |
| club-3090（4×3090，P2P 实测可用）| P2P 传输本身 = prefill +14.4% @10K，**decode 在噪声内** |
| 双 5090 P2P vs SHM 干净 A/B（2026-08）| **+2.67% 均值**（范围 -1.9% ~ +4.7%）|

**→ 我们花在 BAR1 上的所有努力，即使成功，decode 也只会涨个位数百分比。** 这反过来证明了拆掉交换机、转向调优是正确决定。
