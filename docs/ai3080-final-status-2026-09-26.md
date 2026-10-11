# ai3080 最终状态与提速结论（2026-09-26 实测）

> 本文取代此前所有速度/故障结论。全部数据来自内核 `7.0.0-31-generic` + P2P 补丁驱动的**干净状态**（每次关键测量前已确认 `nvidia-smi topo -p2p` 全 OK、MTP 接受率正常）。

---

## 0. 一句话结论

**P2P 已经打通了。** 拆掉 PEX88096、4 卡直插主板后，驱动层 12 对 GPU 全部 P2P 可用。但 P2P 本身只带来 **+2~4%**，因为 Linux 上 llama.cpp 的 allreduce 默认就走 NCCL，瓶颈从来不在互连，而在**每个解码步的内核启动/同步开销**。

真正能立刻拿到的提速：

| 动作 | 收益 |
|---|---|
| `--spec-draft-n-max 3` → **`2`**（tensor 模式） | decode **+7.3%** |
| 长提示场景改用 `--split-mode layer --spec-draft-n-max 6` | prefill **+53%**，总时间 **+7.3%** |
| **永远不要设 `GGML_CUDA_P2P=1`** | 避免乱码 + 避免驱动被污染到必须重启 |

---

## 1. 故障根因：内核 7.0.0-34 + 驱动组合会静默损坏多卡数据

### 现象
4 卡 tensor 模式推理输出乱码（`//////`、`mselandorraasc_HANDL`），单卡正常，且与 llama.cpp 版本、split 模式、`-fa`、KV 类型、MTP 开关全部无关。

### 排查结论
```
内核 7.0.0-31 + 原始补丁驱动 → 'Paris'  51  Jupiter   3/3 全对 ✅
内核 7.0.0-34 + 同一份源码重编 → '//////' 乱码 ❌
```

### 为什么不是"编坏了"
```
kernel-open 全部 211 个 .o 时间戳 = 同一批（2026-09-24 06:54）  → 全新构建，无残留污染
GCC：两个内核都是 15.2.0（CONFIG_GCC_VERSION=150200）
关键 config：完全一致（MODVERSIONS / MODULE_SIG / SMP / PREEMPT / P2PDMA / ZONE_DEVICE ...）
srcversion：完全相同（F110D2947200966FBA4D579）
```

**同一份源码 + 同一个编译器 + 同一份 config，干净重编 → 在 31 上正常，在 34 上损坏数据。** 即内核 7.0.0-34 与 NVIDIA 610.57.04 不兼容。31→34 是 Ubuntu HWE 自动升级引进来的。

### 已实施的修复
```bash
# /etc/default/grub
GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 7.0.0-31-generic"
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=10
# 然后必须 update-grub，否则改 GRUB_DEFAULT 不生效！

# 防止 autoremove 删掉唯一可用的内核
apt-mark hold linux-image-7.0.0-31-generic linux-modules-7.0.0-31-generic \
              linux-headers-7.0.0-31-generic linux-headers-7.0.0-31
```

已验证：重启后自动回到 `7.0.0-31-generic`。

> **注意**：`/lib/modules/7.0.0-34-generic/` 下的 nvidia 模块仍然存在且是坏的。只要不启动 34 就无害；`linux-image-generic` 元包仍指向 34，将来 `apt upgrade` + 手工选 GRUB 项会踩雷。

---

## 2. P2P 现在是通的（硬件层证据齐全）

### nvidia-smi 拓扑能力
```
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3
 GPU0    X      OK      OK      OK
 GPU1    OK      X      OK      OK
 GPU2    OK      OK      X      OK
 GPU3    OK      OK      OK      X
```
全部 `OK`，不是 `CNS`/`NS`/`DR`。（拓扑矩阵显示为 `NODE`，即经主板 PCIe Host Bridge，同 NUMA 节点，交换机已移除。）

### CUDA 层证据
```
cudaDeviceCanAccessPeer   : 12/12 组全部 YES
cudaDeviceEnablePeerAccess: 12/12 组全部 OK
P2P 原子操作              : 12/12 组全部 OK
```

### 但收益很小
```
P2P-ON  (允许 peer copy) : prefill 876.40 t/s | decode 83.82 t/s | AL 2.65
P2P-OFF (NO_PEER_COPY)   : prefill 845.53 t/s | decode 82.20 t/s | AL 2.65
```
即 **prefill +3.7% / decode +2.0%**。

**原因**：`ggml/src/ggml-cuda/ggml-cuda.cu` 中
```c
const char * env = getenv("GGML_CUDA_ALLREDUCE");
if (!env) {
    // Platform default: Linux uses NCCL
#if defined(__linux__)
    ggml_backend_cuda_comm_init_nccl(ret);
```
Linux 默认就是 NCCL；而构建里 `GGML_CUDA_NCCL=ON` 使 `use_peer_access = true`，VMM 池本来就做了 `cuMemSetAccess` 授予 peer 访问。所以 NCCL 早已在用 P2P。

> 附带结论：`NCCL_P2P_DISABLE=1` 现在既无必要也无害（实测 866.58 vs 867.20，噪声范围）。以前必须加它是因为 P2P 本来就不通。

---

## 3. ⚠️ `GGML_CUDA_P2P=1` 是毒药（有硬 A/B 证据）

### 3.1 它直接导致乱码
```
1-base NCCL_P2P_DISABLE=1    : 866.58 / 83.48  'Paris'                ✅
2-NCCL 默认 (允许 P2P)       : 867.20 / 82.97  'Paris'                ✅
3-internal allreduce         : 659.07 / 56.67  'Paris'                ← 慢 24%
4-internal + GGML_CUDA_P2P=1 : 840.12 / 30.50  'mselandorraasc_HANDL'  ❌ 乱码
```

### 3.2 它还会持久污染驱动状态，必须重启才能恢复
```
设 GGML_CUDA_P2P=1 跑过一次之后：
   所有后续 llama.cpp 进程（即使不带该变量）→ MTP 接受率归零、decode 掉 60%
   重复测量 5 次，AL 恒为 0.00000
   （3 次 tensor n=2：33.50 / 33.70 / 33.03 t/s，均 0 accepted / 507 generated）

重启后立刻恢复：
   第1次: prefill 818.61 | decode 82.88 | AL 0.82292 (158/192) len 2.65  ✅
   第2次: prefill 878.35 | decode 83.16 | AL 0.82292 (158/192) len 2.65  ✅
```

**这解释了历史上反复出现的"输出乱码"和"怎么调都慢"两类症状。** 只跑一次就中毒，且症状会延续到之后所有测试，极易误判为"换卡/换配置也没用"。

### 3.3 注意
用户环境（`.bashrc` / `.profile` / `/etc/environment` / `/etc/profile.d/`）**没有**默认导出 `GGML_CUDA_P2P`。但在内核 34 上，驱动自身授予 peer 映射的方式可能等价于此，所以 34 也出现乱码。**结论不变：31 是唯一安全的内核，`GGML_CUDA_P2P` 永远不要设。**

---

## 4. 硬件本身很快 —— 排除"卡不行"

自写 CUDA 带宽测试（1 GiB，20 次迭代，事件计时，`cudaMemcpyAsync` 避免同步偏差）：

```
每卡纯读 : 724.4 GB/s    ← 理论峰值 9501 MHz x2 = 19 Gbps, 320-bit -> 760 GB/s，达成率 95%
每卡纯写 : 720.3 GB/s
D2D 双向 : 686.5 GB/s
```

魔改 20GB 显存健康，带宽满血。**所以 decode 慢不是卡的问题。**

---

## 5. 干净状态完整速度矩阵

统一条件：`Qwen3.8-27B-Q8_0.gguf`，4001 token 提示，256 token 输出（`ignore_eos`），`-c 262144 -b 1024 -ub 512 -fa on --cache-type-k/v f16`，MTP。

### 5.1 tensor 模式 `n_max` 扫描
| n_max | prefill | decode | AL | 接受率 |
|---|---|---|---|---|
| 0（关不掉，见 5.5） | 821.53 | 32.40 | 1.88 | 0.882 (112/127) |
| 1 | 828.82 | 71.86 | 1.90 | 0.903 (121/134) |
| **2** | **813.64** | **82.40** | **2.65** | 0.823 (158/192) |
| 3（原配置） | 815.31 | 76.81 | 2.76 | 0.587 (162/276) |
| 4 | 827.91 | 74.17 | 3.00 | 0.503 (170/338) |

**tensor 模式最优 = n_max=2。** 用户原来的 n_max=3 白白损失 7.3%。

### 5.2 layer 模式 `n_max` 扫描（GPU 采样，验证近乎免费）
| n_max | prefill | decode | AL | 接受率 |
|---|---|---|---|---|
| 2 | 1244.96 | 42.54 | 2.34 | 0.673 (146/217) |
| 4 | 1246.42 | 55.55 | 3.54 | 0.642 (183/285) |
| **6** | **1258.86** | **57.31** | **4.18** | 0.540 (194/359) |
| 8 | 1245.78 | 49.40 | 4.25 | 0.416 (195/469) |
| 12 | 1248.15 | 46.11 | 4.64 | 0.312 (200/642) |

**layer 模式最优 = n_max=6。**

### 5.3 关掉 MTP 的对照
| 模式 | prefill | decode |
|---|---|---|
| tensor 无 MTP | 949.60 | 52.92 |
| layer 无 MTP | 1996.10 | 23.12 |

**MTP 的增益远大于历史记录**：tensor 52.92 → 82.40 = **+56%**；layer 23.12 → 57.31 = **+148%**。（此前记录的 +13.7% 是在被污染状态下测的。）

### 5.4 总时间对比（4001 token 提示 + 256 token 输出）
| 配置 | prefill | decode | 合计 |
|---|---|---|---|
| tensor n_max=2 | 4917 ms | 3095 ms | 8012 ms |
| **layer n_max=6** | **3178 ms** | 4449 ms | **7627 ms** ← 最快 |
| tensor n_max=3（原配置） | 4907 ms | 3320 ms | 8227 ms |
| layer n_max=4 | 3214 ms | 5995 ms | 9209 ms |

**交叉点：提示 token 数 > 输出 token 数的 12 倍时，layer 模式反超。** 这台机器是 262144 上下文的服务器，长文档 / RAG 场景下 layer 优势会急剧放大。

### 5.5 `--spec-draft-n-max 0` 不能关闭 MTP
`config.ini` 里的 `spec-type = draft-mtp` 属于**列表型参数**，CLI 的 `--spec-type none` 会被**追加**而不是覆盖，所以关不掉。传 `n_max=0` 也只是退化成 n_max=1 的畸形状态（32.40 t/s）。**要真正关闭 MTP 必须移走 `config.ini`。**

### 5.6 `-bs / --backend-sampling` 是陷阱
| 配置 | prefill | decode |
|---|---|---|
| layer n=6 + `-bs` | **781.08**（−37%） | 58.65 |
| layer n=6 无 `-bs` | 1248.71 | 57.18 |
| tensor n=2 + `-bs` | 830.55 | 82.07（仍退回 CPU 采样） |

只换来 decode +2.6%，prefill 却掉 37%。**净亏损，不要用。**

### 5.7 `--split-mode row` 不可用
`failed to load model` —— 该模型的权重布局不支持 row 切分。

### 5.8 `GGML_CUDA_ALLREDUCE=internal` 在 4 卡上不可用
```
W internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
```
这个构建的 internal allreduce 流水线**只支持正好 2 个设备**，4 卡会退回低速 butterfly：659.07 / 56.67（比 NCCL 慢 24%）。**4 卡保持默认（NCCL）。**

---

## 6. 瓶颈定位（数据推导）

| 指标 | 数值 | 含义 |
|---|---|---|
| tensor 无 MTP 单 pass | 18.90 ms | 每卡读 7.28 GB → **385 GB/s = 53% 峰值** |
| layer 无 MTP 单 token | 43.26 ms | 4 卡顺序，每卡 7.28 GB → 每卡 ~674 GB/s ≈ **93% 峰值**（纯带宽受限） |
| tensor MTP 每步 | 32.08 ms | = 主 pass 18.90 + 2×草稿 6.6 ms |
| layer MTP 每步 | 72.9 ms | = 主 pass 43.26 + 6×草稿 4.9 ms |

**关键发现**：

1. **layer 模式是纯带宽受限的顺序流水线** —— 一次 verify 5 个 token 和 verify 1 个 token 读的权重完全一样，所以**验证几乎免费**，这就是 layer 需要往大 `n_max` 走的原因。

2. **tensor 模式是计算受限** —— batch 越大越贵，所以 `n_max=2` 最优。

3. **每个草稿 pass 恒定花费 ~5-6.6 ms**，且**与 split 模式无关**（tensor 6.6ms、layer 4.9ms）。而 MTP 模块只有 1/65 的模型权重，理论成本应 <0.2 ms。**这 5ms 是内核启动 + 每步同步的固定开销，不是采样、不是带宽。** 已用 `-bs` 实验排除采样假说。

**理论上限**：若草稿开销能降到接近 0，tensor 可达 18.90/2.65 = 7.13 ms/token ≈ **140 t/s**；layer 可达 **~93 t/s** 且 prefill 保持 1250。这需要改 llama.cpp 源码（把草稿/验证 pass 纳入 CUDA Graph 捕获），属于 PR 范畴，已按用户要求不做。

---

## 7. 推荐命令

### 7.1 通用 / 交互优先（decode 优先，与用户原命令差异最小）
```bash
NCCL_P2P_DISABLE=1 ./build/bin/llama-server \
  -m /home/<user>/models/Qwen3.8-27B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 262144 -ngl -1 \
  --split-mode tensor -ts 1,1,1,1 -b 1024 -ub 512 -t 24 --threads-batch 48 \
  -fa on --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --cont-batching --load-mode mmap --jinja --fit off
```
→ **prefill ~815 / decode ~82.4**（原 n_max=3 是 76.8，**+7.3%**）

### 7.2 长提示 / RAG 优先（总吞吐优先）
同上去掉 `--split-type` 部分，改：
```
--split-mode layer -ts 1,1,1,1 ... --spec-type draft-mtp --spec-draft-n-max 6
```
→ **prefill ~1259（+53%）/ decode ~57.3**，总时间 7627 ms（比 7.1 快 5%，提示越长优势越大）

### 7.3 绝不要用的选项
| 选项 | 后果 |
|---|---|
| `GGML_CUDA_P2P=1` | 输出乱码 + 驱动中毒直到重启 |
| `-bs` / `--backend-sampling` | prefill −37% |
| `--split-mode row` | 模型加载失败 |
| `GGML_CUDA_ALLREDUCE=internal`（4 卡） | 退回 butterfly，慢 24% |
| 内核 `7.0.0-34` | 静默数据损坏 |
| `--spec-draft-n-max 3`（tensor） | 比 2 慢 7.3% |

---

## 8. `config.ini` 覆盖陷阱（务必知悉）

`/home/<user>/.config/llama.cpp/config.ini` 会被自动加载：
```ini
[*]
model            = /home/<user>/models/Qwen3.8-27B-Q8_0.gguf
mmproj           = /home/<user>/models/mmproj-Qwen3.8-27B-f16.gguf
split-mode       = layer          ← 配置文件是 layer
tensor-split     = 1,1,1,1
batch-size       = 2048
ubatch-size      = 1024
cache-type-k/v   = q8_0
spec-type        = draft-mtp      ← 列表型，CLI 追加而非覆盖，关不掉
spec-draft-n-max = 4
image-min-tokens = 1024
```

- **标量型参数**（如 `--spec-draft-n-max`）：CLI 覆盖配置文件 ✅
- **列表型参数**（如 `--spec-type`）：CLI **追加**到配置文件的值上，**无法用 CLI 关闭** ❌

本文所有测量都显式传了 CLI 标量参数，因此不受 `spec-draft-n-max = 4` 影响；`spec-type = draft-mtp` 则始终生效。

---

## 9. 本次会话的机器状态变更汇总

| 项目 | 变更 |
|---|---|
| `/etc/default/grub` | `GRUB_DEFAULT` 指向内核 31，`GRUB_TIMEOUT_STYLE=menu`，`GRUB_TIMEOUT=10` |
| `/boot/grub/grub.cfg` | 已 `update-grub` 重新生成 |
| apt hold | `linux-image/modules/headers-7.0.0-31*` 已 hold |
| 内核 | 固定 `7.0.0-31-generic`，已验证重启后生效 |
| llama.cpp | `/home/<user>/llama.cpp` 主线 **81bc6b8**（`0.5.0-dev build 333`），`git status` 干净无本地改动 |
| 构建产物 | `build/`（P2P peer copy 开）、`build-nopeer/`（`GGML_CUDA_NO_PEER_COPY=ON`）、`build/bin/llama-bench`（本次补编） |
| 清理 | ik_llama.cpp、build-28549、Q4 量化、小模型均已删除 |
| 磁盘 | 68G 已用 / 154G 可用 |

**`/home/<user>/llama-backup-304665f/`（77 MB 旧二进制备份）已可安全删除** —— 主线构建已验证通过且性能更好。
