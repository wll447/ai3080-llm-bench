# ai3080 硬件拓扑与问题全记录

**主机**: `ai3080` (`<TAILSCALE-IP>` / `<LAN-IP>`)
**整理日期**: 2026-09-24

---

## 一、硬件清单

### GPU

| 项 | 值 |
|---|---|
| 型号 | NVIDIA GeForce RTX 3080 |
| **显存** | **20480 MiB (20 GB)** ← 改装卡 |
| 板卡标识 | `A26R RTX 3080 20GB` |
| VBIOS | `94.02.27.00.14`（**2020-10-13，3080 首发版，早于 ReBAR 更新**）|
| Subsystem ID | `10DE 146D` |
| PCI Device ID | `10DE 2206` |
| 数量 | 4 张（bus `c8:00.0` / `cb:00.0` / `cf:00.0` / `d0:00.0`）|
| **BAR1** | **256 MiB**（`supported: 64MB 128MB 256MB`）|

### 平台

| 项 | 值 |
|---|---|
| CPU 平台 | AMD EPYC (Starship/Matisse) |
| Root Complex | 4 个：`00` / `40` / `80` / `c0` |
| 单 NUMA 节点 | 48 逻辑核，126 GB 内存 |
| 主板 BIOS | AMI 2.2 / **2026-01-06**（很新，不是固件老旧问题）|
| BMC/显卡 | ASPEED AST1150 (`d7:00.0`) |
| 网卡 | 2× Intel I226-V |
| NVMe | Lexar NM610 PRO |

### PCIe 交换卡

| 项 | 值 |
|---|---|
| 芯片 | **Broadcom / LSI PEX880xx PCIe Gen 4 Switch** |
| PCI ID | `1000:c010` |
| 端口 | 上行 1× Gen4 x16，下行 4× Gen4 x16 |

---

## 二、拓扑对比

### 情况 A：四卡经 PEX88096 交换卡（**当前**）

```
                   ┌──────────────────────────────────┐
                   │   AMD EPYC (Starship/Matisse)     │
                   │   Root Complex: 00 / 40 / 80 / c0 │
                   └──────────────┬───────────────────┘
                                  │
                        c0:03.1   │   Gen4 x16
                                  │
                   ┌──────────────▼───────────────────┐
                   │   Broadcom PEX880xx PCIe Gen4     │
                   │   Switch  (上行 x16  ← 共享瓶颈)   │
                   └──┬────────┬────────┬─────────────┘
              Gen4 x16│        │        │
        ┌─────────────┘        │        └─────────────┐
        │                      │                      │
   ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
   │ RTX3080 │            │ RTX3080 │            │ RTX3080 │
   │  20GB   │            │  20GB   │            │  20GB   │
   │  c8:00  │            │  cb:00  │            │  cf:00  │
   └─────────┘            └─────────┘            └─────────┘
                                                     ...
                   ┌─────────┐
                   │ RTX3080 │
                   │  20GB   │
                   │  d0:00  │
                   └─────────┘
```

**实测补充（2026-09-24 复核）**：

- **4 张卡全部挂在同一个 root complex `c0` 下**，且**全部**在**同一个上游口 `c0:03.1`** 后面
  （`c8:00.0` / `cb:00.0` / `cf:00.0` / `d0:00.0`；EPYC 的 `00` / `40` / `80` 三个 root complex 完全没用到）
- 上游 `c0:03.1`：`LnkSta: Speed 16GT/s, Width x16` → 唯一的 CPU 通道就是 **Gen4 x16 一条**

```
GPU0 c8:00.0  ← c7:10.0 ← c6:00.0 ┐
GPU1 cb:00.0  ← ca:00.0 ← c9:00.0 ├─ c5 (交换芯片内部) ─ c4:00.0 ─ c0:03.1 ─ CPU (Gen4 x16)
GPU2 cf:00.0  ← ce:00.0 ──────────┤                                    ← 4 卡共用这一条
GPU3 d0:00.0  ← ce:10.0 ──────────┘
```

**特点**：
- ✅ 拓扑变为 `PXB` / `PIX`（GPU 间走交换机，不绕 CPU 宿主桥）
- ✅ 这是 P2P 的**物理前提**
- ❌ **4 卡共享一条 x16 上行** → 聚合带宽被压（实测只有单卡的 1.07 倍）
- ❌ 卡到主机内存的路径多了一跳（延迟）
- ⚠️ **注意**：如果 P2P 真能用，GPU 间流量会留在交换芯片内部，**完全不碰**这条共享上行 —— 那时交换机几乎不花代价。它现在的代价是被"ReBAR 太小导致 P2P 不可用"**放大**出来的（详见 `ai3080-rebar-root-cause.md` 第 6 节）

### 情况 B：四卡直插主板（**80 t/s 那次的配置**）

```
                   ┌──────────────────────────────────┐
                   │   AMD EPYC (Starship/Matisse)     │
                   │   Root Complex: 00 / 40 / 80 / c0 │
                   └──┬────────┬────────┬─────────────┘
              Gen4 x16│        │        │
        ┌─────────────┘        │        └─────────────┐
        │                      │                      │
   ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
   │ RTX3080 │            │ RTX3080 │            │ RTX3080 │
   │  20GB   │            │  20GB   │            │  20GB   │
   └─────────┘            └─────────┘            └─────────┘
                                                     ...
                   ┌─────────┐
                   │ RTX3080 │
                   │  20GB   │
                   └─────────┘
```

**特点**：
- ❌ 拓扑是 `PHB`（走 CPU 宿主桥），P2P 物理上不通
- ❌ 更彻底：4 个插槽分属 EPYC 的 `00` / `40` / `80` / `c0` **四个不同 root complex**
  → 跨 root complex，`nvidia-smi topo` 会报 `NODE` / `SYS`，**P2P 在任何驱动下都不可能**
- ✅ **4 条相互独立的 x16**，无共享上行竞争
- ✅ 卡到主机内存路径更短（少一跳）

> ⚠️ 所以"拆掉交换机"并不是"回到更好的 P2P 配置"——它是**彻底放弃 P2P**，换取 4 倍的主机中转带宽。
> 在 P2P 本来就不可用的现实下，这笔交易是划算的；但要知道代价的性质。

---

## 三、实测性能对比

### 固件层：ReBAR 未生效的铁证（桥窗口）

| 桥 | 对应 GPU | Prefetchable 窗口 |
|---|---|---|
| `c6:00.0` / `c7:10.0` | `c8:00.0` (GPU0) | **288M** |
| `c9:00.0` / `ca:00.0` | `cb:00.0` (GPU1) | **288M** |
| `ce:00.0` | `cf:00.0` (GPU2) | **288M** |
| `ce:10.0` | `d0:00.0` (GPU3) | **288M** |

**288 MB = 256MB (BAR1) + 32MB (BAR3)**，一个字节都不多。
→ 固件**没有留任何增长空间**，这是 ReBAR 未生效的标准指纹。

同时 `c0:03.1` 的 prefetchable 地址在 `0x17fa0000000`（≈1.6TB）处
→ **Above 4G Decoding 是开着的**。所以这**不是**"Above 4G 没开"的问题。

> **Above 4G 打开 ≠ ReBAR 生效**，这是两件独立的事。

完整根因链（为什么 BAR1 只有 256MB）见 **`ai3080-rebar-root-cause.md`**。

### 卡到主机内存的带宽（经交换机实测）

| 测试 | 带宽 |
|---|---|
| **单卡** pinned 内存 | **24.5 GB/s** |
| **4 卡并发聚合** H2D | **26.2 GB/s** ← 没能 4 倍扩展 |
| **4 卡并发聚合** D2H | **12.1 GB/s** |
| 理论（4 条独立 x16）| ~98 GB/s |

**4 卡并发只有单卡的 1.07 倍** → 共享上行 x16 是硬瓶颈。

### 跨卡中转带宽（无 P2P 时的实际路径）

```
GPU i → 主机内存 → GPU j
 12 个方向平均 12.55 GB/s，延迟 2659 µs / 32 MB
```

### llama.cpp 推理性能

**统一命令**（除硬件外参数完全一致）：

```bash
NCCL_P2P_DISABLE=1 ./build/bin/llama-server \
  -m ../models/Qwen3.8-27B-Q8_0.gguf \
  -c 262144 -ngl -1 --split-mode tensor -ts 1,1,1,1 \
  -b 1024 -ub 512 -t 24 --threads-batch 48 -fa on \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --cont-batching --load-mode mmap --jinja --fit off
```

| 硬件 | prefill | decode | 数据来源 |
|---|---|---|---|
| **四卡直插主板** | **828-1000 t/s** | **77-84 t/s** | 你的历史记录 |
| **四卡经 PEX88096** | **327 t/s** | **44.6-45.9 t/s** | 本次实测 |

**差距：约 2 倍。**

---

## 四、遇到的问题（按时间线）

### 问题 1：GPU 完全不可用（已解决）

**现象**：4 卡全部初始化失败，`nvidia-smi` 报 `No devices found`

```
NVRM: GPU0 kgspWaitForGfwBootOk_TU102: failed to wait for GFW boot complete: 0x65
NVRM: GPU0 RmInitAdapter: Cannot initialize GSP firmware RM
NVRM: GPU 0000:d0:00.0: RmInitAdapter failed! (0x62:0x65:2119)
```

**根因**：手工编译的 610 模块与 apt 体系冲突 + GPU GSP 状态卡死

**解决**：清理手工模块 → 装发行版 595 → **冷重启**（只有冷重启能复位 GSP 坏状态）

### 问题 2：P2P 被驱动锁死（已解决授权层）

**现象**：`nvidia-smi topo -p2p rw` 全部 `CNS`

**根因**：NVIDIA 原版驱动的芯片组白名单

**解决**：换 aikitoria 魔改驱动 `610.57.04-p2p-v3` → 授权变全 `OK`

### 问题 3：P2P 授权了但映射失败（已解决）

**现象**：
```
cudaDeviceEnablePeerAccess → "mapping of buffer object failed"
NVRM: kbusSetP2PMailboxBar1Area_GM200: P2P mailbox area expected from RM
      but no valid address is installed
```

**根因**：驱动自身的一个条件判断不一致——`_kbusRequiresP2PMailboxBar1_GM107()`
的第一条件是 `(pcieP2PType != BAR1)`，而驱动默认值就是 `BAR1`，导致 **mailbox 永不分配**，
但 `p2p_api.c` 又要求 mailbox 必须存在。

**解决**：删掉那个条件（1 行补丁）→ 重新编译 → P2P 通了

```
cudaMemcpyPeer 跨卡：24.2 GB/s
```

### 问题 4：NCCL 集合通信挂死（**未解决**）

**现象**：
```
cudaMemcpyPeer       → ✅ 24.2 GB/s
ncclAllReduce        → ❌ 挂死（无报错、无超时）
llama.cpp tensor 模式 → ❌ 挂死
```

**根因**（已定位到硬件层，有外部文档佐证）：

```
BAR1 上限 = 256MB（VBIOS strap 索引 = 2；显存 20GB）
   ↓
aikitoria 补丁要求 static BAR1 mapping 覆盖整个帧缓冲 → 这条路不可达
   ↓
驱动退化成 BAR1 里的 mailbox 小窗口
   ↓
内核直接写对方显存被硬压到 2.67 GB/s
   （实测 1024/8192/32768 blocks 完全一样；本地写 711–848 GB/s）
   （ncu：对等写 12,917 cycles vs 本地 15.61 cycles）
   ↓
NCCL 集合通信还需要「GPU A 写标志到 GPU B，GPU B 内核自旋等」
256MB 窗口撑不住 → 第一次 all-reduce 永久挂死
```

**间接后果**：P2P 对推理**零收益**（外部 4×3090 实测：`-sm tensor` TP=2 61.5→64.2 / 73.5→72.2，噪声级）

**深挖记录**：`ai3080-rebar-root-cause.md`（为什么修不了 + straps 机制 + 唯一的理论绕过路径）

### 问题 5：内核 panic（**已恢复，血的教训**）

**现象**：机器完全挂死，SSH / BMC 全部不可达

**现场**：
```
BUG: Bad page map in process llama-server  pmd:800ba6c345a87cad
exc_general_protection+0x12e/0x410     → 内核 panic
```

**根因**：我用了**不带 NCCL 重新编译**的 llama.cpp + 4 卡 tensor 模式。
ggml 的内部 all-reduce 只支持 2 卡，4 卡退化到未测试的 `meta-backend butterfly`，
触发 ggml 的 VMM + P2P 已知 bug（源码里就有 `VMM Bug fix for P2P access` 注释）→ 页表损坏。

**这是一次严重失误** —— 日志里已经出现 `internal AllReduce init failed (n_devices != 2?)`
的警示，我应该停手。

---

## 五、当前状态

| 项 | 值 |
|---|---|
| 驱动 | 610.57.04（aikitoria p2p 补丁 + 我的 mailbox 修复）|
| GPU | 4 × RTX 3080 20480 MiB |
| **BAR1** | **256 MiB**（VBIOS strap 索引 = 2，见 `ai3080-rebar-root-cause.md`）|
| P2P 授权 | 全 `OK` |
| P2P 拷贝 | 24.2 GB/s ✅ |
| **NCCL 集合通信** | ❌ 挂死 |
| llama.cpp 生产配置 | `NCCL_P2P_DISABLE=1` + `--split-mode tensor` |
| 当前 decode | 44.6-45.9 t/s（经交换机）|

---

## 六、结论与建议

### 核心结论（2026-09-24 修正）

> **真正卡住一切的是 BAR1 = 256MB，不是交换机。**
>
> - 买 PEX88096 → 为了让 P2P 可用：**方向本身没错**（有真实交换芯片才能让 P2P 授权通过）
> - 但 P2P 最终不可用 → 卡在 **ReBAR 太小**，这是 VBIOS 固件门限，**无解**
> - 后果被放大 → P2P 不可用 ⇒ 跨卡流量全走主机内存 ⇒ 4 卡全挤在一条共享上行 x16 上
>
> 换句话说：**"ReBAR 太小"和"交换机拖后腿"是耦合的一个问题，不是两个独立问题。**
> 如果 P2P 能用，GPU 间流量会留在交换机内部，交换机几乎不花代价。

### 立刻可做，且不需要动硬件（按预期收益排序）

| # | 措施 | 为什么 |
|---|---|---|
| **1** | **把 `--split-mode tensor` 换成 `layer`** | 卡间只需传激活值（~7 MB/s），而不是**每层** all-reduce（~1400 MB/s）。你的 `config.ini` 里本来就是 `layer`，只有基准测试用了 `tensor`。**零成本，先试这个。** |
| **2** | **`-ts 1,1,0,0`，只用 2 卡** | 模型 29GB，2× 20GB 装得下。TP 参与者从 4 降到 2，共享上行流量直接减半以上 |
| **3** | **拆掉交换机**（你原本的计划） | 恢复 4 条独立 x16，主机中转聚合带宽从实测 26.2 GB/s → 理论 ~98 GB/s |

### ⚠️ 对比陷阱：decode 那一栏的证据目前不足

| | 历史直插 | 本次经交换机 |
|---|---|---|
| prefill | 828–1000 t/s | 327 t/s |
| decode | 77–84 t/s | 44.6–45.9 t/s |
| **MTP 接受率** | **0.91** | **0.325–0.43** |

投机解码（`--spec-type draft-mtp --spec-draft-n-max 5`）吞吐与接受率几乎成正比。
**接受率 0.91 → 0.35 这一项本身就能解释掉大部分 decode 差距**，与交换机无关。

- ✅ **prefill 828→327 可信** —— prefill 不吃投机解码
- ❌ **decode 77→45 归因于交换机，目前站不住**

复测必须**同时记录接受率**，且用相同 prompt / 相同上下文长度。

### 拆装前的注意事项

1. **先确认主板确实有 4 条独立 x16**（需要 CPU 提供 64 条 Gen4 通道）
2. 记录拆装前的基准数据（本次已测：327 / 45，接受率 0.325–0.43）
3. 拆装后用**完全相同的命令 + 相同 prompt** 对比，并记录接受率
4. 先做完上面第 1、2 条零成本优化再拆 —— 有可能一步就拿到大部分收益

### 复测方案（分三步，按成本从低到高）

**第 1 步（零成本，先做）** —— 只把 `--split-mode tensor` 改成 `layer`，其余一字不改：

```bash
NCCL_P2P_DISABLE=1 ./build/bin/llama-server \
  -m ../models/Qwen3.8-27B-Q8_0.gguf \
  -c 262144 -ngl -1 --split-mode layer -ts 1,1,1,1 \
  -b 1024 -ub 512 -t 24 --threads-batch 48 -fa on \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --cont-batching --load-mode mmap --jinja --fit off
```

**第 2 步** —— 只用 2 卡 TP：`--split-mode tensor -ts 1,1,0,0`

**第 3 步（需要动手）** —— 拆交换机，回到 4 卡直插，用**完全相同命令**复测以上配置

**每次都必须记录三项**：prefill t/s、decode t/s、**MTP 接受率**。

预期判读：

| 观察 | 结论 |
|---|---|
| `layer` 明显快于 `tensor` | 瓶颈是"每层 all-reduce 走主机内存"，与交换机关系不大，改配置即可 |
| 2 卡 TP 明显快于 4 卡 TP | 瓶颈是共享上行争用 → 拆交换机收益明确 |
| 拆装后回到 828-1000 / 77-84（**且接受率也回到 ~0.9**） | 交换机是元凶，确认 |
| 拆装后 prefill 回到 800+ 但 decode 仍 ~45 | decode 的差距是**接受率**造成的，与硬件无关 |

---

## 七、⛔ 禁用清单（血泪教训）

| 禁止操作 | 后果 |
|---|---|
| **不带 NCCL 的 llama.cpp + 4 卡 tensor** | ggml butterfly 路径 → **内核 panic** |
| `GGML_CUDA_P2P=1` 配上述组合 | 触发 VMM bug → Bad page map |
| `RMForceStaticBar1=1` | 打断 GPU 初始化（GSP 引导失败），需冷重启 |
| `RMPcieP2PType=0` | `CanAccessPeer` 变全 0，P2P 反而不可用 |
| 刷零售版 3080 VBIOS | 显存配置不同，**必砖** |
| 在 Tailscale 会话里跑 `netplan apply` | 短暂断网，可能掉 SSH |

---

## 附：关键配置文件

```
/etc/modprobe.d/nvidia-p2p.conf:
  options nvidia NVreg_EnableResizableBar=1
  options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
  options nvidia NVreg_DmaRemapPeerMmio=1

GRUB cmdline:
  amd_iommu=off iommu=off pcie_acs_override=downstream,multifunction
  pci=realloc=on hpmmioprefsize=512G pcie_aspm.policy=performance
```

## 附：备份与回退

| 内容 | 路径 |
|---|---|
| 补丁前源文件 | `/root/p2p-driver-patch-backup/kern_bus_gm107.c.orig` |
| 补丁说明 | `/root/p2p-fix-record/PATCH-NOTES.txt` |
| 打补丁源码树 | `/usr/src/nvidia-p2p-610.57.04/` |
| 原始 610 runfile | `/home/<user>/NVIDIA-Linux-x86_64-610.57.04.run` |
| 595 deb 缓存 | `/var/cache/apt/archives/*595*` |
