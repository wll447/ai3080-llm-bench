# ai3080：ReBAR / BAR1 为什么卡在 256MB —— 完整根因链

**主机**: `ai3080` (`<TAILSCALE-IP>`)
**日期**: 2026-09-24
**状态**: 根因已定位，附证据链与外部出处

---

## 0. 一句话结论

> **BAR1 上限不是 VBIOS 里的一张表，也不是硅片限制，而是 GPU 硬件「straps」位在 PCIe bring-up 时被 VBIOS 写入的一个 5 bit 索引。**
>
> 你这批卡的 VBIOS 是 **`94.02.27.00.14` / 2020-10-13**——RTX 3080 的**首发版**，早于 NVIDIA 2021 年才发布的 ReBAR VBIOS 更新。
> 该 VBIOS 写下的 strap 索引 = **2**，按 envytools 的表就是 **256MB**。
>
> 所以 `lspci` 只能报出 `64MB 128MB 256MB`，没有更大的尺寸可选。

同核心反证：GA102 的 RTX 3090（ReBAR 版 VBIOS）和 RTX A5000 能报出 24–32GB。
**硅片完全支持，差别只在 VBIOS 写进去的那个数字。** 这不是"核心不支持"，不是 GSP 固件，不是驱动，不是主板 BIOS。

---

## 1. 证据一：四张卡的 ReBAR capability（实测）

```
Capabilities: [bb0 v1] Physical Resizable BAR
    BAR 0: current size: 16MB,  supported: 16MB
    BAR 1: current size: 256MB, supported: 64MB 128MB 256MB
    BAR 3: current size: 32MB,  supported: 32MB
```

`c8:00.0` / `cb:00.0` / `cf:00.0` / `d0:00.0` —— **四张完全一致**。
BAR1 上限就是 256MB，supported 列表里根本不存在 512MB / 1GB / … 这些选项。

---

## 2. 证据二：上游桥窗口（决定性）

club-3090 项目给的判据是：**"桥窗口才是真正的诊断，不是 BAR 本身。窗口刚好等于 `BAR1 + BAR3`，就意味着固件没有留任何增长空间——这就是 ReBAR 未生效的样子。"**

实测你这台机器，每个 GPU 前面的桥：

| 桥 | 对应 GPU | Prefetchable 窗口 |
|---|---|---|
| `c6:00.0` / `c7:10.0` | `c8:00.0` (GPU0) | **288M** |
| `c9:00.0` / `ca:00.0` | `cb:00.0` (GPU1) | **288M** |
| `ce:00.0` | `cf:00.0` (GPU2) | **288M** |
| `ce:10.0` | `d0:00.0` (GPU3) | **288M** |

**288 MB = 256MB (BAR1) + 32MB (BAR3)** —— 一个字节都不多。**四张卡全部命中"ReBAR 未生效"的指纹。**

### 同时可以排除"Above 4G 没开"

- 交换机上行口 `c0:03.1`：`LnkSta: Speed 16GT/s, Width x16`（Gen4 x16 正常）
- 所有 prefetchable 地址在 `0x17fa0000000` ≈ **1.6 TB** 处 → **Above 4G Decoding 明确是开着的**

> ⚠️ **Above 4G 打开 ≠ ReBAR 生效。** 这是两件独立的事。你这台机器是"Above 4G 正常、ReBAR 无效"的典型。

---

## 3. 证据三：envytools 的 straps 定义（硬件机制出处）

来源：[envytools — PSTRAPS: straps readout and override](https://envytools.readthedocs.io/en/latest/hw/io/pstraps.html)

GPU 有一组 "straps" 位，在复位后的最初几个周期由板上的上/下拉电阻采样，用来告诉 GPU 自己处于什么配置（显存容量、总线类型、**BAR 尺寸**…）。

- 寄存器区：**`PSTRAPS` @ `BAR0 + 0x101000`**（也叫 PEXTDEV）
- **G80+ 系列的 BAR1 尺寸编码：**
  - Set 0 `bits 14-15` = BAR1 size 第 1 部分
  - Set 1 `bits 20-22` = BAR1 size 第 2 部分
  - **两段相加**得索引，查表：

| 索引 | BAR1 | 索引 | BAR1 |
|---|---|---|---|
| 0 | 64MB | 6 | 4GB |
| 1 | 128MB | 7 | 8GB |
| **2** | **256MB** ← 你的卡 | 8 | 16GB |
| 3 | 512MB | 9 | 32GB |
| 4 | 1GB | 10 | 64GB |
| 5 | 2GB | | |

- `STRAPS0_PRIMARY` / `STRAPS1_PRIMARY` 的 **bit 31 = override enable**：置 1 后可由主机改写 straps 值；置 0 则恢复复位时的原始值。

**你的卡索引 = 2 ⇒ 256MB。和实测精确吻合。**

---

## 4. 证据四：NvStrapsReBar 的证词（为什么 ROM 里搜不到）

来源：[terminatorul/NvStrapsReBar](https://github.com/terminatorul/NvStrapsReBar)

> "It's usually the video BIOS (vBIOS) that should enable ReBAR, but the **vBIOS is digitally signed and can not be modified by end-users (locked-down)**."
>
> "@Xelafic for the first code samples (written in assembly!) and the first test for **using the GPU STRAPS bits**, documented by envytools, **to select the BAR size during PCIe bring-up in UEFI code**."

这两段解释了两个一直没想通的事：

1. **为什么我在 VBIOS ROM 里搜 `0x0015`（Resizable BAR capability）结构，结果是零命中。**
   因为**这个 capability 不是 ROM 里的一张静态表**，而是 GPU 硬件按 strap 值**生成**的。ROM 里只有那个数字（strap 位），没有"列表"。
2. **为什么 NVIDIA 必须发新版签名 VBIOS 才能开 RTX 30 的 ReBAR。**
   因为要改的就是 VBIOS 在设备初始化时写下的 strap 值。

> 这也解释了项目名：`NvStrapsReBar` = "通过写 GPU **straps** 来开 **ReBAR**"。

---

## 5. 为什么"改不了"

| 尝试路径 | 结果 | 原因 |
|---|---|---|
| 刷改过的 VBIOS | ❌ | **注意：不是签名挡住的**（见下方更正）。挡住你的是"不知道改哪几个字节" + 显存配置必须匹配 |

> ### ⚠️ 更正（2026-09-24，重要）
>
> 本文早先把"VBIOS 签名"当作主要障碍。**这对 GeForce 消费卡是错的。**
>
> - **2023 年 8 月起**，社区工具 **OMGVflash**（Veii）和 **nvflashk**（Kefinator）**已绕过 NVIDIA 的固件签名检查**，覆盖 RTX 20 / 30 / 40 系。nvflashk 的描述是"会尝试刷任何 BIOS，哪怕从 3060 刷到 4090"。
>   - <https://github.com/notfromstatefarm/nvflashk>
>   - <https://www.techpowerup.com/312631/>
> - 更直接的反证：**这批卡现在跑的就是改装固件** —— 官方从来没出过 20GB 的 RTX 3080。既然它能存在，就说明**改装固件能刷进去**。
> - **MAC / `csecret(2)` 那套严格校验是 GA100（A100 / CMP 数据中心卡）的情况**，第 13 节讲的都是 GA100。把它套到 GA10x 消费卡上不成立。
>
> **修正后的真实障碍：**
>
> | 障碍 | 说明 |
> |---|---|
> | 不知道改哪几个字节 | ReBAR 不是固件里的一张表，是启动时写进 GPU 的一个 strap 值。GA102 上这个值的位位置**无公开文档**（envytools 只编到 Kepler）|
> | 缺少可比对样本 | 需要"20GB + ReBAR"固件，或拿"零售 10GB ReBAR"固件与你的 20GB 固件做三方 diff；而 20GB 改装让差异遍布全篇，信噪比极差 |
> | 显存配置必须匹配 | 刷 3090（24GB/384-bit）或零售 3080（10GB/320-bit）的固件 → **必砖**，因为它们含各自的内存训练表与容量声明 |
> | 砖机成本 ×4 | 刷错要拆机 + SPI 编程器救援；你有 4 张卡 |


| 主板 BIOS 里开 "Re-Size BAR" | ❌ 无效 | GPU 只声明支持到 256MB，**没有更大的尺寸可选**。开关打开也变不出来 |
| `setpci` 直接写 Resizable BAR 控制寄存器 | ❌ 被忽略 | **现在能解释了**：写一个不在 supported 列表里的尺寸，PCIe 规范要求设备忽略。256MB 已是上限，"改成 512MB"必然被丢弃 |
| `NVreg_EnableResizableBar=1` / `RMForceStaticBar1=1` | ❌ | `RMForceStaticBar1` 要求"BAR1 足以一次性映射整个 FB，否则初始化失败"——256MB 装不下 20GB，所以它直接打断 GSP 引导（已实测两次，需冷重启） |
| 换 ReBAR 版 3080 VBIOS | ⛔ **必砖** | 这是 20GB 改装卡（板号 `A26R`），零售版 3080 VBIOS 的显存配置不同 |

### 唯一存在的用户侧绕过路径（研究级，**未在 Ampere 上验证**）

不放 VBIOS，改从**主板固件**下手：

> 做一个 **UEFI DXE 驱动**，在 PCI 枚举阶段（系统驱动看到设备之前）**直接写 GPU 的 strap 寄存器**
> —— `STRAPS*_PRIMARY` 置 bit 31 打开 override，再写入新的 BAR1 size 索引。

- 该项目**只发布了 Turing 版（GTX 1600 / RTX 2000）**。Ampere 没做，因为 Ampere 有官方 VBIOS。
- 要在 GA102 上做，先得**只读**探测 `BAR0 + 0x101000`，确认 Ampere 是否仍是 envytools 记录的那套布局。
- **风险**：GA102 的 strap 布局无公开文档；对未文档化的 MMIO 写错，会挂卡甚至需要拆机。
- **当前不建议动。**

---

## 6. ReBAR 不够大 → 到底造成了什么（因果链接到性能）

这一段是重点，把"固件问题"和"推理慢 2 倍"接起来：

```
① VBIOS 是 2020 首发版，没写 ReBAR strap
        ↓
② BAR1 上限 = 256MB（显存 20GB）
        ↓
③ aikitoria 补丁的 P2P 路径要求 static BAR1 mapping 覆盖整个帧缓冲
   —— 256MB 装不下 20GB，这条路直接不可达
        ↓
④ 驱动退化成 BAR1 里的 mailbox 小窗口
   （就是那个已修好的 kbusSetP2PMailboxBar1Area_GM200）
        ↓
⑤ 内核「直接写对方显存」被硬压到 2.67 GB/s
   （实测：1024 / 8192 / 32768 blocks 全部一样；本地写 711–848 GB/s）
   ncu 佐证：对等写 12,917 cycles vs 本地 15.61 cycles
        ↓
⑥ NCCL 集合通信额外需要「GPU A 写标志到 GPU B，GPU B 内核自旋等待」
   256MB 窗口撑不住 → 第一次 all-reduce 就永久挂死（实测 + 外部同样现象）
        ↓
⑦ 只能 NCCL_P2P_DISABLE=1 → 跨卡流量全部经主机内存中转
   （实测 12.55 GB/s，2659 µs / 32MB）
        ↓
⑧ 经主机中转的每一个字节，都要穿过交换机那条
   被 4 张卡共享的 Gen4 x16 上行
   （实测：4 卡并发聚合 H2D 仅 26.2 GB/s ≈ 单卡 24.5 GB/s 的 1.07 倍）
        ↓
⑨ 结果：prefill 828 → 327 t/s
```

### 关键推论（这条改变了结论的性质）

> **交换机的代价，是被"ReBAR 卡死"这件事放大出来的。**

- 如果 **P2P 能用**：GPU 间流量全部**留在交换机内部**，根本不碰那条共享上行 → 交换机几乎不花你钱。
- 现实是 **P2P 不能用**：所有跨卡流量被迫绕主机内存 → 4 卡的流量**全部挤在一条 x16 上**。

所以"ReBAR 太小"和"交换机拖后腿"**不是两个独立问题，是耦合的**。你说的"88096 拖后腿"结论方向对，但机制不是"链路长了"这种模糊说法。

---

## 7. 修正后的结论与建议

### 7.1 前提修正：修好 P2P 对 llama.cpp 也几乎零收益

外部 4× RTX 3090 项目的实测（[club-3090 / PCIE_P2P §6](https://github.com/noonghunna/club-3090/blob/master/docs/PCIE_P2P.md)）：

| split mode | 实测（P2P 开 vs 关） |
|---|---|
| `-sm layer` (PP) | **+0.2% decode**, ~+1% prefill |
| `-sm tensor` TP=2 | 61.5→64.2（叙述）/ 73.5→72.2（代码）—— **噪声级** |

> "On the shipped llama.cpp image, patched P2P buys essentially nothing at any split mode."
> 原因：`layer` 模式卡间只传激活值（~7 MB/s），没有 all-reduce 可加速；`tensor` 模式虽有小消息 all-reduce，但收益被别的瓶颈吃掉。

**所以「修 ReBAR 来提升 llama.cpp」这条路本身价值极低。** 这基本是终点。

> ### ⭐ 最强的一条论证：3090 就是你想做的那个实验的对照组
>
> 你的目标是"**让 3080 表现得像 3090 一样**"（BAR1 够大、P2P 能用）。
>
> **而 3090 本身已经被测过了。** 4× RTX 3090，P2P 实测可用 13.17 GB/s，结果：
>
> | split mode | P2P 开 vs 关 |
> |---|---|
> | `-sm layer` | **+0.2%** decode · ~+1% prefill |
> | `-sm tensor` TP=2 | 61.5→64.2 / 73.5→72.2 —— **噪声级** |
>
> **原因**：llama.cpp 单流推理只跑 ~2 GB/s，而链路有 13 GB/s。
> **它根本不是互联带宽受限的负载 —— 给它修路，它也不走。**
>
> → 所以：**即使你把 BAR1 从 256MB 变成 20GB、让 P2P 完全跑通，最好的结果也是 llama.cpp 一动不动。**
> 这个方向的期望收益 ≈ 0，而代价是 4 张卡的砖机风险。

### 7.2 立刻可做、不需要硬件改动（按预期收益排序）

| # | 措施 | 为什么 |
|---|---|---|
| **1** | **`--split-mode layer`（PP）替代 `tensor`** | 卡间只需传激活值（~7 MB/s），而不是**每层** all-reduce（~1400 MB/s）。在"被迫走主机内存 + 共享上行"的环境里，这是最大的一刀。**你的 `config.ini` 里本来就是 `layer`；基准测试用的是 `tensor`。** |
| **2** | **减少参与 TP 的卡数**：`-ts 1,1,0,0` | 模型 29GB，2× 20GB 装得下。TP=2 的 all-reduce 参与者从 4 降到 2，共享上行上的流量直接减半以上 |
| **3** | **拆掉 PEX88096** | 恢复 4 条独立 x16，主机中转聚合带宽从实测 26.2 GB/s 升到理论 ~98 GB/s |

第 3 条是你原本的计划，**方向是对的**——但请先做第 1 条，因为它零成本，可能一步就拿到大部分收益。

### 7.3 ⚠️ 对比时的陷阱（必须承认）

**decode 那一栏的对比目前证据不足：**

| | 历史直插 | 本次经交换机 |
|---|---|---|
| prefill | 828–1000 t/s | 327 t/s |
| decode | 77–84 t/s | 44.6–45.9 t/s |
| **MTP 接受率** | **0.91** | **0.325–0.43** |

投机解码（`--spec-type draft-mtp --spec-draft-n-max 5`）的吞吐与接受率几乎成正比。**接受率从 0.91 掉到 0.35，这一项本身就能解释掉大部分 decode 差距**，与交换机无关。

- **prefill 的 828→327 更可信** —— prefill 不吃投机解码。
- **decode 的 77→45 归因于交换机，目前是站不住的。**

复测时必须**同时记录接受率**，且用相同 prompt / 相同上下文长度。

---

## 8. 复测方案

**步骤 1（零成本，先做）**：把 `--split-mode tensor` 换成 `layer`（其余参数一字不改），跑同一个 prompt。

**步骤 2**：`-ts 1,1,0,0` 只用 2 卡跑 TP=2。

**步骤 3（需要动手）**：拆交换机，回到 4 卡直插，用**完全相同命令**复测三个配置。

每次都要记录：prefill t/s、decode t/s、**MTP 接受率**。

```bash
# 每次测完顺手取一下 BAR1 / 链路状态，确认前提没变
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi -q -d MEMORY | grep -iE "BAR1"
```

> 📌 顺带说明：**空闲时** `lspci` 会显示 `LnkSta: Speed 2.5GT/s (downgraded)`，`nvidia-smi` 也报 `gen.current = 1`。
> 这是 GPU 空闲降速省电的**正常行为**，不是故障 —— 单卡 H2D 实测 24.5 GB/s 已经证明负载下会跑到 Gen4 x16。要在负载下看才对。

---

## 9. 出处

| 内容 | 来源 |
|---|---|
| straps / BAR1 size 编码表、`PSTRAPS @ BAR0+0x101000`、override bit 31 | [envytools — PSTRAPS](https://envytools.readthedocs.io/en/latest/hw/io/pstraps.html) |
| "vBIOS 是签名锁死的"、"用 GPU STRAPS 位在 PCIe bring-up 时选 BAR 尺寸" | [terminatorul/NvStrapsReBar](https://github.com/terminatorul/NvStrapsReBar) |
| "桥窗口才是真正的诊断"、"256MB = pre-ReBAR VBIOS 固件门限"、"P2P 在 llama.cpp 上收益≈0"、"NCCL collective 挂死 → `NCCL_P2P_DISABLE=1`" | [noonghunna/club-3090 — PCIE_P2P.md](https://github.com/noonghunna/club-3090/blob/master/docs/PCIE_P2P.md) |
| GA100 上同样现象（拷贝通、集合通信挂）+ 其解释 | [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) |
| ReBAR capability 在 `[bb0 v1]`、BAR0/1/3 = 16/64/32MB、"advertised but functionally inert" | cmp170hx `docs/hardware/pcie-subsystem.md`（本地存档 `cmp170hx-pcie-subsystem-doc.md`）|
| DevInit 区域 `0x33800`-`0x41200`、MAC 区 `0x2200`-`0x43A00`、"every DEVINIT modification breaks the Booter MAC" | cmp170hx `docs/hardware/vbios.md`（本地存档 `cmp170hx-vbios-doc.md`）|
| **`_nv000358rm` 在 `0x264fb1` 读 strap 寄存器 `0x101000`** | 同上（RM 侧 DEVINIT 解释器）|
| "A latch PHY capability is not [writable]"、`SUPPORTED_LINK_SPEED` "No write port at any privilege" | cmp170hx `docs/hardware/pcie-subsystem.md` |
| "Board straps … 'the straps don't do anything', 'falcon is driving the rewrites'" | 同上（"看着像原因、其实不是"表）|
| **"A switch does not create bandwidth, and because the [card] has no P2P, cards behind one switch cannot bypass the uplink"（点名 PEX88096）** | 同上（Platform and interconnect 表）|
| PP 模式 "moves 10,240 bytes per token per hop … saturating a single PCIe 1.0 lane would need roughly 25,000 tokens per second" | 同上（Bandwidth by configuration）|
| 被动 SlimSAS/MCIO 在 Gen4 下不可靠；只看 `LnkSta` 不看 `LnkCap`；sysfs `max_link_speed` 会撒谎 | 同上 |

---

## 10. 本次实测原始证据（可复查）

```
# 每张卡的 ReBAR capability
Capabilities: [bb0 v1] Physical Resizable BAR
    BAR 1: current size: 256MB, supported: 64MB 128MB 256MB

# 每张卡前面的桥窗口 —— 288M = 256M + 32M，无增长空间
ce:00.0  Prefetchable memory behind bridge: 17fc0000000-17fd1ffffff [size=288M]
ce:10.0  Prefetchable memory behind bridge: 17fa0000000-17fb1ffffff [size=288M]

# 交换机上行 = Gen4 x16（唯一的 CPU 通道）
c0:03.1  LnkSta: Speed 16GT/s, Width x16

# 4 张卡全部在同一个 root complex c0 下
00000000:C8:00.0 / CB:00.0 / CF:00.0 / D0:00.0

# BIOS: AMI 2.2 / 2026-01-06（新版本，不是固件老旧的问题）
# cmdline: amd_iommu=off iommu=off pcie_acs_override=downstream,multifunction
#          pci=realloc=on hpmmioprefsize=512G pcie_aspm.policy=performance
```

---

## 11. 交叉印证：GA100 / 其它 Ampere 上的同一现象

来自 CMP 170HX（GA100）的独立逆向工程文档（本地存档 `cmp170hx-pcie-subsystem-doc.md`、`cmp170hx-vbios-doc.md`）。

### ① Resizable BAR 在**同一个 capability 偏移**上是 "advertised but functionally inert"

| | CMP 170HX (GA100) | 你的 RTX 3080 (GA102) |
|---|---|---|
| ReBAR capability 位置 | **`[bb0 v1]`** | **`[bb0 v1]`** ← 同一个偏移 |
| BAR0 | 16 MB | 16 MB ← 一样 |
| BAR1 | 64 MB（只声明**一个** supported size）| **256 MB**（`64MB 128MB 256MB`）|
| BAR3 | 32 MB | 32 MB ← 一样 |
| 定性 | "advertised but functionally inert"；"BAR1 stays at 64 MiB **no matter what framebuffer size the card reports**" | 完全同一模式 |

> BAR0 / BAR3 尺寸、capability 偏移、以及"ReBAR 存在但无效"的行为，在两个不同的 Ampere 家族（GA100 / GA102）上**完全一致**。
> → **决定 BAR1 上限的机制是跨 Ampere 家族共用的**，不是某张卡或某批改装卡的偶发故障。

### ② strap 表在 ROM 里，而且整个落在 MAC 签名保护区内

GA100 ROM 区域图：

| 区间 | 内容 |
|---|---|
| `0x33800`-`0x41200` | **Configuration / DEVINIT-class tables**（54.5 KB，熵 4.5–6.0，明文）|
| `0x41200`-`0x42A00` | **Strap and training tables** |
| `0x2200`-`0x43A00` | ← **以上全部在 MAC 校验区内** |

原文的结论句：

> "**That is the clean dividing line: memory timings and clock offsets are editable in the image; memory capacity straps are not.**"

→ **strap 表在签名区内 ⇒ 改 VBIOS 镜像这条路被密码学封死。**
与 NvStrapsReBar 的 "vBIOS is digitally signed and can not be modified by end-users" 是同一结论，**来自两条独立证据链**。

### ③ "falcon is driving the rewrites"

> 有人把 A100 的板级 strap 配置抄到 170HX 上 → **开机不认卡**。
> 当事人结论："the straps don't do anything"，"**falcon is driving the rewrites**"。

→ 板上的 strap 电阻**不是**有效值来源 —— **有效 strap 由 Falcon/GSP 在启动时写入**。
这和 envytools 说的"primary / secondary / select 从 ROM 载入"完全一致：**有效值来自固件，不是来自电阻。**

### ④ ⭐ 直接点名 PEX88096 的独立结论

> "PCIe switch fan-out (for example **PEX88096**): **A switch does not create bandwidth, and because the [card] has no P2P, cards behind one switch cannot bypass the uplink.**"

这正是我从你这台机器的实测数据推出来的机制 —— 而且是别人独立写下、并且**点名了同一个交换芯片型号**：

> **没有 P2P ⇒ 卡无法绕过上行口 ⇒ 交换机不创造带宽。**

### ⑤ 对"改用 PP（`--split-mode layer`）"的强力支持

> 关于流水线并行："for pipeline-parallel LLM decode the link is nearly irrelevant (a 5120-hidden-dimension model **moves 10,240 bytes per token per hop**, so saturating a **single PCIe 1.0 lane** would need roughly **25,000 tokens per second**)."

→ PP 模式的卡间流量小到**"跑满一条 PCIe 1.0 单通道都需要 25,000 tok/s"**。
→ 把 `tensor` 换成 `layer`，跨卡流量会小到**根本碰不到那条共享上行**。
→ 所以 7.2 节第 1 条不只是"值得一试"，而是**有明确理论上限的一刀：PP 对互联带宽几乎免疫。**

---

## 12. ⚠️ 一处必须说清的不确定性

envytools 的 BAR1-size strap 编码表（set0 bits14-15 + set1 bits20-22）**明确覆盖到 GK104（Kepler）为止**，文档里没有 GA102。
GA100 的资料也只说到"strap 表在签名区内"，**没有给出 Ampere 的 BAR1 strap 位位置**。

| 结论 | 强度 |
|---|---|
| BAR1 上限 = 256MB；桥窗口无增长空间；ReBAR 失效 | ✅ **实测事实** |
| 该机制跨 Ampere 家族共用（GA100 / GA102 行为一致） | ✅ **实测事实**（两家族对照）|
| VBIOS 镜像不可改（签名/MAC） | ✅ **两条独立来源** |
| **strap 索引具体 = 2** | ⚠️ **高度自洽的推断，非 GA102 直接读数** |

最后一条是"envytools 表里 **2 恰好 = 256MB**"与实测的精确吻合，而不是在 GA102 上读出来的。

要去掉这个不确定性，唯一办法是**只读**探测 `BAR0 + 0x101000`。
但 GA102 布局无公开文档、误操作可能挂卡（170HX 那边抄 strap 就导致过"开机不认卡"），
**不建议在你这台机器上做。**

---

## 13. 最后一道门也是关着的：DevInit 层在 MAC 保护区内

GA100 的 PCIe 配置表**不在传统 x86 VBIOS 里**，而在 SPI flash 中**未加密的 DevInit Falcon 镜像**里：

| 项 | CMP 170HX |
|---|---|
| PCIe config table，flash 偏移 | `0x420ED`（镜像 `0xA20ED`）|
| 那 5 个字节，表内 `+0xC7`–`+0xCB` | flash `0x421B4`-`0x421B8` |
| 抑制标志，表内 `+0x0F` | `0x01` ← **`bra ne` 直接跳过整个 Gen4 编程块** |

### ⭐ 闭环的那一环：DEVINIT 读的就是 `0x101000`

GA100 逆向工程给出的 RM 侧 DEVINIT 解释器细节：

> RM 解释器符号 **`_nv000358rm`**，ASCII opcode 分发，
> **在代码地址 `0x264fb1` 处读取 strap 寄存器 `0x101000`**。

而 envytools 说 BAR1 尺寸位就在 **`PSTRAPS` @ `BAR0 + 0x101000`**。

> **→ 两条独立来源在同一个地址上会合：**
> **DEVINIT（签名 ROM 里的启动脚本）读 `0x101000` 的 strap 值 → strap 值决定 BAR1 尺寸。**

这就把整条链闭合了：

```
VBIOS/DevInit 里的 strap 表  ──写入──▶  PSTRAPS 寄存器 (BAR0+0x101000)
                                              │
                                    DEVINIT 读取 strap (=2)
                                              │
                                    GPU 把 BAR1 声明为 256MB
                                              │
                          lspci: supported: 64MB 128MB 256MB   ← 实测
```

而 **DEVINIT 区域 `0x33800`-`0x41200` 完全落在 MAC 校验区 `0x2200`-`0x43A00` 之内**：

> "**Every DEVINIT modification breaks the Booter MAC because the region ends at `0x41200` < `0x43A00`.**"

→ **改 DEVINIT ⇒ Booter MAC 校验失败 ⇒ 卡不启动。** 这条路是密码学封死的。

原文结论：

> "**Editing those bytes is a closed route. All five sit 100% inside the Davies-Meyer `csecret(2)` MAC range `0x2200`-`0x43C00`**, so a keyless forge is a 2^128 second-preimage, and reflashing an edited image fails the Ampere RSA signature check outright."
>
> ⚠️ "An edited DevInit or VBIOS image is rejected by the signature check and **the card will not boot**. Recovery requires an external programmer."

→ **这是"改 VBIOS/DevInit 不可行"的第三条独立证据**，而且给出的是具体密码学理由，不是"厂商锁了"这种模糊说法。

### 更关键：这类值在原理上就无法运行时修改

原文一句决定性的话：

> "The memory and compute unlocks land because `FEAT_OVR` / FBPA are **ordinary registers inside the 16 MB BAR0 that become writable once their PLM is opened**. **A latched PHY capability is not.**"

以及 `SUPPORTED_LINK_SPEED`（`0x0008808C`）的定性：

> "PHY reflection, marked `R-EVF`. **No write port at any privilege.**"

→ 这些值在启动早期被 **FWSEC-DevInit 编程并 latch（锁存）**，之后**任何特权级别都写不进去**。
→ Resizable BAR 的 "supported sizes" 是 **PCI config space 的 capability 字段**，与 `SUPPORTED_LINK_SPEED` 同属"锁存的设备声明"这一类。
→ **所以 GSP 补丁、驱动补丁、注册表键都改不动它。**

这同时解释了我早先查到的一条驱动源码注释：

```
nvrm_registry.h: RMBar1ApertureSizeMB
  "cannot be GREATER than the physical size of BAR1"
```

——**驱动只能在这个上限之内取一个值，改不了上限本身。**

> 📌 这条很重要，因为它关掉了最后一个"看起来可行"的方向：
> 170HX 社区对**运行时**才能修的东西（如 Gen2/Gen3 链路速率）确实是用 **GSP-RM 补丁**解决的
> （`gsp.bin` 里 `jalr fuse_read`，`li a2, 0x57c` / `0x580`）。
> 但 **BAR1 supported sizes 不是运行时值，而是锁存的 capability** —— 这类东西 GSP 补丁也够不到。

---

## 14. 必须说清的区分：板级电阻 straps ≠ PSTRAPS 寄存器

170HX 文档把 "Board straps" 列在 **"看着像原因、其实不是"** 里：

> 把 A100 的**板级** strap 配置抄到 170HX → **开机不认卡**。
> 当事人结论："the straps don't do anything"，"**falcon is driving the rewrites**"。

**这不是推翻 strap 理论，而是要说清是哪一层：**

| 层 | 内容 | 是不是有效值来源 |
|---|---|---|
| ① 板级电阻 strap | 复位后最初几个周期由板上拉/下拉电阻采样 | ❌ 会被 Falcon/GSP 在启动时**改写** |
| ② `PSTRAPS` 寄存器 (`BAR0+0x101000`) | `STRAPS*_PRIMARY` / `_SECONDARY` / `_SELECT` | ✅ **设备实际使用的有效值** |
| ③ ROM 里的 strap 表 | VBIOS / DevInit 在初始化时把值写进 ② | ✅ 上电默认值的来源，**且在 MAC 保护区内** |

- `NvStrapsReBar` 写的是**第 ② 层（寄存器）**，不是电阻 —— 这就是它能在不改签名 VBIOS 的前提下改变 BAR 尺寸的原因。它必须从**主板 UEFI** 下手，因为要在 PCI 枚举**之前**写入。
- **第 ③ 层在签名保护区内** → 改 VBIOS 这条路封死。

→ 结论不变，但机制表述更准确：**"有效 strap 由固件写入；固件里那张表是签名的。"**

### 那么"从操作系统写 ② 再让内核重新枚举"行不行？

理论上存在：`root` 写 `BAR0+0x101000`，然后 `remove` + `rescan` PCI 设备让内核重读 capability。

**但期望值低、风险高，不建议做：**

| 问题 | 说明 |
|---|---|
| GA102 布局无文档 | envytools 的编码表明确只覆盖到 GK104；GA100 资料也只说"strap 表在签名区内" |
| 可能已经太晚 | 若是 FWSEC-DevInit 阶段锁存（如 `SUPPORTED_LINK_SPEED` 那样），重扫也没用 |
| **写错会挂卡** | 未文档化 MMIO 写；170HX 那边抄 strap 就"开机不认卡" |
| 即便成功也未必有用 | club-3090 明确记录：**满尺寸 32GB BAR1 也是"必要但非充分"** —— 还需要注册表强制 static BAR1 映射；而另一个 2×5090 机器在 P2P 负载下**硬复位** |
| **对 llama.cpp 仍然≈零收益** | 见 7.1 节 |

综合：**收益可能为零，代价是一次内核 panic 甚至拆机。** 你这台机器已经因为我的失误被物理断电恢复过一次，不值得再赌。

---

## 15. 补一条廉价且关键的诊断：AER / 信号完整性

170HX 文档记录：**被动 SlimSAS / MCIO 线材在 Gen4 x8 下不可靠、错误很多**，Gen3 x8 才稳。
club-3090 也说：**"负载下上升的 correctable error 计数是劣质转接线的特征"**，而链路质量问题**在 `topo -m` 里完全看不出来**。

你的 4 张卡都通过**转接线**接到 PEX88096 卡上 —— 这是一个**独立于 P2P / ReBAR 的、可能导致性能损失的嫌疑因素**，而且检查完全免费：

```bash
# 有没有 PCIe 错误（AER）
sudo lspci -vv | grep -A3 -E 'Correctable Error|Uncorrectable Error|AER'
sudo dmesg -T | grep -iE 'AER|pcieport.*error|Corrected error|link.*retrain'

# 必须在负载下读（空闲会降速到 Gen1，读数没有意义）
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gpucurrent,pcie.link.width.current --format=csv
cat /sys/bus/pci/devices/0000:c8:00.0/current_link_speed
```

**诊断规则**（来自 170HX 文档的踩坑记录）：

| 规则 | 原因 |
|---|---|
| **只看 `LnkSta`，不要看 `LnkCap`** | capability 会挂着 Gen4，而链路实际跑在 Gen1 |
| **不要信 sysfs 的 `max_link_speed`** | 已知会报出比当前速率还低的值 |
| **不要用 `nvidia-smi` 的 `PCIe Generation Max` 当证据** | 已知长期报出无意义的值 |
| 三个可信字段 | `lspci -vvs <bdf> \| grep LnkSta`、`/sys/.../current_link_speed`、`nvidia-smi --query-gpu=pcie.link.gen.gpucurrent` |
| **必须在负载下测** | 空闲降速是正常行为（你已经看到 `Speed 2.5GT/s (downgraded)`）|

> **如果发现 correctable error 在增长 → 先换短的、Gen4 认证的转接线。**
> 这可能与交换机是**叠加的两个问题**，而不是同一个。
