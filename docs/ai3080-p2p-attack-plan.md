# ai3080 — 打通 P2P 的攻坚记录与路线图

**日期**: 2026-09-24
**主机**: `ai3080` (LAN `<LAN-IP>` / Tailscale `<TAILSCALE-IP>`)
**目标**: 把 BAR1 从 256MB 改大 → 让 aikitoria 补丁的 static BAR1 mapping 路径可达 → NCCL 集合通信可用 → llama.cpp 提速

---

## 0. 一句话现状

> **平台侧全部正确；签名不是障碍；ROM 已备份。**
> **唯一堵点是 GA102 的 BAR1 尺寸字段在哪 —— 而定位它需要一份 ReBAR 版 GA102 ROM 做 diff。**
> 这一份参照 ROM，需要你用浏览器去下（厂商站和 TPU 都是 JS/反爬，我这边抓不到）。

---

## 1. 已确认的事实（本轮实测）

### 1.1 ROM 已备份 ✅

```
4 张卡 sha256 完全相同 : 1d98e39ddfea3d3b0977e9a0...
ROM 大小               : 156,672 字节 (0x26400)
VBIOS 版本             : 94.02.27.00.14   (@0xB9)
板卡串                 : "A26R RTX 3080 20GB"  (@0x68, @0x7B)
制造日期               : 10/13/20 (@0x38)  ← RTX 3080 首发期，早于 ReBAR 更新
PCIR                   : @0x170 (10DE:2206), @0xF49D, @0xFE1C
NPDS                   : @0xF4A5
本地备份               : A26R-3080-20GB-stock-rom.bin
```

**熵剖面**（每 4KB）：

| 区间 | 性质 |
|---|---|
| `0x00000-0x03FFF` | 代码/数据 |
| `0x05000-0x07FFF` | **结构化表区** ← 候选 |
| `0x09000-0x0EFFF` | 全零 |
| `0x0F000-0x0FFFF` | **结构化表区** + PCIR + NPDS ← 候选 |
| `0x10000-0x23FFF` | 高熵（加密/压缩）80KB |
| `0x25000-0x263FF` | 高熵 |

> 注意：本 ROM **没有 `NVGI`、没有 `RFRD`**，与 GA100 的容器格式不同。

### 1.2 平台侧全部正确 ✅

BIOS（AMI Aptio 2.20.1275 / 2026-01-06）：

| 项 | 值 |
|---|---|
| `Above 4G Decoding` | **Enabled** |
| `Re-Size BAR Support` | **Enabled** ← 关键，本来就是开的 |
| `IOMMU` (AMD CBS) | **Disabled** |
| `ACS Enable` (AMD CBS) | **Disabled** ← 关键，ACS 会强制 P2P 绕 root complex |

内核 cmdline：
```
amd_iommu=off iommu=off pcie_acs_override=downstream,multifunction
pci=realloc=on hpmmioprefsize=512G pcie_aspm.policy=performance
```

→ 这正是 club-3090 #734 描述的场景：**主板开关打开，但 GPU 固件把它静默架空**。

### 1.3 已排除的嫌疑

| 嫌疑 | 实测值 | 判定 |
|---|---|---|
| `FUSE_PCIE_GEN23_DIS` `0x82057c` | `0x00000000` | 链路速率未锁 |
| `FUSE_PCIE_GEN3_DIS` `0x820580` | `0x00000000` | 同上 |
| `FUSE_PCIE_MAGIC_D` `0x820520` | `0x10a80000` | **与公开 RTX 30 值精确吻合** → 地址表正确 |
| `LINK_CAP` `0x88084` | `0x00453d04` | Gen4 x16 ✓（XVE shadow 在 `0x88000`）|
| PCIe AER 错误 | 无 | 转接线/交换机链路干净 |
| MAC/签名 | nvflashk / OMVGflash 已绕过（2023 起）| **不是障碍** |
| 实测 `cudaMemcpyPeer` | **24.19 GB/s** | **硬件通路本身是好的** |

### 1.4 BAR1 现状（问题本体）

```
Capabilities: [bb0 v1] Physical Resizable BAR
    BAR 0: current size: 16MB,  supported: 16MB
    BAR 1: current size: 256MB, supported: 64MB 128MB 256MB   ← 上限锁死
    BAR 3: current size: 32MB,  supported: 32MB

每张卡前方桥 prefetchable 窗口 = 288M = 256MB + 32MB  (无增长空间)
交换机上行口 c4:00.0 窗口    = 1824 MiB  (4 卡共享)
PCIe ReBAR capability 原始值 : 0xbb0=c1c10015 0xbb4=00000100 0xbbc=00001c00
                              (0x1C0 即位图 64/128/256MB)
```

---

## 2. 本轮新增的硬核探测结果

### 2.1 特权掩码（PLM）寄存器**可读**

| 寄存器 | 当前值 | GA100 开锁时写 |
|---|---|---|
| `0x88fe8` `0x88fec` `0x88ff0` (XVE PLM) | `0xffffffcf` | `0xffffffff` |
| `0x88ff4` `0x88ab4` `0x88ff8` | `0xffffff8f` | `0xffffffff` |
| `0x8e1b0` (XP3G_PLM) | `0xffffff8f` | `0xffffffff` |
| `0x8e1b4` `0x8e1b8` `0x8e1bc` | `0xffffffcf` | `0xffffffff` |
| `0x8200d0..0xf4` (OPTB) | `0xffffff8f` / `0xfffffffc` | `0xffffffff` |
| `0x8200f8` `0x8200fc` | **`0xffffffff`** | 已经是开的 |
| `0x823800` | `0xffffff8f` | `0xffffffff` |

### 2.2 PSTRAPS 被特权屏蔽

```
@0x101000 = 0x00400080   real      (PSTRAPS0_PRIMARY)
@0x10100c = 0x02013000   real      (PSTRAPS1_PRIMARY)
其余 30 个            = 0xbadf5040  MASKED
```

`0xbadf5040` = NVIDIA 的**特权屏蔽哨兵值**（GA100 文档记录过）。

### 2.3 ⛔ envytools 的 BAR1 公式在 GA102 上不成立

```
公式: BAR1 index = STRAPS0_PRIMARY[14:15] + STRAPS1_PRIMARY[20:22]
实测: set0[14:15] = 0,  set1[20:22] = 0  →  索引 0 = 64MB
但实际 BAR1 = 256MB
```

→ **envytools 的编码表只到 GK104（Kepler），GA102 换了机制。** 这条假设被证伪。

### 2.4 ⚠️ 宿主写测试：写得到硬件，但语义不可控

```
写入 @0x088fe8 :  0xffffffcf  →  写 0xffffffdf (置 bit4)
读回           :  0xfffffece     ← 只有 bit0 和 bit8 被清掉
还原写 0xffffffcf → 读回仍 0xfffffece（还原失败）
```

- **不是普通存储**（否则读回应等于写入值）
- **也不是统一 W1C**（bit1 同为 1→1 却未被清）
- 结论：**带副作用语义，由硬件/RM 参与，用户态无法预测、无法还原**
- 重启后完全复位（已执行，寄存器全部回到原值，Xid=0，P2P 授权仍全 OK）

### 2.5 ⛔ GA100 的 PLM 开锁手段是**漏洞利用**，我们不具备

GA100 项目的 `xp3gTable` 不是调 API，而是：

> **crafted Falcon signature payload + `kgspSec2PostblTimingRefillPayload(addr, value)` + `kgspExecuteBooterLoad_HAL` + 重试**
> —— 即利用 **SEC2 Booter 签名校验的时间窗口漏洞**，获得任意 BAR0 写原语。

**610.57.04 源码侦察结果：**

| 符号 | 结果 |
|---|---|
| `kgspSec2PostblTimingRefillPayload` | **0 个文件** ❌ |
| `kgspSec2PostblTimingRebuildStockSignature` | **0 个文件** ❌ |
| `kgspExecuteBooterLoad_HAL` | 2 个文件 ✅ |
| `kernel_gsp_tu102.c` 的 `// Execute Booter Load` 返回点 | 第 565-570 行 ✅ |
| `_kbusRequiresP2PMailboxBar1_GM107`（已修）| 1 个文件 ✅ |

→ **重实现这个签名校验漏洞属于原创安全研究，不是打补丁。这条路本质走不通。**

**而 2.4 的测试正好解释了为什么 GA100 非要那个漏洞**：普通 BAR0 写（用户态 mmap 和驱动的 `GPU_REG_WR32` 是同一条 MMIO 路径）语义不可控。

---

## 3. 结论：只剩 VBIOS 一条路

| 路线 | 状态 |
|---|---|
| **A. 驱动/Booter 特权写** | ❌ **需要 SEC2 签名时间窗口漏洞，610.57.04 里不存在该原语** |
| **B. 用户态直接写寄存器** | ❌ 语义不可控，无法预测/还原 |
| **C. 改 VBIOS（推荐）** | ✅ **不需要任何漏洞** |
| D. UEFI DXE 驱动（NvStrapsReBar 式）| ⚠️ 依赖"GA102 的 strap 位在哪"—— 同一个未知数，被 C 解锁后才能做 |

### C 路线可行性 ✅

| 前置条件 | 状态 |
|---|---|
| ROM 备份 | ✅ 有，4 卡一致 |
| 签名 | ✅ 已可绕过（nvflashk / OMVGflash） |
| 平台侧 | ✅ 全部正确 |
| 砖机救援 | ⚠️ **需先买 SPI 编程器 + SOP8 夹子** |
| **参照 ROM** | ⬅️ **缺这个** |

---

## 4. diff 策略（拿到参照 ROM 后怎么做）

**不要直接拿"零售 10GB ReBAR ROM"和"我们的 20GB mod ROM"做 diff** —— 显存配置、板卡配置全不同，差异上万字节，信噪比极差。

**正确做法是两步：**

```
第 1 步：同厂商同型号的两份零售 3080 ROM 对比
         [2020 年非 ReBAR 版]  vs  [2021 年 ReBAR 版]
         → 差异应该很小很干净
         → 直接看出"开启 ReBAR 改的是哪几个字节 / 哪个结构"

第 2 步：把第 1 步找到的特征，在我们 20GB ROM 的对应结构里定位
         （按结构找，不是按偏移找 —— 因为版本不同偏移会差）
```

**需要你下载的样本（放进 `C:\Users\<user>\Documents\` 我就能读）：**

| # | 样本 | 去哪找 |
|---|---|---|
| 1 | RTX 3080 **ReBAR 版** ROM（2021 年后，任意大厂，如 ASUS TUF / MSI Gaming X / Gigabyte Gaming OC）| techpowerup.com/vgabios → 筛 RTX 3080 → 挑 2021+ 的 |
| 2 | RTX 3080 **非 ReBAR 版** ROM（2020 年早期版本）| 同上，挑 2020 年的 |
| 3 | （可选但最好）**同厂商同型号**的上两份 | 差异最小，最好用 |

> TechPowerUp 的列表页和下载按钮都是 JS 渲染 + 反爬，我这边 curl/代理都抓不到，**必须用你的浏览器手动下**。
> 每份 ROM 约 1–2 MB，是 `.rom` 文件。

**另外可问一下卖卡的人**：A26R 20GB 这批卡**有没有出过支持 ReBAR 的 VBIOS**？如果有，那就是一步到位，不用 diff。

---

## 5. 安全与回退

| 项 | 做法 |
|---|---|
| **刷固件前必须** | 买 SPI 编程器（CH341A）+ SOP8 夹子；4 张原厂 ROM 已备份 |
| 逐张推进 | 先在 **1 张卡**上试，成功再推广到其余 3 张 |
| 冷启动复位 | 寄存器类改动重启即复位（已验证）|
| 禁用清单 | `RMForceStaticBar1=1`（打断 GSP 引导）· 不带 NCCL 的 llama.cpp + 4 卡 tensor（内核 panic）· 刷零售 VBIOS 到 20GB 卡（必砖）|

**本地文件**：
- `A26R-3080-20GB-stock-rom.bin` — 原厂 ROM 备份（4 卡一致）
- `ai3080-rebar-root-cause.md` — ReBAR 根因全链
- `ai3080-topology-and-issues.md` — 拓扑与实测数据
- `cmp170hx-*.md` — GA100 逆向工程原文存档（方法学出处）

---

## 6. 备选：不依赖 P2P 的提速（随时可做，零风险）

即使 P2P 攻坚暂停，这些是确认有效的：

| # | 措施 | 原理 |
|---|---|---|
| 1 | `--split-mode tensor` → **`layer`** | 卡间流量从 ~1400 MB/s（每层 all-reduce）降到 ~7 MB/s。**PP 对互联带宽几乎免疫** |
| 2 | `-ts 1,1,0,0` 只用 2 卡 | TP 参与者 4→2，共享上行流量减半以上 |
| 3 | 调 MTP 接受率 | 实测只有 0.325–0.43（历史 0.91），投机解码吞吐与接受率近似成正比 |
