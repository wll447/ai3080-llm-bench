# ai3080 GPU 静态硬件规格

> 采集时间：2026-09-28 · 只读采集，未做任何改动
> 主机：`ai3080` · 内核 `7.0.0-31-generic` x86_64

---

## 一、平台

| 项目 | 值 |
|---|---|
| 主机名 | `ai3080` |
| 操作系统 | Ubuntu 26.04 (resolute) |
| 内核 | `7.0.0-31-generic` x86_64 |
| CPU | **AMD EPYC 7402 24-Core Processor** |
| 核心/线程 | 24 核 / 48 线程，1 插槽 |
| NUMA | 1 个节点 |
| 内存 | **123 GiB**（114 GiB 可用） |
| 内存条 | 8 × 16 GB DDR4-2400 ECC，Micron `36ASF2G72PZ-2G3B1`（6 条）/ `-2G1A2`（2 条） |
| 内存通道 | P0 CHANNEL **A–H**（8 通道全插满） |
| 主板 | **HUANANZHI H12D-8D** rev V2.0 |
| BIOS | American Megatrends **2.2**（2026-01-06） |
| PCIe 架构 | AMD Starship/Matisse，4 个独立 Root Complex（`00` / `40` / `80` / `c0`） |

---

## 二、GPU × 4（四张完全一致）

| 项目 | 值 |
|---|---|
| 型号 | **NVIDIA GeForce RTX 3080** |
| 芯片 | **GA102**（Ampere） |
| Compute Capability | **8.6** (sm_86) |
| PCI ID | `10de:2206` (rev a1) |
| 子系统 ID | `10de:146d` — *GA102 [GeForce RTX 3080 **20GB**]* |
| VBIOS | **94.02.27.00.14**（含 20GB 改装补丁） |
| 显存 | **20480 MiB**（428 MiB 保留） |
| 显存类型 | GDDR6X，显存时钟 **9501 MHz**（= 19 Gbps） |
| 显存总线 | 320-bit → 理论带宽 **760 GB/s**（实测纯读 724 GB/s，达成率 95%） |
| 功耗 | 最小 **100 W** / 默认 **320 W** / 最大 **320 W** |
| 核心时钟上限 | Graphics **2100 MHz** · SM **2100 MHz** · Video 1950 MHz |
| 温度阈值 | 目标 83 °C · 最大运行 93 °C · 降频 95 °C · 关机 98 °C |
| ECC | N/A（GeForce 不支持） |
| Compute Mode | Default |
| 序列号 | N/A（GeForce 不上报） |
| 持久模式 | Enabled |

### 逐卡标识

| # | PCI 地址 | GPU UUID |
|---|---|---|
| 0 | `0000:01:00.0` | `<GPU-UUID>` |
| 1 | `0000:41:00.0` | `<GPU-UUID>` |
| 2 | `0000:81:00.0` | `<GPU-UUID>` |
| 3 | `0000:c4:00.0` | `<GPU-UUID>` |

### PCIe 链路

| 项目 | 值 |
|---|---|
| 链路能力 `LnkCap` | **Speed 16 GT/s (Gen4) × Width x16** |
| 当前速率（空闲） | `2.5 GT/s (downgraded)` — ASPM 省电降速，满载回 16 GT/s |
| ASPM | L0s L1，Exit Latency L0s <512ns / L1 <4us |
| 挂载方式 | 每卡独立 Root Complex 的 Root Port（非交换机） |
| PCIe 代际最大 | **Gen4** |

### BAR 配置（关键）

```
Region 0: 16 MB   (寄存器)
Region 1: 256 MB  (BAR1，64-bit prefetchable)   ← 显存映射窗口
Region 3:  32 MB  (64-bit prefetchable)
Region 5: 128 B   (I/O)
Expansion ROM: 512 KB (disabled)
```

**Resizable BAR 能力表（写在 VBIOS 里）：**

```
Capabilities: [bb0 v1] Physical Resizable BAR
    BAR 0: current size:  16MB, supported:  16MB
    BAR 1: current size: 256MB, supported:  64MB 128MB 256MB   ← 上限就是 256MB
    BAR 3: current size:  32MB, supported:  32MB
```

- **BAR1 当前 = 256 MiB，已经是固件允许的最大值**
- 支持列表里**没有** 512MB / 1GB / 8GB / 16GB 等大尺寸项
- 因此：`NVreg_EnableResizableBar=1` 已设且已生效至上限，**再调驱动/GRUB 参数也无法突破**，因为可选项里不存在更大尺寸
- 对比：一张 ReBAR 可用的 RTX 3060 其 BAR1 可达 16384 MiB

---

## 三、驱动与内核模块

| 项目 | 值 |
|---|---|
| 驱动版本 | **610.57.04**（开源内核模块 Open Kernel Module） |
| 构建信息 | `Release Build (root@ai3080) Thu Sep 24 01:01:00 AM UTC 2026` |
| 模块路径 | `/lib/modules/7.0.0-31-generic/kernel/drivers/video/nvidia.ko` |
| srcversion | `F110D2947200966FBA4D579`（含 P2P 补丁） |
| vermagic | `7.0.0-31-generic SMP preempt mod_unload modversions` |
| 编译 GCC | 15.2.0 (Ubuntu 15.2.0-16ubuntu1) |
| License | Dual MIT/GPL |
| CUDA | 13.2（`/usr/local/cuda-13.2`） |

### 驱动参数

```ini
# /etc/modprobe.d/nvidia.conf
options nvidia NVreg_EnableResizableBar=1

# /etc/modprobe.d/nvidia-p2p.conf
options nvidia NVreg_EnableResizableBar=1
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
options nvidia NVreg_DmaRemapPeerMmio=1
```

### 内核启动参数

```
pcie_aspm.policy=performance amd_iommu=off iommu=off
pcie_acs_override=downstream,multifunction pci=realloc=on hpmmioprefsize=512G
```

---

## 四、P2P 能力（实测）

```
nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3
 GPU0    X      OK      OK      OK
 GPU1    OK      X      OK      OK
 GPU2    OK      OK      X      OK
 GPU3    OK      OK      OK      X
```

- 12 对方向全部 `OK`（非 CNS / NS / DR）
- `cudaDeviceCanAccessPeer` / `cudaDeviceEnablePeerAccess` / P2P 原子操作：12/12 全部通过
- 实测 `cudaMemcpyPeer` 带宽 ≈ **24 GB/s**，接近 PCIe Gen4 x16 线速

---

## 五、其它 PCI 设备

| 地址 | 设备 |
|---|---|
| `c1:00.0` | Intel Ethernet Controller I226-V |
| `c2:00.0` | Intel Ethernet Controller I226-V |
| `c3:00.0` | Lexar NM610 PRO NVMe SSD (DRAM-less) |
| `c5-c6:00.0` | ASPEED Graphics Family（BMC 显示） |
| `44/45:00.0` | AMD FCH SATA Controller [AHCI] |

---

## 六、需要注意的两点

**1. 曾拆除的 Broadcom PEX88096 交换机**
现在 4 卡各自挂在独立 Root Complex 的 Root Port 上（`00`/`40`/`80`/`c0` 四个域），
不是通过 PLX 交换芯片。交换机在位时 prefill 仅 327 t/s、MTP 接受率 0.33；
拆除后 860 t/s / 0.625。

**2. BAR1 = 256 MB 是固件天花板，不是配置问题**
这直接影响 P2P mailbox 的实现方式 —— 驱动补丁（`kern_bus_gm107.c`）正是为此而做：
去掉原条件让 P2P mailbox 无条件在 BAR1 里预留（512 KB）。
补丁与重建步骤见 `~/nvidia-p2p-driver-kit/`。

---

## 七、GPU-Z 格式对照表

> 四张卡数值完全一致。`来源` 列说明该值从何而来：
> **实测** = 本机读出的；**规格** = GA102/RTX 3080 的公开硬件规格（软件不暴露）。

| GPU-Z 字段 | 值 | 来源 |
|---|---|---|
| **Name** | NVIDIA GeForce RTX 3080 | 实测 |
| **GPU** | GA102（RTX 3080 SKU = GA102-200） | 规格 |
| **Revision** | A1 | 实测（PCI rev `a1`） |
| **Technology** | 8 nm（Samsung 8N） | 规格 |
| **Die Size** | 628 mm² | 规格 |
| **Release Date** | Sep 17, 2020 | 规格 |
| **Transistors** | 28300M | 规格 |
| **BIOS Version** | 94.02.27.00.14 | 实测 |
| **UEFI** | ☑ 有 EFI 映像（PCIR CodeType=0x03 @0x00FE1C） | 实测 |
| **Subvendor** | NVIDIA（`10DE`） | 实测 |
| **Device ID** | `10DE 2206 - 10DE 146D` | 实测 |
| **ROPs / TMUs** | 96 / 272 | 规格 |
| **Bus Interface** | PCIe x16 4.0 @ x16 1.1（空闲；满载 x16 4.0） | 实测 |
| **Shaders** | 8704 Unified（68 SM × 128） | **实测** |
| **DirectX Support** | 12 (12_2) | 规格 |
| **Pixel Fillrate** | 164.2 GPixel/s | 计算（96 ROP × 1710 MHz） |
| **Texture Fillrate** | 465.1 GTexel/s | 计算（272 TMU × 1710 MHz） |
| **Memory Type** | GDDR6X（Micron — GDDR6X 为美光独家） | 规格 |
| **Bus Width** | 320 bit | **实测** |
| **Memory Size** | 20480 MB（可用 20052 MiB，428 MiB 保留） | 实测 |
| **Bandwidth** | 760.0 GB/s（19 Gbps × 320 ÷ 8） | 计算 |
| **Driver Version** | 610.57.04（Linux，Open Kernel Module） | 实测 |
| **Driver Date** | 2026-07-29（安装包） / 2026-09-24（模块构建） | 实测 |
| **Digital Signature** | N/A（Linux 无 WHQL 概念） | — |
| **GPU Clock** | Graphics **1710 MHz** / Memory 9251 MHz / Boost 1710 MHz（当前 P2） | 实测 |
| **Default Clock** | Graphics 1440 MHz（基频）/ Memory 9501 MHz / Boost **1710 MHz** | 规格 + 实测 |
| **NVIDIA SLI** | Disabled（RTX 3080 无 NVLink 接口） | 规格 |
| **Resizable BAR** | 能力结构存在，但上限 256 MB = 默认值 → **无大 BAR 收益** | 实测 |
| **Computing** | CUDA ☑ / OpenCL ☑ / DirectCompute — / DirectML —（后两者为 Windows 专有） | 实测 |
| **Technologies** | Vulkan ☑（1.4.341）/ Ray Tracing ☑（68 RT Core）/ PhysX —（Linux 不支持）/ OpenGL ☑ | 实测 |

### 补充实测数据（CUDA 属性）

| 属性 | 值 |
|---|---|
| Compute Capability | 8.6 |
| SM 数 | **68** |
| CUDA 核心 | **8704** |
| L2 缓存 | **5120 KB**（5 MB） |
| 常量内存 | 64 KB |
| 每 SM 共享内存 | 100 KB |
| 每 SM 最大块数 | 16 |
| 每 SM 寄存器文件 | 65536 |
| warp 大小 | 32 |
| 最大线程/块 | 1024 |
| 最大线程/SM | 1536 |
| 异步引擎数 | 2 |
| ECC 启用 | 0（否） |
| 统一寻址 | 是 |
| 纹理对齐 | 512 |

### 显存时钟的两种口径（避免混淆）

| 口径 | 值 |
|---|---|
| nvidia-smi（NVIDIA 口径，= WCK） | 最大 **9501 MHz** / 当前 9251 MHz |
| 等效数据率 | **19.0 Gbps**（最大）/ 18.5 Gbps（当前） |
| **带宽** | **760.0 GB/s** |
| 实测纯读带宽 | **724.4 GB/s**（达成率 95%） |

> GPU-Z 在 Windows 上对 GDDR6X 显示约 1188 MHz —— 那是同一物理量的另一种换算口径，不是矛盾。
