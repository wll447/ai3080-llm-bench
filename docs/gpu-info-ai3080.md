# GPU-Z 风格显卡信息报告 — ai3080

主机: `ai3080` (`<user>@<TAILSCALE-IP>`) · 采集时间: 2026-09-23 13:58 UTC
采集方式: `nvidia-smi -q` / `lspci -vv` / sysfs / VBIOS ROM 读取

---

## 一、显卡总览

| | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| **型号** | GeForce RTX 3080 | GeForce RTX 3080 | GeForce RTX 3080 | GeForce RTX 3080 |
| **板卡标识** | RTX 3080 20GB | RTX 3080 20GB | RTX 3080 20GB | RTX 3080 20GB |
| **品牌** | GeForce | GeForce | GeForce | GeForce |
| **架构** | Ampere | Ampere | Ampere | Ampere |
| **芯片** | GA102 | GA102 | GA102 | GA102 |
| **修订** | A1 (rev a1) | A1 | A1 | A1 |
| **PCI 位置** | C8:00.0 | CB:00.0 | CF:00.0 | D0:00.0 |
| **显存** | 20480 MiB | 20480 MiB | 20480 MiB | 20480 MiB |
| **VBIOS** | 94.02.27.00.14 | 同 | 同 | 同 |

**共性**: 4 张卡完全相同 —— 同型号、同 VBIOS、同子系统 ID、同供电规格。

---

## 二、GPU 详细规格（GPU-Z 栏目对照）

### 基本信息

| 栏目 | 值 |
|---|---|
| Name | NVIDIA GeForce RTX 3080 |
| Board / 板型 | **A26R RTX 3080 20GB**（VBIOS 内字符串） |
| GPU | **GA102** |
| Device ID | `10DE 2206` |
| Subsystem ID | `10DE 146D`（`GA102 [GeForce RTX 3080 20GB]`） |
| Revision | A1 |
| Technology | 8 nm (Samsung) |
| Die Size | 628 mm² |
| Transistors | 28,300 million |
| Release Date | 2020-09-17（零售 3080）；本卡 VBIOS 为 2020-08-25 |
| Bus Interface | PCIe 4.0 x16（物理） |
| **Compute Capability** | **sm_8.6** |
| CUDA Cores | 8704 |
| RT Cores | 68（第 2 代） |
| Tensor Cores | 272（第 3 代） |
| GPU Part Number | `2206-200-A1` |
| Board ID | `0xc800` |

### 显存（Memory）

| 栏目 | 值 |
|---|---|
| Memory Size | **20480 MiB (20 GB)** |
| Memory Type | **GDDR6X (Micron)** |
| Memory Bus Width | **320-bit** |
| Memory Clock (实际) | 405 MHz（空闲 P8） |
| Memory Clock (最大) | **9501 MHz** |
| 等效数据率 | **19 Gbps** (GDDR6X) |
| **显存带宽** | **约 760 GB/s**（19 Gbps × 320-bit ÷ 8） |
| 已用 / 可用 | 1 MiB / 20054 MiB |
| Reserved | 426 MiB |

> ⚠️ **BAR1 显存窗口 = 256 MiB / 20480 MiB** —— 这是 P2P 不通的直接原因（见第五节）

### 频率（Clocks）

| 栏目 | 空闲值 | 最大值 |
|---|---|---|
| Graphics Clock | 210 MHz | **2100 MHz** |
| SM Clock | 210 MHz | **2100 MHz** |
| Memory Clock | 405 MHz | **9501 MHz** |
| Video Clock | 555 MHz | — |
| **Boost Clock（VBIOS）** | — | **1710 MHz** |
| **Base Clock（VBIOS）** | — | **1440 MHz** |

### PCIe 接口

| 栏目 | 值 |
|---|---|
| 物理接口 | PCIe 4.0 x16 |
| **Link Cap（最大能力）** | `Speed 16GT/s, Width x16` |
| **Link Sta（空闲当前）** | `Speed 2.5GT/s (downgraded), Width x16` |
| ASN 支持 | ASPM L0s L1 |
| 上游连接 | Broadcom/LSI **PEX880xx** PCIe Gen4 Switch |

> 空闲时降速至 2.5 GT/s 是 **正常的省电行为**，负载下会自动升到 16 GT/s。

### 供电与温度

| 栏目 | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| Power Draw（空闲） | 4.12 W | 9.40 W | 9.86 W | 6.07 W |
| Default / Max Limit | 320.00 W | 320.00 W | 320.00 W | 320.00 W |
| Min Limit | 100.00 W | 100.00 W | 100.00 W | 100.00 W |
| Enforced Limit | 320.00 W | 320.00 W | 320.00 W | 320.00 W |
| Adj. Range（VBIOS） | -69% / +0% | 同 | 同 | 同 |
| GPU 温度（空闲） | 26 °C | 26 °C | 30 °C | 27 °C |
| 显存温度 | N/A | N/A | N/A | N/A |
| 风扇转速 | 0 % | 0 % | 0 % | 0 %（低负载停转） |
| Thermal Limit（VBIOS） | 83 °C 额定 / 91 °C 最大 | 同 | 同 | 同 |

### 功能特性

| 特性 | 状态 |
|---|---|
| CUDA | ✅ 支持（sm_8.6） |
| DirectX | 12 Ultimate (12_2) |
| Vulkan | 支持 |
| OpenGL | 4.6 |
| OpenCL | 3.0 |
| Ray Tracing | ✅ 支持（第 2 代 RT Core） |
| DLSS | ✅ 支持 |
| NVENC / NVDEC | ✅（第 7 代 NVENC） |
| **NVLink** | ❌ **不支持**（消费级 GA102 无 NVLink） |
| **ECC** | ❌ 不支持（N/A） |
| **SLI** | ❌ 不支持 |
| Resizable BAR | ⚠️ **能力存在但上限 256MB** |
| MIG | ❌ N/A |
| 虚拟化 | None |
| **P2P (Peer-to-Peer)** | ❌ **CNS（Chipset not supported）** |

### 输出接口（VBIOS 记录）

`1× HDMI 3× DisplayPort`

---

## 三、驱动与系统环境

| 栏目 | 值 |
|---|---|
| Driver Version | **595.91.07** |
| Driver 类型 | Open Kernel Module (DKMS) |
| CUDA Version | **13.2** |
| CUDA Toolkit 路径 | `/usr/local/cuda-13.2/bin/nvcc` |
| NVIDIA-SMI 版本 | 595.91.07 |
| 内核 | `7.0.0-31-generic` |
| 操作系统 | Ubuntu 26.04 (resolute) |
| Persistence Mode | **Enabled** (×4) |
| nvidia-persistenced | active |
| IRQ | 101 / 104 / 105 / 106 |
| Minor Number | 0 / 1 / 2 / 3 |

---

## 四、GPU UUID（唯一标识）

```
GPU 0: <GPU-UUID>
GPU 1: <GPU-UUID>
GPU 2: <GPU-UUID>
GPU 3: <GPU-UUID>
```

---

## 五、关键限制（P2P 相关）

### BAR1 内存窗口

```
GPU0  BAR1 Memory Usage
        Total : 256 MiB        ← 上限
        Used  : 1 MiB
        Free  : 255 MiB       （显存实际 20480 MiB）
```

PCIe Resizable BAR 能力原始数据：
```
Capabilities: [bb0 v1] Physical Resizable BAR
    BAR 0: current size: 16MB,  supported: 16MB
    BAR 1: current size: 256MB, supported: 64MB 128MB 256MB
    BAR 3: current size: 32MB,  supported: 32MB
```

### P2P 拓扑与授权

拓扑（**88096 生效**，走交换机不走 CPU 宿主桥）：
```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      PXB     PXB     PXB
GPU1    PXB      X      PXB     PXB
GPU2    PXB     PXB      X      PIX
GPU3    PXB     PIX     PXB      X
```

P2P 授权（**不通**）：
```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      CNS     CNS     CNS
GPU1    CNS      X      CNS     CNS
GPU2    CNS     CNS      X      CNS
GPU3    CNS     CNS     CNS      X
```

**两道独立阻塞**：
1. `CNS` —— 驱动芯片组表不认 PEX880xx（PCI ID `1000:c010`）
2. `BAR1 = 256MB` —— VBIOS 固件级上限，装不下 20GB 显存

---

## 六、VBIOS ROM 信息

| 栏目 | 值 |
|---|---|
| VBIOS Version | **94.02.27.00.14** |
| 内部板卡标识 | `A26R RTX 3080 20GB` |
| 版权字符串 | `Copyright (C) 1996-2020 NVIDIA Corp.` |
| 构建日期 | 2020-08-25 |
| PCI ID | `10DE 2206` |
| Subsystem ID | `10DE 146D` |
| Board Power Limit | Target 320.0 W / Limit 320.0 W |
| Board Connectors | 1× HDMI, 3× DisplayPort |
| Memory Support | GDDR6X, Micron |
| UEFI Supported | Yes |

**ROM 读取（`/sys` 方式，截断至 156672 字节，仅供识别用）**：
```
$ ls -la /tmp/gpu0.rom
-rw-r--r-- 1 root root 156672 /tmp/gpu0.rom
$ md5sum /tmp/gpu0.rom
74833db0a26c363821e4b6cad4d86e5c
```

> ⚠️ 这是**截断的**读取（`/sys/.../rom` 只暴露前 156672 字节）。
> 完整备份请用 `nvflash --save`（需要完整 ~976 KB）。

**TechPowerUp 上的对应条目**（已确认完全匹配）:
https://www.techpowerup.com/vgabios/282572/282572
— aliexpress RTX 3080 20 GB, VBIOS `94.02.27.00.14`, `10DE 2206 / 10DE 146D`,
20480 MB GDDR6X, 2020-10-13, UEFI Yes

---

## 七、四卡一致性核对

| 项目 | 结果 |
|---|---|
| 型号 | ✅ 4/4 完全一致 |
| VBIOS | ✅ 4/4 均为 94.02.27.00.14 |
| Subsystem ID | ✅ 4/4 均为 10DE 146D |
| 显存容量 | ✅ 4/4 均为 20480 MiB |
| 供电上限 | ✅ 4/4 均为 320 W |
| Link Width | ✅ 4/4 均为 x16 |
| BAR1 上限 | ⚠️ 4/4 均为 256 MB（一致的限制） |

**结论**：4 张卡是同一批次、同一固件的同款产品。
