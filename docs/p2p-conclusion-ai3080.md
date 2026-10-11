# P2P 可行性结论报告 — ai3080

**日期**: 2026-09-23
**主机**: `ai3080` (`<user>@<TAILSCALE-IP>`)
**硬件**: 4× RTX 3080 20GB (改装卡, VBIOS `94.02.27.00.14`) + Broadcom PEX88096 交换卡
**问题**: P2P 能否在不用大 BAR1 的情况下实现？

---

## 结论

> **不能。** P2P **必须**有 BAR1 区域，这是 NVIDIA 驱动内核层的硬性设计，
> 与用原版还是魔改驱动无关。改装卡的 VBIOS 把 BAR1 锁死在 256MB，
> 因此本机 P2P 无法实现。

但过程中有重大进展，值得完整记录。

---

## 一、已取得的进展（魔改驱动确实解锁了授权层）

### 原版驱动 vs 魔改驱动对比

| 测试项 | 原版驱动 (595.91.07) | **魔改驱动 (610.57.04-p2p-v3)** |
|---|---|---|
| `nvidia-smi topo -p2p rw` | `CNS` ×12 全部拒绝 | ✅ **`OK` ×12 全部通过** |
| `cudaDeviceCanAccessPeer` | 全 `0` | ✅ **全 `1`** |
| `cudaDeviceEnablePeerAccess` | `peer access is not supported` | ⚠️ `mapping of buffer object failed` |

**这说明用户的判断是正确的**：P2P 首先是被原版驱动的芯片组白名单（`CNS`）锁住的，
魔改驱动通过绕过该判定，**确实解开了授权层**。

### 使用的魔改驱动

- 仓库: `aikitoria/open-gpu-kernel-modules`
- 分支: **`610.57.04-p2p-v3`**（与驱动版本精确匹配）
- 补丁文件: `kernel-open/nvidia/nv-p2p.c`、`nv-p2p.h`、`rmp2pdefines.h`、`nv-pci.c`
- 编译: 48 核，`make modules` 全绿，0 error

---

## 二、为什么最终仍然失败：代码级证据

`cudaDeviceEnablePeerAccess` 触发内核报错：

```
NVRM: kbusSetP2PMailboxBar1Area_GM200: P2P mailbox area expected from RM
      but no valid address is installed  gpu=3
NVRM: nvAssertOkFailedNoLog: Assertion failed: Generic Error: Invalid state
      [NV_ERR_INVALID_STATE] (0x00000040) returned from status @ p2p_api.c:445
```

### 调用链

```
p2p_api.c:442   kbusSetP2PMailboxBar1Area_HAL(pLocalGpu, ...)      ← 失败点
p2p_api.c:447   kbusSetP2PMailboxBar1Area_HAL(pRemoteGpu, ...)
       ↓
kern_bus_gm200.c:798   kbusSetP2PMailboxBar1Area_GM200()
```

### 失败原因（`kern_bus_gm200.c`）

```c
if (!kbusIsP2pMailboxClientAllocated(pKernelBus))
{
    if (pKernelBus->p2pPcie.writeMailboxBar1Addr ==
        PCIE_P2P_INVALID_WRITE_MAILBOX_ADDR)
    {
        NV_PRINTF(LEVEL_ERROR,
                  "P2P mailbox area expected from RM but no valid address is installed gpu=%u\n",
                  gpuGetInstance(pGpu));
        return NV_ERR_INVALID_STATE;      // ← 我们命中的分支
    }
    return NV_OK;
}
```

**驱动期望 RM（Resource Manager）已经在 BAR1 里装好 P2P mailbox 区域，但实际没有。**
`writeMailboxBar1Addr` 仍是 `PCIE_P2P_INVALID_WRITE_MAILBOX_ADDR`。

### 三种 P2P 连接类型都依赖 BAR1

```c
// p2p_api.c:411-437  连接类型选择
NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PCI_BAR1_SUPPORTED → P2P_CONNECTIVITY_PCIE_BAR1
NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PROP_SUPPORTED     → P2P_CONNECTIVITY_PCIE_PROPRIETARY

// p2p_api.c:439-453  mailbox 只在此条件下分配
if ((bP2PWriteCapable || bP2PReadCapable) &&
    p2pConnectionType == P2P_CONNECTIVITY_PCIE_PROPRIETARY)
{
    status = kbusSetP2PMailboxBar1Area_HAL(pLocalGpu,  ...);
    status = kbusSetP2PMailboxBar1Area_HAL(pRemoteGpu, ...);
}
```

- `P2P_CONNECTIVITY_PCIE_BAR1` —— 名字即 BAR1
- `P2P_CONNECTIVITY_PCIE_PROPRIETARY` —— **mailbox 区域同样建在 BAR1 里**（函数名 `...MailboxBar1Area...`）

补丁自身也硬依赖 BAR1 映射：

```c
// nv-p2p.c:417
NvBool force_pcie = !!(flags & NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING);
// nv-p2p.h:202
#define NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING 1
```

### 相关新增 registry 键

```
src/nvidia/interface/nvrm_registry.h:1187
    NV_REG_STR_RM_64KB_BAR1_MAPPINGS    "RM64KBBAR1Mappings"
```

### 尝试过的 registry 键

| 键 | 结果 |
|---|---|
| `NVreg_EnableResizableBar=1` | 保留（无害） |
| `PeerMappingOverride=1` | 无效，同样的 mailbox 报错 |
| `RMForceStaticBar1=1` | ❌ **有害** —— 导致 4 张卡全部 GSP 引导失败，需冷重启恢复 |
| `NVreg_DmaRemapPeerMmio=1` | 保留（无害） |

---

## 三、为什么 BAR1 无法扩大（早前已验证）

1. **VBIOS 里没有可编辑的 Resizable BAR 能力结构**
   完整解剖 ROM：扩展能力 ID `0x15` 全文件仅 1 处（随机数据），配置空间字节 `09 00 14 01` 零匹配。
   该能力由芯片硬连线或运行时固件构建，**ROM 中不存在可改的字段**。

2. **即使有，字段位宽也不够**
   Supported Sizes 位图 12 位（bit 4–15），bit n = 2^n MB → **上限 2^11 MB = 2 GB**。
   而 P2P 需要 20 GB。

3. **改装卡无原厂 ReBAR 固件** —— 商家明确回复没有。

---

## 四、一个重要的附带发现（踩坑记录）

### `RMForceStaticBar1=1` 会直接搞坏 GPU

本次复现了**用户最初故障的成因**：

```
加回 RMForceStaticBar1=1 → 重载驱动 →
NVRM: GPU0 kgspWaitForGfwBootOk_TU102: failed to wait for GFW boot complete: 0x65
NVRM: GPU0 kgspWaitForGfwBootOk_TU102: (the GPU may be in a bad state and may need to be reset)
NVRM: GPU0 RmInitAdapter: Cannot initialize GSP firmware RM
NVRM: GPU 0000:d0:00.0: RmInitAdapter failed! (0x62:0x65:2119)
→ nvidia-smi: No devices found
```

**结论**：
- 该键使 GPU 进入 GSP 坏状态，**卸载/重载模块无法恢复，只有冷重启能复位**
- 排查期间反复 modprobe 会让故障看起来"永久化"，掩盖真实原因
- **已从配置中移除该键**，不要再加回来

---

## 五、当前系统状态（已恢复稳定）

```
index, name, driver_version, vbios_version, memory.total, persistence_mode
0, NVIDIA GeForce RTX 3080, 610.57.04, 94.02.27.00.14, 20480 MiB, Enabled
1, NVIDIA GeForce RTX 3080, 610.57.04, 94.02.27.00.14, 20480 MiB, Enabled
2, NVIDIA GeForce RTX 3080, 610.57.04, 94.02.27.00.14, 20480 MiB, Enabled
3, NVIDIA GeForce RTX 3080, 610.57.04, 94.02.27.00.14, 20480 MiB, Enabled
```

| 项 | 值 |
|---|---|
| 驱动 | **610.57.04**（aikitoria `610.57.04-p2p-v3` 魔改内核模块 + 610 用户态） |
| 内核模块位置 | `/lib/modules/7.0.0-31-generic/kernel/drivers/video/` |
| 源码树 | `/usr/src/nvidia-p2p-610.57.04/`（供内核升级后重建） |
| 设备节点 | 已由 `nvidia-modprobe` 创建，并配 udev 规则自动重建 |
| `nvidia-persistenced` | enabled |
| Persistence Mode | Enabled ×4 |
| P2P 授权 | `topo -p2p rw` = **OK ×12** |
| P2P 实际可用 | ❌ 否（mailbox BAR1 分配失败） |

### 配置文件

```
/etc/modprobe.d/nvidia-p2p.conf:
    options nvidia NVreg_EnableResizableBar=1
    options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
    options nvidia NVreg_DmaRemapPeerMmio=1

/etc/udev/rules.d/70-nvidia-nodes.rules:
    KERNEL=="nvidia", RUN+="/usr/bin/nvidia-modprobe -c 0 -u"
```

---

## 六、回退路径

| 目标 | 方法 |
|---|---|
| 回 595 发行版驱动 | `apt-get install nvidia-driver-595-open`（deb 在 apt 缓存中） |
| 回 610 官方驱动 | `sh /home/<user>/NVIDIA-Linux-x86_64-610.57.04.run` |
| 配置备份 | `/root/gpu-p2p-backup-*/`、`/root/gpu-recovery-*/` |
| 编译源码 | `/home/<user>/p2p-build/open-gpu-kernel-modules-610.57.04-p2p-v3/` |

---

## 七、最终建议

**P2P 在本机（4× RTX 3080 20GB 改装卡）上无法实现**，根本原因是 BAR1 被 VBIOS 锁在 256MB，
而 P2P 的 mailbox 和数据映射都必须放在 BAR1 里。

务实的选择：

1. **接受无 P2P，正常使用 4 张卡**。当前系统完全可用，4 卡跑独立任务（数据并行、
   多实例推理）无任何问题。这是这批卡的正确用法。

2. **走 NCCL 的共享内存中转路径**。经过 88096 交换芯片，比走 CPU 内存通道快；
   `NCCL_P2P_DISABLE=1` 即可（当前驱动默认也是这个行为）。
   张量并行会慢，但能跑。

3. **真要 P2P 就换卡**。需要 BAR1 能覆盖显存的卡：RTX 3090 / 4090 /
   RTX A 系列 / 专业卡。3090 的 ReBAR VBIOS 发布于 2021 年，BAR1 可达 24GB。

4. **保留魔改驱动的意义**：虽然 P2P 走不通，但它是目前唯一能让这批卡在
   610.57.04 上正常工作的驱动组合，且 `topo -p2p rw` 已显示 OK。
   如果将来换到支持大 BAR1 的卡，这套魔改驱动可直接复用。
