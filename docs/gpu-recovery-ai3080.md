# ai3080 GPU 恢复记录

**主机**: `<user>@<TAILSCALE-IP>` (hostname `ai3080`)
**日期**: 2026-09-23
**结果**: 4× RTX 3080 20GB 从"完全不可用"恢复为完全可用

---

## 一、故障现象（修复前）

```
$ nvidia-smi
No devices were found

$ ls /proc/driver/nvidia/gpus/
0000:c8:00.0  0000:cb:00.0  0000:cf:00.0  0000:d0:00.0     # 驱动能枚举
# 但 Video BIOS 显示为 ??.??.??.??.??，GPU Firmware: N/A
```

- 4 张卡累计 `RmInitAdapter failed!` 约 **80 次**
- `dmesg` 中约 **600 万行** `gpuHandleSanityCheckRegReadError_GM107: Possible bad register read: addr: 0x110100, regvalue: 0xbadf5720`
- 核心报错：

```
NVRM: GPU0 kflcnWaitForHalt_TU102: Timeout waiting for Falcon to halt
NVRM: GPU0 gpuWaitForGfwBootComplete_TU102: GSP failed to halt with GFW_BOOT: (progress 0xff)
NVRM: GPU0 kgspWaitForGfwBootOk_TU102: failed to wait for GFW boot complete: 0x65 VBIOS version 94.02.27.00.14
NVRM: GPU0 kgspWaitForGfwBootOk_TU102: (the GPU may be in a bad state and may need to be reset)
NVRM: GPU0 RmInitAdapter: Cannot initialize GSP firmware RM
NVRM: GPU 0000:d0:00.0: RmInitAdapter failed! (0x62:0x65:2119)
```

## 二、根因

**驱动状态与 GPU 内部状态双双损坏**，且互相拖累：

1. `/lib/modules/7.0.0-31-generic/kernel/drivers/video/nvidia.ko` 是**手工塞入的 610.57.04**
   （由 `nvidia-installer` / runfile 安装，**不属于任何 dpkg 包**），与 apt 体系脱节。
2. 驱动被反复手动装载 / 卸载（12:06 一轮、12:31 一轮），GPU 的 GSP 协处理器被留在
   **不可恢复的坏状态**。内核明确提示 `the GPU may need to be reset` —— 仅有**冷重启**能复位。
3. 当日 12:04 那次开机**驱动被 blacklist**（`/etc/modprobe.d/blacklist-nvidia.conf`），
   全程未加载 → **从未出现过"干净开机 + 匹配固件的驱动"的组合**。
4. 另有手工加的 P2P 配置干扰：`RMForceStaticBar1=1`（与 `EnableResizableBar=1` 冲突）。

### 排除项（已实测证伪）

| 假设 | 结论 |
|---|---|
| 驱动版本 610.57.04 有 bug | ❌ 换 595.91.07 后**报错完全相同**（同一点、同错误码 0x62:0x65） |
| `RMForceStaticBar1=1` 是元凶 | ❌ 注释掉后故障**完全复现** |
| GSP 固件文件缺失 | ❌ `/lib/firmware/nvidia/610.57.04/gsp_ga10x.bin` 和 `595.91.07/gsp_ga10x.bin` 都在 |
| `NVreg_EnableGpuFirmware=0` 可绕过 | ❌ 参数被接受但驱动仍走 GSP 路径 |
| VBIOS 与 GSP 固件不兼容 | ❌ 修复后同一份 VBIOS `94.02.27.00.14` 工作正常 |
| IOMMU / ACS 干扰 | ❌ IOMMU 已关、ACS 全树已 disabled |

## 三、修复步骤

```bash
# 1. 备份（恢复前状态）
#    /root/gpu-recovery-<timestamp>/
#      - modprobe.d/                   原始配置
#      - modules-nvidia-custom.tar.gz  手工 610 模块
#      - nvidia-smi-before.txt / dmesg-before.txt

# 2. 卸载全部 nvidia 内核模块
modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia

# 3. 删除手工塞入的 610 模块（DKMS 报 "not newer than 610.57.04" 拒绝降级安装）
rm -f /lib/modules/$(uname -r)/kernel/drivers/video/nvidia{,-uvm,-modeset,-drm,-peermem}.ko

# 4. 安装发行版驱动（595 包已在 apt 缓存中，含为 7.0.0-31 预编译的模块）
apt-get install -y nvidia-driver-595-open

# 5. DKMS 装机 + depmod
dkms install -m nvidia -v 595.91.07 -k $(uname -r) --force
depmod -a
dpkg --configure -a

# 6. 清理干扰配置
#    删除 /etc/modprobe.d/blacklist-nvidia.conf（原本阻止开机加载驱动）
#    注释 /etc/modprobe.d/nvidia-p2p.conf 中的 RMForceStaticBar1=1

# 7. 冷重启（关键：复位 GPU 的 GSP 坏状态）
systemctl reboot

# 8. 固化
nvidia-smi -pm 1
systemctl unmask --now nvidia-persistenced
```

## 四、修复后状态

```
$ nvidia-smi
|   0  NVIDIA GeForce RTX 3080  | 00000000:C8:00.0 |  1MiB / 20480MiB |
|   1  NVIDIA GeForce RTX 3080  | 00000000:CB:00.0 |  1MiB / 20480MiB |
|   2  NVIDIA GeForce RTX 3080  | 00000000:CF:00.0 |  1MiB / 20480MiB |
|   3  NVIDIA GeForce RTX 3080  | 00000000:D0:00.0 |  1MiB / 20480MiB |
```

| 项目 | 值 |
|---|---|
| 驱动 | 595.91.07 (open kernel module, DKMS) |
| VBIOS | `94.02.27.00.14`（现在可正常读取） |
| 显存 | 20480 MiB × 4 |
| Compute Capability | sm_8.6 |
| Persistence Mode | Enabled × 4 |
| `nvidia-persistenced` | active |
| NVRM 报错 | **0 条** |
| bad register read | **0 条** |
| PCIe 链路 | Gen4 x16 (max)，空闲时降速至 Gen1（正常省电行为） |
| CUDA | 13.2，`/usr/local/cuda-13.2/bin/nvcc` |
| CUDA 计算实测 | 4/4 卡 malloc + memset + memcpy 全部通过 |

CUDA 验证输出：
```
device_count=4
GPU0 NVIDIA GeForce RTX 3080 sm_8.6 mem=19.6GB  compute+memcpy OK
GPU1 NVIDIA GeForce RTX 3080 sm_8.6 mem=19.6GB  compute+memcpy OK
GPU2 NVIDIA GeForce RTX 3080 sm_8.6 mem=19.6GB  compute+memcpy OK
GPU3 NVIDIA GeForce RTX 3080 sm_8.6 mem=19.6GB  compute+memcpy OK
```

## 五、P2P 现状（未解决，另有阻塞）

拓扑已由 88096 交换机优化到位：

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      PXB     PXB     PXB
GPU1    PXB      X      PXB     PXB
GPU2    PXB     PXB      X      PIX
GPU3    PXB     PIX     PXB      X
```

`PXB`/`PIX` = 走 PCIe 交换机，**不经过 CPU 宿主桥**（若是 `PHB` 就说明白买了 88096）。

但 P2P 授权仍是 `CNS`（Chipset not supported）：

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      CNS     CNS     CNS
GPU1    CNS      X      CNS     CNS
GPU2    CNS     CNS      X      CNS
GPU3    CNS     CNS     CNS      X
```

**两道独立阻塞：**

1. **`CNS`** —— 驱动芯片组表未识别 PEX880xx（PCI ID `1000:c010`，
   NVIDIA allowlist 通常按 PLX `10b5` 87xx/97xx 匹配）。需魔改内核模块绕过。
2. **BAR1 = 256MB** —— `supported: 64MB 128MB 256MB`，是 **VBIOS 固件级上限**，
   主板 BIOS / 驱动 / 内核参数均无法突破。魔改驱动的 P2P 原理是把**整个显存映射进 BAR1**，
   256MB 窗口装不下 20GB 显存 → **这条路在当前固件下走不通**。

## 六、后续

- **P2P 需要**：aikitoria/open-gpu-kernel-modules 的 `610.57.04-p2p` 系列分支（或对应 595 分支），
  且**必须**先解决 BAR1 上限 —— 需向改装卡卖家索取支持 ReBAR 的 20G 版 VBIOS。
- **禁止**：刷零售版 3080（10GB/12GB）的 VBIOS —— 显存配置不同，必砖。
- **回退 610**：`/home/<user>/NVIDIA-Linux-x86_64-610.57.04.run` 仍在，可重新安装。

## 七、关键经验

> **卸载/重载 NVIDIA 驱动模块无法复位 GPU 的 GSP 坏状态。**
> 一旦出现 `kgspWaitForGfwBootOk` / `the GPU may need to be reset`，
> 后续所有驱动加载都会持续失败，**必须冷重启**才能恢复。
> 排查期间反复 modprobe 会让故障看起来"永久化"，掩盖真实原因。
