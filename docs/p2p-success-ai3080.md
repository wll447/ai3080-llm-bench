# P2P 突破：4× RTX 3080 20GB 改装卡 P2P 成功启用

**日期**: 2026-09-24
**主机**: `ai3080` (`<user>@<TAILSCALE-IP>`)
**硬件**: 4× RTX 3080 20GB 改装卡（VBIOS `94.02.27.00.14`）+ Broadcom PEX88096 交换卡
**结果**: ✅ **P2P 完全可用，无需换卡**

---

## 一、最终结果

```
cudaDeviceCanAccessPeer   : 12 个方向全部 = 1
cudaDeviceEnablePeerAccess: 12 个方向全部 "no error"
nvidia-smi topo -p2p rw   : 全部 OK
```

### 实测带宽

```
直达 GPU-GPU 传输（P2P）：
  GPU0->GPU1  16.53 GB/s      GPU1->GPU3  26.10 GB/s
  GPU0->GPU2  24.26 GB/s      GPU2->GPU3  26.10 GB/s
  GPU0->GPU3  26.10 GB/s      GPU3->GPU0  24.38 GB/s
  ... 12 个方向平均 24.19 GB/s

4 卡并发聚合：19.45 GB/s（每卡 4.86 GB/s）
```

### 与之前方案的对比

| 路径 | 单向带宽 | 延迟 |
|---|---|---|
| **P2P 直连（现在）** | **~24.2 GB/s** | 微秒级 |
| 内存中转（之前） | 12.55 GB/s | 2659 µs / 32MB |

**P2P 比内存中转快约 2 倍，且延迟降低三个数量级。**

---

## 二、根因：驱动自身的一个 bug，不是硬件限制

### 之前所有分析都指向"BAR1 只有 256MB 所以不行"——**这是错的**

真正的原因是 NVIDIA 开源内核模块里的一处**条件判断不一致**。

### 问题代码

**`kernel_bif.c`** —— 驱动的默认值：

```c
pKernelBif->pcieP2PType = NV_REG_STR_RM_PCIEP2P_TYPE_BAR1;   // 默认 = BAR1
```

**`kern_bus_gm107.c:846`** —— 是否分配 P2P mailbox 的判定：

```c
_kbusRequiresP2PMailboxBar1_GM107(...)
{
    return (pKernelBif->pcieP2PType != NV_REG_STR_RM_PCIEP2P_TYPE_BAR1)   // ★
           && (!READS_DISABLED || !WRITES_DISABLED)
           && IS_GFID_PF(gfid)
           && !kbusIsP2pMailboxClientAllocated(pKernelBus);
}
```

**第一项在默认配置下恒为 false** → `pcieP2PType == BAR1` → 整个函数返回 false → **mailbox 从不被分配**。

**`p2p_api.c:439`** —— 但启用 P2P 时又要求 mailbox 必须存在：

```c
if ((bP2PWriteCapable || bP2PReadCapable) &&
    p2pConnectionType == P2P_CONNECTIVITY_PCIE_PROPRIETARY)
{
    status = kbusSetP2PMailboxBar1Area_HAL(..., mailboxBar1Addr, mailboxTotalSize);
    // 失败：writeMailboxBar1Addr == PCIE_P2P_INVALID_WRITE_MAILBOX_ADDR
}
```

**矛盾**：驱动选择 BAR1 传输 → 跳过 mailbox 分配 → 启用 P2P 时却要求 mailbox 有效 → 失败。

报错正是：
```
NVRM: kbusSetP2PMailboxBar1Area_GM200: P2P mailbox area expected from RM
      but no valid address is installed
```

### 关键验证实验

| 配置 | mailbox 分配 | CanAccessPeer | EnablePeerAccess |
|---|---|---|---|
| 默认（BAR1 类型） | ❌ 跳过 | ✅ 全 1 | ❌ mapping failed |
| 强制 mailbox 类型 | ✅ 分配 | ❌ 全 0 | ❌ not supported |

**两个配置各对一半** —— 证明矛盾确实存在，且与 BAR1 大小无关。

---

## 三、修复：删掉一个条件

**文件**：`src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm107.c`
**函数**：`_kbusRequiresP2PMailboxBar1_GM107()`

```c
// 删除前
return (pKernelBif->pcieP2PType != NV_REG_STR_RM_PCIEP2P_TYPE_BAR1)
       && (!READS_DISABLED || !WRITES_DISABLED)
       && IS_GFID_PF(gfid)
       && !kbusIsP2pMailboxClientAllocated(pKernelBus);

// 删除后 —— 去掉第一个条件
return (!READS_DISABLED || !WRITES_DISABLED)
       && IS_GFID_PF(gfid)
       && !kbusIsP2pMailboxClientAllocated(pKernelBus);
```

### 为什么这个改动是安全的

- mailbox 只需 **512KB**（`PCIE_P2P_WRITE_MAILBOX_SIZE` 64KB × `P2P_MAX_NUM_PEERS` 8）
- 你的 BAR1 有 256MB，**完全装得下**
- BAR1 P2P 本身并不使用这块 mailbox —— 它只是被那些仍然引用 `writeMailboxBar1Addr` 的代码路径需要
- 不改固件、不改 VBIOS、不动硬件

---

## 四、当前配置

### 驱动

```
内核模块: 610.57.04 (aikitoria 610.57.04-p2p-v3 + 本补丁)
用户态:   610.57.04
源码树:   /usr/src/nvidia-p2p-610.57.04/          ← 含补丁，供内核升级后重建
构建树:   /home/<user>/p2p-build/open-gpu-kernel-modules-610.57.04-p2p-v3/
```

### `/etc/modprobe.d/nvidia-p2p.conf`

```
options nvidia NVreg_EnableResizableBar=1
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
options nvidia NVreg_DmaRemapPeerMmio=1
```

### 系统层

| 项 | 值 |
|---|---|
| IOMMU | `amd_iommu=off iommu=off` |
| ACS | 已禁用（pcie_acs_override + 硬件层） |
| Persistence Mode | Enabled ×4 |
| 设备节点 | nvidia-modprobe + udev 规则 |

---

## 五、回退方法

```bash
# 1. 恢复原始（未打补丁）源文件
cp /root/p2p-driver-patch-backup/kern_bus_gm107.c.orig \
   /usr/src/nvidia-p2p-610.57.04/src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm107.c

# 2. 重新编译安装
cd /usr/src/nvidia-p2p-610.57.04
make modules -j$(nproc) && sudo make modules_install && sudo depmod -a

# 3. 重启
sudo systemctl reboot
```

**其他回退层级**：
- 回 595 发行版驱动：`apt-get install nvidia-driver-595-open`（deb 在缓存中）
- 回 610 官方驱动：`sh /home/<user>/NVIDIA-Linux-x86_64-610.57.04.run`
- 配置备份：`/root/gpu-p2p-backup-*/`、`/root/gpu-recovery-*/`

---

## 六、⛔ 禁用清单（已验证有害）

| 操作 | 后果 |
|---|---|
| `RMForceStaticBar1=1` | **打断 GPU 初始化**（GSP 引导失败），需冷重启恢复。已验证两次 |
| `RMPcieP2PType=0`（强制 mailbox） | CanAccessPeer 变全 0，P2P 反而不可用 |
| 刷零售版 3080 VBIOS | 显存配置不同，必砖 |

---

## 七、经验教训

### 我犯的错误

在这条路上我**反复下过 4 次错误结论**，每次都因为**测量工具本身无效**：

1. `setpci` 写寄存器"无变化" → 误判"硬件拒绝大 BAR1"（实际是写不进去）
2. ROM 搜 `0x15` 零匹配 → 误判"固件里没有"（实际是搜索模式不对）
3. 推论"限制在签名固件里" → 被 GSP 全家族共用的事实推翻
4. 多次断言"只能换卡" → **最终被一行补丁推翻**

### 真正有效的做法

- **交叉验证多个信息源**：源码 + 实测 + 社区案例，缺一不可
- **对测量工具本身做 sanity check**：先用已知可写/可读的对象验证工具有效，再下结论
- **不轻易说"不可能"**：驱动是软件，软件就有 bug，bug 就能修
- **深挖代码而不是猜**：从 `cudaDeviceEnablePeerAccess` 的报错一路追到 `kern_bus_gm107.c:846`，才找到真正的矛盾

### 最终结论

> **"BAR1 只有 256MB 所以 P2P 不可能"——这个结论是错的。**
> 真正的原因是 NVIDIA 驱动里 mailbox 分配与 P2P 类型选择的耦合 bug。
> 删掉一个条件，P2P 就通了，带宽 24.2 GB/s。

---

## 八、后续建议

1. **跑真实负载验证** —— 建议用 NCCL 测试 (`nccl-tests`) 验证集合通信：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=SYS ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
   ```
2. **张量并行现在可行了** —— vLLM/llama.cpp 的 TP 模式可以试；注意 4 卡 PCIe 场景下 vLLM 会禁用 custom all-reduce（`world_size > 2` 且无 NVLink），收益走 NCCL 通道
3. **内核升级后需重建** —— 源码树已在 `/usr/src/`，但 DKMS 未注册（因补丁版本号与官方冲突）。内核升级后手动 `make modules && make modules_install`
4. **建议把补丁反馈给 aikitoria** —— 这是上游驱动的通用 bug，对其他 20/30 系改装卡同样适用
