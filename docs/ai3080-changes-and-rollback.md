# ai3080 变更清单与回退步骤

**日期**: 2026-09-24
**本次会话共做了 3 类变更**：① GPU 驱动修复　② P2P 补丁　③ 网络修复
每一步都可独立回退。

---

## 一、本次全部变更清单

| # | 变更 | 位置 | 影响 |
|---|---|---|---|
| 1 | 595 驱动 → 610.57.04 魔改驱动 | 内核模块 + 用户态 | GPU 从"完全不可用"恢复 |
| 2 | **P2P mailbox 补丁**（删 1 个条件） | `kern_bus_gm107.c` | P2P 从失败 → 24.2 GB/s |
| 3 | ai3080 IPv4 默认路由补全 | `/etc/netplan/00-installter-config.yaml` | 修复 IPv4 断网 |
| 4 | ai3080 SOCKS5 代理 | `/etc/systemd/system/ss-local.service` | 新增功能 |
| 5 | 本机 sing-box 代理 | `Documents\ds-harness-desktop\proxy\config.json` | 新增功能 |

### 已确认无害的配置

```
/etc/modprobe.d/nvidia-p2p.conf:
  options nvidia NVreg_EnableResizableBar=1
  options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
  options nvidia NVreg_DmaRemapPeerMmio=1
```

---

## 二、回退步骤（按需选择）

### 回退 ①：撤销 P2P 补丁（保留 610 驱动）

**症状**：P2P 出问题，或想验证补丁是否是某个新问题的原因。

```bash
# 1. 恢复未打补丁的源文件
sudo cp /root/p2p-driver-patch-backup/kern_bus_gm107.c.orig \
        /usr/src/nvidia-p2p-610.57.04/src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm107.c

# 2. 重新编译安装
cd /usr/src/nvidia-p2p-610.57.04
sudo make modules -j$(nproc) && sudo make modules_install && sudo depmod -a

# 3. 重启（模块重载不足以复位 GPU）
sudo systemctl reboot
```

**验证回退成功**：
```bash
cudaDeviceEnablePeerAccess → mapping of buffer object failed   # 回到补丁前状态
```

---

### 回退 ②：撤销全部驱动改动，回到 595 发行版驱动

**症状**：610 + 魔改驱动出严重问题，想回到最保守状态。

```bash
# 1. 卸载当前手工模块
sudo modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo rm -f /lib/modules/$(uname -r)/kernel/drivers/video/nvidia{,-uvm,-modeset,-drm,-peermem}.ko
sudo depmod -a

# 2. 用 apt 缓存离线重装 595（缓存里有现成 deb）
sudo dpkg -i /var/cache/apt/archives/linux-modules-nvidia-595-open-7.0.0-31-generic_*.deb
sudo dpkg -i /var/cache/apt/archives/nvidia-driver-595-open_*.deb
sudo apt-get install -f -y

# 3. 重启
sudo systemctl reboot
```

**注意**：回 595 会**失去 P2P**（原版驱动的 `CNS` 锁）。

---

### 回退 ③：撤销 610 驱动，回到官方 610.57.04

```bash
# 用你本地的 runfile 重装官方版
sudo sh /home/<user>/NVIDIA-Linux-x86_64-610.57.04.run
# 同样会失去 P2P
```

---

### 回退 ④：撤销 ai3080 的网络改动

**⚠️ 慎重**：这个改动是**修复**了一个真实故障（IPv4 断网），回退会让 ai3080 回到 IPv4 断网状态。

```bash
# 恢复备份的 netplan（备份在 /root/netplan-backup-*.yaml）
sudo ls /root/netplan-backup-*.yaml
sudo cp /root/netplan-backup-<时间戳>.yaml /etc/netplan/00-installer-config.yaml
sudo chmod 600 /etc/netplan/00-installer-config.yaml
sudo netplan apply      # 注意：会短暂断网
```

**临时恢复 IPv4 断网（仅当前会话）**：
```bash
sudo ip route del default via <LAN-IP> dev enp194s0
```

---

### 回退 ⑤：停用/卸载 ai3080 代理

```bash
# 停用但保留
sudo systemctl stop ss-local
sudo systemctl disable ss-local

# 完全删除
sudo systemctl disable --now ss-local
sudo rm /etc/systemd/system/ss-local.service
sudo systemctl daemon-reload
sudo apt-get remove --purge -y shadowsocks-libev
```

---

### 回退 ⑥：停用本机 sing-box 代理

```powershell
Stop-Process -Name sing-box -Force
# 配置在 C:\Users\<user>\Documents\ds-harness-desktop\proxy\config.json
```

---

## 三、备份位置索引

| 内容 | 路径 |
|---|---|
| 补丁前源文件 | `/root/p2p-driver-patch-backup/kern_bus_gm107.c.orig` |
| 补丁说明 | `/root/p2p-fix-record/PATCH-NOTES.txt` |
| 补丁后模块 | `/root/p2p-fix-record/nvidia.ko` |
| GPU 恢复前状态 | `/root/gpu-recovery-*/` |
| P2P 配置备份 | `/root/gpu-p2p-backup-*/` |
| netplan 备份 | `/root/netplan-backup-*.yaml` |
| 原始 610 runfile | `/home/<user>/NVIDIA-Linux-x86_64-610.57.04.run` |
| 595 deb 缓存 | `/var/cache/apt/archives/*595*` |
| 打补丁的源码树 | `/usr/src/nvidia-p2p-610.57.04/` |

---

## 四、验证当前状态是否健康

一条命令全查：

```bash
# 1) GPU 在位 + 驱动版本
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# 2) BAR1（应为 256 MiB，不变）
nvidia-smi -q | grep -A2 "BAR1 Memory Usage" | grep Total

# 3) P2P 授权（应为全 OK）
nvidia-smi topo -p2p rw

# 4) P2P 实际可用性（12 方向应全部 no error）
/tmp/peer_test

# 5) 路由（应有 default via <LAN-IP>）
ip route show default

# 6) 代理
systemctl is-active ss-local
```

### 健康状态参照表

| 项 | 健康值 |
|---|---|
| 驱动 | 610.57.04 |
| GPU 数量 | 4，各 20480 MiB |
| BAR1 | 256 MiB |
| `topo -p2p rw` | 全部 OK |
| `peer_test` | 12 方向全部 `no error` |
| P2P 带宽 | ~24 GB/s |
| 默认路由 | `default via <LAN-IP> dev enp194s0` |
| ss-local | active |

---

## 五、当前验证结果（本次）

### llama-server 启动测试 ✅

```
0.29.114.813 I srv  llama_server: model loaded
0.29.114.818 I srv  llama_server: listening on http://0.0.0.0:8080

4 卡显存占用：
  GPU0 14283 MiB | GPU1 12695 MiB | GPU2 13147 MiB | GPU3 16985 MiB

推理实测：
  prompt eval : 92.0 t/s
  generation  : 32.7 t/s
  延迟        : 1.13 s

错误日志：无
```

**结论：P2P 补丁未引入任何回归，模型加载与推理完全正常。**

### 模型与环境

```
模型    : /home/<user>/models/Qwen3.8-27B-Q8_0.gguf  (29 GB)
mmproj  : /home/<user>/models/mmproj-Qwen3.8-27B-f16.gguf
配置    : /home/<user>/.config/llama.cpp/config.ini
          split-mode = layer       ← 流水线并行
          tensor-split = 1,1,1,1
          ctx-size = 262144
二进制  : /usr/local/bin/llama-server (0.4.0-dev, 带 NCCL)
```

---

## 六、⚠️ 禁用清单（已验证有害，不要做）

| 操作 | 后果 |
|---|---|
| 加 `RMForceStaticBar1=1` | 打断 GPU 初始化（GSP 引导失败），需冷重启。已验证两次 |
| 加 `RMPcieP2PType=0` | CanAccessPeer 变全 0，P2P 反而不可用 |
| 刷零售版 3080 VBIOS | 显存配置不同，必砖 |
| `netplan apply`（在 Tailscale 会话中） | 会短暂断网，可能丢失 SSH 连接 |

---

## 七、待办 / 建议

1. **验证张量并行**：当前配置是 `split-mode = layer`（流水线，卡间流量极小）。
   现在 P2P 可用，可以试 `split-mode = tensor`（需要 P2P），预期能提速。
2. **跑 NCCL 基准**：`nccl-tests` 的 `all_reduce_perf`，测真实集合通信性能。
3. **反馈补丁给上游**：这是 aikitoria/open-gpu-kernel-modules 的通用 bug，
   GitHub 现已可用，可提 PR。
4. **内核升级提醒**：源码在 `/usr/src/nvidia-p2p-610.57.04/`，但**未注册 DKMS**
   （因补丁版本号与官方冲突）。内核升级后需手动重建：
   ```bash
   cd /usr/src/nvidia-p2p-610.57.04 && sudo make modules -j$(nproc) && sudo make modules_install && sudo depmod -a
   ```
