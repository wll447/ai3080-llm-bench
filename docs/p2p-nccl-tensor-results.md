# P2P 实测报告：NCCL 与张量并行

**日期**: 2026-09-24
**主机**: `ai3080` — 4× RTX 3080 20GB + PEX88096 + 魔改驱动 610.57.04 + mailbox 补丁

---

## 一、结论摘要

| 路径 | 状态 | 性能 |
|---|---|---|
| **P2P 授权** | ✅ `topo -p2p rw` 全 OK | — |
| **拷贝引擎 P2P**（`cudaMemcpyPeer`）| ✅ **可用** | **24.2 GB/s** |
| **内核直写 P2P**（kernel → peer 显存）| ⚠️ 功能通但**极慢** | **2.67 GB/s**（本地写 848 GB/s，差 **318 倍**）|
| **NCCL 集合通信（P2P 启用）** | ❌ **挂死** | — |
| **NCCL（`NCCL_P2P_DISABLE=1`）** | ✅ 可用 | ~1.3 GB/s busbw |
| **llama.cpp `layer` 模式** | ✅ 可用 | 15.6 t/s 生成 |
| **llama.cpp `tensor` 模式（P2P）** | ❌ **挂死** | — |
| **llama.cpp `tensor` + P2P 禁用** | ✅ 可用 | **17.4 t/s** 生成 |

**核心发现：P2P 在半通状态** —— 拷贝引擎路径正常，但**内核直写路径性能崩坏**，而 NCCL 恰恰依赖后者。

---

## 二、关键证据

### 1. NCCL 挂死的位置

`NCCL_DEBUG=INFO` 显示连接**全部成功**，挂在之后的数据传输：

```
Channel 02/0 : 0[0] -> 1[1] via P2P/direct pointer
Channel 01/0 : 2[2] -> 3[3] via P2P/direct pointer
[0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
[1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
[2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
[3] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
→ 然后挂死，第一次 all-reduce 永不完成
```

注意传输方式：**`P2P/direct pointer`** = GPU 内核直接写对端显存。

### 2. 两种 P2P 路径的性能天壤之别

```cuda
// 路径 A：拷贝引擎（DMA）
cudaMemcpyPeerAsync(dB, 1, dA, 0, BYTES);     → 24.2 GB/s   ✅

// 路径 B：内核直写（NCCL 用的）
write_peer<<<...>>>(dB, 42.0f, N);            → 2.67 GB/s   ⚠️

// 对照：本地写
write_peer<<<...>>>(dA, 42.0f, N);            → 848 GB/s
```

**318 倍差距。** 内核直写走的是极慢的路径，NCCL 的 ring all-reduce 在这种延迟下会超时/挂死。

### 3. 功能上"通"但性能不通

精细验证（三个观测点）：

```
GPU0 内核写 7.0 进 GPU1 显存：
  GPU0 peer read  → 7.0   ✅ 值到了
  GPU1 local read → 7.0   ✅ 对端看得到
  host read       → 1.0   ⚠️ 主机侧陈旧（缓存/测量假象）
```

**数据确实传过去了，但代价是极慢的路径。**

---

## 三、NCCL 基准完整数据

### A) P2P 启用 → 挂死

```
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=SYS ./all_reduce_perf -b 1M -e 256M -f 2 -g 4
→ rc=124 (timeout)，无任何结果输出
```

### B) `NCCL_P2P_DISABLE=1` → 可用

```
      size      algbw    busbw
     1048576     0.93     1.39
     4194304     0.99     1.49
    16777216     0.96     1.44
    67108864     0.95     1.42
   268435456     0.89     1.33
```

### C) `NCCL_P2P_LEVEL=PXB` → 仍挂死

### D) 尝试过但无效的 registry 键

| 键 | 效果 |
|---|---|
| `RM64KBBAR1Mappings=1` | ❌ 对端写仍 2.67 GB/s，NCCL 仍挂死 |

---

## 四、llama.cpp 实测

### 启动对比

| 模式 | 结果 | 加载时间 |
|---|---|---|
| `split-mode layer` | ✅ 正常 | 29 秒 |
| `split-mode tensor`（P2P）| ❌ 挂死 | — |
| `split-mode tensor` + `NCCL_P2P_DISABLE=1` | ✅ 正常 | **7.4 秒** |

### 生成性能（均在 P2P 禁用下测得）

| 模式 | prompt | generate |
|---|---|---|
| `layer` | 52.9 t/s | 15.6 t/s |
| `tensor` | 39.8 t/s | **17.4 t/s** |

**注意**：两个数字都是 **P2P 禁用**状态下测的，所以**还没吃到 P2P 收益**。

### 显存占用（layer 模式，4 卡均分）

```
GPU0 14283 MiB | GPU1 12695 MiB | GPU2 13147 MiB | GPU3 16985 MiB
```

---

## 五、当前可用配置

### 推荐（无 P2P 加速，但稳定）

```bash
# llama.cpp：张量并行 + 禁用 NCCL P2P
NCCL_P2P_DISABLE=1 llama-server --split-mode tensor -ngl 999 -c 262144
```

或在 `.config/llama.cpp/config.ini` 里加环境变量 / 用 launcher 脚本导出。

### 关键环境变量

| 变量 | 用途 |
|---|---|
| `NCCL_P2P_DISABLE=1` | **必须**。否则 NCCL 挂死 |
| `NCCL_DEBUG=INFO` | 排查时用 |

### ⚠️ 不要设

| 变量 | 后果 |
|---|---|
| `NCCL_SHM_DISABLE=1` | 会禁用 P2P 禁用后依赖的回退路径 |

---

## 六、为什么还是半通：技术推断

`cudaMemcpyPeer` 走**拷贝引擎（DMA）**，需要的是 BAR1 里的**地址映射**——补丁修好 mailbox 后这条通了。

而内核直写需要 GPU 的 **P2P 写引擎**走 mailbox 协议做对端写确认。这条路径**功能存在但性能崩坏**，推测原因：

1. **BAR1 仅 256MB**：P2P 写需要频繁的目标地址翻译，窗口太小导致翻译开销爆炸
2. **mailbox 协议开销**：每次写都要走 mailbox 握手，而 mailbox 在慢速路径上
3. 补丁只解决了"映射能否建立"，没解决"数据传输是否高效"

**这与社区经验一致**：club-3090 文档记载同类硬件上 NCCL 的 P2P 传输会挂，解法就是 `NCCL_P2P_DISABLE=1`。

---

## 七、下一步可选方向

如果想让 NCCL 真正用上 P2P，需要解决**内核直写性能**问题：

| 方向 | 说明 | 难度 |
|---|---|---|
| 改补丁用拷贝引擎路径 | 让 NCCL 走 DMA 而非内核直写 | 高（要改 NCCL 或驱动） |
| 深入分析 mailbox 慢在哪 | 用 nvprof/nsight 看 kernel peer write 的瓶颈 | 中 |
| 接受现状 | 用 `NCCL_P2P_DISABLE=1`，P2P 收益仅限 `cudaMemcpyPeer` 类应用 | 低 |
| 换卡 | 有原生 ReBAR 的卡（3090/4090）无此问题 | 高成本 |

**务实建议**：如果主要跑推理，当前 `tensor + NCCL_P2P_DISABLE=1` 已可用（17.4 t/s），比 layer 模式快 11%。P2P 的完整收益需要进一步攻内核直写路径。

---

## 八、当前系统状态

| 项 | 值 |
|---|---|
| 驱动 | 610.57.04（魔改 + mailbox 补丁）|
| GPU | 4 × RTX 3080 20480 MiB |
| BAR1 | 256 MiB |
| P2P 授权 | 全 OK |
| `cudaMemcpyPeer` | 24.2 GB/s |
| 内核直写 | 2.67 GB/s ⚠️ |
| NCCL（P2P 开）| 挂死 |
| NCCL（P2P 关）| 1.3 GB/s busbw |
| Persistence | Enabled（systemd `nvidia-pm` 单元，开机自动）|
| 路由 | `default via <LAN-IP>`（netplan 已持久化）|
| 代理 | `ss-local` active，127.0.0.1:1080 |

### 配置

```
/etc/modprobe.d/nvidia-p2p.conf:
  options nvidia NVreg_EnableResizableBar=1
  options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
  options nvidia NVreg_DmaRemapPeerMmio=1
```

### 新增文件

| 路径 | 内容 |
|---|---|
| `/etc/systemd/system/nvidia-pm.service` | 开机自动开 persistence mode |
| `/etc/systemd/system/ss-local.service` | SOCKS5 代理 |
| `/home/<user>/nccl-tests/` | NCCL 基准测试（已编译）|
