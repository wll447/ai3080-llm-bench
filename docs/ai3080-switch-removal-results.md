# ai3080：拆掉 PEX88096 前后的实测对比（最终结果）

**日期**: 2026-09-24
**结论**: **交换机就是元凶，拆掉后完全恢复到历史水平。**

---

## 1. 拓扑变化

| | 拆之前（经 PEX88096） | 拆之后（直插主板） |
|---|---|---|
| GPU 落点 | 4 卡全在 root complex `c0`，挤在**一个**上游口 `c0:03.1` 后面 | 4 卡**各占一个** root complex：`00` / `40` / `80` / `c0` |
| 中间设备 | Broadcom PEX880xx 交换机（2 层虚拟桥） | 无，直连 AMD 根端口 |
| 主机↔卡带宽 | 4 卡共享 1 条 Gen4 x16，实测聚合 **26.2 GB/s** | 4 条独立 Gen4 x16，理论 **~98 GB/s** |
| P2P 可能性 | 授权 OK，但集合通信挂死 | **物理上不可能**（跨 root complex）—— 但反正本来也不可用 |

新 bus ID：`01:00.0` / `41:00.0` / `81:00.0` / `c4:00.0`，全部 **Gen4 x16**。

BAR1 仍为 **256 MiB**（GPU 固件未变，符合预期）。

---

## 2. ★ 核心结果：llama.cpp 实测

**同一命令、同一 prompt（4001 token）、同一输出长度（256 token）、MTP=3**

| 指标 | 经 PEX88096 | **直插主板** | 你的历史记录 | 判定 |
|---|---|---|---|---|
| **prefill** | 327 t/s | **860.7 t/s** | 828–1000 | ✅ **完全命中** |
| **decode** | 44.6–45.9 | **78.3 t/s** | 77–84 | ✅ **完全命中** |
| MTP 接受率 | 0.325–0.43 | **0.625** | 0.91 | ✅ 也恢复了 |

**提升：prefill +163%，decode +71%。**

### 附带发现

**MTP 接受率也从 0.33–0.43 恢复到 0.625。** draft（MTP）模型同样跑 4 卡 tensor 并行，交换机的额外延迟同样拖累了它的采样质量。**这解释了之前"接受率异常低"的疑问 —— 也是交换机造成的。**

---

## 3. `layer` vs `tensor` 干净对比

条件：4001 token prompt，`ignore_eos` 固定 256 token 输出，其余完全相同。

| split-mode | prefill | decode | MTP 接受率 |
|---|---|---|---|
| **`layer`**（PP，流水线并行） | **1400.1 t/s** | 52.7 t/s | 0.722 |
| **`tensor`**（TP，张量并行） | 831.2 t/s | **75.5 t/s** | 0.587 |

### 结论：**各赢一半，按负载选**

| 场景 | 推荐 | 原因 |
|---|---|---|
| **单流交互 / 聊天（decode 敏感）** | **`tensor`** | TP 每层并行；PP 单序列 decode 要 4 卡**串行**跑 |
| **长上下文 / 批量 prefill（吞吐敏感）** | **`layer`** | PP 无 all-reduce，不跨卡同步 |
| **多并发请求（`--cont-batching` 多序列）** | 优先试 **`layer`** | PP 在多序列下能真正流水起来，不再是串行 |

> ⚠️ 注意：`layer` 的接受率**更高**（0.722 vs 0.587）却仍然 decode 更慢 —— 串行惩罚比表面更大。

**你的历史记录（828–1000 / 77–84）精确对应 `tensor`**，与你提供的基准命令一致。

---

## 4. 途中遇到的问题与修复：内核升级打断驱动

拆装重启后 `nvidia-smi` 报 `couldn't communicate with the NVIDIA driver`。

**原因**：内核从 `7.0.0-31` 升级到 `7.0.0-34`，而打补丁的 610.57.04 模块（`.run` 方式安装，非 DKMS）**只编到了 31**。

**修复（无需重启）**：

```bash
cd /usr/src/nvidia-p2p-610.57.04
make modules -j$(nproc)                      # 针对当前内核重编（约 30 秒，复用核心 RM 对象）
cp kernel-open/nvidia*.ko /lib/modules/7.0.0-34-generic/kernel/drivers/video/
depmod -a 7.0.0-34-generic
modprobe nvidia && modprobe nvidia-uvm
```

模块产物在 `kernel-open/*.ko`（**不在**子目录里）。

**善后已做**：

```bash
apt-mark hold linux-image-7.0.0-34-generic linux-headers-7.0.0-34-generic
# GRUB_DEFAULT 也从临时的 31 恢复为 0
```

> 建议：以后要么保持 `apt-mark hold`，要么给这份打补丁的源码配 DKMS，否则每次内核升级都会重演。

---

## 5. 完整结论链

```
① 交换机把 4 卡压成 1 条共享 Gen4 x16（实测聚合 26.2 GB/s ≈ 单卡的 1.07 倍）
② P2P 因 BAR1=256MB 不可用 → 跨卡流量必须走主机内存
③ 于是所有跨卡流量都挤在那一条共享上行上  ← 交换机在此处造成真实损失
④ prefill 327 → 860，decode 45 → 78（拆掉后）
```

**这也回过头解释了 P2P 攻坚的意义**：如果 BAR1 够大、P2P 能用，跨卡流量会留在卡间，那条共享上行就不会成为瓶颈 —— 交换机的损失本可以避免。**但因为 BAR1 无法改大（已逐条证伪），交换机的成本无法回收，所以拆掉是唯一正确的选择。**

---

## 6. 复测脚本（可直接复用）

```bash
# 启动（tensor = 单流最优）
pkill -f llama-server; sleep 3
cd /home/<user>/llama.cpp
sudo -u <user> env NCCL_P2P_DISABLE=1 \
  ./build/bin/llama-server \
  -m /home/<user>/models/Qwen3.8-27B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 262144 -ngl -1 \
  --split-mode tensor -ts 1,1,1,1 -b 1024 -ub 512 -t 24 --threads-batch 48 \
  -fa on --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --cont-batching --load-mode mmap --jinja --fit off &

# 基准（4001 token prompt，固定 256 token 输出）
P=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 400)")
curl -s http://127.0.0.1:8080/completion -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json,sys
print(json.dumps({'prompt':sys.argv[1],'n_predict':256,'temperature':0,
                  'top_k':1,'cache_prompt':False,'ignore_eos':True}))" "$P")"
# 看日志：/tmp/bench_tensor2.log 里的 prompt eval time / eval time / draft acceptance
```

---

## 7. 本机状态快照

| 项 | 值 |
|---|---|
| 内核 | `7.0.0-34-generic`（已 hold）|
| 驱动 | `610.57.04`（aikitoria P2P 补丁 + mailbox 修复，已为 34 重编）|
| GPU | 4× RTX 3080 20480 MiB，**Gen4 x16 各占一个 root complex** |
| BAR1 | 256 MiB（固件锁定，无解）|
| 交换机 | **已拆除** |
| llama-server | 运行中，`tensor` 模式，MTP=3，端口 8080 |
