# Strata 基准报告（2026-10，ai3080，3× RTX 3080 20GB）

本文件把分阶段实验的**验证过的数字**合并成一份可引用的记录。方法学先说清楚，后面所有表格都按同一口径。

---

## 1. 引擎与模型

| 项 | 值 |
|---|---|
| 引擎 | **Strata 0.1.41**（原 0.1.40.1；上游 tag `v0.1.41`） |
| 模型 | **Qwen3.8-Flash-Next UD-Q4_K_XL**（Unsloth，revision `38bb39e`，4 分片共 **111.3 GB**） |
| pack | 约 1.4 GB（`iq_pack.py --compat-bf16` 产物，含 `dense.bin`） |
| MTP 草稿层 | `mtp/rt`（投机解码） |
| 视觉编码器 | `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（865 MiB）+ 自编译 `strata-vision` |
| 上下文 | 262,144（`--kv int8 --kv-resident 32768`） |

对比：被替换掉的 llama.cpp + Qwen3.8-27B Q8_0 基线为 decode ~84 t/s、prefill ~855 t/s（4 卡张量并行）。27B-Q8 每 token 读 ~29 GB，80 t/s 需持续 2.3 TB/s ≈ 4×3080 的聚合带宽天花板。

---

## 2. 定稿配置

```jsonc
{
  "gpu": [0, 1, 2],              // 3 卡（第 4 卡已物理拆除）
  "layer_split": "auto",         // 自动分层 → K=15,32
  "host": "0.0.0.0",             // ⚠️ setup 会把它删掉，删了就只监听 127.0.0.1
  "port": 8080,
  "vision": { "gpu": true, "max_tokens": 1024 },
  "env": { "STRATA_SPLIT_OWN": "1" },
  "args": [
    "--pack", "<pack>", "--native", "<shard1.gguf>",
    "--expert-profile", "<profile.bin>",
    "--expert-cache", "auto", "--prefill", "auto",
    "--spec", "4", "--spec-min-p", "0.5", "--mtp", "<rt>",
    "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768",
    "--remote-expert-opt",
    "--conversation-cache-mib", "8192",
    "--pcie-frac", "0.00",
    "--vision", "--vram-reserve-mib", "700"
  ]
}
// 顶层键 "parallel" 关闭（= 请求排队）。它是运行配置的顶层键，
// server.py 的 parallel_args() 会翻成引擎的 --batch —— 不是引擎参数，塞进 args 无效。
```

### 各参数怎么定下来的

| 参数 | 结论 | 依据 |
|---|---|---|
| `STRATA_SPLIT_OWN=1` | 采纳，**prefill +10.3%** | A/B：各 stage 用自己的提示缓冲，不借缓存槽位 |
| `--pcie-frac 0.00` | 采纳（官方 calibrate 结果） | 自动探测原本给 0.55；calibrate 实测 0.00 最快 |
| `--pool-workers 23` | 采纳 | calibrate：23 → 92.3 t/s；15 → 90.1；12 → 88.7；6 → 82.3 |
| `--spec-min-p 0.50` | 采纳 | calibrate 同时给出 |
| `--conversation-cache-mib 8192` | 采纳 | 多会话复用；实测 2 万 token 级恢复 |
| `--remote-expert-opt` | 保留 | setup 会丢，需手工补 |
| `parallel` | **关闭** | 单用户下无收益且占显存（每 slot 约 0.6 GB 专家缓存） |

---

## 3. 方法学（结果可信的前提）

1. **提示必须唯一**——每条提示带唯一随机串。否则对话缓存命中会让 prefill 被"短路"，测出的数字无意义（曾因此报废一整轮 A/B）。
2. **就绪判定用 HTTP `/health`**（`"loaded": true`）。
   ⚠️ `ready:` 横幅只由 `serve/server.py` 打印到 **tmux 面板**，**不写进 `strata-<model>.log`**——用日志等就绪会白等满超时（踩过 600 s）。
3. **预热后再测**，冷启动首请求的数字不可引用。
4. 长提示跑 3 次取中位，decode 跑 2 次取中位。
5. `decode` 与内容强相关（取决于草稿接受率）：58–120 t/s 都属正常范围。

---

## 4. 主结果

### 4.1 吞吐

| 卡数 | Prefill（长唯一提示，中位） | Decode（全新 256 token，中位） |
|---|---:|---:|
| 2 卡 | 1,339 tok/s | — |
| **3 卡（定稿）** | **1,941 tok/s** | **64.4 tok/s** |
| 4 卡 | 1,721 tok/s（19.6k 提示，非同一提示长度） | — |

引擎自报 decode 79.9–93.2 t/s；可预测内容（代码重复、引用）可达 ~120 t/s。

**Prefill 的物理模型**：≈ 1 / (最慢 stage 的层数)。3 卡 `K=15,32` 时最慢 17 层 → 24/17 = 1.41；实测 1,941/1,339 = 1.45。**卡数只改变专家缓存池，不改变 prefill 上限。**

### 4.2 并发表（`parallel: 4`，每路唯一提示，聚合 = 总输出 token ÷ 墙钟）

| 卡数 | N=1 | N=2 | N=4 |
|---|---:|---:|---:|
| 3 卡 | 59.1 t/s | 67.3（×1.14） | 80.8（×1.37） |
| 4 卡 | 70.4 t/s | 75.0（×1.07） | 97.9（×1.39） |

4 卡相对 3 卡：N=1 **+19.1%**、N=2 **+11.4%**、N=4 **+21.2%**。

**结论**：第 4 张卡的价值集中在 N=4；N=2 时几乎白给（+7%）。**单人使用关闭 `parallel`。**

### 4.3 结构参数（同一份模型，只改卡数）

| 卡数 | 分层 | 专家槽 | 路由覆盖 |
|---|---|---:|---:|
| 4 卡 | `K=12,25,37` | 18,394 | 98.6% |
| **3 卡** | **`K=15,32`** | **13,825** | **98.4%** |
| 2 卡 | `K=24` | 9,025 | 97.0% |
| 1 卡 | — | 约 13–15 GB 缓存 | — |

其他运行时量（3 卡）：专家 arena **73,450 MiB（71.73 GiB）pin 在 RAM**；`pool_workers=23`；`vram_free_mib≈411–432`；推理期 `file_blobs`/`file_mb` **恒为 0**。

### 4.4 上下文

| 声明窗口 | 专家槽 | 缓存显存 |
|---|---:|---:|
| 131,072 | 18,394 | 54,969 MiB |
| 262,144 | 18,279（**−0.6%**） | 54,628 MiB |

→ **把窗口开大几乎免费**，因为 `--kv-resident 32768` 只让 32,768 个 KV cell 常驻显存（不随窗口变），其余从 pinned RAM 流。

**决定保留 262,144，不上 1M**：rope scaling 是**全局**的（缩放后的角度与 yarn 幅度修正作用于**每一个位置**，等于变成"略有不同的模型"）；KV 的 RAM 从 ~3.6 GB 涨到 ~14.4 GB；冷读一次 1M 约 8.8 分钟（按实测 1,900–2,000 t/s 折算，属推算）。

### 4.5 视觉（0.1.41 起）

| 项 | 值 |
|---|---|
| 就绪时间 | 125 s |
| 编码器预热 | 1024 image token 用时 **0.6 s** |
| 代价 | `--vram-reserve-mib 700` → 专家槽 13,825 → **13,244（−4.2%）**，覆盖 98.4% → 98.2% |
| 实测 | 512×256（红方块 + "STRATA" 字样）→ 正确答出 `1) Red 2) STRATA`，2.68 s |
| 文本正确性 | `Paris` / `51` / `Jupiter` 全对 |

⚠️ **`max_tokens` 给太小会被思考块吃光**：`max_tokens=32` 时 17×3 只答出 `'5'`（`finish_reason=length`）；给 200 才正常返回 `51`。

---

## 5. 查找表（PLE / n-gram 表）放哪

引擎有专门的开关（`strata --help`）：

```
--ple-io direct|mmap|ram   n-gram table reads
   direct (默认): unbuffered SSD reads, the table never enters RAM or the file cache
   ram:           the whole table locked in RAM at start (mlock)
--ple-row-cache N          热点行缓存，90 B/行（默认 1048576 ≈ 94 MB）
--ple-inflight N           未完成 SSD 读（默认 256）
```

官方文档（`docs/DETAILS.md`）：**SSD 上的 28.8 GB n-gram 表，每 token 只读几行**，`direct` 是"为 SSD 设计"的默认；官方**只在机械盘上自动改 `ram`**（"Setup adds `--ple-io ram` itself on a rotational disk"）。

本机盘是 NVMe（`ROTA=0`）→ 属 `direct` 的目标场景，因此**保持默认**：
- 开 `ram` 需 **mlock 28.8 GB 不可回收**，会挤掉对话缓存（当前 available 仅约 32 GiB，而对话缓存要求至少 2.5 GiB 空闲）；
- 想优化也只能调 `--ple-row-cache`（默认 94 MB）。

---

## 6. 运维踩坑（全部实测）

| 坑 | 症状 | 根因 / 修法 |
|---|---|---|
| `setup.sh --setup` 重写配置 | 手加参数消失 | 实测丢掉 `--remote-expert-opt`、`--conversation-cache-mib 8192`、`--pcie-frac 0.00` **以及 `"host": "0.0.0.0"`**（丢了只监听 127.0.0.1，局域网直接连不上）。它只保留 `env.STRATA_SPLIT_OWN` 与 `open_browser`，并自留 `strata-<model>.json.bak`。**跑 `--setup` 后必须 diff 备份**。`./update.sh`（= `git pull` + `setup.py --update`）则不动配置。 |
| memlock 太小 | 无法 pin 专家 | `ulimit -Hl` = 8192（8 MiB）会挡住 71.73 GiB 的 pinning → 写 `/etc/security/limits.d/` 并重新登录；**systemd 单元必须显式 `LimitMEMLOCK=infinity`**（systemd 不读 limits.d）。 |
| 加载被节流 | 专家加载慢 **24 倍**（1.42 GiB/s → 0.05 GiB/s） | 官方记录：启动阶段被施加 CPU/IO 限制会触发。**systemd 单元不要加 CPUQuota / IOWeight / Sandbox / Private\*。** |
| 就绪判定 | 脚本白等 600 s | `ready:` 只进 tmux 面板 → 用 HTTP `/health`。 |
| HugePages 缺失 | `MAP_HUGETLB unavailable (needed 36727 2 MiB pages, vm.nr_hugepages=0)` | 回退到 THP（未进一步调优）。 |
| 重启即失 | 机器重启后无任何推理服务 | 跑在 detached tmux 会话里、无 systemd 单元；llama-server 单元已删。**待办：建 systemd 单元。** |
| `git fetch` 被掐断 | `GnuTLS recv error (-110)` | 需挂本机 ss-local 代理（`socks5h://127.0.0.1:1080`）才能 `git pull`。 |
| 视觉在 UD-Q4_K_XL 上不可用（≤0.1.40.1） | `/health` 显示 `"images": false` | 官方在 0.1.41 才接线（commit `af76b97`，issue #967；社区报告可用 #971）。**不是模型/硬件限制**——同一模型在另一台机器（gufo 引擎）本来就带图。 |

---

## 7. 复现

- 基准/切换脚本见仓库 `scripts/`（`strata_conc_ab.py` 即并发表所用的脚本）。
- 官方文档：[Strata repo](https://github.com/Niko1221/Strata) 的 `docs/DETAILS.md`、`docs/UNSLOTH_Q4.md`、`docs/MODELS.md`。
- 数据本机实测，日期见各原始文档；标注"推算"的条目未直接测量。

---

## 8. 免责与说明

- 全部数字为**单机单次实验**，不同提示内容/温度会显著改变 decode；对比时请保持同一口径。
- 文档已脱敏，不含内网地址、凭据与设备序列号。
