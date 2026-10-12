# Strata 升级记录：v0.1.41 → v0.1.43（2026-10-11）

本机：ai3080，3× RTX 3080 20GB（`sm_86`），Qwen3.8-Flash-Next UD-Q4_K_XL，262,144 上下文。

## 升级内容

```
本地 v0.1.41 (fb58e0d, 2026-10-08)  →  v0.1.43 (7695859a, 2026-10-11)
落后 307 个提交；新 tag：v0.1.42、v0.1.43
方式：官方 ./update.sh（= git pull --ff-only + setup.py --update；不启动模型、不动模型文件）
      git 必须走代理，否则 TLS 被掐断（GnuTLS recv error -110）
编译：ninja -j 24，138/138 目标通过；配置 json 未被改动
```

## 与本机相关的新特性（官方版本说明 + 提交）

| 变更 | 说明 |
|---|---|
| `STRATA_ROUTE_TAIL_SKIP=7` 在 CUDA 上默认开启 | 官方称 decode +13~16% |
| adaptive-tier copies overlap（CUDA） | v0.1.43 默认 |
| NaN 自愈护栏（#879 / #1892） | head logits 非有限值时丢缓存并从 token 0 重读，开销 <0.2% |
| CUDA `sm_86`：Q3_K 交错 kernel + sm_86 表 | 位逐相同；A4000 decode +2.0~2.7%（本机专家是 Q4_K/Q5_1，未必受益） |
| `STRATA_ROUTE_RESIDENT`（opt-in） | decode +1.6~2.4%（32 GB 卡）/ +5.7~7.8%（16 GB 等效） |
| prompt chunk 的 step records/token ids 改从 pinned 缓冲取 | #1712 |
| 修复 | #1576 卡序、#1616 stage 大小、#1595 CPU share、#1603 并发请求记账 |
| 新工具 | `tools/strata_bench.py` |

## 验证结果

| 项 | 结果 |
|---|---|
| 启动 | 就绪 **95 s** |
| 正确性三件套（Paris / 51 / Jupiter） | ✅ 全对（`max_tokens` 给足 200，否则思考块会吃掉预算） |
| 视觉 | ✅ `"images": true`；512×256 红方块 + "STRATA" 字样 → `1) Red 2) STRATA`，2.12 s；**旧的 `strata-vision` 未重编译也兼容** |
| 配置 | ✅ 手加参数全在（`--conversation-cache-mib 8192`、`--pcie-frac 0.00`、`--spec-min-p 0.5`、`host 0.0.0.0` 等） |
| 新特性生效 | ✅ 日志出现 `route tail skip: 46478 missed experts skipped` |

## Decode 对比（同一提示、同一方法学）

| 状态 | decode 中位 |
|---|---:|
| v0.1.41（升级前，已跑热） | **83.5 tok/s** |
| v0.1.43（刚启动，专家缓存命中率 90.5%） | 78.0 tok/s |
| v0.1.43（**跑热后**，命中率回到 98.3%） | **82.8 tok/s** |

→ **净变化 −0.8%，在噪声内**（单轮波动 ±7%）。

> ⚠️ **方法学教训（重要）**：刚启动的引擎会因为**专家缓存还没自适应**而虚低约 6.6%。**基准必须跑热再测**（预热若干轮，等命中率回到 96-99%）。

**官方称的 CUDA `route tail skip` +13~16% 未在本机复现。** 推断原因：该优化跳过的是**未命中专家**，而本机专家缓存命中率高达 **98.3%**，尾巴里几乎没有可跳的；官方的数字来自别的模型/卡（多为专家更小、miss 更多的 Q2_0 / IQ3_XXS 档）。

## 长文档口径的基线（v0.1.43，91.9k token 提示）

| 指标 | 值（4 次重复） |
|---|---|
| Prefill | **2,176 – 2,183 tok/s**（波动仅 **0.35%**） |
| 长上下文 decode | 92.6 – 93.6 tok/s |
| 草稿接受率 | 67.5% – 74.5%（**与输出内容强相关，不可跨提示比较**） |
| 同前缀追问（对话缓存） | 81,920 token 命中、只读 9,983 → **7.42 – 7.49 s**（冷读 42 s） |

→ 长文档口径的重复性远好于短提示（≤1%），是可靠的 A/B 平台。

## 回滚

```sh
cd ~/Strata && git checkout v0.1.41 && ./update.sh
cp strata-unsloth-ud-q4_k_xl.json.bak-before-v0143 strata-unsloth-ud-q4_k_xl.json
```

## 结论

**保留 v0.1.43**：性能中性（无回归），换取 NaN 自愈护栏与一批修复。
