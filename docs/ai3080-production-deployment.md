# ai3080 生产部署状态（2026-09-26 完成）

## 1. 交付结果

| 项目 | 状态 |
|---|---|
| 内核 | `7.0.0-31-generic`（唯一已安装内核，`GRUB_DEFAULT=0` 必然启动它） |
| 启动参数 | `pcie_aspm.policy=performance amd_iommu=off iommu=off pcie_acs_override=downstream,multifunction pci=realloc=on hpmmioprefsize=512G` 全部在位 |
| 驱动 | NVIDIA 610.57.04 开源内核模块 + P2P 补丁，`nvidia-smi topo -p2p` 4 卡 12 对全 `OK` |
| 服务 | `llama-server.service`，`enabled` + `active`，`Restart=always` |
| 开机自启 | **已验证**：系统启动后 **10 秒**服务即启动（`14:45:30` 启动 → `14:45:40` 服务 active） |
| 端口 | `0.0.0.0:8080` |
| 实测性能 | prefill **854.62 t/s** / decode **83.78 t/s** / MTP 接受率 0.82292（158/192，mean len 2.65） |
| 正确性 | Paris / 51 / Jupiter 三项全对 |
| 磁盘 | **51G 已用 / 232G，171G 可用**（累计清理释放 **17 GB**） |
| 对外端口 | 仅 `0.0.0.0:8080`（+ ssh 22）。**LAN 上零个临时端口** |

---

## 2. 生产配置（唯一真源：systemd 单元）

`/etc/systemd/system/llama-server.service`

```ini
[Unit]
Description=llama.cpp server - Qwen3.8-27B Q8_0 on 4x RTX 3080 (tensor split)
After=network.target

[Service]
Type=simple
User=<user>
Group=<user>
WorkingDirectory=/home/<user>/llama.cpp
ExecStart=/home/<user>/llama.cpp/build/bin/llama-server \
    -m /home/<user>/models/Qwen3.8-27B-Q8_0.gguf \
    --mmproj /home/<user>/models/mmproj-Qwen3.8-27B-f16.gguf \
    --host 0.0.0.0 --port 8080 \
    -c 262144 -ngl -1 \
    --split-mode tensor -ts 1,1,1,1 \
    -b 1024 -ub 512 -t 24 --threads-batch 48 \
    -fa on --cache-type-k f16 --cache-type-v f16 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --cont-batching --load-mode mmap --jinja --fit off \
    --image-min-tokens 1024
Restart=always
RestartSec=10
TimeoutStartSec=900
TimeoutStopSec=90
KillSignal=SIGINT
LimitNOFILE=1048576
OOMScoreAdjust=-500
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llama-server

[Install]
WantedBy=multi-user.target
```

### 补充单元：`/etc/systemd/system/llama-server.service.d/10-nccl-loopback.conf`

```ini
[Service]
# 单机 4 卡：NCCL 的 bootstrap 只走回环，不要占用局域网临时端口
Environment=NCCL_SOCKET_IFNAME=lo
```

**为什么必须加**：不加时 llama-server 会把 **25 个临时 TCP 端口绑定到 `<LAN-IP>`**（NCCL 的 bootstrap/transport 监听 socket），等于在局域网上开了一堆可探测的端口。单机 4 卡完全不需要走网卡。

加了之后：LAN 暴露端口 **25 → 0**，NCCL 全部收敛到回环，性能无损失（prefill 877 / decode 83.8）。

`~/.config/llama.cpp/config.ini` 已重写为**完全相同**的取值，因此手工运行 `llama-server` 与 systemd 启动行为一致。

### 为什么是这套参数

| 参数 | 取值 | 依据 |
|---|---|---|
| `--split-mode` | `tensor` | decode 82.4 vs layer 57.3（layer 换来 prefill +53% 但 decode −30%） |
| `--spec-draft-n-max` | **`2`** | 3 → 76.8、4 → 74.2；**2 是最优 82.4**，比默认 3 快 7.3% |
| `-b/-ub` | `1024/512` | 2048/1024 会让接受率从 0.823 掉到 0.662，decode −10~12% |
| `--cache-type-k/v` | `f16` | 与量化 KV 速度无差异，f16 精度更高 |
| `NCCL_P2P_DISABLE` | **不设** | P2P 已打通，设与不设实测无差别（866.6 vs 867.2），属历史调试残留 |
| `--spec-type draft-mtp` | 保留 | 比无 MTP 提速 **+56%**（52.92 → 82.40） |

---

## 3. 日常操作

```bash
systemctl status llama-server          # 查看状态
systemctl restart llama-server         # 重启（加载约 30 秒）
systemctl stop llama-server            # 停止
systemctl disable llama-server         # 取消开机自启
journalctl -u llama-server -f          # 实时日志
journalctl -u llama-server -n 100      # 最近 100 行
curl http://127.0.0.1:8080/health      # 健康检查
```

对外接口：`http://<LAN-IP>:8080`（OpenAI 兼容 `/v1/chat/completions`）

---

## 4. ⚠️ 维护红线

### 4.1 绝对不要设置 `GGML_CUDA_P2P=1`
硬 A/B 证据：
```
不设 → 'Paris'   83.48 t/s   ✅
设了 → 乱码      30.50 t/s   ❌
```
而且**会持久污染驱动状态**：跑过一次之后，所有后续 llama.cpp 进程（即使不带该变量）MTP 接受率归零、decode 掉 60%，重复 5 次全部复现，**只有重启能恢复**。

### 4.2 不要装/启动其它内核
内核 `7.0.0-34` 与 610.57.04 驱动组合会**静默损坏多卡数据**（同一份源码、同一编译器、同一份 config，干净重编：31 正常、34 乱码）。已清除 34 及内核元包，31 已 `apt-mark hold`。

**恢复已被清除的内核元包（如需内核更新）：**
```bash
sudo apt-mark unhold linux-image-7.0.0-31-generic linux-modules-7.0.0-31-generic \
                      linux-headers-7.0.0-31-generic linux-headers-7.0.0-31
sudo apt install linux-generic-hwe-26.04
# 之后必须用 --split-mode tensor 实测多卡输出是否仍然正确
```

### 4.3 不要用这些选项（均有实测反证）
| 选项 | 后果 |
|---|---|
| `-bs` / `--backend-sampling` | prefill −37%（1249 → 781），decode 仅 +2.6% |
| `--split-mode row` | 该模型加载失败 |
| `GGML_CUDA_ALLREDUCE=internal` | 此构建只支持 2 卡，4 卡退回 butterfly，慢 24% |
| `GGML_CUDA_ALLREDUCE=none` | 会退化成 meta-backend 蝴蝶归约 |

### 4.4 长提示场景的替代方案
提示 token 数 > 输出 token 数 **12 倍** 时，layer 模式总时间反超：
```
--split-mode layer --spec-draft-n-max 6
→ prefill 1259 t/s（+53%）| decode 57.3 t/s（−30%）
```
4001+256 场景下 7627 ms vs tensor 的 8012 ms。RAG / 长文档可考虑。

---

## 5. P2P 驱动补丁包（不可再生内容）

`/home/<user>/nvidia-p2p-driver-kit/`
```
p2p-gm107.patch            1.5 KB   —— 修改 kern_bus_gm107.c 的 unified diff（含完整上下文与原因注释）
patch-libcuda-p2p.py       4.6 KB   —— 修改用户态 libcuda.so 的 P2P 判断
BUILD.md                            —— 从零重建的完整步骤 + 下载地址 + 运行环境要求
PATCH-NOTES.txt            2.1 KB   —— 当初定位此 bug 的原始记录（症状/根因/修复/代价/实测结果）
kern_bus_gm107.c.orig    229829 B   —— 未打补丁的原始文件（参照基准）
kern_bus_gm107.c.patched 230411 B   —— 打过补丁后的文件

已交叉校验：`kern_bus_gm107.c.orig` 用 `p2p-gm107.patch` 可干净还原出 `.patched`（`patch -p1 --dry-run` 通过）。
```

原两棵源码树（`/usr/src/nvidia-p2p-610.57.04` 554M + `~/open-gpu-kernel-modules` 684M）已删除，补丁内容完整保存于上述文件。

---

## 6. 清理明细

### 已删除（累计释放 13 GB）
| 项目 | 体积 |
|---|---|
| **`/root/` 下 8 个上一轮排查的备份目录** | **240M** |
| ↳ `gpu-p2p-backup-20260923-142718`（dkms 595 模块 tar + kernel-open 构建 + 包清单） | 161M |
| ↳ `gpu-recovery-20260923-134816`（恢复前 dmesg/nvidia-smi + 模块 tar + modprobe.d 副本） | 36M |
| ↳ `p2p-fix-record`（补丁记录 + nvidia.ko 副本） | 37M |
| ↳ `p2p-driver-patch-backup` / `nvidia-config-backup` / `p2p-test-mailbox` / `p2p-test-prehopper` / `Documents/Nsight` / `netplan-backup` | 2.5M |
| **systemd journal**（本次全部调试日志） | **3.8G** |
| **`/var/cache/apt/archives`** | **453M** |
| **`/var/cuda-repo-ubuntu2404-13-2-local`**（CUDA 本地 apt 源，安装介质）+ 对应 `.list` | **3.9G** |
| **三个内核 PPA**（`canonical-kernel-team/ppa`、`canonical-kernel-team/unstable`、`vicamo/ppa-2133799`）—— 34/39 内核的来源，也是隐患源头 | — |
| **`/var/crash/` 全部崩溃转储**（含 `nvidia-dkms-595-open`、sudo 崩溃、add-apt-repository 崩溃） | 1.2M |
| **残缺 journal**（硬断电/内核 panic 留下的 `*.journal~`） | 55M |
| **孤立 drop-in** `/etc/systemd/system/nvidia-persistenced.service.d/`（该单元并不存在，毫无作用） | — |
| `cuda-repo-...13.2.1...deb` | 3.9G |
| `open-gpu-kernel-modules/`（原始源码树） | 684M |
| `/usr/src/nvidia-p2p-610.57.04/`（补丁源码树） | 554M |
| `p2p-build/` | 554M |
| `~/.cache/` | 584M |
| `nccl_2.29.7.../` | 555M |
| `nccl_2.29.7....txz` | 375M |
| `nccl-tests/` | 293M |
| `llama.cpp/build-nopeer/` | 187M |
| `llama.cpp-0.4.0.zip` / `master.zip` | 42M |
| `llama-bench` 及 impl 库 | — |
| `llama.log` / `llama_server.log` / `llama_tensor.log` | — |
| `start_llama*.sh` / `run_server.py` | — |
| `Documents/NVIDIA Nsight Compute` / `.nv/ComputeCache` | 2.2M |
| `/lib/modules/7.0.0-34-generic`（含坏掉的 nvidia 模块） | 100M |
| `/lib/modules/7.0.0-30-generic` | 7.3M |
| 内核 34 全套包 + 内核元包 + `linux-tools` / `linux-perf` / `bpftool` | — |
| 所有 `/tmp` 调试脚本与日志 | — |

### 保留（理由）
| 项目 | 体积 | 理由 |
|---|---|---|
| `NVIDIA-Linux-x86_64-610.57.04.run` | 442M | 驱动安装介质，也是 `patch-libcuda-p2p.py` 的作用对象，离线重建必需 |
| `llama.cpp/`（含 `build/`） | 681M | 运行体本身 |
| `models/` | 29G | 模型与 mmproj |

### 待你决定
- **`llama` snap**（antonmedv，1.4.0）—— 与本服务无关的第三方 LLM 工具，未在运行。要删：`sudo snap remove llama`
- **`hwctl` snap**（ce-certification-qa）—— 硬件认证测试工具，socket-activated 且 inactive。要删：`sudo snap remove hwctl`
- **`~/.bash_history`**（46 KB）—— 这次排查的完整命令记录。要清：`truncate -s 0 ~/.bash_history`

---

## 7. 本次事故与修复（我造成的，已修复）

清 `rc` 状态包时连带删掉了 `/etc/default/grub`（该文件归属 `grub-pc`），导致**内核启动参数一次性丢失**，`iommu=off` / `pcie_acs_override` 等 P2P 依赖项消失。

**已完整恢复并验证**：`/etc/default/grub` 重建（含全部原始参数 + `GRUB_DISTRIBUTOR` 恢复 `Ubuntu` 命名），`update-grub` 重新生成，重启后 `/proc/cmdline` 六项 P2P 参数全部在位，`nvidia-smi topo -p2p` 12 对仍全 `OK`，服务 10 秒自启。

> 教训：`apt purge` 处理 `rc` 状态包前应先确认 `dpkg -S` 归属，`grub-pc` 的 conffile 列表包含 `/etc/default/grub`。

---

## 8. GRUB 现状

```bash
$ grep -vE '^#|^$' /etc/default/grub
GRUB_DEFAULT=0
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=5
GRUB_DISTRIBUTOR=`( . /etc/os-release && echo ${NAME} )`
GRUB_CMDLINE_LINUX_DEFAULT="pcie_aspm.policy=performance amd_iommu=off iommu=off pcie_acs_override=downstream,multifunction pci=realloc=on hpmmioprefsize=512G"
GRUB_CMDLINE_LINUX=""
```

`GRUB_DEFAULT=0` 是健壮写法：**系统只装了 `7.0.0-31-generic` 一个内核**（`/lib/modules/` 已验证），
且 31 已 `apt-mark hold`、内核元包已清除，不会再有新内核进来把第一项换掉。

---

## 9. 内核 34 清除核验（逐项）

| 检查点 | 结果 |
|---|---|
| 正在运行的内核 | `7.0.0-31-generic` |
| `dpkg` 中 34 的内核包（image / modules / headers / generic） | **0 个** ✅ |
| `/boot/vmlinuz-*`、`initrd.img-*`、`config-*` | 只有 31 ✅ |
| `/lib/modules/` | 只有 `7.0.0-31-generic` ✅ |
| `/usr/lib/modules/` | 只有 `7.0.0-31-generic` ✅ |
| `/usr/src/linux-headers-*` | 只有 31 ✅ |
| 34 的内核模块目录（含那份会损坏数据的 nvidia.ko） | 已删除 ✅ |
| GRUB 菜单项 | 只有 31（普通 + recovery）+ UEFI 设置 ✅ |
| `dpkg` rc 状态残留包 | 0 个 ✅ |
| 内核元包（`linux-generic` / `linux-image-generic` / `linux-headers-generic`） | 0 个 ✅ |

### 唯一剩下的 "34" 字样：`linux-libc-dev`
```
ii  linux-libc-dev:amd64  7.0.0-34.34
Description: Linux Kernel Headers for development
```
**这不是内核，是给用户态程序编译用的内核头文件**（`/usr/include/linux/*`）。它与运行中的内核版本不一致是完全正常的，删掉会破坏编译环境。**保留。**

`linux-tools-7.0.0-34*`、`linux-perf`、`bpftool` 等调试工具已一并清除。

### 仍然存在的 "backup/old" 文件名（均为发行版自带，非本次排查产物）
```
/etc/.resolv.conf.systemd-resolved.bak     systemd-resolved 自带
/etc/apt/sources.list.d/ubuntu.sources.curtin.orig   cloud-init/curtin
/etc/apt/preferences.d.save/               Ubuntu Pro ESM 偏好设置（目录）
/etc/xml/{catalog,xml-core.xml,polkitd.xml}.old      xml-core 包自带
/var/backups/                              dpkg/apt 系统备份（标准机制）
/etc/lvm/backup/                           LVM 元数据备份（重要，勿删）
```

---

## 10. 最终干净度审计（重启后端到端实测）

### 已核实干净的项

| 检查项 | 结果 |
|---|---|
| 内核 | 只有 `7.0.0-31-generic`；`/boot`、`/lib/modules`、`/usr/lib/modules`、`/usr/src` 均只剩 31 |
| 非 31 内核包 | **0 个** |
| GRUB 菜单 | 只有 31（普通 + recovery）+ UEFI |
| 内核启动参数 | 6 项 P2P 依赖参数全部在位 |
| P2P | 4 卡 12 对全部 `OK` |
| apt 源 | 只剩 `ubuntu.sources`、`tailscale.list`（+ 系统自带 `.curtin.orig`）；**PPA 残留 0 个** |
| `apt update` | 退出码 0，无告警 |
| `/root` | 只剩 `.bashrc .config .gnupg .profile .ssh snap` |
| `/tmp` | 空（仅剩 systemd/snap 私有目录） |
| `/var/crash` | 清空；**apport 已 mask**（桌面崩溃上报，服务器不需要） |
| journal | 3.9G → **61.8M** |
| `/var/log` | 208M → **97M** |
| 监听端口 | 仅 `0.0.0.0:8080` + ssh 22；**LAN 暴露的临时端口 0 个**（NCCL 已收敛到 `127.0.0.1`） |
| 磁盘 | 68G → **51G 已用**，171G 可用 |
| 服务 | `enabled` + `active`，重启后 **11 秒**自动启动，3/3 推理正确，prefill 877 / decode 83.8 |

### 保留的正当组件（不是调试痕迹）

| 组件 | 理由 |
|---|---|
| `nvidia-pm.service` | 开启 GPU persistence mode（4 卡均为 `Enabled`），对常驻推理服务有益 |
| `tailscaled` | 你的远程接入方式（`<TAILSCALE-IP>` 就是 SSH 入口） |
| `/etc/modprobe.d/nvidia-p2p.conf` | P2P 必需 |
| `/etc/udev/rules.d/70-nvidia-nodes.rules` | 生成 `/dev/nvidia*` 设备节点，驱动必需 |
| `/opt/nvidia`、`/usr/local/cuda-13.2` | 驱动与 CUDA 运行库本体 |
| `NVIDIA-Linux-x86_64-610.57.04.run` | 驱动安装介质（`patch-libcuda-p2p.py` 的作用对象），离线重建必需 |

### 仍属你个人、我未擅自处理的项

| 项目 | 现状 | 如需清理 |
|---|---|---|
| `ss-local.service`（shadowsocks 代理） | active，监听 `127.0.0.1:1080` | `sudo systemctl disable --now ss-local` |
| `llama` snap（antonmedv，1.4.0） | 已装未运行，与本服务无关 | `sudo snap remove llama` |
| `hwctl` snap（0.11.1） | socket-activated 且 inactive | `sudo snap remove hwctl` |
| `~/.bash_history`（1161 行） | 完整记录了本次排查过程 | `truncate -s 0 ~/.bash_history` |
| `apport` 想恢复 | 已 mask | `sudo systemctl unmask apport && sudo systemctl enable --now apport` |

### 一个使用提示

你用 `tail -n 0 -F /home/<user>/llama_tensor.log` 在跟日志，但**那个文件已经不存在了**（现在是 systemd 托管，日志走 journal）。请改用：

```bash
journalctl -u llama-server -f          # 实时
journalctl -u llama-server -n 200      # 最近 200 行
```


