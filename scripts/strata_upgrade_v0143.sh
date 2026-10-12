#!/bin/bash
# ai3080: Strata v0.1.41 → v0.1.43 升级（官方 update.sh，不启动模型）
set -u
cd /home/<user>/Strata

echo "===== 0. 备份配置 ====="
cp -v strata-unsloth-ud-q4_k_xl.json strata-unsloth-ud-q4_k_xl.json.bak-before-v0143

echo
echo "===== 1. 停服务 ====="
tmux kill-session -t strata-262k 2>/dev/null || true
pkill -INT -f '[s]erve/server.py' 2>/dev/null || true
pkill -INT -f '[e]ngine/strata' 2>/dev/null || true
sleep 8
if pgrep -f '[s]erve/server.py' >/dev/null; then echo "警告：server.py 仍在"; else echo "server.py 已停"; fi
if pgrep -f '[e]ngine/strata' >/dev/null; then echo "警告：engine 仍在"; else echo "engine 已停"; fi
curl -s -m 3 http://127.0.0.1:8080/health && echo "警告：8080 仍有响应" || echo "8080 已关闭"

echo
echo "===== 2. update.sh（走 ss-local 代理）====="
export ALL_PROXY=socks5h://127.0.0.1:1080
export https_proxy=socks5h://127.0.0.1:1080
timeout 3000 ./update.sh > /tmp/strata_update.log 2>&1
rc=$?
unset ALL_PROXY https_proxy
echo "update.sh 退出码 = $rc"
echo "--- 输出尾部 30 行 ---"
tail -30 /tmp/strata_update.log

echo
echo "===== 3. 版本确认 ====="
echo -n "git:    "; git describe --tags --always
echo -n "engine: "; ./engine/strata --version 2>&1 | head -1

echo
echo "===== 4. 配置是否被动过（应无差异）====="
if diff strata-unsloth-ud-q4_k_xl.json.bak-before-v0143 strata-unsloth-ud-q4_k_xl.json; then
  echo "(无差异 ✅ 手加参数都在)"
fi

echo
echo "===== 5. 关键文件时间戳（看谁被重编译）====="
ls -la --time-style=+%H:%M:%S engine/strata engine/strata-vision 2>/dev/null
echo "--- 现在时间: $(date +%H:%M:%S) ---"
