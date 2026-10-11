#!/bin/bash
# ai3080：机器重启后把 Strata 拉起来（HTTP /health 探测就绪）
set -u
cd /home/<user>/Strata
tmux kill-session -t strata-262k 2>/dev/null || true
tmux new-session -d -s strata-262k 'cd /home/<user>/Strata && exec ./run-unsloth-ud-q4_k_xl.sh'
echo "=== 已在 detached tmux 会话 strata-262k 启动 ==="

READY=no
for i in $(seq 1 72); do
  if curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"loaded": true'; then
    echo "=== READY，用时约 $((i*5)) s ==="
    READY=yes
    break
  fi
  sleep 5
done
[ "$READY" = no ] && echo "=== 6 分钟内未就绪 ==="

echo "--- health ---"
curl -s -m 5 http://127.0.0.1:8080/health; echo
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- tmux ---"
tmux ls 2>&1
echo "--- 本次启动的分层与缓存 ---"
tail -400 /home/<user>/Strata/strata-unsloth-ud-q4_k_xl.log | grep -a -e 'layer split auto: K=' -e 'caches hold' -e 'experts loaded' | tail -4
echo "--- 日志尾 ---"
tail -3 /home/<user>/Strata/strata-unsloth-ud-q4_k_xl.log
echo "--- 内存 ---"
free -g | head -2
