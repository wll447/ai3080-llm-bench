#!/bin/bash
# ai3080: 重启 Strata（262144 上下文 + prefill auto:32768 + conversation-cache 8192 + 思考档 medium）
# 在远程用 `base64 -d | bash -s` 执行，避免引号/变量展开问题。
set -u
LOG=$HOME/Strata/strata-unsloth-ud-q4_k_xl.log
N=$(wc -l < "$LOG")
echo "=== old log lines: $N ==="

PID=$(pgrep -f 'serve/server.py' | head -1)
echo "=== stopping server.py pid=${PID:-none} ==="
if [ -n "${PID:-}" ]; then kill -INT "$PID"; fi
for _ in $(seq 1 15); do
  pgrep -f 'serve/server.py' >/dev/null || break
  sleep 2
done
if pgrep -f 'serve/server.py' >/dev/null; then
  echo "=== server.py still alive -> SIGTERM ==="
  pkill -TERM -f 'serve/server.py'
  sleep 8
fi
pgrep -af 'serve/server.py' || echo "=== old server stopped ==="

# 引擎子进程若残留，一并收拾（否则新实例会因显存/端口起不来）
EPID=$(pgrep -f 'engine/strata' | head -1)
if [ -n "${EPID:-}" ]; then
  echo "=== leftover engine pid=$EPID -> SIGINT ==="
  kill -INT "$EPID"
  for _ in $(seq 1 10); do pgrep -f 'engine/strata' >/dev/null || break; sleep 2; done
  pgrep -af 'engine/strata' && { echo "=== engine still alive -> SIGTERM ==="; pkill -TERM -f 'engine/strata'; sleep 5; }
fi
pgrep -af 'engine/strata' || echo "=== no engine running ==="

sleep 2
tmux kill-session -t strata-262k 2>/dev/null
tmux new-session -d -s strata-262k "cd $HOME/Strata && exec ./run-unsloth-ud-q4_k_xl.sh"
echo "=== launched in detached tmux session 'strata-262k' ==="

READY=no
for i in $(seq 1 54); do
  if tail -n +$((N+1)) "$LOG" | grep -q 'ready: http'; then
    echo "=== READY after about $((i*10))s ==="
    READY=yes
    break
  fi
  sleep 10
done
[ "$READY" = no ] && echo "=== NOT ready within ~9 min ==="

echo "=== new log tail ==="
tail -n +$((N+1)) "$LOG" | tail -25
echo "=== tmux ==="
tmux ls
echo "=== processes ==="
pgrep -af 'serve/server.py' | head -2
pgrep -af 'engine/strata' | head -1 | cut -c1-120
