#!/bin/bash
# ai3080: short-read 4 组 + 抖音组合 的 A/B（每组重启→预热→测）
# 指标：decode tok/s（3 轮中位）、草稿接受率（日志）、短提示延迟（max_tokens=1 墙钟 + 引擎 prompt_ms）
set -u
cd /home/<user>/Strata
CFG=strata-unsloth-ud-q4_k_xl.json
LOG=strata-unsloth-ud-q4_k_xl.log
PY=./.venv/bin/python

cp -f "$CFG" "${CFG}.bak-before-ab-shortread"
echo "已备份配置 -> ${CFG}.bak-before-ab-shortread"

set_args() {   # $1=short-read("" 表示删除)  $2=spec-min-p  $3=mtp-hnorm("" 表示删除)
  $PY - "$1" "$2" "$3" <<'PY'
import json, sys
sr, smp, hn = sys.argv[1], sys.argv[2], sys.argv[3]
p = "strata-unsloth-ud-q4_k_xl.json"
d = json.load(open(p)); a = d["args"]

def setarg(flag, val):
    if val == "":
        while flag in a:
            i = a.index(flag)
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                del a[i:i + 2]
            else:
                del a[i:i + 1]
        return
    if flag in a:
        i = a.index(flag)
        if i + 1 < len(a) and not a[i + 1].startswith("--"):
            a[i + 1] = val
        else:
            a.insert(i + 1, val)
    else:
        a.extend([flag, val])

setarg("--short-read", sr)
setarg("--spec-min-p", smp)
setarg("--mtp-hnorm", hn)
json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
print("  生效: --short-read=%s  --spec-min-p=%s  --mtp-hnorm=%s"
      % (sr or "(默认64)", smp, hn or "(pooled)"))
PY
}

restart() {
  tmux kill-session -t strata-262k 2>/dev/null || true
  pkill -INT -f '[s]erve/server.py' 2>/dev/null || true
  pkill -INT -f '[e]ngine/strata' 2>/dev/null || true
  sleep 6
  tmux new-session -d -s strata-262k 'cd /home/<user>/Strata && exec ./run-unsloth-ud-q4_k_xl.sh'
  for i in $(seq 1 84); do
    curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"loaded": true' && return 0
    sleep 5
  done
  return 1
}

measure() {
  $PY - <<'PY'
import json, time, urllib.request, uuid, statistics
MODEL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
API = "http://127.0.0.1:8080/v1/chat/completions"
def ask(p, mt):
    b = {"model": MODEL, "temperature": 0, "max_tokens": mt, "messages": [{"role": "user", "content": p}]}
    r = urllib.request.Request(API, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=1800)); dt = time.time() - t0
    return dt, d["usage"]["completion_tokens"]

# 预热：1 短 + 3 长（让专家缓存自适应）
ask("Say hi.", 8)
for i in range(3):
    ask("Warm-up %s: describe a calm sea in about 200 words." % uuid.uuid4().hex[:8], 256)

# ① decode：3 轮 256 token 唯一 nonce
dec = []
for i in range(3):
    k = uuid.uuid4().hex[:8]
    dt, ct = ask("Watch log %s entry %d: describe a storm at the lighthouse in about 200 words." % (k, i), 256)
    dec.append(ct / dt)
print("  decode: %s  -> 中位 %.1f tok/s" % (" / ".join("%.1f" % x for x in dec), statistics.median(dec)))

# ② 短提示延迟：5 条 max_tokens=1
lat = []
for i in range(5):
    k = uuid.uuid4().hex[:8]
    t0 = time.time()
    ask("Log %s item %d: reply with the single word OK." % (k, i), 1)
    lat.append((time.time() - t0) * 1000)
print("  短提示(1tok)墙钟: %s ms  -> 中位 %.0f ms" % (" / ".join("%.0f" % x for x in lat), statistics.median(lat)))

# ③ 引擎自报的 prompt 计时（最后 5 条请求）
try:
    m = json.load(urllib.request.urlopen("http://127.0.0.1:8080/metrics", timeout=10))
    rs = m.get("requests", [])[-5:]
    pms = [r.get("prompt_ms") for r in rs if r.get("prompt_ms") is not None]
    pts = [r.get("prompt_tokens") for r in rs]
    if pms:
        print("  引擎 prompt_ms: %s  -> 中位 %.0f ms (prompt_tokens=%s)"
              % (" / ".join("%.0f" % x for x in pms), statistics.median(pms), pts))
except Exception as e:
    print("  metrics 读取失败:", e)
PY
}

run_arm() {  # $1=标签 $2=short-read $3=spec-min-p $4=mtp-hnorm
  echo
  echo "##################### ARM: $1 #####################"
  set_args "$2" "$3" "$4"
  if ! restart; then echo "!! $1 启动失败"; return; fi
  sleep 3
  L=$(wc -l < "$LOG")
  measure
  echo "  --- 本组草稿接受率（新增日志）---"
  tail -n +$((L + 1)) "$LOG" | grep -a 'drafts accepted' | tail -4 | sed 's/^/  /'
}

run_arm "A 基线（默认 short-read=64 / spec-min-p 0.5 / hnorm pooled）" "" "0.5" ""
run_arm "B short-read=32" "32" "0.5" ""
run_arm "C short-read=0（关闭）" "0" "0.5" ""
run_arm "D short-read=640" "640" "0.5" ""
run_arm "E 抖音组合（short-read=32 + spec-min-p=0.8 + hnorm stream）" "32" "0.8" "stream"

echo
echo "##################### 恢复原配置并重启 #####################"
cp -f "${CFG}.bak-before-ab-shortread" "$CFG"
restart && echo "已恢复原配置并启动"
$PY -c 'import json;d=json.load(open("strata-unsloth-ud-q4_k_xl.json"));print("  恢复后 args:"," ".join(d["args"]))'
curl -s -m 5 http://127.0.0.1:8080/health; echo
echo "ALL DONE"
