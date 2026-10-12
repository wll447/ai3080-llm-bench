#!/bin/bash
# ai3080: 长文档口径 A/B —— 基线 / spec-min-p 0.8 / mtp-hnorm stream / prefill auto:32768
set -u
cd /home/<user>/Strata
CFG=strata-unsloth-ud-q4_k_xl.json
LOG=strata-unsloth-ud-q4_k_xl.log
PY=./.venv/bin/python

cp -f "$CFG" "${CFG}.bak-before-ab-longdoc"
echo "已备份配置 -> ${CFG}.bak-before-ab-longdoc"

echo "=== 生成长文档提示（唯一，约 30k token）==="
$PY - <<'PY'
import uuid, random
random.seed(20261011)
lines = []
for i in range(2200):
    k = uuid.uuid4().hex[:8]
    lines.append("Entry %s #%d: the keeper logged wind %d kn, swell %d m, visibility %d m at station %d."
                 % (k, i, random.randint(2, 45), random.randint(0, 6), random.randint(50, 9000), random.randint(1, 40)))
txt = "\n".join(lines)
open("/tmp/longdoc.txt", "w").write(txt)
print("  行数 %d，字符 %d" % (len(lines), len(txt)))
PY

set_args() {   # $1=spec-min-p  $2=mtp-hnorm("" 删除)  $3=prefill值
  $PY - "$1" "$2" "$3" <<'PY'
import json, sys
smp, hn, pf = sys.argv[1], sys.argv[2], sys.argv[3]
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
setarg("--spec-min-p", smp)
setarg("--mtp-hnorm", hn)
setarg("--prefill", pf)
json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
print("  生效: --spec-min-p=%s  --mtp-hnorm=%s  --prefill=%s" % (smp, hn or "(pooled)", pf))
assert a[a.index("--spec-min-p") + 1] == smp, "spec-min-p 未写入"
assert ("--mtp-hnorm" in a) == (hn != ""), "mtp-hnorm 未写入"
assert a[a.index("--prefill") + 1] == pf, "prefill 未写入"
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
import json, time, urllib.request, uuid
MODEL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
API = "http://127.0.0.1:8080/v1/chat/completions"
DOC = open("/tmp/longdoc.txt").read()
def ask(p, mt):
    b = {"model": MODEL, "temperature": 0, "max_tokens": mt, "messages": [{"role": "user", "content": p}]}
    r = urllib.request.Request(API, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=3600)); dt = time.time() - t0
    return dt, d
ask("Say hi.", 8)   # 预热
# ① 长文档首轮
dt, d = ask(DOC + "\n\nQuestion %s: in one sentence, what did the keeper log?" % uuid.uuid4().hex[:8], 256)
u = d["usage"]
print("  长文档首轮: 提示 %s tok, 输出 %s tok, 墙钟 %.1fs" % (u.get("prompt_tokens"), u.get("completion_tokens"), dt))
# ② 同前缀追问（测对话缓存复用）
dt2, d2 = ask(DOC + "\n\nQuestion %s: reply with just the word OK." % uuid.uuid4().hex[:8], 8)
u2 = d2["usage"]
print("  同前缀追问: 提示 %s tok, 输出 %s tok, 墙钟 %.2fs" % (u2.get("prompt_tokens"), u2.get("completion_tokens"), dt2))
PY
}

run_arm() {  # $1=标签 $2=smp $3=hnorm $4=prefill
  echo
  echo "##################### ARM: $1 #####################"
  set_args "$2" "$3" "$4"
  if ! restart; then echo "!! $1 启动失败"; return; fi
  sleep 3
  L=$(wc -l < "$LOG")
  measure
  echo "  --- 本组新增日志（关键行）---"
  tail -n +$((L + 1)) "$LOG" | grep -a -e 'prompt .* tokens =' -e 'reads in' | tail -6 | sed 's/^/  /'
}

# 基线已由 4 次重复实验确立（prefill 2175-2183 tok/s、decode 92.6-93.6、追问 7.42-7.49 s），本轮不重复跑
run_arm "F spec-min-p=0.8" "0.8" "" "auto"
run_arm "G mtp-hnorm=stream" "0.5" "stream" "auto"
run_arm "H prefill auto:32768" "0.5" "" "auto:32768"

echo
echo "##################### 恢复原配置并重启 #####################"
cp -f "${CFG}.bak-before-ab-longdoc" "$CFG"
restart && echo "已恢复原配置并启动"
$PY -c 'import json;d=json.load(open("strata-unsloth-ud-q4_k_xl.json"));print("  恢复后 args:"," ".join(d["args"]))'
curl -s -m 5 http://127.0.0.1:8080/health; echo
echo "ALL DONE"
