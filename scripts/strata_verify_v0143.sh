#!/bin/bash
# ai3080: v0.1.43 升级后验证（启动 + 正确性 + 吞吐对比 + 视觉）
set -u
cd /home/<user>/Strata

echo "===== 1. 启动 ====="
tmux kill-session -t strata-262k 2>/dev/null || true
tmux new-session -d -s strata-262k 'cd /home/<user>/Strata && exec ./run-unsloth-ud-q4_k_xl.sh'
for i in $(seq 1 84); do
  if curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"loaded": true'; then
    echo "READY，约 $((i*5)) s"; break
  fi
  sleep 5
done

echo
echo "===== 2. health（看 images 与版本）====="
curl -s -m 5 http://127.0.0.1:8080/health; echo
curl -s -m 5 http://127.0.0.1:8080/metrics | ./.venv/bin/python -c 'import sys,json;d=json.load(sys.stdin);e=d["engine"];print("engine=%s slots=%s cache=%sMiB arena=%sMiB pcie=%s spec_min_p=%s conv_cache=%sMiB" % (e["version"],e["expert_slots"],e["expert_cache_mib"],e["arena_mib"],e["pcie_frac"],e["spec_min_p"],e["conversation_cache_mib"]))' 2>/dev/null

echo
echo "===== 3. 正确性三件套 + decode（与升级前同一口径）+ 图片 ====="
./.venv/bin/python - <<'PY'
import base64, json, time, urllib.request, uuid, statistics
MODEL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
API = "http://127.0.0.1:8080/v1/chat/completions"

def ask(msgs, mt):
    b = {"model": MODEL, "temperature": 0, "max_tokens": mt, "messages": msgs}
    r = urllib.request.Request(API, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=1800)); dt = time.time() - t0
    return dt, d

# 正确性（max_tokens 给足，避免被思考块吃掉）
for q, exp in [("Answer in one word: the capital of France is", "Paris"),
               ("What is 17*3? Answer with just the number.", "51"),
               ("Answer in one word: the largest planet in the solar system is", "Jupiter")]:
    dt, d = ask([{"role": "user", "content": q}], 200)
    got = (d["choices"][0]["message"].get("content") or "").strip()
    print("  %-8s -> %-12r (%.1fs)" % (exp, got[:40], dt))

# decode 3 轮（与升级前完全相同的提示模板）
ask([{"role": "user", "content": "Say hi."}], 8)
res = []
for i in range(3):
    k = uuid.uuid4().hex[:8]
    dt, d = ask([{"role": "user",
                  "content": "Watch log %s entry %d: describe a storm at the lighthouse in about 200 words." % (k, i)}], 256)
    ct = d["usage"]["completion_tokens"]
    res.append(ct / dt)
    print("  decode 第%d轮: %.1f tok/s (%d tok / %.1fs)" % (i + 1, ct / dt, ct, dt))
med = statistics.median(res)
print("  >>> v0.1.43 中位 decode = %.1f tok/s   （升级前 v0.1.41 = 83.5）" % med)
print("  >>> 变化 = %+.1f%%" % ((med / 83.5 - 1) * 100))

# 图片（复用之前的测试图；没有就现造）
try:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 256), "white")
    dr = ImageDraw.Draw(img)
    dr.rectangle([20, 30, 140, 150], fill=(220, 30, 30))
    dr.text((180, 120), "STRATA", fill="black")
    img.save("/tmp/vision_test.png")
except Exception as e:
    print("  造图失败:", e)
try:
    b64 = base64.b64encode(open("/tmp/vision_test.png", "rb").read()).decode()
    msg = {"role": "user", "content": [
        {"type": "text", "text": "Answer briefly: 1) what color is the square? 2) what word is written?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}
    dt, d = ask([msg], 128)
    print("  【图片】%.2fs -> %r" % (dt, (d["choices"][0]["message"].get("content") or "")[:120]))
except Exception as e:
    print("  【图片】失败:", e)
PY

echo
echo "===== 4. 日志里的结构/新特性 ====="
tail -1200 strata-unsloth-ud-q4_k_xl.log | grep -a -e 'layer split auto: K=' -e 'expert cache auto' -e 'route tail' -e 'ROUTE_TAIL' | tail -5
echo "--- 日志尾 3 行 ---"
tail -3 strata-unsloth-ud-q4_k_xl.log
echo "--- 显存 ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- 内存 ---"
free -g | head -2
