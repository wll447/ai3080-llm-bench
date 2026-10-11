#!/bin/bash
# 验证：文本正确性 + 真的发一张图
set -u
cd /home/<user>/Strata
PY=./.venv/bin/python
[ -x "$PY" ] || PY=python3

echo "=== 1. 造一张测试图（红方块 + 黑字 STRATA）==="
$PY - <<'PY'
from PIL import Image, ImageDraw
img = Image.new("RGB", (512, 256), "white")
d = ImageDraw.Draw(img)
d.rectangle([20, 30, 140, 150], fill=(220, 30, 30))
d.text((180, 120), "STRATA", fill="black")
img.save("/tmp/vision_test.png")
print("图已生成 /tmp/vision_test.png 512x256")
PY

echo
echo "=== 2. 文本正确性三件套 + 图片测试 ==="
$PY - <<'PY'
import base64, json, time, urllib.request
MODEL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"

def ask(msgs, mt=64):
    body = {"model": MODEL, "temperature": 0, "max_tokens": mt, "messages": msgs}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=600))
    return time.time() - t0, d["choices"][0]["message"]["content"]

for q, exp in [("Answer in one word: the capital of France is", "Paris"),
               ("Answer in one word: the largest planet in the solar system is", "Jupiter"),
               ("What is 17*3? Answer with just the number.", "51")]:
    try:
        dt, ans = ask([{"role": "user", "content": q}], 32)
        print("  %.2fs  期望 %-8s 实得 %r" % (dt, exp, ans.strip()[:40]))
    except Exception as e:
        print("  FAILED:", e)

try:
    b64 = base64.b64encode(open("/tmp/vision_test.png", "rb").read()).decode()
    msg = {"role": "user", "content": [
        {"type": "text", "text": "Answer briefly: 1) what color is the square? 2) what word is written in the image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}
    dt, ans = ask([msg], 128)
    print("  【图片】%.2fs -> %r" % (dt, ans.strip()[:250]))
except Exception as e:
    print("  【图片】FAILED:", e)
PY

echo
echo "=== 3. 引擎/编码器日志里的图片痕迹 ==="
tail -60 strata-unsloth-ud-q4_k_xl.log | grep -a -i -e vision -e image | tail -8
echo
echo "=== 4. 显存（编码器上卡后）==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
