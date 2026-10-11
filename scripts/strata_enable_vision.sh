#!/bin/bash
# Phase 4：给 ai3080 的 Strata 打开看图能力（GPU 编码器）
set -u
cd /home/<user>/Strata

export HF_ENDPOINT=https://hf-mirror.com
unset ALL_PROXY all_proxy https_proxy HTTPS_PROXY http_proxy HTTP_PROXY 2>/dev/null || true

echo "===== 跑 setup（--vision gpu --no-start）====="
timeout 2400 ./setup.sh --setup --family unsloth --model UD-Q4_K_XL \
  --gpus 0,1,2 --context 262144 --vision gpu --no-start --yes \
  < /dev/null > /tmp/setup_vision.log 2>&1
rc=$?
echo "setup 退出码 = $rc"
echo "--- 输出尾部 60 行 ---"
tail -60 /tmp/setup_vision.log

echo
echo "===== 配置差异（备份 vs 现在）====="
if diff strata-unsloth-ud-q4_k_xl.json.bak-before-vision strata-unsloth-ud-q4_k_xl.json; then
  echo "(无差异，手加参数都在)"
fi

echo
echo "===== 关键字段现状 ====="
python3 - <<'PY'
import json
d = json.load(open("strata-unsloth-ud-q4_k_xl.json"))
for k in ("model_name", "gpu", "layer_split", "port", "env", "vision", "tokenizer"):
    if k in d:
        print(f"{k} = {json.dumps(d[k], ensure_ascii=False)}")
print("args =", " ".join(d.get("args", [])))
PY

echo
echo "===== 视觉助手 / 编码器文件 ====="
find . -maxdepth 3 -iname '*vision*' -not -path './.git/*' 2>/dev/null | head -8
find /home/<user>/Strata-data -maxdepth 4 -iname '*mmproj*' -printf '%s %p\n' 2>/dev/null | head -5
echo "（上面若有 mmproj 文件即编码器已到位，约 865 MiB）"
