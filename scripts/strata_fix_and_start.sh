#!/bin/bash
# 修回 setup 丢掉的参数 + 启动 + 验证视觉
set -u
cd /home/<user>/Strata
cp strata-unsloth-ud-q4_k_xl.json strata-unsloth-ud-q4_k_xl.json.bak-post-vision-setup

python3 - <<'PY'
import json, collections
p = "strata-unsloth-ud-q4_k_xl.json"
d = json.load(open(p), object_pairs_hook=collections.OrderedDict)

d["args"] = [
 "--pack", "/home/<user>/Strata-data/packs/unsloth-ud-q4_k_xl",
 "--native", "/home/<user>/Strata-data/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
 "--expert-profile", "/home/<user>/Strata/data/expert-profile.bin",
 "--expert-cache", "auto",
 "--prefill", "auto",
 "--spec", "4",
 "--spec-min-p", "0.5",
 "--mtp", "/home/<user>/Strata-data/mtp/rt",
 "--max-context", "262144",
 "--kv", "int8",
 "--kv-resident", "32768",
 "--remote-expert-opt",
 "--conversation-cache-mib", "8192",
 "--pcie-frac", "0.00",
 "--vision",
 "--vram-reserve-mib", "700",
]

out = collections.OrderedDict()
for k, v in d.items():
    out[k] = v
    if k == "layer_split":
        out["host"] = "0.0.0.0"
if "host" not in out:
    out["host"] = "0.0.0.0"
json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
print("配置已写回")
PY

echo "=== 修正后 ==="
python3 -c 'import json;d=json.load(open("strata-unsloth-ud-q4_k_xl.json"));print("host   =",d.get("host"));print("gpu    =",d.get("gpu"));print("env    =",d.get("env"));print("vision =",json.dumps(d.get("vision"),ensure_ascii=False));print("args   ="," ".join(d["args"]))'

echo
echo "=== 启动 ==="
tmux kill-session -t strata-262k 2>/dev/null || true
tmux new-session -d -s strata-262k 'cd /home/<user>/Strata && exec ./run-unsloth-ud-q4_k_xl.sh'
for i in $(seq 1 72); do
  if curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"loaded": true'; then
    echo "=== READY，约 $((i*5)) s ==="
    break
  fi
  sleep 5
done

echo "--- health（看 images 字段）---"
curl -s -m 5 http://127.0.0.1:8080/health; echo
echo "--- GPU 显存 ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- 分层 / 专家槽 ---"
tail -800 /home/<user>/Strata/strata-unsloth-ud-q4_k_xl.log | grep -a -e 'layer split auto: K=15' -e 'expert cache auto' | tail -3
echo "--- 视觉编码器相关日志 ---"
tail -800 /home/<user>/Strata/strata-unsloth-ud-q4_k_xl.log | grep -a -i -e vision -e mmproj | tail -6
echo "--- 日志尾 ---"
tail -4 /home/<user>/Strata/strata-unsloth-ud-q4_k_xl.log
echo "--- 内存 ---"
free -g | head -2
echo "--- 监听地址（应含 0.0.0.0）---"
ss -ltn | grep 8080
