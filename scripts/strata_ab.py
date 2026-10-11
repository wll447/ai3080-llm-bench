#!/usr/bin/env python3
"""ai3080 Strata A/B 实验台：改运行配置 -> 重启 -> 等就绪 -> 跑固定基准 -> 打印关键日志与指标。

用法示例:
  strata_ab.py --label A                                  # 现状: 4 卡 + remote-expert-opt
  strata_ab.py --label B --gpu 0,1,2                       # 只用 3 卡
  strata_ab.py --label C --drop --remote-expert-opt        # 4 卡但不带 remote-expert-opt
  strata_ab.py --label D --gpu 0 --add --expert-cache-device1 4000 ...

基准对所有变体完全一致：
  测试1 256 token 生成（decode 为主）
  测试2 约 2.7 万 token 的长提示 + 64 token（prefill 为主，跨过 prompt chunk 借还的边界）
两端都用 temperature=0，避免共享设置里的采样默认干扰可比性。
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

HOME = pathlib.Path.home()
CFG = HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.json"
LOG = HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.log"
SESSION = "strata-262k"
API = "http://127.0.0.1:8080/v1/chat/completions"

PARA = ("The harbour lights were dim that evening, and the fog rolled in from the north-east in slow grey banks. "
        "Sailors say the beam must sweep the water twice before a boat may turn, and the keeper counts the turns aloud. "
        "Below the tower, the rocks are black and wet, and the tide makes a sound like breathing in a sleeping room. ")
LONG_PROMPT = (PARA * 280) + "Question: answer in one word - what colour is the fog described above?"


def log_lines():
    try:
        return LOG.read_text(errors="replace").splitlines()
    except OSError:
        return []


def apply_cfg(gpu, drops, adds):
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    if gpu:
        cfg["gpu"] = [int(x) for x in gpu.split(",")]
    a = cfg["args"]
    for flag in drops:
        if flag in a:
            i = a.index(flag)
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                del a[i:i + 2]
            else:
                del a[i]
    for flag, val in adds:
        if flag in a:
            a[a.index(flag) + 1] = val
        else:
            a += [flag, val]
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def stop_it():
    for pat in ("serve/server.py", "engine/strata"):
        subprocess.run(["pkill", "-INT", "-f", pat], check=False)
    for _ in range(20):
        if subprocess.run(["pgrep", "-f", "serve/server.py"], capture_output=True).returncode != 0:
            break
        time.sleep(2)
    subprocess.run(["pkill", "-TERM", "-f", "serve/server.py"], check=False)
    subprocess.run(["pkill", "-TERM", "-f", "engine/strata"], check=False)
    time.sleep(3)


def start_it():
    subprocess.run(["tmux", "kill-session", "-t", SESSION], check=False)
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION,
                    "cd %s/Strata && exec ./run-unsloth-ud-q4_k_xl.sh" % HOME], check=False)


def wait_ready(n0, limit_s=600):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        fresh = log_lines()[n0:]
        for line in fresh:
            if "ready: http" in line:
                return time.time() - t0, fresh
        time.sleep(5)
    return None, log_lines()[n0:]


def post(body, timeout=900):
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, time.time() - t0


def bench():
    res = {}
    out, wall = post({"model": "x", "temperature": 0, "max_tokens": 256,
                      "messages": [{"role": "user",
                                    "content": "Write a plain 200-word description of a lighthouse at night."}]})
    res["decode"] = (out["usage"]["completion_tokens"], out["usage"]["prompt_tokens"], wall)
    out, wall = post({"model": "x", "temperature": 0, "max_tokens": 64,
                      "messages": [{"role": "user", "content": LONG_PROMPT}]})
    res["longprompt"] = (out["usage"]["prompt_tokens"], out["usage"]["completion_tokens"], wall,
                         out["choices"][0]["message"]["content"][:60])
    return res


def show(fresh):
    keys = ("prompt chunk auto", "expert cache auto", "layer split auto: K=", "layer split: CUDA",
            "prompts read slower", "expert arena:", "loaded ", "KV streaming:", "other apps use the Chat settings")
    print("--- 启动要点 ---")
    for line in fresh:
        if any(k in line for k in keys):
            print(line[:200])
    print("--- 最近请求（引擎自报） ---")
    for line in log_lines()[-14:]:
        if "strata serve: prompt" in line or "hit rate" in line or "KV streaming" in line:
            print(line[:220])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu")
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--add", nargs=2, action="append", default=[])
    ap.add_argument("--no-restart", action="store_true")
    args = ap.parse_args()

    if not args.no_restart:
        cfg = apply_cfg(args.gpu, args.drop, args.add)
        print("=== 变体 %s: gpu=%s drops=%s adds=%s ===" % (args.label, cfg["gpu"], args.drop, args.add))
        n0 = len(log_lines())
        stop_it()
        start_it()
        secs, fresh = wait_ready(n0)
    else:
        print("=== 变体 %s（不重启，沿用当前实例）===" % args.label)
        fresh = log_lines()[-40:]
        secs = 0.0

    if secs is None:
        print("!!! 600 秒内没有 ready，放弃这个变体")
        print("\n".join(fresh[-20:]))
        return 1
    print("=== 就绪耗时 %.0f s ===" % secs)

    r = bench()
    ct, pt, wall = r["decode"]
    print("测试1 decode: completion=%d prompt=%d wall=%.2fs -> %.1f tok/s(含TTFT)" % (ct, pt, wall, ct / wall))
    lpt, lct, lwall, ans = r["longprompt"]
    print("测试2 长提示: prompt=%d completion=%d wall=%.2fs -> prompt %.1f tok/s / 端到端 %.1f tok/s answer=[%s]"
          % (lpt, lct, lwall, lpt / lwall, (lpt + lct) / lwall, ans))
    time.sleep(3)
    show(fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
